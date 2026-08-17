#ifndef SKIPLIST_MEMTABLE_H
#define SKIPLIST_MEMTABLE_H

#include <random>
#include <mutex>

#include "columnar_memtable.h"  // Includes FlushIterator and CompactingIterator definitions

#include <atomic>
#include <vector>
#include <memory>
#include <algorithm>

// A thread-safe, lock-free arena for concurrent allocations.
class ConcurrentArena {
public:
    static constexpr size_t kDefaultBlockSize = 64 * 1024;

    ConcurrentArena()
        : current_block_(new Block(kDefaultBlockSize)), allocated_bytes_(kDefaultBlockSize) {}

    ~ConcurrentArena() {
        // The linked-list of blocks will be cleaned up automatically by unique_ptr.
        // We just need to delete the head of the list.
        Block* block = current_block_.load(std::memory_order_relaxed);
        while (block != nullptr) {
            Block* next = block->next.load(std::memory_order_relaxed);
            delete block;
            block = next;
        }
    }

    // No copying or moving.
    ConcurrentArena(const ConcurrentArena&) = delete;
    ConcurrentArena& operator=(const ConcurrentArena&) = delete;

    char* AllocateRaw(size_t bytes) {
        // Add padding for alignment. A common practice.
        const size_t align = alignof(std::max_align_t);
        bytes = (bytes + align - 1) & ~(align - 1);

        while (true) {
            Block* current = current_block_.load(std::memory_order_acquire);
            size_t old_pos = current->pos.fetch_add(bytes, std::memory_order_relaxed);

            if (old_pos + bytes <= current->size) {
                // Success! We found space in the current block.
                return current->data.get() + old_pos;
            } else {
                // The current block is full. We need to allocate a new one.
                // It's possible multiple threads notice this at the same time.
                // Only one will succeed in replacing the current_block_.
                
                // Roll back the fetch_add, though this is not strictly necessary
                // as the wasted space is at the end of a full block.
                // current->pos.fetch_sub(bytes, std::memory_order_relaxed);

                size_t new_block_size = std::max(bytes, kDefaultBlockSize);
                Block* new_block = new Block(new_block_size);
                new_block->next.store(current, std::memory_order_relaxed);
                
                // Try to swap the current block with our new one.
                // If another thread already swapped it, `current` will be stale,
                // and the CAS will fail. In that case, we just loop again
                // and try to allocate from the new block installed by the other thread.
                if (current_block_.compare_exchange_strong(current, new_block, 
                                                           std::memory_order_release,
                                                           std::memory_order_acquire)) {
                    allocated_bytes_.fetch_add(new_block_size, std::memory_order_relaxed);
                } else {
                    // Another thread won the race. Delete the block we allocated but didn't use.
                    delete new_block;
                }
                // In either case (CAS success or failure), we retry the allocation in the next loop iteration.
            }
        }
    }

    std::string_view AllocateAndCopy(std::string_view data) {
        char* mem = AllocateRaw(data.size());
        memcpy(mem, data.data(), data.size());
        return {mem, data.size()};
    }

    size_t ApproximateMemoryUsage() const { return allocated_bytes_.load(std::memory_order_relaxed); }

private:
    struct Block {
        std::unique_ptr<char[]> data;
        const size_t size;
        std::atomic<size_t> pos;
        // Blocks are stored as a singly-linked list for cleanup.
        std::atomic<Block*> next; 

        explicit Block(size_t s) : data(new char[s]), size(s), pos(0), next(nullptr) {}
    };

    // The head of the block list, where allocations happen.
    // This is the main point of contention, handled by atomics.
    std::atomic<Block*> current_block_;
    std::atomic<size_t> allocated_bytes_;
};

namespace SkipListImpl {

class ConcurrentSkipList {
   public:
    static constexpr int kMaxHeight = 12;

    struct Node {
        RecordRef record;
        uint64_t key_prefix;
        // The forward array must be flexible. This is a common C-style trick.
        std::atomic<Node*> forward[1];

        // Factory function to correctly allocate a node of a specific height.
        static Node* New(ConcurrentArena& arena, std::string_view key, std::string_view value, RecordType type,
                         int height) {
            const size_t node_size = sizeof(Node) + sizeof(std::atomic<Node*>) * (height - 1);
            const size_t aligned_node_size =
                (node_size + alignof(std::max_align_t) - 1) & ~(alignof(std::max_align_t) - 1);
            char* mem = arena.AllocateRaw(aligned_node_size + key.size() + value.size());
            Node* node = new (mem) Node();
            char* key_mem = mem + aligned_node_size;
            if (!key.empty()) memcpy(key_mem, key.data(), key.size());
            char* value_mem = key_mem + key.size();
            if (!value.empty()) memcpy(value_mem, value.data(), value.size());
            node->record = {{key_mem, key.size()}, {value_mem, value.size()}, type};
            node->key_prefix = load_u64_prefix(key);
            // Initialize forward pointers to null.
            for (int i = 0; i < height; ++i) {
                node->forward[i].store(nullptr, std::memory_order_relaxed);
            }
            return node;
        }
    };

    class Iterator;

    explicit ConcurrentSkipList() 
        : head_(Node::New(arena_, "", "", RecordType::Put, kMaxHeight)), 
          max_height_(1) {}

    void Insert(std::string_view key, std::string_view value, RecordType type);
    std::optional<RecordRef> Find(std::string_view key) const;
    Iterator begin() const;
    size_t ApproximateMemoryUsage() const { return arena_.ApproximateMemoryUsage(); }
    size_t size() const { return size_.load(std::memory_order_relaxed); }

   private:
    int RandomHeight();
    void FindInsertionSplice(std::string_view key, uint64_t key_prefix, Node** predecessors, Node** successors) const;
    static bool KeyIsBefore(const Node* node, uint64_t prefix, std::string_view key) {
        return node->key_prefix < prefix || (node->key_prefix == prefix && node->record.key < key);
    }
    static bool KeyIsBeforeOrEqual(const Node* node, uint64_t prefix, std::string_view key) {
        return node->key_prefix < prefix || (node->key_prefix == prefix && node->record.key <= key);
    }

    ConcurrentArena arena_;
    Node* const head_;
    std::atomic<int> max_height_;
    std::atomic<size_t> size_{0};
};

class ConcurrentSkipList::Iterator {
   public:
    explicit Iterator(const Node* node) : node_(node) {}
    RecordRef operator*() const { return node_->record; }
    void Next() {
        if (node_) node_ = node_->forward[0].load(std::memory_order_acquire);
    }
    bool IsValid() const { return node_ != nullptr; }

   private:
    const Node* node_;
};

inline ConcurrentSkipList::Iterator ConcurrentSkipList::begin() const {
    return Iterator(head_->forward[0].load(std::memory_order_acquire));
}

inline int ConcurrentSkipList::RandomHeight() {
    static thread_local uint64_t random_state = [] {
        std::random_device device;
        uint64_t seed = (static_cast<uint64_t>(device()) << 32) ^ device();
        return seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
    }();
    auto next_random = [&] {
        random_state ^= random_state >> 12;
        random_state ^= random_state << 25;
        random_state ^= random_state >> 27;
        return random_state * 0x2545f4914f6cdd1dULL;
    };

    // A branching factor of four is a better match for a 12-level list than
    // the old 1/2 probability, which left hundreds of nodes on the top level
    // at 500k entries.
    int height = 1;
    while (height < kMaxHeight && (next_random() & 3U) == 0) ++height;
    return height;
}

inline void ConcurrentSkipList::FindInsertionSplice(std::string_view key, uint64_t key_prefix, Node** predecessors,
                                                     Node** successors) const {
    Node* current = head_;
    const int current_height = max_height_.load(std::memory_order_acquire);
    for (int level = current_height - 1; level >= 0; --level) {
        Node* next = current->forward[level].load(std::memory_order_acquire);
        while (next != nullptr && KeyIsBeforeOrEqual(next, key_prefix, key)) {
            current = next;
            next = current->forward[level].load(std::memory_order_acquire);
        }
        predecessors[level] = current;
        successors[level] = next;
    }
    for (int level = current_height; level < kMaxHeight; ++level) {
        predecessors[level] = head_;
        successors[level] = head_->forward[level].load(std::memory_order_acquire);
    }
}

inline void ConcurrentSkipList::Insert(std::string_view key, std::string_view value, RecordType type) {
    const int height = RandomHeight();
    Node* new_node = Node::New(arena_, key, value, type, height);
    const uint64_t key_prefix = new_node->key_prefix;
    Node* predecessors[kMaxHeight];
    Node* successors[kMaxHeight];

    // Linearize the insertion exactly once at level zero.  The old code
    // allocated and inserted a second logical record whenever an upper-level
    // CAS failed.
    while (true) {
        FindInsertionSplice(key, key_prefix, predecessors, successors);
        for (int level = 0; level < height; ++level) {
            new_node->forward[level].store(successors[level], std::memory_order_relaxed);
        }
        Node* expected = successors[0];
        if (predecessors[0]->forward[0].compare_exchange_weak(expected, new_node, std::memory_order_release,
                                                              std::memory_order_acquire)) {
            break;
        }
    }
    size_.fetch_add(1, std::memory_order_relaxed);

    int observed_height = max_height_.load(std::memory_order_relaxed);
    while (observed_height < height &&
           !max_height_.compare_exchange_weak(observed_height, height, std::memory_order_release,
                                              std::memory_order_relaxed)) {
    }

    // No nodes are removed, so the node is safe to publish one upper level at
    // a time.  A failed CAS only requires recomputing that level's splice.
    for (int level = 1; level < height; ++level) {
        while (true) {
            new_node->forward[level].store(successors[level], std::memory_order_relaxed);
            Node* expected = successors[level];
            if (predecessors[level]->forward[level].compare_exchange_weak(
                    expected, new_node, std::memory_order_release, std::memory_order_acquire)) {
                break;
            }
            FindInsertionSplice(key, key_prefix, predecessors, successors);
        }
    }
}

inline std::optional<RecordRef> ConcurrentSkipList::Find(std::string_view key) const {
    Node* x = head_;
    const uint64_t key_prefix = load_u64_prefix(key);
    // Standard search from top-left.
    for (int i = max_height_.load(std::memory_order_acquire) - 1; i >= 0; --i) {
        Node* next = x->forward[i].load(std::memory_order_acquire);
        while (next != nullptr && KeyIsBefore(next, key_prefix, key)) {
            x = next;
            next = x->forward[i].load(std::memory_order_acquire);
        }
    }

    // Now x is the predecessor of the first node with a key >= `key`.
    // We traverse at level 0 to find the latest version.
    Node* current = x->forward[0].load(std::memory_order_acquire);
    std::optional<RecordRef> last_found = std::nullopt;
    while (current != nullptr && current->record.key == key) {
        last_found = current->record;
        current = current->forward[0].load(std::memory_order_acquire);
    }
    return last_found;
}

class SkipListFlushIterator {
   public:
    explicit SkipListFlushIterator(std::shared_ptr<const ConcurrentSkipList> source)
        : iter_(source ? source->begin() : ConcurrentSkipList::Iterator(nullptr)) {}

    bool IsValid() const { return iter_.IsValid(); }
    RecordRef Get() const { return *iter_; }
    void Next() { iter_.Next(); }

   private:
    ConcurrentSkipList::Iterator iter_;
};
}  // namespace SkipListImpl

class SkipListMemTable {
   public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::vector<GetResult>;

    explicit SkipListMemTable(size_t, bool, std::shared_ptr<Sorter> = nullptr, size_t batch_worker_count = 1)
        : skiplist_(std::make_shared<SkipListImpl::ConcurrentSkipList>()),
          batch_worker_count_(std::max<size_t>(1, batch_worker_count)),
          batch_pool_(batch_worker_count_ > 1 ? std::make_shared<ThreadPool>(batch_worker_count_) : nullptr) {}

    ~SkipListMemTable() = default;
    SkipListMemTable(const SkipListMemTable&) = delete;
    SkipListMemTable& operator=(const SkipListMemTable&) = delete;

    void WaitForBackgroundWork() {}
    void WaitForPendingBackgroundWork() {}
    void WaitForBackgroundBacklogAtMost(size_t) {}

    std::unique_ptr<CompactingIterator> NewCompactingIterator() {
        auto raw_iter = std::make_unique<SkipListImpl::SkipListFlushIterator>(skiplist_);
        return std::make_unique<CompactingIterator>(std::move(raw_iter));
    }

    void Put(std::string_view key, std::string_view value) { skiplist_->Insert(key, value, RecordType::Put); }

    void Delete(std::string_view key) { skiplist_->Insert(key, "", RecordType::Delete); }

    GetResult Get(std::string_view key) const {
        auto result = skiplist_->Find(key);
        if (result.has_value()) {
            return (result->type == RecordType::Put) ? GetResult(result->value) : std::nullopt;
        }
        return std::nullopt;
    }

    void PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
        if (!batch_pool_ || batch.size() < 2) {
            for (const auto& [key, value] : batch) Put(key, value);
            return;
        }

        const size_t task_count = std::min(batch_worker_count_, batch.size());
        const size_t items_per_task = (batch.size() + task_count - 1) / task_count;
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (size_t task_idx = 0; task_idx < task_count; ++task_idx) {
            const size_t begin = task_idx * items_per_task;
            const size_t end = std::min(batch.size(), begin + items_per_task);
            if (begin == end) break;
            futures.emplace_back(batch_pool_->Submit([this, &batch, begin, end] {
                for (size_t i = begin; i < end; ++i) Put(batch[i].first, batch[i].second);
            }));
        }
        for (auto& future : futures) {
            future.get();
        }
    }

    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const {
        MultiGetResult results(keys.size());
        auto get_range = [this, &keys, &results](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                results[i] = Get(keys[i]);
            }
        };

        if (!batch_pool_ || keys.size() < 2) {
            get_range(0, keys.size());
            return results;
        }

        const size_t task_count = std::min(batch_worker_count_, keys.size());
        const size_t items_per_task = (keys.size() + task_count - 1) / task_count;
        std::vector<std::future<void>> futures;
        futures.reserve(task_count);
        for (size_t task_idx = 0; task_idx < task_count; ++task_idx) {
            const size_t begin = task_idx * items_per_task;
            const size_t end = std::min(keys.size(), begin + items_per_task);
            if (begin == end) break;
            futures.emplace_back(batch_pool_->Submit([get_range, begin, end] { get_range(begin, end); }));
        }

        for (auto& future : futures) {
            future.get();
        }
        return results;
    }

    size_t GetSortedBlockNum() const { return skiplist_->size() == 0 ? 0 : 1; }
    size_t GetSortedRecordCount() const { return skiplist_->size(); }
    size_t GetActiveRecordCount() const { return 0; }
    size_t GetPendingSealedBlockNum() const { return 0; }
    size_t GetPendingSealedRecordCount() const { return 0; }
    size_t ApproximateMemoryUsage() const { return skiplist_->ApproximateMemoryUsage(); }
    size_t BatchWorkerCount() const { return batch_worker_count_; }

   private:
    std::shared_ptr<SkipListImpl::ConcurrentSkipList> skiplist_;
    const size_t batch_worker_count_;
    std::shared_ptr<ThreadPool> batch_pool_;
};

#endif  // SKIPLIST_MEMTABLE_H
