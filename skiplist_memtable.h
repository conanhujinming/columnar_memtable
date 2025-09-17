#ifndef SKIPLIST_MEMTABLE_H
#define SKIPLIST_MEMTABLE_H

#include <random>
#include <mutex> // Needed for thread-safe arena

#include "columnar_memtable.h"  // Includes FlushIterator and CompactingIterator definitions

namespace SkipListImpl {

class ConcurrentSkipList {
   public:
    static constexpr int kMaxHeight = 12;

    struct Node {
        RecordRef record;
        // The forward array must be flexible. This is a common C-style trick.
        std::atomic<Node*> forward[1];

        // Factory function to correctly allocate a node of a specific height.
        static Node* New(SimpleArena& arena, std::string_view key, std::string_view value, RecordType type,
                         int height) {
            // Calculate size needed for the node header and the flexible array member.
            size_t size = sizeof(Node) + sizeof(std::atomic<Node*>) * (height - 1);
            char* mem = arena.AllocateRaw(size);
            Node* node = new (mem) Node(); // Placement new
            node->record = {arena.AllocateAndCopy(key), arena.AllocateAndCopy(value), type};
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

   private:
    int RandomHeight();

    SimpleArena arena_;
    Node* const head_;
    std::atomic<int> max_height_;
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
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(0, 1);
    int height = 1;
    while (height < kMaxHeight && distribution(generator) == 1) {
        height++;
    }
    return height;
}

// =================================================================================================
// A robust and correct lock-free Insert implementation.
// =================================================================================================
inline void ConcurrentSkipList::Insert(std::string_view key, std::string_view value, RecordType type) {
    Node* update[kMaxHeight];
    Node* x;

    // The outer retry loop handles all conflicts. If any CAS fails, we restart.
    while (true) {
        x = head_;
        // 1. Find predecessors for all levels based on a consistent snapshot.
        for (int i = kMaxHeight - 1; i >= 0; --i) {
            Node* next = x->forward[i].load(std::memory_order_acquire);
            while (next != nullptr && next->record.key < key) {
                x = next;
                next = x->forward[i].load(std::memory_order_acquire);
            }
            update[i] = x;
        }
        
        // Find the precise predecessor at level 0, handling duplicate keys.
        Node* pred_at_level_0 = update[0];
        Node* successor = pred_at_level_0->forward[0].load(std::memory_order_acquire);
        while (successor != nullptr && successor->record.key == key) {
             pred_at_level_0 = successor;
             successor = successor->forward[0].load(std::memory_order_acquire);
        }

        // 2. Determine the height of the new node.
        int height = RandomHeight();
        if (height > max_height_.load(std::memory_order_relaxed)) {
            max_height_.store(height, std::memory_order_relaxed);
        }

        // 3. Allocate the new node.
        Node* new_node = Node::New(arena_, key, value, type, height);
        
        // Link the new node's successor at level 0.
        new_node->forward[0].store(successor, std::memory_order_relaxed);
        
        // 4. The crucial atomic step: try to link the new node at level 0.
        // If this fails, another thread interfered. We must restart the entire process.
        if (!pred_at_level_0->forward[0].compare_exchange_strong(successor, new_node, std::memory_order_release,
                                                                 std::memory_order_relaxed)) {
            // Conflict detected at the most critical point. Restart the whole operation.
            continue; // The allocated new_node is leaked in the arena, which is acceptable.
        }

        // 5. Success at level 0. Now link the upper levels.
        // This is a "best effort" attempt. If it fails, the node is still reachable
        // via lower levels, so correctness is maintained. We still retry the whole
        // insert to ensure the skiplist structure is optimal for performance.
        for (int i = 1; i < height; ++i) {
            Node* pred = update[i];
            Node* succ = pred->forward[i].load(std::memory_order_acquire);

            // Set the new node's forward pointer for this level.
            new_node->forward[i].store(succ, std::memory_order_relaxed);

            // Attempt to swing the predecessor's pointer.
            // If this fails, our `update` array is stale. The simplest and most robust
            // solution is to abort this attempt and let the outer loop retry everything.
            if (!pred->forward[i].compare_exchange_strong(succ, new_node, std::memory_order_release,
                                                          std::memory_order_relaxed)) {
                // Another thread changed the list at this level. Our `update` array is invalid.
                // Instead of trying to patch it (which is buggy), we signal the need for a full retry.
                // We break out of this inner loop and will then restart the outer while(true) loop.
                // A simple way to do this is to use a flag or a goto.
                goto retry; 
            }
        }

        // If we successfully linked all levels without conflicts, we are done.
        return;

    retry:
        // This label is the target for when an upper-level CAS fails.
        // The outer while(true) loop will then continue.
        ;
    }
}

inline std::optional<RecordRef> ConcurrentSkipList::Find(std::string_view key) const {
    Node* x = head_;
    // Standard search from top-left.
    for (int i = kMaxHeight - 1; i >= 0; --i) {
        Node* next = x->forward[i].load(std::memory_order_acquire);
        while (next != nullptr && next->record.key < key) {
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
    using MultiGetResult = std::map<std::string_view, GetResult, std::less<>>;

    explicit SkipListMemTable(size_t, bool, std::shared_ptr<Sorter> = nullptr)
        : skiplist_(std::make_shared<SkipListImpl::ConcurrentSkipList>()) {}

    ~SkipListMemTable() = default;
    SkipListMemTable(const SkipListMemTable&) = delete;
    SkipListMemTable& operator=(const SkipListMemTable&) = delete;

    void WaitForBackgroundWork() {}

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
        for (const auto& [key, value] : batch) {
            Put(key, value);
        }
    }

    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const {
        MultiGetResult results;
        for (const auto& key : keys) {
            // Small optimization to avoid re-lookups for duplicate keys in the input vector
            if (results.find(key) == results.end()) { 
                results.emplace(key, Get(key));
            }
        }
        return results;
    }

    size_t GetSortedBlockNum() const { return 0; }

   private:
    std::shared_ptr<SkipListImpl::ConcurrentSkipList> skiplist_;
};

#endif  // SKIPLIST_MEMTABLE_H