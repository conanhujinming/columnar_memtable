#ifndef COLUMNAR_MEMTABLE_H
#define COLUMNAR_MEMTABLE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>  // For the thread ID pool
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#define XXH_INLINE_ALL
#include "xxhash.h"

// --- Forward Declarations ---
enum class RecordType;
struct RecordRef;
class ColumnarBlock;
class Sorter;
class SortedColumnarBlock;
class FlushIterator;
class CompactingIterator;
class FlashActiveBlock;
class BloomFilter;
class ColumnarRecordArena;
class ColumnarMemTable;

// --- Core Utility Structures ---
struct XXHasher {
    std::size_t operator()(const std::string_view key) const noexcept { return XXH3_64bits(key.data(), key.size()); }
};

class SpinLock {
   public:
    void lock() noexcept {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                return;
            }
            while (lock_.load(std::memory_order_relaxed)) {
                __builtin_ia32_pause();
            }
        }
    }
    void unlock() noexcept { lock_.store(false, std::memory_order_release); }

   private:
    std::atomic<bool> lock_ = {false};
};

inline uint64_t load_u64_prefix(std::string_view sv) {
    if (sv.size() >= 8) {
        uint64_t prefix;
        memcpy(&prefix, sv.data(), 8);
        return prefix;
    }
    uint64_t prefix = 0;
    if (!sv.empty()) {
        memcpy(&prefix, sv.data(), sv.size());
    }
    return prefix;
}

enum class RecordType { Put, Delete };
struct RecordRef {
    std::string_view key;
    std::string_view value;
    RecordType type;
};

// --- Bloom Filter ---
class BloomFilter {
   public:
    explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01);
    void Add(std::string_view key);
    bool MayContain(std::string_view key) const;
    size_t ApproximateMemoryUsage() const { return bits_.capacity() / 8; }

   private:
    static std::array<uint64_t, 2> Hash(std::string_view key);
    std::vector<bool> bits_;
    int num_hashes_;
};
inline BloomFilter::BloomFilter(size_t n, double p) {
    if (n == 0) n = 1;
    size_t m = -1.44 * n * std::log(p);
    bits_ = std::vector<bool>((m + 7) & ~7, false);
    num_hashes_ = 0.7 * (double(bits_.size()) / n);
    if (num_hashes_ < 1) num_hashes_ = 1;
    if (num_hashes_ > 8) num_hashes_ = 8;
}
inline void BloomFilter::Add(std::string_view key) {
    auto h = Hash(key);
    for (int i = 0; i < num_hashes_; ++i) {
        uint64_t hash = h[0] + i * h[1];
        if (!bits_.empty()) bits_[hash % bits_.size()] = true;
    }
}
inline bool BloomFilter::MayContain(std::string_view key) const {
    if (bits_.empty()) return true;
    auto h = Hash(key);
    for (int i = 0; i < num_hashes_; ++i) {
        uint64_t hash = h[0] + i * h[1];
        if (!bits_[hash % bits_.size()]) return false;
    }
    return true;
}
inline std::array<uint64_t, 2> BloomFilter::Hash(std::string_view key) {
    XXH128_hash_t const hash_val = XXH3_128bits(key.data(), key.size());
    return {hash_val.low64, hash_val.high64};
}

// --- Columnar MemTable Components ---
struct StoredRecord {
    RecordRef record;
    std::atomic<bool> ready{false};
};

// Manages thread IDs, allowing for efficient recycling to keep the ID range small.
class ThreadIdManager {
   public:
    static constexpr size_t kMaxThreads = 256;

    static uint32_t GetId() {
        thread_local ThreadIdRecycler instance;
        return instance.id;
    }

   private:
    struct ThreadIdRecycler {
        uint32_t id;
        ThreadIdRecycler() {
            std::lock_guard<SpinLock> lock(pool_lock_);
            if (!recycled_ids_.empty()) {
                id = recycled_ids_.front();
                recycled_ids_.pop_front();
            } else {
                id = next_id_.fetch_add(1, std::memory_order_relaxed);
                if (id >= kMaxThreads) {
                    next_id_.fetch_sub(1, std::memory_order_relaxed);
                    throw std::runtime_error("Exceeded kMaxThreads. Increase the compile-time constant.");
                }
            }
        }
        ~ThreadIdRecycler() {
            std::lock_guard<SpinLock> lock(pool_lock_);
            recycled_ids_.push_back(id);
        }
    };
    // Use inline static to define and initialize in header for C++17+
    static inline std::atomic<uint32_t> next_id_{0};
    static inline std::deque<uint32_t> recycled_ids_;
    static inline SpinLock pool_lock_;
};

// Thread-local arena for storing record data (keys and values) efficiently.
// Data is allocated in chunks to reduce fragmentation and contention.
class ColumnarRecordArena {
   private:
    friend class ColumnarMemTable;
    struct DataChunk {
        static constexpr size_t kRecordCapacity = 256;
        static constexpr size_t kBufferCapacity = 32 * 1024;
        // Combine record index and buffer position into one atomic to prevent races.
        // High 32 bits for record index, low 32 bits for buffer position.
        std::atomic<uint64_t> positions_{0};
        std::array<StoredRecord, kRecordCapacity> records;
        alignas(64) char buffer[kBufferCapacity];
    };

    struct alignas(64) ThreadLocalData {
        std::vector<std::unique_ptr<DataChunk>> chunks;
        DataChunk* current_chunk = nullptr;
        // Add a lock to prevent multiple threads from switching the chunk for the same TLS simultaneously.
        SpinLock chunk_switch_lock;
        ThreadLocalData() { AddNewChunk(); }
        void AddNewChunk() {
            chunks.push_back(std::make_unique<DataChunk>());
            current_chunk = chunks.back().get();
        }
    };

   public:
    ColumnarRecordArena();
    ~ColumnarRecordArena();
    const StoredRecord* AllocateAndAppend(std::string_view key, std::string_view value, RecordType type);
    std::vector<const StoredRecord*> AllocateAndAppendBatch(
        const std::vector<std::pair<std::string_view, std::string_view>>& batch, RecordType type);
    size_t size() const { return size_.load(std::memory_order_acquire); }
    size_t ApproximateMemoryUsage() const { return memory_usage_.load(std::memory_order_relaxed); }
    uint32_t GetMaxThreadIdSeen() const { return max_tid_seen_.load(std::memory_order_acquire); }
    const std::array<std::atomic<ThreadLocalData*>, ThreadIdManager::kMaxThreads>& GetAllTlsData() const {
        return all_tls_data_;
    }

   private:
    ThreadLocalData* GetTlsData();

    std::array<std::atomic<ThreadLocalData*>, ThreadIdManager::kMaxThreads> all_tls_data_{};
    std::vector<ThreadLocalData*> owned_tls_data_;
    SpinLock owner_lock_;
    std::atomic<size_t> size_;
    std::atomic<size_t> memory_usage_{0};
    std::atomic<uint32_t> max_tid_seen_{0};
};

inline ColumnarRecordArena::ColumnarRecordArena() : size_(0) {}
inline ColumnarRecordArena::~ColumnarRecordArena() {
    std::lock_guard<SpinLock> lock(owner_lock_);
    for (auto* ptr : owned_tls_data_) {
        delete ptr;
    }
}
inline ColumnarRecordArena::ThreadLocalData* ColumnarRecordArena::GetTlsData() {
    uint32_t tid = ThreadIdManager::GetId();
    uint32_t current_max = max_tid_seen_.load(std::memory_order_relaxed);
    while (tid > current_max) {
        if (max_tid_seen_.compare_exchange_weak(current_max, tid, std::memory_order_release,
                                                std::memory_order_relaxed)) {
            break;
        }
    }

    ThreadLocalData* my_data = all_tls_data_[tid].load(std::memory_order_acquire);
    if (my_data == nullptr) {
        auto* new_data = new ThreadLocalData();
        ThreadLocalData* expected_null = nullptr;
        if (all_tls_data_[tid].compare_exchange_strong(expected_null, new_data, std::memory_order_release,
                                                       std::memory_order_acquire)) {
            std::lock_guard<SpinLock> lock(owner_lock_);
            owned_tls_data_.push_back(new_data);
            my_data = new_data;
            memory_usage_.fetch_add(sizeof(ThreadLocalData) + sizeof(DataChunk), std::memory_order_relaxed);
        } else {
            delete new_data;
            my_data = expected_null;
        }
    }
    return my_data;
}

// Rewritten AllocateAndAppend with atomic 64-bit CAS for race-free allocation.
// The logic now internally handles chunk switches, simplifying the caller.
inline const StoredRecord* ColumnarRecordArena::AllocateAndAppend(std::string_view key, std::string_view value,
                                                                  RecordType type) {
    ThreadLocalData* tls_data = GetTlsData();
    size_t required_size = key.size() + value.size();

    // Prevent allocation of single items larger than the buffer.
    if (required_size > DataChunk::kBufferCapacity) {
        return nullptr;
    }

    uint32_t record_idx;
    uint32_t buffer_offset;

    while (true) {
        DataChunk* chunk = tls_data->current_chunk;

        uint64_t old_pos = chunk->positions_.load(std::memory_order_relaxed);
        while (true) {
            uint32_t old_ridx = static_cast<uint32_t>(old_pos >> 32);
            uint32_t old_bpos = static_cast<uint32_t>(old_pos);

            if (old_ridx >= DataChunk::kRecordCapacity || old_bpos + required_size > DataChunk::kBufferCapacity) {
                break;  // Chunk is full, need to switch.
            }

            uint64_t new_pos = (static_cast<uint64_t>(old_ridx + 1) << 32) | (old_bpos + required_size);

            if (chunk->positions_.compare_exchange_weak(old_pos, new_pos, std::memory_order_acq_rel)) {
                record_idx = old_ridx;
                buffer_offset = old_bpos;
                goto allocation_success;  // Exit both loops
            }
            // CAS failed, another thread updated positions_. Retry with the new old_pos.
        }

        // If we're here, the chunk is full. Acquire lock to switch it.
        std::lock_guard<SpinLock> lock(tls_data->chunk_switch_lock);
        // Re-check if another thread already switched the chunk while we waited for the lock.
        if (chunk == tls_data->current_chunk) {
            tls_data->AddNewChunk();
            memory_usage_.fetch_add(sizeof(DataChunk), std::memory_order_relaxed);
        }
        // Loop again to try allocating in the new chunk.
    }

allocation_success:
    DataChunk* final_chunk = tls_data->current_chunk;  // The chunk where allocation succeeded
    char* key_mem = final_chunk->buffer + buffer_offset;
    memcpy(key_mem, key.data(), key.size());
    char* val_mem = key_mem + key.size();
    memcpy(val_mem, value.data(), value.size());

    StoredRecord& record_slot = final_chunk->records[record_idx];
    record_slot.record = {{key_mem, key.size()}, {val_mem, value.size()}, type};
    record_slot.ready.store(true, std::memory_order_release);

    size_.fetch_add(1, std::memory_order_release);
    memory_usage_.fetch_add(required_size + sizeof(StoredRecord), std::memory_order_relaxed);
    return &record_slot;
}

// Rewritten batch allocation with the same robust CAS-based approach.
inline std::vector<const StoredRecord*> ColumnarRecordArena::AllocateAndAppendBatch(
    const std::vector<std::pair<std::string_view, std::string_view>>& batch, RecordType type) {
    std::vector<const StoredRecord*> results;
    if (batch.empty()) return results;
    results.reserve(batch.size());

    ThreadLocalData* tls_data = GetTlsData();
    size_t batch_offset = 0;

    while (batch_offset < batch.size()) {
        DataChunk* chunk = tls_data->current_chunk;

        uint64_t old_pos = chunk->positions_.load(std::memory_order_relaxed);
        uint32_t allocated_record_idx = 0;
        uint32_t allocated_buffer_pos = 0;
        uint32_t records_to_alloc = 0;
        size_t buffer_needed = 0;

        while (true) {  // CAS loop
            uint32_t old_ridx = static_cast<uint32_t>(old_pos >> 32);
            uint32_t old_bpos = static_cast<uint32_t>(old_pos);

            records_to_alloc = 0;
            buffer_needed = 0;
            for (size_t i = batch_offset; i < batch.size(); ++i) {
                const auto& [key, value] = batch[i];
                size_t item_size = key.size() + value.size();
                if (old_ridx + records_to_alloc < DataChunk::kRecordCapacity &&
                    old_bpos + buffer_needed + item_size <= DataChunk::kBufferCapacity) {
                    records_to_alloc++;
                    buffer_needed += item_size;
                } else {
                    break;
                }
            }

            if (records_to_alloc == 0) {
                break;  // Not enough space for even one item, need to switch chunk.
            }

            uint64_t new_pos = (static_cast<uint64_t>(old_ridx + records_to_alloc) << 32) | (old_bpos + buffer_needed);

            if (chunk->positions_.compare_exchange_weak(old_pos, new_pos, std::memory_order_acq_rel)) {
                allocated_record_idx = old_ridx;
                allocated_buffer_pos = old_bpos;
                goto batch_allocation_success;
            }
        }

        // If we're here, the chunk is full for this sub-batch.
        {
            std::lock_guard<SpinLock> lock(tls_data->chunk_switch_lock);
            if (chunk == tls_data->current_chunk) {
                // Handle oversized items that will never fit.
                const auto& [key, value] = batch[batch_offset];
                if (key.size() + value.size() > DataChunk::kBufferCapacity) {
                    batch_offset++;  // Skip this item and continue.
                    continue;
                }
                tls_data->AddNewChunk();
                memory_usage_.fetch_add(sizeof(DataChunk), std::memory_order_relaxed);
            }
            continue;  // Retry with the new chunk.
        }

    batch_allocation_success:
        size_t current_buffer_offset_in_batch = 0;
        for (uint32_t i = 0; i < records_to_alloc; ++i) {
            const auto& [key, value] = batch[batch_offset + i];
            size_t item_size = key.size() + value.size();

            char* key_mem = chunk->buffer + allocated_buffer_pos + current_buffer_offset_in_batch;
            memcpy(key_mem, key.data(), key.size());
            char* val_mem = key_mem + key.size();
            memcpy(val_mem, value.data(), value.size());

            StoredRecord& record_slot = chunk->records[allocated_record_idx + i];
            record_slot.record = {{key_mem, key.size()}, {val_mem, value.size()}, type};
            record_slot.ready.store(true, std::memory_order_release);

            results.push_back(&record_slot);
            current_buffer_offset_in_batch += item_size;
        }

        size_.fetch_add(records_to_alloc, std::memory_order_release);
        memory_usage_.fetch_add(buffer_needed + records_to_alloc * sizeof(StoredRecord), std::memory_order_relaxed);
        batch_offset += records_to_alloc;
    }

    return results;
}

class ConcurrentStringHashMap {
   public:
    static constexpr uint8_t EMPTY_TAG = 0xFF, LOCKED_TAG = 0xFE;

   private:
    struct alignas(64) Slot {
        std::atomic<uint8_t> tag;
        uint64_t full_hash;
        std::string_view key;
        std::atomic<const StoredRecord*> record;
    };
    std::unique_ptr<Slot[]> slots_;
    size_t capacity_, capacity_mask_;
    XXHasher hasher_;

   public:
    ConcurrentStringHashMap(const ConcurrentStringHashMap&) = delete;
    ConcurrentStringHashMap& operator=(const ConcurrentStringHashMap&) = delete;
    static size_t calculate_power_of_2(size_t n) { return n == 0 ? 1 : 1UL << (64 - __builtin_clzll(n - 1)); }
    explicit ConcurrentStringHashMap(size_t build_size);
    void Insert(std::string_view key, const StoredRecord* new_record);
    const StoredRecord* Find(std::string_view key) const;
};

inline ConcurrentStringHashMap::ConcurrentStringHashMap(size_t build_size) {
    size_t capacity = calculate_power_of_2(build_size * 1.5 + 64);
    capacity_ = capacity;
    capacity_mask_ = capacity - 1;
    slots_ = std::make_unique<Slot[]>(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
        slots_[i].record.store(nullptr, std::memory_order_relaxed);
    }
}
inline void ConcurrentStringHashMap::Insert(std::string_view key, const StoredRecord* new_record) {
    uint64_t hash = hasher_(key);
    uint8_t tag = (hash >> 56);
    if (tag >= LOCKED_TAG) tag = 0;
    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;
    while (true) {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
        if (current_tag == tag && slots_[pos].full_hash == hash && slots_[pos].key == key) {
            slots_[pos].record.store(new_record, std::memory_order_release);
            return;
        }
        if (current_tag == EMPTY_TAG) {
            uint8_t expected_empty = EMPTY_TAG;
            if (slots_[pos].tag.compare_exchange_strong(expected_empty, LOCKED_TAG, std::memory_order_acq_rel)) {
                slots_[pos].key = key;
                slots_[pos].full_hash = hash;
                slots_[pos].record.store(new_record, std::memory_order_relaxed);
                slots_[pos].tag.store(tag, std::memory_order_release);
                return;
            }
            continue;
        }
        pos = (pos + 1) & capacity_mask_;
        if (pos == initial_pos) {
            throw std::runtime_error("ConcurrentStringHashMap is full. Consider increasing capacity.");
        }
    }
}
inline const StoredRecord* ConcurrentStringHashMap::Find(std::string_view key) const {
    uint64_t hash = hasher_(key);
    uint8_t tag = (hash >> 56);
    if (tag >= LOCKED_TAG) tag = 0;
    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;
    do {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
        if (current_tag == EMPTY_TAG) return nullptr;
        if (current_tag == tag && slots_[pos].full_hash == hash && slots_[pos].key == key) {
            const StoredRecord* rec = slots_[pos].record.load(std::memory_order_acquire);
            if (rec && rec->ready.load(std::memory_order_acquire)) {
                return rec;
            }
            return nullptr;
        }
        pos = (pos + 1) & capacity_mask_;
    } while (pos != initial_pos);
    return nullptr;
}
class FlashActiveBlock {
    friend class ColumnarMemTable;

   public:
    explicit FlashActiveBlock(size_t cap) : index_(cap) {}
    ~FlashActiveBlock() = default;

    // With the new Arena, TryAdd only fails if the block is sealed or the item is too large.
    // The caller no longer needs to retry on chunk-full conditions.
    bool TryAdd(std::string_view key, std::string_view value, RecordType type) {
        if (is_sealed()) return false;
        const StoredRecord* record_ptr = data_log_.AllocateAndAppend(key, value, type);
        if (record_ptr) {
            index_.Insert(record_ptr->record.key, record_ptr);
            return true;
        }
        return false;
    }

    bool TryAddBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch, RecordType type) {
        if (is_sealed()) return false;
        auto record_ptrs = data_log_.AllocateAndAppendBatch(batch, type);
        if (is_sealed()) return false;  // Re-check after allocation
        for (const auto* record_ptr : record_ptrs) {
            if (record_ptr) {
                index_.Insert(record_ptr->record.key, record_ptr);
            }
        }
        return true;
    }

    std::optional<RecordRef> Get(std::string_view key) const {
        const StoredRecord* record_ptr = index_.Find(key);
        return record_ptr ? std::optional<RecordRef>(record_ptr->record) : std::nullopt;
    }

    size_t size() const { return data_log_.size(); }
    size_t ApproximateMemoryUsage() const { return data_log_.ApproximateMemoryUsage() + sizeof(index_); }
    void Seal() { sealed_.store(true, std::memory_order_release); }
    bool is_sealed() const { return sealed_.load(std::memory_order_acquire); }

   private:
    ColumnarRecordArena data_log_;
    ConcurrentStringHashMap index_;
    std::atomic<bool> sealed_{false};
};
class ColumnarBlock {
   public:
    class SimpleArena {
       public:
        char* AllocateRaw(size_t bytes) {
            if (current_block_idx_ < 0 || blocks_[current_block_idx_].pos + bytes > blocks_[current_block_idx_].size) {
                size_t bs = std::max(bytes, (size_t)4096);
                blocks_.emplace_back(bs);
                current_block_idx_++;
            }
            Block& b = blocks_[current_block_idx_];
            char* r = b.data.get() + b.pos;
            b.pos += bytes;
            return r;
        }
        std::string_view AllocateAndCopy(std::string_view d) {
            char* m = AllocateRaw(d.size());
            if (!d.empty()) memcpy(m, d.data(), d.size());
            return {m, d.size()};
        }
        size_t ApproximateMemoryUsage() const {
            size_t total = 0;
            for (const auto& block : blocks_) total += block.size;
            return total;
        }

       private:
        struct Block {
            std::unique_ptr<char[]> data;
            size_t pos, size;
            explicit Block(size_t s) : data(new char[s]), pos(0), size(s) {}
        };
        std::vector<Block> blocks_;
        int current_block_idx_ = -1;
    };
    SimpleArena arena;
    std::vector<std::string_view> keys, values;
    std::vector<RecordType> types;
    void Add(std::string_view k, std::string_view v, RecordType t) {
        keys.push_back(arena.AllocateAndCopy(k));
        values.push_back(arena.AllocateAndCopy(v));
        types.push_back(t);
    }
    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }
    void Clear() {
        keys.clear();
        values.clear();
        types.clear();
        arena = SimpleArena();
    }
    size_t ApproximateMemoryUsage() const {
        size_t total = arena.ApproximateMemoryUsage();
        total += keys.capacity() * sizeof(std::string_view);
        total += values.capacity() * sizeof(std::string_view);
        total += types.capacity() * sizeof(RecordType);
        return total;
    }
};
class Sorter {
   public:
    virtual ~Sorter() = default;
    virtual std::vector<uint32_t> Sort(const ColumnarBlock& block) const = 0;
};
class StdSorter : public Sorter {
   public:
    std::vector<uint32_t> Sort(const ColumnarBlock& block) const override {
        if (block.empty()) return {};
        std::vector<uint32_t> indices(block.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(),
                         [&block](uint32_t a, uint32_t b) { return block.keys[a] < block.keys[b]; });
        return indices;
    }
};
class ParallelRadixSorter : public Sorter {
   public:
    std::vector<uint32_t> Sort(const ColumnarBlock& block) const override {
        if (block.empty()) return {};
        std::vector<uint32_t> indices(block.size());
        std::iota(indices.begin(), indices.end(), 0);
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
        radix_sort_msd_parallel(indices.begin(), indices.end(), 0, num_threads, block);
        return indices;
    }

   private:
    static constexpr size_t kSequentialSortThreshold = 2048;
    static constexpr size_t kRadixAlphabetSize = 256;
    using Iterator = std::vector<uint32_t>::iterator;
    static inline int get_char_at(std::string_view s, size_t depth) {
        return depth < s.size() ? static_cast<unsigned char>(s[depth]) : -1;
    }
    void radix_sort_msd_sequential(Iterator begin, Iterator end, size_t depth, const ColumnarBlock& block) const {
        if (static_cast<size_t>(std::distance(begin, end)) <= 1) return;
        if (static_cast<size_t>(std::distance(begin, end)) <= kSequentialSortThreshold) {
            std::stable_sort(begin, end, [&](uint32_t a, uint32_t b) {
                return block.keys[a].substr(std::min(depth, block.keys[a].size())) <
                       block.keys[b].substr(std::min(depth, block.keys[b].size()));
            });
            return;
        }
        std::vector<uint32_t> buckets[kRadixAlphabetSize];
        std::vector<uint32_t> finished_strings;
        for (auto it = begin; it != end; ++it) {
            int char_code = get_char_at(block.keys[*it], depth);
            if (char_code == -1) {
                finished_strings.push_back(*it);
            } else {
                buckets[char_code].push_back(*it);
            }
        }
        auto current = begin;
        std::copy(finished_strings.begin(), finished_strings.end(), current);
        current += finished_strings.size();
        for (size_t i = 0; i < kRadixAlphabetSize; ++i) {
            if (!buckets[i].empty()) {
                auto bucket_begin = current;
                std::copy(buckets[i].begin(), buckets[i].end(), bucket_begin);
                current += buckets[i].size();
                radix_sort_msd_sequential(bucket_begin, current, depth + 1, block);
            }
        }
    }
    void radix_sort_msd_parallel(Iterator begin, Iterator end, size_t depth, unsigned int num_threads,
                                 const ColumnarBlock& block) const {
        const size_t size = std::distance(begin, end);
        if (size <= kSequentialSortThreshold || num_threads <= 1) {
            radix_sort_msd_sequential(begin, end, depth, block);
            return;
        }
        std::vector<size_t> bucket_counts(kRadixAlphabetSize + 1, 0);
        for (auto it = begin; it != end; ++it) {
            bucket_counts[get_char_at(block.keys[*it], depth) + 1]++;
        }
        std::vector<size_t> bucket_offsets(kRadixAlphabetSize + 2, 0);
        for (size_t i = 0; i < kRadixAlphabetSize + 1; ++i) {
            bucket_offsets[i + 1] = bucket_offsets[i] + bucket_counts[i];
        }
        std::vector<uint32_t> sorted_output(size);
        std::vector<size_t> current_offsets = bucket_offsets;
        for (auto it = begin; it != end; ++it) {
            uint32_t val = *it;
            int char_code = get_char_at(block.keys[val], depth);
            sorted_output[current_offsets[char_code + 1]++] = val;
        }
        std::copy(sorted_output.begin(), sorted_output.end(), begin);
        std::vector<std::future<void>> futures;
        for (size_t i = 1; i < kRadixAlphabetSize + 1; ++i) {
            size_t bucket_size = bucket_counts[i];
            if (bucket_size == 0) continue;
            Iterator bucket_begin = begin + bucket_offsets[i];
            Iterator bucket_end = begin + bucket_offsets[i + 1];
            if (futures.size() < num_threads - 1 && bucket_size > kSequentialSortThreshold) {
                // Improved thread distribution logic.
                futures.push_back(std::async(
                    std::launch::async, [this, bucket_begin, bucket_end, depth, num_threads, &block, &futures] {
                        unsigned int threads_for_child = std::max(1u, num_threads / (unsigned int)(futures.size() + 1));
                        radix_sort_msd_parallel(bucket_begin, bucket_end, depth + 1, threads_for_child, block);
                    }));
            } else {
                radix_sort_msd_sequential(bucket_begin, bucket_end, depth + 1, block);
            }
        }
        for (auto& f : futures) {
            f.get();
        }
    }
};
class SortedColumnarBlock {
   public:
    class Iterator;
    static constexpr size_t kSparseIndexSampleRate = 16;
    explicit SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block, const Sorter& sorter,
                                 bool build_bloom_filter = true);
    bool MayContain(std::string_view key) const;
    std::optional<RecordRef> Get(std::string_view key) const;
    std::string_view min_key() const { return min_key_; }
    std::string_view max_key() const { return max_key_; }
    Iterator begin() const;
    bool empty() const { return sorted_indices_.empty(); }
    size_t size() const { return sorted_indices_.size(); }
    size_t ApproximateMemoryUsage() const {
        size_t usage = sizeof(*this);
        if (block_data_) usage += block_data_->ApproximateMemoryUsage();
        usage += sorted_indices_.capacity() * sizeof(uint32_t);
        if (bloom_filter_) usage += bloom_filter_->ApproximateMemoryUsage();
        usage += sparse_index_.capacity() * sizeof(std::pair<std::string_view, size_t>);
        return usage;
    }

   private:
    friend class Iterator;
    std::shared_ptr<ColumnarBlock> block_data_;
    std::vector<uint32_t> sorted_indices_;
    std::string_view min_key_, max_key_;
    std::unique_ptr<BloomFilter> bloom_filter_;
    std::vector<std::pair<std::string_view, size_t>> sparse_index_;
};
inline SortedColumnarBlock::SortedColumnarBlock(std::shared_ptr<ColumnarBlock> b, const Sorter& s,
                                                bool build_bloom_filter)
    : block_data_(std::move(b)) {
    sorted_indices_ = s.Sort(*block_data_);
    if (sorted_indices_.empty()) {
        min_key_ = {};
        max_key_ = {};
        return;
    }
    min_key_ = block_data_->keys[sorted_indices_.front()];
    max_key_ = block_data_->keys[sorted_indices_.back()];

    constexpr size_t kBloomFilterThreshold = 256 * 1024;
    if (build_bloom_filter && block_data_->size() < kBloomFilterThreshold) {
        bloom_filter_ = std::make_unique<BloomFilter>(block_data_->size());
        for (size_t i = 0; i < block_data_->size(); ++i) {
            bloom_filter_->Add(block_data_->keys[i]);
        }
    }

    sparse_index_.reserve(sorted_indices_.size() / kSparseIndexSampleRate + 1);
    for (size_t i = 0; i < sorted_indices_.size(); i += kSparseIndexSampleRate)
        sparse_index_.emplace_back(block_data_->keys[sorted_indices_[i]], i);
}

inline bool SortedColumnarBlock::MayContain(std::string_view key) const {
    if (empty() || key < min_key_ || key > max_key_) return false;
    if (!bloom_filter_) {
        return true;
    }
    return bloom_filter_->MayContain(key);
}

inline std::optional<RecordRef> SortedColumnarBlock::Get(std::string_view key) const {
    if (!MayContain(key)) return std::nullopt;

    auto sparse_it = std::lower_bound(sparse_index_.begin(), sparse_index_.end(), key,
                                      [](const auto& a, auto b) { return a.first < b; });
    auto start_it = sorted_indices_.begin();
    if (sparse_it != sparse_index_.begin()) start_it += (sparse_it - 1)->second;

    auto end_it = sorted_indices_.end();
    if (sparse_it != sparse_index_.end()) {
        end_it = sorted_indices_.begin() + sparse_it->second + kSparseIndexSampleRate;
        if (end_it > sorted_indices_.end()) end_it = sorted_indices_.end();
    }

    auto it = std::lower_bound(start_it, end_it, key,
                               [&](uint32_t i, std::string_view k) { return block_data_->keys[i] < k; });

    if (it == end_it || block_data_->keys[*it] != key) {
        return std::nullopt;
    }

    // stable_sort ensures that for identical keys, the one inserted later appears later in the sorted list.
    // We want the latest version, so find the end of the range of equal keys and take the one just before it.
    auto range_end =
        std::upper_bound(it, end_it, key, [&](std::string_view k, uint32_t i) { return k < block_data_->keys[i]; });

    uint32_t latest_idx = *std::prev(range_end);
    return RecordRef{block_data_->keys[latest_idx], block_data_->values[latest_idx], block_data_->types[latest_idx]};
}

class SortedColumnarBlock::Iterator {
   public:
    Iterator(const SortedColumnarBlock* b, size_t p) : block_(b), pos_(p) {}
    RecordRef operator*() const {
        uint32_t i = block_->sorted_indices_[pos_];
        return {block_->block_data_->keys[i], block_->block_data_->values[i], block_->block_data_->types[i]};
    }
    void Next() { ++pos_; }
    bool IsValid() const { return block_ && pos_ < block_->sorted_indices_.size(); }

   private:
    const SortedColumnarBlock* block_;
    size_t pos_;
};
inline SortedColumnarBlock::Iterator SortedColumnarBlock::begin() const { return Iterator(this, 0); }
class FlushIterator {
   public:
    explicit FlushIterator(const std::vector<std::shared_ptr<const SortedColumnarBlock>>& sources);
    bool IsValid() const { return !min_heap_.empty(); }
    RecordRef Get() const { return min_heap_.top().record; }
    void Next();

   private:
    struct HeapNode {
        RecordRef record;
        uint64_t key_prefix;
        size_t source_index;
        // Use original logic with key_prefix optimization
        bool operator>(const HeapNode& o) const {
            if (key_prefix != o.key_prefix) return key_prefix > o.key_prefix;
            if (record.key != o.record.key) return record.key > o.record.key;
            // To ensure stability, use source index as a tie-breaker.
            // Iterators created later (for newer data) should have a higher index.
            // A smaller source_index means older data, so it should have higher priority (come first).
            return source_index > o.source_index;
        }
    };
    std::vector<SortedColumnarBlock::Iterator> iterators_;
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> min_heap_;
};
inline FlushIterator::FlushIterator(const std::vector<std::shared_ptr<const SortedColumnarBlock>>& sources) {
    iterators_.reserve(sources.size());
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i]) {
            iterators_.emplace_back(sources[i]->begin());
        } else {
            iterators_.emplace_back(nullptr, 0);
        }
        if (iterators_.back().IsValid()) {
            RecordRef rec = *iterators_.back();
            uint64_t prefix = load_u64_prefix(rec.key);
            min_heap_.push({rec, prefix, i});
        }
    }
}
inline void FlushIterator::Next() {
    if (!IsValid()) return;
    HeapNode n = min_heap_.top();
    min_heap_.pop();
    iterators_[n.source_index].Next();
    if (iterators_[n.source_index].IsValid()) {
        RecordRef rec = *iterators_[n.source_index];
        uint64_t prefix = load_u64_prefix(rec.key);
        min_heap_.push({rec, prefix, n.source_index});
    }
}
class CompactingIterator {
   public:
    template <typename It>
    explicit CompactingIterator(std::unique_ptr<It> s);
    bool IsValid() const { return is_valid_; }
    RecordRef Get() const { return current_record_; }
    void Next() { FindNext(); }

   private:
    struct ItConcept {
        virtual ~ItConcept() = default;
        virtual bool IsValid() const = 0;
        virtual RecordRef Get() const = 0;
        virtual void Next() = 0;
    };
    template <typename It>
    struct ItWrapper final : public ItConcept {
        explicit ItWrapper(std::unique_ptr<It> i) : iter_(std::move(i)) {}
        bool IsValid() const override { return iter_->IsValid(); }
        RecordRef Get() const override { return iter_->Get(); }
        void Next() override { iter_->Next(); }
        std::unique_ptr<It> iter_;
    };
    void FindNext();
    std::unique_ptr<ItConcept> source_;
    RecordRef current_record_;
    bool is_valid_ = false;
};
template <typename It>
inline CompactingIterator::CompactingIterator(std::unique_ptr<It> s)
    : source_(std::make_unique<ItWrapper<It>>(std::move(s))) {
    FindNext();
}
inline void CompactingIterator::FindNext() {
    while (source_->IsValid()) {
        RecordRef latest_record = source_->Get();
        source_->Next();
        while (source_->IsValid() && source_->Get().key == latest_record.key) {
            latest_record = source_->Get();
            source_->Next();
        }
        if (latest_record.type == RecordType::Put) {
            current_record_ = latest_record;
            is_valid_ = true;
            return;
        }
    }
    is_valid_ = false;
}
class ColumnarMemTable : public std::enable_shared_from_this<ColumnarMemTable> {
   public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::map<std::string_view, GetResult, std::less<>>;
    ~ColumnarMemTable() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_background_thread_ = true;
        }
        queue_cond_.notify_one();
        if (background_thread_.joinable()) background_thread_.join();
    }
    ColumnarMemTable(const ColumnarMemTable&) = delete;
    ColumnarMemTable& operator=(const ColumnarMemTable&) = delete;

    // Adopt factory pattern for safe lifetime management with background thread.
    static std::shared_ptr<ColumnarMemTable> Create(
        size_t active_block_size_bytes = 16 * 1024 * 48, bool enable_compaction = false,
        std::shared_ptr<Sorter> sorter = std::make_shared<ParallelRadixSorter>(), size_t num_shards = 16) {
        struct MakeSharedEnabler : public ColumnarMemTable {
            MakeSharedEnabler(size_t active_block_size_bytes, bool enable_compaction, std::shared_ptr<Sorter> sorter,
                              size_t num_shards)
                : ColumnarMemTable(active_block_size_bytes, enable_compaction, std::move(sorter), num_shards) {}
        };
        auto table = std::make_shared<MakeSharedEnabler>(active_block_size_bytes, enable_compaction, std::move(sorter),
                                                         num_shards);
        table->StartBackgroundThread();
        return table;
    }

    void Put(std::string_view key, std::string_view value) { Insert(key, value, RecordType::Put); }
    void Delete(std::string_view key) { Insert(key, "", RecordType::Delete); }
    GetResult Get(std::string_view key) const;
    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const;
    void PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch);
    void WaitForBackgroundWork();
    std::unique_ptr<CompactingIterator> NewCompactingIterator();
    size_t ApproximateMemoryUsage() const;

   private:
    // Make constructor private for factory pattern.
    explicit ColumnarMemTable(size_t active_block_size_bytes, bool enable_compaction, std::shared_ptr<Sorter> sorter,
                              size_t num_shards)
        : active_block_threshold_(std::max((size_t)1, active_block_size_bytes / 116)),
          enable_compaction_(enable_compaction),
          sorter_(std::move(sorter)),
          num_shards_(num_shards > 0 ? 1UL << (63 - __builtin_clzll(num_shards - 1)) : 1),
          shard_mask_(num_shards_ - 1) {
        for (size_t i = 0; i < num_shards_; ++i) {
            shards_.push_back(std::make_unique<Shard>(active_block_threshold_));
        }
    }
    void StartBackgroundThread() { background_thread_ = std::thread(&ColumnarMemTable::BackgroundWorkerLoop, this); }

    struct ImmutableState {
        using SortedBlockList = std::vector<std::shared_ptr<const SortedColumnarBlock>>;
        using SealedBlockList = std::vector<std::shared_ptr<FlashActiveBlock>>;
        std::shared_ptr<const SealedBlockList> sealed_blocks;
        std::shared_ptr<const SortedBlockList> blocks;
        ImmutableState()
            : sealed_blocks(std::make_shared<const SealedBlockList>()),
              blocks(std::make_shared<const SortedBlockList>()) {}
    };
    struct alignas(64) Shard {
        std::shared_ptr<FlashActiveBlock> active_block_;
        std::shared_ptr<const ImmutableState> immutable_state_;
        std::atomic<uint64_t> version_{0};
        SpinLock seal_mutex_;
        Shard(size_t active_block_threshold) {
            active_block_ = std::make_shared<FlashActiveBlock>(active_block_threshold);
            immutable_state_ = std::make_shared<const ImmutableState>();
        }
    };
    struct BackgroundWorkItem {
        std::shared_ptr<FlashActiveBlock> block;
        std::unique_ptr<std::promise<void>> promise;
        size_t shard_idx;
    };

    const size_t active_block_threshold_;
    const bool enable_compaction_;
    std::shared_ptr<Sorter> sorter_;
    const size_t num_shards_;
    const size_t shard_mask_;
    std::vector<std::unique_ptr<Shard>> shards_;
    XXHasher hasher_;

    // Added object pool for ColumnarBlocks
    std::vector<std::unique_ptr<ColumnarBlock>> columnar_block_pool_;
    std::mutex pool_mutex_;

    std::vector<BackgroundWorkItem> sealed_blocks_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::thread background_thread_;
    std::atomic<bool> stop_background_thread_{false};

    size_t GetShardIdx(std::string_view key) const { return hasher_(key) & shard_mask_; }
    void Insert(std::string_view key, std::string_view value, RecordType type);
    void SealActiveBlockIfNeeded(size_t shard_idx);
    void BackgroundWorkerLoop();
    void ProcessBlocksForShard(size_t shard_idx, const std::vector<std::shared_ptr<FlashActiveBlock>>& sealed_blocks);
    std::shared_ptr<ColumnarBlock> GetPooledColumnarBlock();

    std::shared_ptr<FlashActiveBlock> GetActiveBlockForThread(size_t shard_idx, bool force_refresh = false) const;
    std::shared_ptr<const ImmutableState> GetImmutableStateForThread(size_t shard_idx,
                                                                     bool force_refresh = false) const;
};

// --- Implementation ---
inline void ColumnarMemTable::Insert(std::string_view k, std::string_view v, RecordType t) {
    const size_t shard_idx = GetShardIdx(k);
    auto current_block = GetActiveBlockForThread(shard_idx);
    // The new arena handles chunk-full retries internally, so this loop mainly handles
    // the case where a block is sealed by another thread while we are trying to add to it.
    while (!current_block->TryAdd(k, v, t)) {
        current_block = GetActiveBlockForThread(shard_idx, true);
    }
    if (current_block->size() >= active_block_threshold_) {
        SealActiveBlockIfNeeded(shard_idx);
    }
}
inline ColumnarMemTable::GetResult ColumnarMemTable::Get(std::string_view key) const {
    const size_t shard_idx = GetShardIdx(key);
    auto active_block = GetActiveBlockForThread(shard_idx);
    if (auto r = active_block->Get(key)) {
        return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
    }
    auto s = GetImmutableStateForThread(shard_idx);
    if (s->sealed_blocks) {
        for (auto it = s->sealed_blocks->rbegin(); it != s->sealed_blocks->rend(); ++it) {
            if (auto r = (*it)->Get(key)) {
                return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
            }
        }
    }
    if (s->blocks) {
        for (auto it = s->blocks->rbegin(); it != s->blocks->rend(); ++it) {
            if (auto r = (*it)->Get(key)) {
                return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
            }
        }
    }
    return std::nullopt;
}
inline void ColumnarMemTable::PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
    if (batch.empty()) return;
    std::vector<std::vector<std::pair<std::string_view, std::string_view>>> sharded_batches(num_shards_);
    for (const auto& [key, value] : batch) {
        sharded_batches[GetShardIdx(key)].emplace_back(key, value);
    }
    std::vector<std::future<void>> futures;
    futures.reserve(num_shards_);
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
        if (sharded_batches[shard_idx].empty()) continue;
        futures.emplace_back(std::async(std::launch::async, [this, shard_idx, &sub_batch = sharded_batches[shard_idx]] {
            auto current_block = GetActiveBlockForThread(shard_idx);
            // Use improved fallback logic
            if (!current_block->TryAddBatch(sub_batch, RecordType::Put)) {
                current_block = GetActiveBlockForThread(shard_idx, true);
                for (const auto& [key, value] : sub_batch) {
                    while (!current_block->TryAdd(key, value, RecordType::Put)) {
                        current_block = GetActiveBlockForThread(shard_idx, true);
                    }
                }
            }
            if (current_block->size() >= active_block_threshold_) {
                SealActiveBlockIfNeeded(shard_idx);
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
}
inline ColumnarMemTable::MultiGetResult ColumnarMemTable::MultiGet(const std::vector<std::string_view>& keys) const {
    if (keys.empty()) return {};
    std::vector<std::vector<std::string_view>> sharded_keys(num_shards_);
    std::vector<size_t> active_shards;
    active_shards.reserve(num_shards_);
    for (const auto& key : keys) {
        size_t shard_idx = GetShardIdx(key);
        if (sharded_keys[shard_idx].empty()) {
            active_shards.push_back(shard_idx);
        }
        sharded_keys[shard_idx].push_back(key);
    }
    auto process_shards = [this](const std::vector<size_t>& shards_to_process,
                                 const std::vector<std::vector<std::string_view>>& all_sharded_keys) -> MultiGetResult {
        MultiGetResult local_results;
        for (const size_t shard_idx : shards_to_process) {
            auto active_block = GetActiveBlockForThread(shard_idx);
            auto s = GetImmutableStateForThread(shard_idx);
            for (const auto& key : all_sharded_keys[shard_idx]) {
                if (auto r = active_block->Get(key)) {
                    local_results[key] = (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                    continue;
                }
                if (s->sealed_blocks) {
                    bool found = false;
                    for (auto it = s->sealed_blocks->rbegin(); it != s->sealed_blocks->rend(); ++it) {
                        if (auto r = (*it)->Get(key)) {
                            local_results[key] = (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                }
                if (s->blocks) {
                    bool found = false;
                    for (auto it = s->blocks->rbegin(); it != s->blocks->rend(); ++it) {
                        if (auto r = (*it)->Get(key)) {
                            local_results[key] = (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                }
            }
        }
        return local_results;
    };
    unsigned int num_workers = std::thread::hardware_concurrency();
    if (active_shards.size() < 2 || num_workers < 2) {
        MultiGetResult final_results = process_shards(active_shards, sharded_keys);
        for (const auto& key : keys) {
            final_results.try_emplace(key, std::nullopt);
        }
        return final_results;
    } else {
        if (num_workers > active_shards.size()) {
            num_workers = active_shards.size();
        }
        std::vector<std::vector<size_t>> workloads(num_workers);
        for (size_t i = 0; i < active_shards.size(); ++i) {
            workloads[i % num_workers].push_back(active_shards[i]);
        }
        std::vector<std::future<MultiGetResult>> futures;
        futures.reserve(num_workers);
        for (const auto& workload : workloads) {
            if (workload.empty()) continue;
            futures.emplace_back(
                std::async(std::launch::async, process_shards, std::cref(workload), std::cref(sharded_keys)));
        }
        MultiGetResult final_results;
        for (auto& f : futures) {
            final_results.merge(f.get());
        }
        for (const auto& key : keys) {
            final_results.try_emplace(key, std::nullopt);
        }
        return final_results;
    }
}
inline void ColumnarMemTable::SealActiveBlockIfNeeded(size_t shard_idx) {
    auto& shard = *shards_[shard_idx];
    auto current_b_sp = std::atomic_load(&shard.active_block_);
    if (current_b_sp->size() < active_block_threshold_ || current_b_sp->is_sealed()) return;
    std::shared_ptr<FlashActiveBlock> sealed_block;
    {
        std::lock_guard<SpinLock> lock(shard.seal_mutex_);
        current_b_sp = std::atomic_load(&shard.active_block_);
        if (current_b_sp->size() < active_block_threshold_ || current_b_sp->is_sealed()) return;
        current_b_sp->Seal();
        sealed_block = current_b_sp;
        auto new_active_block = std::make_shared<FlashActiveBlock>(active_block_threshold_);
        auto old_s = std::atomic_load(&shard.immutable_state_);
        auto new_s = std::make_shared<ImmutableState>();
        new_s->blocks = old_s->blocks;
        auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>(*old_s->sealed_blocks);
        new_sealed_list->push_back(sealed_block);
        new_s->sealed_blocks = new_sealed_list;
        std::atomic_exchange(&shard.active_block_, new_active_block);
        std::atomic_store(&shard.immutable_state_, std::shared_ptr<const ImmutableState>(new_s));
        shard.version_.fetch_add(1, std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> ql(queue_mutex_);
        sealed_blocks_queue_.push_back({std::move(sealed_block), nullptr, shard_idx});
    }
    queue_cond_.notify_one();
}
inline void ColumnarMemTable::WaitForBackgroundWork() {
    auto promise = std::make_unique<std::promise<void>>();
    auto future = promise->get_future();
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        for (size_t i = 0; i < num_shards_; ++i) {
            auto& shard = *shards_[i];
            std::lock_guard<SpinLock> seal_lock(shard.seal_mutex_);
            auto ab = std::atomic_load(&shard.active_block_);
            if (ab->size() > 0 && !ab->is_sealed()) {
                ab->Seal();
                auto new_b = std::make_shared<FlashActiveBlock>(active_block_threshold_);
                auto old_s = std::atomic_load(&shard.immutable_state_);
                auto new_s = std::make_shared<ImmutableState>();
                new_s->blocks = old_s->blocks;
                auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>(*old_s->sealed_blocks);
                new_sealed_list->push_back(ab);
                new_s->sealed_blocks = new_sealed_list;
                std::atomic_exchange(&shard.active_block_, new_b);
                std::atomic_store(&shard.immutable_state_, std::shared_ptr<const ImmutableState>(new_s));
                shard.version_.fetch_add(1, std::memory_order_release);
                sealed_blocks_queue_.push_back({std::move(ab), nullptr, i});
            }
        }
        sealed_blocks_queue_.push_back({nullptr, std::move(promise), 0});
    }
    queue_cond_.notify_one();
    future.wait();
}
inline std::unique_ptr<CompactingIterator> ColumnarMemTable::NewCompactingIterator() {
    WaitForBackgroundWork();
    std::vector<std::shared_ptr<const SortedColumnarBlock>> all_blocks;
    for (const auto& shard_ptr : shards_) {
        auto s = std::atomic_load(&shard_ptr->immutable_state_);
        if (s->blocks && !s->blocks->empty()) {
            all_blocks.insert(all_blocks.end(), s->blocks->begin(), s->blocks->end());
        }
    }
    auto flush_iterator = std::make_unique<FlushIterator>(all_blocks);
    return std::make_unique<CompactingIterator>(std::move(flush_iterator));
}
inline void ColumnarMemTable::BackgroundWorkerLoop() {
    while (true) {
        std::vector<BackgroundWorkItem> work_items;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this] { return !sealed_blocks_queue_.empty() || stop_background_thread_; });
            if (stop_background_thread_ && sealed_blocks_queue_.empty()) return;
            work_items.swap(sealed_blocks_queue_);
        }
        std::map<size_t, std::vector<std::shared_ptr<FlashActiveBlock>>> work_by_shard;
        std::vector<std::unique_ptr<std::promise<void>>> promises;
        for (auto& item : work_items) {
            if (item.block) {
                work_by_shard[item.shard_idx].push_back(std::move(item.block));
            }
            if (item.promise) {
                promises.push_back(std::move(item.promise));
            }
        }
        for (auto const& [shard_idx, blocks] : work_by_shard) {
            try {
                ProcessBlocksForShard(shard_idx, blocks);
            } catch (const std::exception& e) {
                std::cerr << "Exception in background thread for shard " << shard_idx << ": " << e.what() << std::endl;
            }
        }
        for (auto& p : promises) {
            p->set_value();
        }
    }
}
inline void ColumnarMemTable::ProcessBlocksForShard(
    size_t shard_idx, const std::vector<std::shared_ptr<FlashActiveBlock>>& sealed_blocks) {
    if (sealed_blocks.empty()) return;
    std::vector<std::shared_ptr<const SortedColumnarBlock>> new_sorted_blocks;
    new_sorted_blocks.reserve(sealed_blocks.size());
    for (const auto& sealed_b : sealed_blocks) {
        auto cb = GetPooledColumnarBlock();  // Use the pool
        const auto& arena_to_copy = sealed_b->data_log_;
        uint32_t active_threads = arena_to_copy.GetMaxThreadIdSeen() + 1;
        if (active_threads > ThreadIdManager::kMaxThreads) active_threads = ThreadIdManager::kMaxThreads;
        for (uint32_t thread_idx = 0; thread_idx < active_threads; ++thread_idx) {
            const auto* tls_data = arena_to_copy.GetAllTlsData()[thread_idx].load(std::memory_order_acquire);
            if (!tls_data) continue;
            for (const auto& chunk_ptr : tls_data->chunks) {
                uint64_t final_pos = chunk_ptr->positions_.load(std::memory_order_relaxed);
                uint32_t max_idx = static_cast<uint32_t>(final_pos >> 32);
                if (max_idx > ColumnarRecordArena::DataChunk::kRecordCapacity)
                    max_idx = ColumnarRecordArena::DataChunk::kRecordCapacity;
                for (uint32_t i = 0; i < max_idx; ++i) {
                    const auto& record_slot = chunk_ptr->records[i];
                    // Robust check for record readiness.
                    while (!record_slot.ready.load(std::memory_order_acquire)) {
                        __builtin_ia32_pause();
                    }
                    cb->Add(record_slot.record.key, record_slot.record.value, record_slot.record.type);
                }
            }
        }
        if (!cb->empty()) {
            new_sorted_blocks.push_back(
                std::make_shared<const SortedColumnarBlock>(std::move(cb), *sorter_, !enable_compaction_));
        }
    }
    auto& shard = *shards_[shard_idx];
    std::lock_guard<SpinLock> lock(shard.seal_mutex_);
    auto old_s = std::atomic_load(&shard.immutable_state_);
    auto new_s = std::make_shared<ImmutableState>();
    auto new_sorted_list = std::make_shared<ImmutableState::SortedBlockList>();
    // Keep the original compaction logic
    if (enable_compaction_) {
        std::vector<std::shared_ptr<const SortedColumnarBlock>> to_merge;
        if (old_s->blocks) to_merge.insert(to_merge.end(), old_s->blocks->begin(), old_s->blocks->end());
        to_merge.insert(to_merge.end(), new_sorted_blocks.begin(), new_sorted_blocks.end());
        if (!to_merge.empty()) {
            auto compacted_block = GetPooledColumnarBlock();
            CompactingIterator it(std::make_unique<FlushIterator>(to_merge));
            while (it.IsValid()) {
                RecordRef r = it.Get();
                compacted_block->Add(r.key, r.value, r.type);
                it.Next();
            }
            if (!compacted_block->empty()) {
                new_sorted_list->push_back(
                    std::make_shared<const SortedColumnarBlock>(std::move(compacted_block), *sorter_, false));
            }
        }
    } else {
        if (old_s->blocks) *new_sorted_list = *old_s->blocks;
        new_sorted_list->insert(new_sorted_list->end(), new_sorted_blocks.begin(), new_sorted_blocks.end());
    }
    new_s->blocks = std::move(new_sorted_list);
    auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>();
    if (old_s->sealed_blocks) {
        for (const auto& b : *old_s->sealed_blocks) {
            bool was_processed = false;
            for (const auto& pb : sealed_blocks)
                if (b == pb) {
                    was_processed = true;
                    break;
                }
            if (!was_processed) new_sealed_list->push_back(b);
        }
    }
    new_s->sealed_blocks = std::move(new_sealed_list);
    std::atomic_store(&shard.immutable_state_, std::shared_ptr<const ImmutableState>(new_s));
}
// Implement object pooling for ColumnarBlocks
inline std::shared_ptr<ColumnarBlock> ColumnarMemTable::GetPooledColumnarBlock() {
    std::weak_ptr<ColumnarMemTable> weak_self = shared_from_this();
    auto recycler_deleter = [weak_self](ColumnarBlock* ptr) {
        if (auto shared_self = weak_self.lock()) {
            ptr->Clear();
            std::lock_guard<std::mutex> lock(shared_self->pool_mutex_);
            shared_self->columnar_block_pool_.emplace_back(ptr);
        } else {
            delete ptr;
        }
    };
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!columnar_block_pool_.empty()) {
        std::unique_ptr<ColumnarBlock> block_ptr = std::move(columnar_block_pool_.back());
        columnar_block_pool_.pop_back();
        return std::shared_ptr<ColumnarBlock>(block_ptr.release(), recycler_deleter);
    }
    return std::shared_ptr<ColumnarBlock>(new ColumnarBlock(), recycler_deleter);
}
inline size_t ColumnarMemTable::ApproximateMemoryUsage() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        auto active = std::atomic_load(&shard->active_block_);
        total += active->ApproximateMemoryUsage();
        auto immutable = std::atomic_load(&shard->immutable_state_);
        if (immutable->sealed_blocks) {
            for (const auto& block : *immutable->sealed_blocks) {
                total += block->ApproximateMemoryUsage();
            }
        }
        if (immutable->blocks) {
            for (const auto& block : *immutable->blocks) {
                total += block->ApproximateMemoryUsage();
            }
        }
    }
    return total;
}
inline std::shared_ptr<FlashActiveBlock> ColumnarMemTable::GetActiveBlockForThread(size_t shard_idx,
                                                                                   bool force_refresh) const {
    thread_local std::vector<std::shared_ptr<FlashActiveBlock>> active_block_cache;
    thread_local std::vector<uint64_t> last_seen_version;
    if (active_block_cache.size() != num_shards_) {
        active_block_cache.resize(num_shards_);
        last_seen_version.assign(num_shards_, -1);
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || active_block_cache[shard_idx] == nullptr || last_seen_version[shard_idx] != current_version) {
        active_block_cache[shard_idx] = std::atomic_load(&shard.active_block_);
        last_seen_version[shard_idx] = current_version;
    }
    return active_block_cache[shard_idx];
}
inline std::shared_ptr<const ColumnarMemTable::ImmutableState> ColumnarMemTable::GetImmutableStateForThread(
    size_t shard_idx, bool force_refresh) const {
    thread_local std::vector<std::shared_ptr<const ImmutableState>> immutable_state_cache;
    thread_local std::vector<uint64_t> last_seen_version;
    if (immutable_state_cache.size() != num_shards_) {
        immutable_state_cache.resize(num_shards_);
        last_seen_version.assign(num_shards_, -1);
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || immutable_state_cache[shard_idx] == nullptr ||
        last_seen_version[shard_idx] != current_version) {
        immutable_state_cache[shard_idx] = std::atomic_load(&shard.immutable_state_);
        last_seen_version[shard_idx] = current_version;
    }
    return immutable_state_cache[shard_idx];
}

#endif  // COLUMNAR_MEMTABLE_H