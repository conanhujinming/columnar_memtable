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
#include <limits>
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
#include "thread_pool.h"
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
        memcpy(&prefix, sv.data(), sizeof(prefix));
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return __builtin_bswap64(prefix);
#else
        return prefix;
#endif
    }
    uint64_t prefix = 0;
    const size_t count = sv.size();
    if (count == 0) return 0;
    for (size_t i = 0; i < count; ++i) {
        prefix = (prefix << 8) | static_cast<unsigned char>(sv[i]);
    }
    prefix <<= (8 - count) * 8;
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
    using PreparedHash = std::array<uint64_t, 2>;
    explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01);
    void Add(std::string_view key);
    void AddHash(uint64_t hash);
    bool MayContain(std::string_view key) const;
    bool MayContainHash(const PreparedHash& hash) const;
    static PreparedHash PrepareHash(uint64_t hash);
    size_t ApproximateMemoryUsage() const { return bits_.capacity() / 8; }

   private:
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
    AddHash(XXH3_64bits(key.data(), key.size()));
}
inline void BloomFilter::AddHash(uint64_t hash) {
    const auto h = PrepareHash(hash);
    for (int i = 0; i < num_hashes_; ++i) {
        const uint64_t bit_hash = h[0] + i * h[1];
        if (!bits_.empty()) bits_[bit_hash % bits_.size()] = true;
    }
}
inline bool BloomFilter::MayContain(std::string_view key) const {
    return MayContainHash(PrepareHash(XXH3_64bits(key.data(), key.size())));
}
inline bool BloomFilter::MayContainHash(const PreparedHash& h) const {
    if (bits_.empty()) return true;
    for (int i = 0; i < num_hashes_; ++i) {
        const uint64_t bit_hash = h[0] + i * h[1];
        if (!bits_[bit_hash % bits_.size()]) return false;
    }
    return true;
}
inline BloomFilter::PreparedHash BloomFilter::PrepareHash(uint64_t hash) {
    uint64_t delta = hash ^ 0x9e3779b97f4a7c15ULL;
    delta ^= delta >> 30;
    delta *= 0xbf58476d1ce4e5b9ULL;
    delta ^= delta >> 27;
    delta *= 0x94d049bb133111ebULL;
    delta ^= delta >> 31;
    return {hash, delta | 1ULL};
}

// --- Columnar MemTable Components ---
struct StoredRecord {
    RecordRef record;
    std::atomic<bool> ready{false};
};

struct HashedBatchEntry {
    std::string_view key;
    std::string_view value;
    uint64_t hash;
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
    friend class FlashActiveBlock;
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
    std::vector<const StoredRecord*> AllocateAndAppendBatch(const std::vector<HashedBatchEntry>& batch, size_t offset,
                                                            size_t count, RecordType type);
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

    if (required_size > DataChunk::kBufferCapacity) {
        throw std::length_error("record is larger than ColumnarRecordArena::DataChunk::kBufferCapacity");
    }

    uint32_t record_idx;
    uint32_t buffer_offset;
    DataChunk* allocated_chunk = nullptr;

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
                allocated_chunk = chunk;
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
    char* key_mem = allocated_chunk->buffer + buffer_offset;
    memcpy(key_mem, key.data(), key.size());
    char* val_mem = key_mem + key.size();
    memcpy(val_mem, value.data(), value.size());

    StoredRecord& record_slot = allocated_chunk->records[record_idx];
    record_slot.record = {{key_mem, key.size()}, {val_mem, value.size()}, type};
    record_slot.ready.store(true, std::memory_order_release);

    size_.fetch_add(1, std::memory_order_release);
    return &record_slot;
}

// Rewritten batch allocation with the same robust CAS-based approach.
inline std::vector<const StoredRecord*> ColumnarRecordArena::AllocateAndAppendBatch(
    const std::vector<HashedBatchEntry>& batch, size_t offset, size_t count, RecordType type) {
    std::vector<const StoredRecord*> results;
    if (offset >= batch.size() || count == 0) return results;
    const size_t batch_end = std::min(batch.size(), offset + count);
    results.reserve(batch_end - offset);

    ThreadLocalData* tls_data = GetTlsData();
    size_t batch_offset = offset;

    while (batch_offset < batch_end) {
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
            for (size_t i = batch_offset; i < batch_end; ++i) {
                const auto& entry = batch[i];
                const auto key = entry.key;
                const auto value = entry.value;
                size_t item_size = key.size() + value.size();
                if (item_size > DataChunk::kBufferCapacity) {
                    throw std::length_error("record is larger than ColumnarRecordArena::DataChunk::kBufferCapacity");
                }
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
                tls_data->AddNewChunk();
                memory_usage_.fetch_add(sizeof(DataChunk), std::memory_order_relaxed);
            }
            continue;  // Retry with the new chunk.
        }

    batch_allocation_success:
        size_t current_buffer_offset_in_batch = 0;
        for (uint32_t i = 0; i < records_to_alloc; ++i) {
            const auto& entry = batch[batch_offset + i];
            const auto key = entry.key;
            const auto value = entry.value;
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
        batch_offset += records_to_alloc;
    }

    return results;
}

class ConcurrentStringHashMap {
   public:
    static constexpr uint8_t EMPTY_TAG = 0xFF, LOCKED_TAG = 0xFE;

   private:
    struct Slot {
        std::atomic<const StoredRecord*> record;
        std::atomic<uint8_t> tag;
    };
    static_assert(sizeof(Slot) <= 16, "active hash slots should remain cache compact");
    std::unique_ptr<Slot[]> slots_;
    size_t capacity_, capacity_mask_;
    XXHasher hasher_;

   public:
    ConcurrentStringHashMap(const ConcurrentStringHashMap&) = delete;
    ConcurrentStringHashMap& operator=(const ConcurrentStringHashMap&) = delete;
    static size_t calculate_power_of_2(size_t n) {
        if (n <= 1) return 1;
        return 1UL << (64 - __builtin_clzll(n - 1));
    }
    explicit ConcurrentStringHashMap(size_t build_size);
    void Insert(std::string_view key, const StoredRecord* new_record);
    void Insert(std::string_view key, uint64_t hash, const StoredRecord* new_record);
    const StoredRecord* Find(std::string_view key) const;
    const StoredRecord* Find(std::string_view key, uint64_t hash) const;
    void PrefetchForInsert(uint64_t hash) const { __builtin_prefetch(&slots_[hash & capacity_mask_], 1, 1); }
    size_t ApproximateMemoryUsage() const { return sizeof(*this) + capacity_ * sizeof(Slot); }
};

inline ConcurrentStringHashMap::ConcurrentStringHashMap(size_t build_size) {
    size_t capacity = calculate_power_of_2(build_size + build_size / 2 + 64);
    capacity_ = capacity;
    capacity_mask_ = capacity - 1;
    slots_ = std::make_unique<Slot[]>(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
        slots_[i].record.store(nullptr, std::memory_order_relaxed);
    }
}
inline void ConcurrentStringHashMap::Insert(std::string_view key, const StoredRecord* new_record) {
    Insert(key, hasher_(key), new_record);
}
inline void ConcurrentStringHashMap::Insert(std::string_view key, uint64_t hash, const StoredRecord* new_record) {
    uint8_t tag = (hash >> 56);
    if (tag >= LOCKED_TAG) tag = 0;
    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;
    while (true) {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
        if (current_tag == LOCKED_TAG) {
            // The slot's key has not been published yet.  Advancing could let
            // two concurrent inserts of the same key occupy different slots,
            // so wait and re-check this probe position.
            __builtin_ia32_pause();
            continue;
        }
        if (current_tag == tag) {
            const StoredRecord* current_record = slots_[pos].record.load(std::memory_order_acquire);
            if (current_record && current_record->record.key == key) {
                slots_[pos].record.store(new_record, std::memory_order_release);
                return;
            }
        }
        if (current_tag == EMPTY_TAG) {
            uint8_t expected_empty = EMPTY_TAG;
            if (slots_[pos].tag.compare_exchange_strong(expected_empty, LOCKED_TAG, std::memory_order_acq_rel)) {
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
inline const StoredRecord* ConcurrentStringHashMap::Find(std::string_view key) const { return Find(key, hasher_(key)); }
inline const StoredRecord* ConcurrentStringHashMap::Find(std::string_view key, uint64_t hash) const {
    uint8_t tag = (hash >> 56);
    if (tag >= LOCKED_TAG) tag = 0;
    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;
    do {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
        if (current_tag == EMPTY_TAG) return nullptr;
        if (current_tag == tag) {
            const StoredRecord* rec = slots_[pos].record.load(std::memory_order_acquire);
            if (rec && rec->record.key == key && rec->ready.load(std::memory_order_acquire)) {
                return rec;
            }
        }
        pos = (pos + 1) & capacity_mask_;
    } while (pos != initial_pos);
    return nullptr;
}
class FlashActiveBlock {
    friend class ColumnarMemTable;

   public:
    explicit FlashActiveBlock(size_t cap) : data_log_(std::make_shared<ColumnarRecordArena>()), index_(cap) {}
    ~FlashActiveBlock() = default;

    // With the new Arena, TryAdd only fails if the block is sealed or the item is too large.
    // The caller no longer needs to retry on chunk-full conditions.
    bool TryAdd(std::string_view key, std::string_view value, RecordType type) {
        return TryAdd(key, value, type, XXHasher{}(key));
    }

    bool TryAdd(std::string_view key, std::string_view value, RecordType type, uint64_t hash) {
        if (!BeginWrite()) return false;
        try {
            const StoredRecord* record_ptr = data_log_->AllocateAndAppend(key, value, type);
            index_.Insert(record_ptr->record.key, hash, record_ptr);
            EndWrite();
            return true;
        } catch (...) {
            EndWrite();
            throw;
        }
    }

    bool TryAddBatch(const std::vector<HashedBatchEntry>& batch, size_t offset, size_t count,
                     RecordType type) {
        if (!BeginWrite()) return false;
        try {
            auto record_ptrs = data_log_->AllocateAndAppendBatch(batch, offset, count, type);
            constexpr size_t kPrefetchDistance = 8;
            for (size_t i = 0; i < record_ptrs.size(); ++i) {
                if (i + kPrefetchDistance < record_ptrs.size()) {
                    index_.PrefetchForInsert(batch[offset + i + kPrefetchDistance].hash);
                }
                const auto* record_ptr = record_ptrs[i];
                index_.Insert(record_ptr->record.key, batch[offset + i].hash, record_ptr);
            }
            EndWrite();
            return true;
        } catch (...) {
            EndWrite();
            throw;
        }
    }

    std::optional<RecordRef> Get(std::string_view key) const {
        const StoredRecord* record_ptr = index_.Find(key);
        return record_ptr ? std::optional<RecordRef>(record_ptr->record) : std::nullopt;
    }

    std::optional<RecordRef> Get(std::string_view key, uint64_t hash) const {
        const StoredRecord* record_ptr = index_.Find(key, hash);
        return record_ptr ? std::optional<RecordRef>(record_ptr->record) : std::nullopt;
    }

    size_t size() const { return data_log_->size(); }
    size_t ApproximateMemoryUsage() const {
        return data_log_->ApproximateMemoryUsage() + index_.ApproximateMemoryUsage();
    }
    void Seal() {
        sealed_.store(true, std::memory_order_release);
        while (active_writers_.load(std::memory_order_acquire) != 0) __builtin_ia32_pause();
    }
    bool is_sealed() const { return sealed_.load(std::memory_order_acquire); }
    void AppendReferencesTo(ColumnarBlock& destination) const;

   private:
    bool BeginWrite() {
        if (is_sealed()) return false;
        active_writers_.fetch_add(1, std::memory_order_acq_rel);
        if (!is_sealed()) return true;
        active_writers_.fetch_sub(1, std::memory_order_release);
        return false;
    }
    void EndWrite() { active_writers_.fetch_sub(1, std::memory_order_release); }

    std::shared_ptr<ColumnarRecordArena> data_log_;
    ConcurrentStringHashMap index_;
    std::atomic<uint32_t> active_writers_{0};
    std::atomic<bool> sealed_{false};
};

class ColumnarBlock {
   public:
    class SimpleArena {
       public:
        void Reserve(size_t bytes) {
            if (bytes == 0 || current_block_idx_ >= 0) return;
            blocks_.emplace_back(bytes);
            current_block_idx_ = 0;
        }
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
    std::vector<uint64_t> key_prefixes;
    std::vector<RecordType> types;
    void Reserve(size_t count) {
        keys.reserve(count);
        values.reserve(count);
        key_prefixes.reserve(count);
        types.reserve(count);
    }
    void Add(std::string_view k, std::string_view v, RecordType t) {
        keys.push_back(arena.AllocateAndCopy(k));
        values.push_back(arena.AllocateAndCopy(v));
        key_prefixes.push_back(load_u64_prefix(k));
        types.push_back(t);
    }
    void SetBackingArena(std::shared_ptr<const ColumnarRecordArena> backing_arena) {
        backing_arena_ = std::move(backing_arena);
    }
    void AddReference(std::string_view k, std::string_view v, RecordType t) {
        keys.push_back(k);
        values.push_back(v);
        key_prefixes.push_back(load_u64_prefix(k));
        types.push_back(t);
    }
    void ReorderColumns(const std::vector<uint32_t>& order) {
        std::vector<std::string_view> sorted_keys;
        std::vector<std::string_view> sorted_values;
        std::vector<uint64_t> sorted_prefixes;
        std::vector<RecordType> sorted_types;
        sorted_keys.reserve(order.size());
        sorted_values.reserve(order.size());
        sorted_prefixes.reserve(order.size());
        sorted_types.reserve(order.size());
        for (const uint32_t index : order) {
            sorted_keys.push_back(keys[index]);
            sorted_values.push_back(values[index]);
            sorted_prefixes.push_back(key_prefixes[index]);
            sorted_types.push_back(types[index]);
        }
        keys = std::move(sorted_keys);
        values = std::move(sorted_values);
        key_prefixes = std::move(sorted_prefixes);
        types = std::move(sorted_types);
    }
    void MaterializePayloadInCurrentOrder() {
        SimpleArena sorted_arena;
        size_t payload_bytes = 0;
        for (size_t i = 0; i < keys.size(); ++i) payload_bytes += keys[i].size() + values[i].size();
        sorted_arena.Reserve(payload_bytes);
        for (size_t i = 0; i < keys.size(); ++i) {
            keys[i] = sorted_arena.AllocateAndCopy(keys[i]);
            values[i] = sorted_arena.AllocateAndCopy(values[i]);
        }
        arena = std::move(sorted_arena);
        backing_arena_.reset();
    }
    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }
    void Clear() {
        keys.clear();
        values.clear();
        key_prefixes.clear();
        types.clear();
        backing_arena_.reset();
        arena = SimpleArena();
    }
    size_t ApproximateMemoryUsage() const {
        size_t total = arena.ApproximateMemoryUsage();
        total += keys.capacity() * sizeof(std::string_view);
        total += values.capacity() * sizeof(std::string_view);
        total += key_prefixes.capacity() * sizeof(uint64_t);
        total += types.capacity() * sizeof(RecordType);
        if (backing_arena_) total += backing_arena_->ApproximateMemoryUsage();
        return total;
    }

   private:
    std::shared_ptr<const ColumnarRecordArena> backing_arena_;
};

inline void FlashActiveBlock::AppendReferencesTo(ColumnarBlock& destination) const {
    destination.SetBackingArena(data_log_);
    const uint32_t active_threads =
        std::min<uint32_t>(data_log_->GetMaxThreadIdSeen() + 1, ThreadIdManager::kMaxThreads);
    for (uint32_t thread_idx = 0; thread_idx < active_threads; ++thread_idx) {
        const auto* tls_data = data_log_->GetAllTlsData()[thread_idx].load(std::memory_order_acquire);
        if (!tls_data) continue;
        for (const auto& chunk_ptr : tls_data->chunks) {
            uint32_t max_idx = static_cast<uint32_t>(chunk_ptr->positions_.load(std::memory_order_relaxed) >> 32);
            max_idx = std::min<uint32_t>(max_idx, ColumnarRecordArena::DataChunk::kRecordCapacity);
            for (uint32_t i = 0; i < max_idx; ++i) {
                const auto& record_slot = chunk_ptr->records[i];
                while (!record_slot.ready.load(std::memory_order_acquire)) __builtin_ia32_pause();
                destination.AddReference(record_slot.record.key, record_slot.record.value, record_slot.record.type);
            }
        }
    }
}

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
        std::sort(indices.begin(), indices.end(), [&block](uint32_t a, uint32_t b) {
            if (block.key_prefixes[a] != block.key_prefixes[b]) {
                return block.key_prefixes[a] < block.key_prefixes[b];
            }
            if (block.keys[a] != block.keys[b]) return block.keys[a] < block.keys[b];
            // Equivalent to stable ordering for duplicate keys, without
            // stable_sort's auxiliary merge buffer.
            return a < b;
        });
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
    bool MayContain(std::string_view key, const BloomFilter::PreparedHash& hash) const;
    std::optional<RecordRef> Get(std::string_view key) const;
    std::optional<RecordRef> Get(std::string_view key, const BloomFilter::PreparedHash& hash) const;
    std::string_view min_key() const { return min_key_; }
    std::string_view max_key() const { return max_key_; }
    Iterator begin() const;
    bool empty() const { return !block_data_ || block_data_->empty(); }
    size_t size() const { return block_data_ ? block_data_->size() : 0; }
    size_t ApproximateMemoryUsage() const {
        size_t usage = sizeof(*this);
        if (block_data_) usage += block_data_->ApproximateMemoryUsage();
        if (bloom_filter_) usage += bloom_filter_->ApproximateMemoryUsage();
        usage += sparse_index_.capacity() * sizeof(std::pair<std::string_view, size_t>);
        return usage;
    }

   private:
    friend class Iterator;
    std::shared_ptr<ColumnarBlock> block_data_;
    std::string_view min_key_, max_key_;
    std::unique_ptr<BloomFilter> bloom_filter_;
    std::vector<std::pair<std::string_view, size_t>> sparse_index_;
};
inline SortedColumnarBlock::SortedColumnarBlock(std::shared_ptr<ColumnarBlock> b, const Sorter& s,
                                                bool build_bloom_filter)
    : block_data_(std::move(b)) {
    auto sort_order = s.Sort(*block_data_);
    if (sort_order.empty()) {
        min_key_ = {};
        max_key_ = {};
        return;
    }
    block_data_->ReorderColumns(sort_order);
    block_data_->MaterializePayloadInCurrentOrder();
    min_key_ = block_data_->keys.front();
    max_key_ = block_data_->keys.back();

    if (build_bloom_filter) {
        bloom_filter_ = std::make_unique<BloomFilter>(block_data_->size());
        for (size_t i = 0; i < block_data_->size(); ++i) {
            bloom_filter_->AddHash(XXH3_64bits(block_data_->keys[i].data(), block_data_->keys[i].size()));
        }
    }

    sparse_index_.reserve(block_data_->size() / kSparseIndexSampleRate + 1);
    for (size_t i = 0; i < block_data_->size(); i += kSparseIndexSampleRate)
        sparse_index_.emplace_back(block_data_->keys[i], i);
}

inline bool SortedColumnarBlock::MayContain(std::string_view key) const {
    return MayContain(key, BloomFilter::PrepareHash(XXH3_64bits(key.data(), key.size())));
}

inline bool SortedColumnarBlock::MayContain(std::string_view key, const BloomFilter::PreparedHash& hash) const {
    if (empty() || key < min_key_ || key > max_key_) return false;
    if (!bloom_filter_) {
        return true;
    }
    return bloom_filter_->MayContainHash(hash);
}

inline std::optional<RecordRef> SortedColumnarBlock::Get(std::string_view key) const {
    return Get(key, BloomFilter::PrepareHash(XXH3_64bits(key.data(), key.size())));
}

inline std::optional<RecordRef> SortedColumnarBlock::Get(
    std::string_view key, const BloomFilter::PreparedHash& hash) const {
    if (!MayContain(key, hash)) return std::nullopt;

    auto sparse_it = std::lower_bound(sparse_index_.begin(), sparse_index_.end(), key,
                                      [](const auto& a, auto b) { return a.first < b; });
    auto start_it = block_data_->keys.begin();
    if (sparse_it != sparse_index_.begin()) start_it += (sparse_it - 1)->second;

    auto end_it = block_data_->keys.end();
    if (sparse_it != sparse_index_.end()) {
        end_it = block_data_->keys.begin() + sparse_it->second + kSparseIndexSampleRate;
        if (end_it > block_data_->keys.end()) end_it = block_data_->keys.end();
    }

    auto it = std::lower_bound(start_it, end_it, key);

    if (it == end_it || *it != key) {
        return std::nullopt;
    }

    // The sorter's original-index tie-break ensures that for identical keys,
    // the one inserted later appears later in the sorted list.
    // We want the latest version, so find the end of the range of equal keys and take the one just before it.
    auto range_end = std::upper_bound(it, end_it, key);

    const size_t latest_idx = static_cast<size_t>(std::prev(range_end) - block_data_->keys.begin());
    return RecordRef{block_data_->keys[latest_idx], block_data_->values[latest_idx], block_data_->types[latest_idx]};
}

class SortedColumnarBlock::Iterator {
   public:
    Iterator(const SortedColumnarBlock* b, size_t p) : block_(b), pos_(p) {}
    RecordRef operator*() const {
        return {block_->block_data_->keys[pos_], block_->block_data_->values[pos_], block_->block_data_->types[pos_]};
    }
    uint64_t KeyPrefix() const { return block_->block_data_->key_prefixes[pos_]; }
    void Next() { ++pos_; }
    bool IsValid() const { return block_ && pos_ < block_->size(); }

   private:
    const SortedColumnarBlock* block_;
    size_t pos_;
};
inline SortedColumnarBlock::Iterator SortedColumnarBlock::begin() const { return Iterator(this, 0); }
class FlushIterator {
   public:
    explicit FlushIterator(const std::vector<std::shared_ptr<const SortedColumnarBlock>>& sources);
    bool IsValid() const { return winner_tree_.size() > 1 && winner_tree_[1] != kNoSource; }
    RecordRef Get() const { return current_records_[winner_tree_[1]]; }
    void Next();

   private:
    static constexpr size_t kNoSource = std::numeric_limits<size_t>::max();
    size_t EarlierSource(size_t left, size_t right) const;
    std::vector<SortedColumnarBlock::Iterator> iterators_;
    std::vector<RecordRef> current_records_;
    std::vector<uint64_t> current_prefixes_;
    std::vector<size_t> winner_tree_;
    size_t leaf_count_ = 1;
};
inline FlushIterator::FlushIterator(const std::vector<std::shared_ptr<const SortedColumnarBlock>>& sources) {
    iterators_.reserve(sources.size());
    current_records_.resize(sources.size());
    current_prefixes_.resize(sources.size());
    while (leaf_count_ < sources.size()) leaf_count_ <<= 1;
    winner_tree_.assign(leaf_count_ * 2, kNoSource);
    for (size_t i = 0; i < sources.size(); ++i) {
        if (sources[i]) {
            iterators_.emplace_back(sources[i]->begin());
        } else {
            iterators_.emplace_back(nullptr, 0);
        }
        if (iterators_.back().IsValid()) {
            current_records_[i] = *iterators_.back();
            current_prefixes_[i] = iterators_.back().KeyPrefix();
            winner_tree_[leaf_count_ + i] = i;
        }
    }
    for (size_t node = leaf_count_; node-- > 1;) {
        winner_tree_[node] = EarlierSource(winner_tree_[node * 2], winner_tree_[node * 2 + 1]);
    }
}
inline size_t FlushIterator::EarlierSource(size_t left, size_t right) const {
    if (left == kNoSource) return right;
    if (right == kNoSource) return left;
    if (current_prefixes_[left] != current_prefixes_[right]) {
        return current_prefixes_[left] < current_prefixes_[right] ? left : right;
    }
    if (current_records_[left].key != current_records_[right].key) {
        return current_records_[left].key < current_records_[right].key ? left : right;
    }
    // Lower source indices contain older blocks and must be emitted first so
    // CompactingIterator retains the newest value from an equal-key run.
    return std::min(left, right);
}
inline void FlushIterator::Next() {
    if (!IsValid()) return;
    const size_t source_index = winner_tree_[1];
    iterators_[source_index].Next();
    if (iterators_[source_index].IsValid()) {
        current_records_[source_index] = *iterators_[source_index];
        current_prefixes_[source_index] = iterators_[source_index].KeyPrefix();
    } else {
        winner_tree_[leaf_count_ + source_index] = kNoSource;
    }
    for (size_t node = (leaf_count_ + source_index) / 2; node != 0; node /= 2) {
        winner_tree_[node] = EarlierSource(winner_tree_[node * 2], winner_tree_[node * 2 + 1]);
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
        virtual bool Pop(RecordRef& record) = 0;
    };
    template <typename It>
    struct ItWrapper final : public ItConcept {
        explicit ItWrapper(std::unique_ptr<It> i) : iter_(std::move(i)) {}
        bool Pop(RecordRef& record) override {
            if (!iter_->IsValid()) return false;
            record = iter_->Get();
            iter_->Next();
            return true;
        }
        std::unique_ptr<It> iter_;
    };
    void FindNext();
    std::unique_ptr<ItConcept> source_;
    RecordRef current_record_;
    RecordRef lookahead_record_;
    bool has_lookahead_ = false;
    bool is_valid_ = false;
};
template <typename It>
inline CompactingIterator::CompactingIterator(std::unique_ptr<It> s)
    : source_(std::make_unique<ItWrapper<It>>(std::move(s))) {
    FindNext();
}
inline void CompactingIterator::FindNext() {
    while (true) {
        RecordRef latest_record;
        if (has_lookahead_) {
            latest_record = lookahead_record_;
            has_lookahead_ = false;
        } else if (!source_->Pop(latest_record)) {
            is_valid_ = false;
            return;
        }

        RecordRef next_record;
        while (source_->Pop(next_record)) {
            if (next_record.key != latest_record.key) {
                lookahead_record_ = next_record;
                has_lookahead_ = true;
                break;
            }
            latest_record = next_record;
        }
        if (latest_record.type == RecordType::Put) {
            current_record_ = latest_record;
            is_valid_ = true;
            return;
        }
    }
}
class ColumnarMemTable : public std::enable_shared_from_this<ColumnarMemTable> {
   public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::vector<GetResult>;
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
        std::shared_ptr<Sorter> sorter = std::make_shared<ParallelRadixSorter>(), size_t num_shards = 16,
        size_t batch_worker_count = 1, bool async_background_processing = true) {
        struct MakeSharedEnabler : public ColumnarMemTable {
            MakeSharedEnabler(size_t active_block_size_bytes, bool enable_compaction, std::shared_ptr<Sorter> sorter,
                              size_t num_shards, size_t batch_worker_count, bool async_background_processing)
                : ColumnarMemTable(active_block_size_bytes, enable_compaction, std::move(sorter), num_shards,
                                   batch_worker_count, async_background_processing) {}
        };
        auto table = std::make_shared<MakeSharedEnabler>(active_block_size_bytes, enable_compaction, std::move(sorter),
                                                         num_shards, batch_worker_count, async_background_processing);
        if (async_background_processing) table->StartBackgroundThread();
        return table;
    }

    void Put(std::string_view key, std::string_view value) { Insert(key, value, RecordType::Put); }
    void Delete(std::string_view key) { Insert(key, "", RecordType::Delete); }
    GetResult Get(std::string_view key) const;
    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const;
    void PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch);
    void WaitForBackgroundWork();
    void WaitForPendingBackgroundWork();
    void WaitForBackgroundBacklogAtMost(size_t maximum_pending_blocks);
    std::unique_ptr<CompactingIterator> NewCompactingIterator();
    size_t ApproximateMemoryUsage() const;
    size_t GetActiveRecordCount() const;
    size_t GetPendingSealedBlockNum() const;
    size_t GetPendingSealedRecordCount() const;
    size_t GetSortedBlockNum() const;
    size_t GetSortedRecordCount() const;
    size_t BatchWorkerCount() const { return batch_worker_count_; }

   private:
    // Make constructor private for factory pattern.
    explicit ColumnarMemTable(size_t active_block_size_bytes, bool enable_compaction, std::shared_ptr<Sorter> sorter,
                              size_t num_shards, size_t batch_worker_count, bool async_background_processing)
        : instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed)),
          active_block_threshold_(std::max((size_t)1, active_block_size_bytes / 116)),
          enable_compaction_(enable_compaction),
          sorter_(std::move(sorter)),
          num_shards_(ConcurrentStringHashMap::calculate_power_of_2(std::max<size_t>(1, num_shards))),
          shard_mask_(num_shards_ - 1),
          batch_worker_count_(std::max<size_t>(1, batch_worker_count)),
          batch_pool_(batch_worker_count_ > 1 ? std::make_shared<ThreadPool>(batch_worker_count_) : nullptr),
          async_background_processing_(async_background_processing) {
        for (size_t i = 0; i < num_shards_; ++i) {
            shards_.push_back(std::make_unique<Shard>(CreateActiveBlock()));
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
        explicit Shard(std::shared_ptr<FlashActiveBlock> active_block) {
            active_block_ = std::move(active_block);
            immutable_state_ = std::make_shared<const ImmutableState>();
        }
    };
    struct BackgroundWorkItem {
        std::shared_ptr<FlashActiveBlock> block;
        std::unique_ptr<std::promise<void>> promise;
        size_t shard_idx;
    };

    static inline std::atomic<uint64_t> next_instance_id_{1};
    const uint64_t instance_id_;
    const size_t active_block_threshold_;
    const bool enable_compaction_;
    std::shared_ptr<Sorter> sorter_;
    const size_t num_shards_;
    const size_t shard_mask_;
    const size_t batch_worker_count_;
    std::shared_ptr<ThreadPool> batch_pool_;
    const bool async_background_processing_;
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
    void PutBatchForShard(size_t shard_idx, const std::vector<HashedBatchEntry>& batch);
    void SealActiveBlockIfNeeded(size_t shard_idx);
    void SealNonEmptyActiveBlocksLocked();
    void BackgroundWorkerLoop();
    void ProcessBackgroundWorkItems(std::vector<BackgroundWorkItem> work_items);
    void ProcessBlocksForShard(size_t shard_idx,
                               const std::vector<std::shared_ptr<FlashActiveBlock>>& sealed_blocks);
    std::shared_ptr<ColumnarBlock> GetPooledColumnarBlock();
    std::shared_ptr<FlashActiveBlock> CreateActiveBlock() const;

    std::shared_ptr<FlashActiveBlock> GetActiveBlockForThread(size_t shard_idx, bool force_refresh = false) const;
    std::shared_ptr<const ImmutableState> GetImmutableStateForThread(size_t shard_idx,
                                                                     bool force_refresh = false) const;
};

// --- Implementation ---
inline std::shared_ptr<FlashActiveBlock> ColumnarMemTable::CreateActiveBlock() const {
    return std::make_shared<FlashActiveBlock>(active_block_threshold_);
}

inline void ColumnarMemTable::Insert(std::string_view k, std::string_view v, RecordType t) {
    const uint64_t hash = hasher_(k);
    const size_t shard_idx = hash & shard_mask_;
    auto current_block = GetActiveBlockForThread(shard_idx);
    // The new arena handles chunk-full retries internally, so this loop mainly handles
    // the case where a block is sealed by another thread while we are trying to add to it.
    while (!current_block->TryAdd(k, v, t, hash)) {
        current_block = GetActiveBlockForThread(shard_idx, true);
    }
    if (current_block->size() >= active_block_threshold_) {
        SealActiveBlockIfNeeded(shard_idx);
    }
}
inline ColumnarMemTable::GetResult ColumnarMemTable::Get(std::string_view key) const {
    const uint64_t hash = hasher_(key);
    const auto bloom_hash = BloomFilter::PrepareHash(hash);
    const size_t shard_idx = hash & shard_mask_;
    auto active_block = GetActiveBlockForThread(shard_idx);
    if (auto r = active_block->Get(key, hash)) {
        return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
    }
    auto s = GetImmutableStateForThread(shard_idx);
    if (s->sealed_blocks) {
        for (auto it = s->sealed_blocks->rbegin(); it != s->sealed_blocks->rend(); ++it) {
            if (auto r = (*it)->Get(key, hash)) {
                return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
            }
        }
    }
    if (s->blocks) {
        for (auto it = s->blocks->rbegin(); it != s->blocks->rend(); ++it) {
            if (auto r = (*it)->Get(key, bloom_hash)) {
                return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
            }
        }
    }
    return std::nullopt;
}
inline void ColumnarMemTable::PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
    if (batch.empty()) return;
    std::vector<std::vector<HashedBatchEntry>> sharded_batches(num_shards_);
    const size_t expected_per_shard = batch.size() / num_shards_ + 1;
    for (auto& shard_batch : sharded_batches) shard_batch.reserve(expected_per_shard);
    for (const auto& [key, value] : batch) {
        const uint64_t hash = hasher_(key);
        sharded_batches[hash & shard_mask_].push_back({key, value, hash});
    }

    if (!batch_pool_) {
        for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
            if (!sharded_batches[shard_idx].empty()) PutBatchForShard(shard_idx, sharded_batches[shard_idx]);
        }
        return;
    }

    std::vector<std::future<void>> futures;
    futures.reserve(std::min(num_shards_, batch_worker_count_));
    for (size_t shard_idx = 0; shard_idx < num_shards_; ++shard_idx) {
        if (sharded_batches[shard_idx].empty()) continue;
        futures.emplace_back(batch_pool_->Submit(
            [this, shard_idx, &sub_batch = sharded_batches[shard_idx]] { PutBatchForShard(shard_idx, sub_batch); }));
    }
    for (auto& f : futures) {
        f.get();
    }
}

inline void ColumnarMemTable::PutBatchForShard(size_t shard_idx, const std::vector<HashedBatchEntry>& batch) {
    size_t offset = 0;
    while (offset < batch.size()) {
        auto current_block = GetActiveBlockForThread(shard_idx);
        const size_t current_size = current_block->size();
        if (current_block->is_sealed() || current_size >= active_block_threshold_) {
            SealActiveBlockIfNeeded(shard_idx);
            current_block = GetActiveBlockForThread(shard_idx, true);
            continue;
        }

        const size_t count = std::min(batch.size() - offset, active_block_threshold_ - current_size);
        if (!current_block->TryAddBatch(batch, offset, count, RecordType::Put)) {
            // A concurrent seal can race with the append. Retrying in the newer
            // block preserves the same visibility rule as scalar Put.
            GetActiveBlockForThread(shard_idx, true);
            continue;
        }
        offset += count;
        if (current_block->size() >= active_block_threshold_) SealActiveBlockIfNeeded(shard_idx);
    }
}
inline ColumnarMemTable::MultiGetResult ColumnarMemTable::MultiGet(const std::vector<std::string_view>& keys) const {
    if (keys.empty()) return {};
    struct IndexedKey {
        std::string_view key;
        uint64_t hash;
        size_t result_index;
    };
    MultiGetResult results(keys.size());
    std::vector<std::vector<IndexedKey>> sharded_keys(num_shards_);
    std::vector<size_t> active_shards;
    active_shards.reserve(num_shards_);
    for (size_t index = 0; index < keys.size(); ++index) {
        const auto key = keys[index];
        const uint64_t hash = hasher_(key);
        const size_t shard_idx = hash & shard_mask_;
        if (sharded_keys[shard_idx].empty()) active_shards.push_back(shard_idx);
        sharded_keys[shard_idx].push_back({key, hash, index});
    }
    auto process_shards = [this, &sharded_keys, &results](const std::vector<size_t>& shards_to_process) {
        for (const size_t shard_idx : shards_to_process) {
            auto active_block = GetActiveBlockForThread(shard_idx);
            auto s = GetImmutableStateForThread(shard_idx);
            for (const auto& entry : sharded_keys[shard_idx]) {
                const auto key = entry.key;
                const uint64_t hash = entry.hash;
                const auto bloom_hash = BloomFilter::PrepareHash(hash);
                if (auto r = active_block->Get(key, hash)) {
                    results[entry.result_index] =
                        (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                    continue;
                }
                if (s->sealed_blocks) {
                    bool found = false;
                    for (auto it = s->sealed_blocks->rbegin(); it != s->sealed_blocks->rend(); ++it) {
                        if (auto r = (*it)->Get(key, hash)) {
                            results[entry.result_index] =
                                (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                }
                if (s->blocks) {
                    bool found = false;
                    for (auto it = s->blocks->rbegin(); it != s->blocks->rend(); ++it) {
                        if (auto r = (*it)->Get(key, bloom_hash)) {
                            results[entry.result_index] =
                                (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                }
            }
        }
    };
    if (active_shards.size() < 2 || !batch_pool_) {
        process_shards(active_shards);
    } else {
        const size_t num_workers = std::min(batch_worker_count_, active_shards.size());
        std::vector<std::vector<size_t>> workloads(num_workers);
        for (size_t i = 0; i < active_shards.size(); ++i) {
            workloads[i % num_workers].push_back(active_shards[i]);
        }
        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);
        for (const auto& workload : workloads) {
            if (workload.empty()) continue;
            futures.emplace_back(batch_pool_->Submit([&process_shards, workload] { process_shards(workload); }));
        }
        for (auto& future : futures) future.get();
    }
    return results;
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
        auto new_active_block = CreateActiveBlock();
        auto old_s = std::atomic_load(&shard.immutable_state_);
        auto new_s = std::make_shared<ImmutableState>();
        new_s->blocks = old_s->blocks;
        auto new_sealed_list = std::make_shared<ImmutableState::SealedBlockList>(*old_s->sealed_blocks);
        new_sealed_list->push_back(sealed_block);
        new_s->sealed_blocks = new_sealed_list;
        std::atomic_store(&shard.immutable_state_, std::shared_ptr<const ImmutableState>(new_s));
        // Publish the sealed block before replacing the active pointer.  A Get
        // may temporarily inspect the old block twice, but can never miss it in
        // a gap between the two snapshots.
        std::atomic_exchange(&shard.active_block_, new_active_block);
        shard.version_.fetch_add(1, std::memory_order_release);
    }
    {
        std::lock_guard<std::mutex> ql(queue_mutex_);
        sealed_blocks_queue_.push_back({std::move(sealed_block), nullptr, shard_idx});
    }
    queue_cond_.notify_one();
}
inline void ColumnarMemTable::SealNonEmptyActiveBlocksLocked() {
    for (size_t i = 0; i < num_shards_; ++i) {
        auto& shard = *shards_[i];
        std::lock_guard<SpinLock> seal_lock(shard.seal_mutex_);
        auto active_block = std::atomic_load(&shard.active_block_);
        if (active_block->size() == 0 || active_block->is_sealed()) continue;
        active_block->Seal();
        auto new_block = CreateActiveBlock();
        auto old_state = std::atomic_load(&shard.immutable_state_);
        auto new_state = std::make_shared<ImmutableState>();
        new_state->blocks = old_state->blocks;
        auto new_sealed_list =
            std::make_shared<ImmutableState::SealedBlockList>(*old_state->sealed_blocks);
        new_sealed_list->push_back(active_block);
        new_state->sealed_blocks = new_sealed_list;
        std::atomic_store(&shard.immutable_state_, std::shared_ptr<const ImmutableState>(new_state));
        std::atomic_exchange(&shard.active_block_, new_block);
        shard.version_.fetch_add(1, std::memory_order_release);
        sealed_blocks_queue_.push_back({std::move(active_block), nullptr, i});
    }
}

inline void ColumnarMemTable::WaitForBackgroundWork() {
    if (!async_background_processing_) {
        std::vector<BackgroundWorkItem> work_items;
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            SealNonEmptyActiveBlocksLocked();
            work_items.swap(sealed_blocks_queue_);
        }
        ProcessBackgroundWorkItems(std::move(work_items));
        return;
    }

    auto promise = std::make_unique<std::promise<void>>();
    auto future = promise->get_future();
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        SealNonEmptyActiveBlocksLocked();
        sealed_blocks_queue_.push_back({nullptr, std::move(promise), 0});
    }
    queue_cond_.notify_one();
    future.wait();
}
inline void ColumnarMemTable::WaitForPendingBackgroundWork() {
    if (!async_background_processing_) {
        std::vector<BackgroundWorkItem> work_items;
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            work_items.swap(sealed_blocks_queue_);
        }
        ProcessBackgroundWorkItems(std::move(work_items));
        return;
    }

    auto promise = std::make_unique<std::promise<void>>();
    auto future = promise->get_future();
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        sealed_blocks_queue_.push_back({nullptr, std::move(promise), 0});
    }
    queue_cond_.notify_one();
    future.wait();
}
inline void ColumnarMemTable::WaitForBackgroundBacklogAtMost(size_t maximum_pending_blocks) {
    if (GetPendingSealedBlockNum() <= maximum_pending_blocks) return;
    if (!async_background_processing_) {
        WaitForPendingBackgroundWork();
        return;
    }
    std::unique_lock<std::mutex> queue_lock(queue_mutex_);
    queue_cond_.wait(queue_lock, [this, maximum_pending_blocks] {
        return stop_background_thread_.load(std::memory_order_acquire) ||
               GetPendingSealedBlockNum() <= maximum_pending_blocks;
    });
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
        ProcessBackgroundWorkItems(std::move(work_items));
        queue_cond_.notify_all();
    }
}
inline void ColumnarMemTable::ProcessBackgroundWorkItems(std::vector<BackgroundWorkItem> work_items) {
    std::map<size_t, std::vector<std::shared_ptr<FlashActiveBlock>>> work_by_shard;
    std::vector<std::unique_ptr<std::promise<void>>> promises;
    for (auto& item : work_items) {
        if (item.block) work_by_shard[item.shard_idx].push_back(std::move(item.block));
        if (item.promise) promises.push_back(std::move(item.promise));
    }
    auto process_one_shard = [this](size_t shard_idx,
                                    const std::vector<std::shared_ptr<FlashActiveBlock>>& blocks) {
        try {
            ProcessBlocksForShard(shard_idx, blocks);
        } catch (const std::exception& error) {
            std::cerr << "Exception while processing shard " << shard_idx << ": " << error.what() << std::endl;
        }
    };

    if (batch_pool_ && work_by_shard.size() > 1) {
        // Reuse the same bounded pool as PutBatch.  Foreground shard tasks and
        // background sorting therefore share one worker budget instead of
        // creating an unaccounted set of sorting threads.
        std::vector<std::future<void>> futures;
        futures.reserve(work_by_shard.size());
        for (auto& [shard_idx, blocks] : work_by_shard) {
            futures.emplace_back(batch_pool_->Submit(
                [process_one_shard, shard_idx, blocks = std::move(blocks)] {
                    process_one_shard(shard_idx, blocks);
                }));
        }
        for (auto& future : futures) future.get();
    } else {
        for (const auto& [shard_idx, blocks] : work_by_shard) process_one_shard(shard_idx, blocks);
    }
    for (auto& promise : promises) promise->set_value();
}
inline void ColumnarMemTable::ProcessBlocksForShard(
    size_t shard_idx, const std::vector<std::shared_ptr<FlashActiveBlock>>& sealed_blocks) {
    if (sealed_blocks.empty()) return;
    std::vector<std::shared_ptr<const SortedColumnarBlock>> new_sorted_blocks;
    new_sorted_blocks.reserve(sealed_blocks.size());
    for (const auto& sealed_b : sealed_blocks) {
        auto cb = GetPooledColumnarBlock();  // Use the pool
        cb->Reserve(sealed_b->size());
        sealed_b->AppendReferencesTo(*cb);
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
    // Invalidate per-thread immutable-state caches after sealed blocks become
    // sorted blocks.  The seal path already increments this same version.
    shard.version_.fetch_add(1, std::memory_order_release);
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
inline size_t ColumnarMemTable::GetActiveRecordCount() const {
    size_t total = 0;
    for (const auto& shard : shards_) total += std::atomic_load(&shard->active_block_)->size();
    return total;
}
inline size_t ColumnarMemTable::GetPendingSealedBlockNum() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        auto state = std::atomic_load(&shard->immutable_state_);
        if (state->sealed_blocks) total += state->sealed_blocks->size();
    }
    return total;
}
inline size_t ColumnarMemTable::GetPendingSealedRecordCount() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        auto state = std::atomic_load(&shard->immutable_state_);
        if (state->sealed_blocks) {
            for (const auto& block : *state->sealed_blocks) total += block->size();
        }
    }
    return total;
}
inline size_t ColumnarMemTable::GetSortedBlockNum() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        auto state = std::atomic_load(&shard->immutable_state_);
        if (state->blocks) total += state->blocks->size();
    }
    return total;
}
inline size_t ColumnarMemTable::GetSortedRecordCount() const {
    size_t total = 0;
    for (const auto& shard : shards_) {
        auto state = std::atomic_load(&shard->immutable_state_);
        if (state->blocks) {
            for (const auto& block : *state->blocks) total += block->size();
        }
    }
    return total;
}
inline std::shared_ptr<FlashActiveBlock> ColumnarMemTable::GetActiveBlockForThread(size_t shard_idx,
                                                                                   bool force_refresh) const {
    struct Cache {
        uint64_t instance_id = 0;
        std::vector<std::shared_ptr<FlashActiveBlock>> blocks;
        std::vector<uint64_t> versions;
    };
    thread_local Cache cache;
    if (cache.instance_id != instance_id_ || cache.blocks.size() != num_shards_) {
        cache.instance_id = instance_id_;
        cache.blocks.assign(num_shards_, nullptr);
        cache.versions.assign(num_shards_, std::numeric_limits<uint64_t>::max());
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || cache.blocks[shard_idx] == nullptr || cache.versions[shard_idx] != current_version) {
        cache.blocks[shard_idx] = std::atomic_load(&shard.active_block_);
        cache.versions[shard_idx] = current_version;
    }
    return cache.blocks[shard_idx];
}
inline std::shared_ptr<const ColumnarMemTable::ImmutableState> ColumnarMemTable::GetImmutableStateForThread(
    size_t shard_idx, bool force_refresh) const {
    struct Cache {
        uint64_t instance_id = 0;
        std::vector<std::shared_ptr<const ImmutableState>> states;
        std::vector<uint64_t> versions;
    };
    thread_local Cache cache;
    if (cache.instance_id != instance_id_ || cache.states.size() != num_shards_) {
        cache.instance_id = instance_id_;
        cache.states.assign(num_shards_, nullptr);
        cache.versions.assign(num_shards_, std::numeric_limits<uint64_t>::max());
    }
    const auto& shard = *shards_[shard_idx];
    uint64_t current_version = shard.version_.load(std::memory_order_acquire);
    if (force_refresh || cache.states[shard_idx] == nullptr || cache.versions[shard_idx] != current_version) {
        cache.states[shard_idx] = std::atomic_load(&shard.immutable_state_);
        cache.versions[shard_idx] = current_version;
    }
    return cache.states[shard_idx];
}

#endif  // COLUMNAR_MEMTABLE_H
