#ifndef COLUMNAR_MEMTABLE_H
#define COLUMNAR_MEMTABLE_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <future>

#define XXH_INLINE_ALL
#include "xxhash.h"

// --- Forward Declarations ---
enum class RecordType;
struct RecordRef;
class ColumnarBlock;
class Sorter;
class StdSorter;
class SortedColumnarBlock;
class FlushIterator;
class CompactingIterator;
class FlashActiveBlock;
class BloomFilter;
class ColumnarRecordArena;
class ConcurrentRawArena;

// --- Core Utility Structures ---
struct XXHasher {
    std::size_t operator()(const std::string_view key) const noexcept { return XXH3_64bits(key.data(), key.size()); }
};

inline uint64_t load_u64_prefix(std::string_view sv) {
    if (sv.size() >= 8) {
        uint64_t prefix;
        // Use memcpy for strict-aliasing safe type punning
        memcpy(&prefix, sv.data(), 8);
        return prefix;
    }
    // For keys shorter than 8 bytes, pad with zeros
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

// --- Bloom Filter (Unchanged) ---
class BloomFilter {
   public:
    explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01);
    void Add(std::string_view key);
    bool MayContain(std::string_view key) const;
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
};

class ColumnarMemTable; 
class ColumnarRecordArena {
   private:
    friend class ColumnarMemTable; 
    friend class Iterator;
    struct DataChunk {
        static constexpr size_t kRecordCapacity = 256;
        static constexpr size_t kBufferCapacity = 32 * 1024;
        uint32_t write_idx{0}, buffer_pos{0};
        std::array<StoredRecord, kRecordCapacity> records;
        alignas(64) char buffer[kBufferCapacity];
    };
    struct alignas(64) ThreadLocalData {
        std::vector<std::unique_ptr<DataChunk>> chunks;
        DataChunk* current_chunk = nullptr;
        ThreadLocalData() { AddNewChunk(); }
        void AddNewChunk() {
            chunks.push_back(std::make_unique<DataChunk>());
            current_chunk = chunks.back().get();
        }
    };

   public:
    static std::map<uint64_t, ThreadLocalData*>& GetTlsMap() {
        thread_local std::map<uint64_t, ThreadLocalData*> tls_map;
        return tls_map;
    }

    class Iterator;
    ColumnarRecordArena() : id_(next_id_.fetch_add(1, std::memory_order_relaxed)), size_(0) {}
    ~ColumnarRecordArena() {
        auto& tls_map = GetTlsMap();  // 使用辅助函数

        if (tls_map.count(id_)) {
            tls_map.erase(id_);
        }
    }
    const StoredRecord* AllocateAndAppend(std::string_view key, std::string_view value, RecordType type);
    size_t size() const { return size_.load(std::memory_order_relaxed); }
    Iterator begin() const;
    Iterator end() const;

   private:
    ThreadLocalData* GetTlsData();
    static std::atomic<uint64_t> next_id_;
    const uint64_t id_;
    std::mutex registration_mutex_;
    std::vector<std::unique_ptr<ThreadLocalData>> all_tls_data_;
    std::atomic<size_t> size_;
};

inline std::atomic<uint64_t> ColumnarRecordArena::next_id_{0};
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
class ColumnarRecordArena::Iterator {
   public:
    const RecordRef& operator*() const {
        return arena_->all_tls_data_[tls_idx_]->chunks[chunk_idx_]->records[record_idx_].record;
    }
    Iterator& operator++() {
        advance();
        return *this;
    }
    bool operator!=(const Iterator& other) const {
        return tls_idx_ != other.tls_idx_ || chunk_idx_ != other.chunk_idx_ || record_idx_ != other.record_idx_;
    }

   private:
    friend class ColumnarRecordArena;
    Iterator(const ColumnarRecordArena* arena, size_t tls_idx, size_t chunk_idx, size_t record_idx)
        : arena_(arena), tls_idx_(tls_idx), chunk_idx_(chunk_idx), record_idx_(record_idx) {}
    const ColumnarRecordArena* arena_;
    size_t tls_idx_;
    size_t chunk_idx_;
    size_t record_idx_;
    void advance() {
        if (tls_idx_ >= arena_->all_tls_data_.size()) return;
        record_idx_++;
        while (true) {
            const auto& tls_data = arena_->all_tls_data_[tls_idx_];
            if (chunk_idx_ < tls_data->chunks.size()) {
                if (record_idx_ < tls_data->chunks[chunk_idx_]->write_idx) return;
                chunk_idx_++;
                record_idx_ = 0;
            } else {
                tls_idx_++;
                chunk_idx_ = 0;
                record_idx_ = 0;
                if (tls_idx_ >= arena_->all_tls_data_.size()) return;
            }
        }
    }
};
inline ColumnarRecordArena::Iterator ColumnarRecordArena::begin() const {
    size_t start_tls = 0, start_chunk = 0, start_record = 0;
    while (start_tls < all_tls_data_.size()) {
        const auto& tls_data = all_tls_data_[start_tls];
        while (start_chunk < tls_data->chunks.size()) {
            if (start_record < tls_data->chunks[start_chunk]->write_idx)
                return Iterator(this, start_tls, start_chunk, start_record);
            start_chunk++;
            start_record = 0;
        }
        start_tls++;
        start_chunk = 0;
    }
    return end();
}
inline ColumnarRecordArena::Iterator ColumnarRecordArena::end() const {
    return Iterator(this, all_tls_data_.size(), 0, 0);
}

inline ColumnarRecordArena::ThreadLocalData* ColumnarRecordArena::GetTlsData() {
    auto& tls_map = GetTlsMap();

    if (tls_map.find(id_) == tls_map.end()) {
        std::lock_guard<std::mutex> lock(registration_mutex_);
        if (tls_map.find(id_) == tls_map.end()) {
            all_tls_data_.push_back(std::make_unique<ThreadLocalData>());
            tls_map[id_] = all_tls_data_.back().get();
        }
    }
    return tls_map[id_];
}

inline const StoredRecord* ColumnarRecordArena::AllocateAndAppend(std::string_view key, std::string_view value,
                                                                  RecordType type) {
    ThreadLocalData* tls_data = GetTlsData();
    size_t required_size = key.size() + value.size();
    DataChunk* chunk = tls_data->current_chunk;
    if (chunk->buffer_pos + required_size > DataChunk::kBufferCapacity ||
        chunk->write_idx >= DataChunk::kRecordCapacity) {
        tls_data->AddNewChunk();
        chunk = tls_data->current_chunk;
    }
    char* key_mem = chunk->buffer + chunk->buffer_pos;
    memcpy(key_mem, key.data(), key.size());
    char* val_mem = key_mem + key.size();
    memcpy(val_mem, value.data(), value.size());
    StoredRecord& record = chunk->records[chunk->write_idx];
    record.record = {{key_mem, key.size()}, {val_mem, value.size()}, type};
    chunk->buffer_pos += required_size;
    chunk->write_idx++;
    size_.fetch_add(1, std::memory_order_relaxed);
    return &record;
}
inline ConcurrentStringHashMap::ConcurrentStringHashMap(size_t build_size) {
    size_t capacity = calculate_power_of_2(build_size * 3.0);
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
        if (pos == initial_pos) return;
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
            return slots_[pos].record.load(std::memory_order_acquire);
        }
        pos = (pos + 1) & capacity_mask_;
    } while (pos != initial_pos);
    return nullptr;
}
class FlashActiveBlock {
    friend class ColumnarMemTable;
   public:
    explicit FlashActiveBlock(size_t cap) : index_(cap) {}
    ~FlashActiveBlock() {}
    bool TryAdd(std::string_view key, std::string_view value, RecordType type);
    std::optional<RecordRef> Get(std::string_view key) const;
    size_t size() const { return data_log_.size(); }
    ColumnarRecordArena::Iterator begin() const { return data_log_.begin(); }
    ColumnarRecordArena::Iterator end() const { return data_log_.end(); }
    void Seal() { sealed_.store(true, std::memory_order_release); }
    bool is_sealed() const { return sealed_.load(std::memory_order_acquire); }

   private:
    ColumnarRecordArena data_log_;
    ConcurrentStringHashMap index_;
    std::atomic<bool> sealed_{false};
};
inline bool FlashActiveBlock::TryAdd(std::string_view key, std::string_view value, RecordType type) {
    if (is_sealed()) return false;
    const StoredRecord* record_ptr = data_log_.AllocateAndAppend(key, value, type);
    if (record_ptr) index_.Insert(record_ptr->record.key, record_ptr);
    return record_ptr != nullptr;
}
inline std::optional<RecordRef> FlashActiveBlock::Get(std::string_view key) const {
    const StoredRecord* record_ptr = index_.Find(key);
    return record_ptr ? std::optional<RecordRef>(record_ptr->record) : std::nullopt;
}
class ColumnarBlock {
   public:
    class SimpleArena {
       public:
        char* AllocateRaw(size_t bytes);
        std::string_view AllocateAndCopy(std::string_view data);

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
    void Add(std::string_view k, std::string_view v, RecordType t);
    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }
    void Clear() {
        keys.clear();
        values.clear();
        types.clear();
        arena = SimpleArena(); 
    }
};
inline char* ColumnarBlock::SimpleArena::AllocateRaw(size_t bytes) {
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
inline std::string_view ColumnarBlock::SimpleArena::AllocateAndCopy(std::string_view d) {
    char* m = AllocateRaw(d.size());
    if (!d.empty()) memcpy(m, d.data(), d.size());
    return {m, d.size()};
}
inline void ColumnarBlock::Add(std::string_view k, std::string_view v, RecordType t) {
    keys.push_back(arena.AllocateAndCopy(k));
    values.push_back(arena.AllocateAndCopy(v));
    types.push_back(t);
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
        if (static_cast<size_t>(std::distance(begin, end)) <= 1) {
            return;
        }

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
    
    void radix_sort_msd_parallel(Iterator begin, Iterator end, size_t depth, unsigned int num_threads, const ColumnarBlock& block) const {
        const size_t size = std::distance(begin, end);
        if (size <= kSequentialSortThreshold || num_threads <= 1) {
            radix_sort_msd_sequential(begin, end, depth, block);
            return;
        }

        // 1. Count frequencies of each character.
        // bucket_counts[0] is for finished strings (char_code -1)
        // bucket_counts[1] is for char 0, etc.
        std::vector<size_t> bucket_counts(kRadixAlphabetSize + 1, 0);
        for (auto it = begin; it != end; ++it) {
            bucket_counts[get_char_at(block.keys[*it], depth) + 1]++;
        }
        
        // 2. Calculate starting offsets for each bucket in the output.
        std::vector<size_t> bucket_offsets(kRadixAlphabetSize + 2, 0);
        for(size_t i = 0; i < kRadixAlphabetSize + 1; ++i) {
            bucket_offsets[i+1] = bucket_offsets[i] + bucket_counts[i];
        }

        // 3. Distribute elements into a temporary output buffer based on their bucket.
        // This is the core fix: using a separate output buffer avoids data corruption.
        std::vector<uint32_t> sorted_output(size);
        std::vector<size_t> current_offsets = bucket_offsets; // Copy offsets to use as write cursors.

        for (auto it = begin; it != end; ++it) {
            uint32_t val = *it;
            int char_code = get_char_at(block.keys[val], depth);
            sorted_output[current_offsets[char_code + 1]++] = val;
        }
        
        // 4. Copy the correctly sorted data from the buffer back to the original range.
        std::copy(sorted_output.begin(), sorted_output.end(), begin);

        // 5. Parallel recursive sorting on each bucket.
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        // Start from i=1 (char_code 0). Bucket 0 (char_code -1) is already sorted.
        for (size_t i = 1; i < kRadixAlphabetSize + 1; ++i) {
            size_t bucket_size = bucket_counts[i];
            if (bucket_size == 0) continue;

            Iterator bucket_begin = begin + bucket_offsets[i];
            Iterator bucket_end = begin + bucket_offsets[i+1];
            
            if (threads.size() < num_threads - 1 && bucket_size > kSequentialSortThreshold) {
                 threads.emplace_back([this, bucket_begin, bucket_end, depth, num_threads, &block] {
                    unsigned int threads_for_child = (num_threads + 1) / 2;
                    radix_sort_msd_parallel(bucket_begin, bucket_end, depth + 1, threads_for_child, block);
                 });
            } else { 
                radix_sort_msd_sequential(bucket_begin, bucket_end, depth + 1, block);
            }
        }

        for (auto& t : threads) {
            t.join();
        }
    }
};

class SortedColumnarBlock {
   public:
    class Iterator;
    static constexpr size_t kSparseIndexSampleRate = 16;
    explicit SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block, const Sorter& sorter);
    bool MayContain(std::string_view key) const;
    std::optional<RecordRef> Get(std::string_view key) const;
    std::string_view min_key() const { return min_key_; }
    std::string_view max_key() const { return max_key_; }
    Iterator begin() const;
    bool empty() const { return sorted_indices_.empty(); }
    size_t size() const { return sorted_indices_.size(); }

   private:
    friend class Iterator;
    std::shared_ptr<ColumnarBlock> block_data_;
    std::vector<uint32_t> sorted_indices_;
    std::string_view min_key_, max_key_;
    std::unique_ptr<BloomFilter> bloom_filter_;
    std::vector<std::pair<std::string_view, size_t>> sparse_index_;
};
inline SortedColumnarBlock::SortedColumnarBlock(std::shared_ptr<ColumnarBlock> b, const Sorter& s)
    : block_data_(std::move(b)) {
    sorted_indices_ = s.Sort(*block_data_);

    if (sorted_indices_.empty()) {
        min_key_ = {};
        max_key_ = {};
        return;
    }

    min_key_ = block_data_->keys[sorted_indices_.front()];
    max_key_ = block_data_->keys[sorted_indices_.back()];
    bloom_filter_ = std::make_unique<BloomFilter>(block_data_->size());
    for (const auto& k : block_data_->keys) bloom_filter_->Add(k);
    sparse_index_.reserve(sorted_indices_.size() / kSparseIndexSampleRate + 1);
    for (size_t i = 0; i < sorted_indices_.size(); i += kSparseIndexSampleRate)
        sparse_index_.emplace_back(block_data_->keys[sorted_indices_[i]], i);
}

inline bool SortedColumnarBlock::MayContain(std::string_view key) const {
    if (empty() || key < min_key_ || key > max_key_) return false;
    return bloom_filter_->MayContain(key);
}
inline std::optional<RecordRef> SortedColumnarBlock::Get(std::string_view key) const {
    if (empty()) return std::nullopt;
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
    if (it != sorted_indices_.end() && *it < block_data_->keys.size() && block_data_->keys[*it] == key) {
        uint32_t i = *it;
        return RecordRef{block_data_->keys[i], block_data_->values[i], block_data_->types[i]};
    }
    return std::nullopt;
}
class SortedColumnarBlock::Iterator {
   public:
    Iterator(const SortedColumnarBlock* b, size_t p) : block_(b), pos_(p) {}
    RecordRef operator*() const {
        uint32_t i = block_->sorted_indices_[pos_];
        return {block_->block_data_->keys[i], block_->block_data_->values[i], block_->block_data_->types[i]};
    }
    void Next() { ++pos_; }
    bool IsValid() const { return pos_ < block_->sorted_indices_.size(); }

   private:
    const SortedColumnarBlock* block_;
    size_t pos_;
};
inline SortedColumnarBlock::Iterator SortedColumnarBlock::begin() const { return Iterator(this, 0); }

class FlushIterator {
   public:
    explicit FlushIterator(std::vector<std::shared_ptr<const SortedColumnarBlock>> sources);
    bool IsValid() const { return !min_heap_.empty(); }
    RecordRef Get() const { return min_heap_.top().record; }
    void Next();

   private:
    struct HeapNode {
        RecordRef record;
        uint64_t key_prefix;
        size_t source_index;
        size_t iterator_index;

        bool operator>(const HeapNode& o) const {
            if (key_prefix != o.key_prefix) {
                return key_prefix > o.key_prefix;
            }
            if (record.key != o.record.key) {
                return record.key > o.record.key;
            }
            return source_index > o.source_index;
        }
    };

    std::vector<std::shared_ptr<const SortedColumnarBlock>> sources_;
    std::vector<SortedColumnarBlock::Iterator> iterators_;
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> min_heap_;
};

inline FlushIterator::FlushIterator(std::vector<std::shared_ptr<const SortedColumnarBlock>> s)
    : sources_(std::move(s)) {
    iterators_.reserve(sources_.size());
    for (size_t i = 0; i < sources_.size(); ++i) {
        if (!sources_[i] || sources_[i]->empty()) {
            continue;
        }
        iterators_.emplace_back(sources_[i]->begin());
        size_t current_iterator_index = iterators_.size() - 1;
        RecordRef rec = *iterators_.back();
        
        uint64_t prefix = load_u64_prefix(rec.key);
        min_heap_.push({rec, prefix, i, current_iterator_index});
    }
}

inline void FlushIterator::Next() {
    if (!IsValid()) return;
    HeapNode n = min_heap_.top();
    min_heap_.pop();
    iterators_[n.iterator_index].Next();
    if (iterators_[n.iterator_index].IsValid()) {
        RecordRef rec = *iterators_[n.iterator_index];
        uint64_t prefix = load_u64_prefix(rec.key);
        min_heap_.push({rec, prefix, n.source_index, n.iterator_index});
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
    is_valid_ = false;
    while (source_->IsValid()) {
        RecordRef rec = source_->Get();
        std::string_view key = rec.key;
        source_->Next();
        while (source_->IsValid() && source_->Get().key == key) {
            rec = source_->Get();
            source_->Next();
        }
        if (rec.type == RecordType::Put) {
            current_record_ = rec;
            is_valid_ = true;
            return;
        }
    }
}
class ColumnarMemTable {
   public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::map<std::string_view, GetResult, std::less<>>;
    explicit ColumnarMemTable(size_t active_block_size_bytes = 16 * 1024 * 48, bool enable_compaction = false,
                              std::shared_ptr<Sorter> sorter = std::make_shared<ParallelRadixSorter>()) // Use new sorter
    : active_block_threshold_(std::max((size_t)1, active_block_size_bytes / 116)),
      enable_compaction_(enable_compaction),
      sorter_(std::move(sorter)) {
        active_block_ = std::make_shared<FlashActiveBlock>(active_block_threshold_);
        immutable_state_ = std::make_shared<const ImmutableState>();
        background_thread_ = std::thread(&ColumnarMemTable::BackgroundWorkerLoop, this);
    }
    ~ColumnarMemTable();
    ColumnarMemTable(const ColumnarMemTable&) = delete;
    ColumnarMemTable& operator=(const ColumnarMemTable&) = delete;
    void Put(std::string_view key, std::string_view value);
    void Delete(std::string_view key);
    GetResult Get(std::string_view key) const;
    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const;
    void PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch);
    void WaitForBackgroundWork();
    std::unique_ptr<CompactingIterator> NewCompactingIterator();

   private:
    struct ImmutableState {
        using SortedBlockList = std::vector<std::shared_ptr<const SortedColumnarBlock>>;
        
        using MetaInfo = std::tuple<std::string_view, std::string_view, std::shared_ptr<const SortedColumnarBlock>>;
        
        std::shared_ptr<const SortedBlockList> blocks;
        std::shared_ptr<const std::vector<MetaInfo>> read_meta_cache;
        ImmutableState()
            : blocks(std::make_shared<const SortedBlockList>()),
            read_meta_cache(std::make_shared<const std::vector<MetaInfo>>()) {}
    };
    std::unique_ptr<FlushIterator> NewRawFlushIterator();
    std::vector<std::shared_ptr<ColumnarBlock>> columnar_block_pool_;
    std::mutex pool_mutex_;
    void Insert(std::string_view key, std::string_view value, RecordType type);
    void SealActiveBlockIfNeeded();
    void BackgroundWorkerLoop();
    void ProcessBlocks(std::vector<std::shared_ptr<ColumnarBlock>> blocks);
    FlashActiveBlock* GetActiveBlockForThread(bool force_refresh = false) const;
    const size_t active_block_threshold_;
    const bool enable_compaction_;
    std::shared_ptr<Sorter> sorter_;
    std::shared_ptr<FlashActiveBlock> active_block_;
    std::shared_ptr<const ImmutableState> immutable_state_;
    alignas(64) std::atomic<uint64_t> seal_sequence_{0};
    std::mutex seal_mutex_;
    std::vector<std::shared_ptr<ColumnarBlock>> sealed_blocks_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::thread background_thread_;
    std::atomic<bool> stop_background_thread_{false};
    bool background_thread_processing_{false};
};

inline ColumnarMemTable::~ColumnarMemTable() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_background_thread_ = true;
    }
    queue_cond_.notify_one();
    if (background_thread_.joinable()) background_thread_.join();
}

inline FlashActiveBlock* ColumnarMemTable::GetActiveBlockForThread(bool force_refresh) const {
    thread_local const ColumnarMemTable* last_memtable_instance = nullptr;
    thread_local FlashActiveBlock* active_block_cache = nullptr;
    thread_local uint64_t last_seen_seal_sequence = -1;

    uint64_t current_sequence = seal_sequence_.load(std::memory_order_acquire);

    if (force_refresh || last_memtable_instance != this || last_seen_seal_sequence != current_sequence) {
        active_block_cache = std::atomic_load(&active_block_).get();
        last_seen_seal_sequence = current_sequence;
        last_memtable_instance = this;
    }

    return active_block_cache;
}

inline void ColumnarMemTable::Insert(std::string_view k, std::string_view v, RecordType t) {
    FlashActiveBlock* current_block = GetActiveBlockForThread();
    while (!current_block->TryAdd(k, v, t)) {
        current_block = GetActiveBlockForThread(true);  // Force refresh
    }
    if (current_block->size() >= active_block_threshold_) SealActiveBlockIfNeeded();
}

inline void ColumnarMemTable::Put(std::string_view k, std::string_view v) { Insert(k, v, RecordType::Put); }
inline void ColumnarMemTable::Delete(std::string_view k) { Insert(k, "", RecordType::Delete); }
inline void ColumnarMemTable::PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
    for (const auto& [k, v] : batch) Insert(k, v, RecordType::Put);
}

inline ColumnarMemTable::GetResult ColumnarMemTable::Get(std::string_view key) const {
    FlashActiveBlock* active_block = GetActiveBlockForThread();
    if (auto r = active_block->Get(key)) return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
    auto s = std::atomic_load(&immutable_state_);
    thread_local const ImmutableState* last_s = nullptr;
    thread_local const std::vector<ImmutableState::MetaInfo>* cache = nullptr;
    if (last_s != s.get()) {
        last_s = s.get();
        cache = last_s->read_meta_cache.get();
    }
    if (cache) {
        for (auto it = cache->rbegin(); it != cache->rend(); ++it) {
            const auto& [min_k, max_k, ptr] = *it; 
            if (key >= min_k && key <= max_k) {
                if (ptr->MayContain(key)) {
                    if (auto r = ptr->Get(key))
                        return (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt;
                }
            }
        }
    }
    return std::nullopt;
}
inline ColumnarMemTable::MultiGetResult ColumnarMemTable::MultiGet(const std::vector<std::string_view>& keys) const {
    MultiGetResult results;
    if (keys.empty()) return results;
    std::vector<std::string_view> remaining_keys;
    remaining_keys.reserve(keys.size());
    FlashActiveBlock* active_block = GetActiveBlockForThread();
    for (const auto& k : keys) {
        if (!results.count(k)) {
            if (auto r = active_block->Get(k))
                results.emplace(k, (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt);
            else
                remaining_keys.push_back(k);
        }
    }
    if (remaining_keys.empty()) return results;
    auto global_state = std::atomic_load(&immutable_state_);
    thread_local const ImmutableState* last_seen_state = nullptr;
    thread_local const std::vector<ImmutableState::MetaInfo>* local_meta_cache_ptr = nullptr;
    if (last_seen_state != global_state.get()) {
        last_seen_state = global_state.get();
        local_meta_cache_ptr = last_seen_state->read_meta_cache.get();
    }
    if (!local_meta_cache_ptr || local_meta_cache_ptr->empty()) return results;
    std::sort(remaining_keys.begin(), remaining_keys.end());
    for (auto block_it = local_meta_cache_ptr->rbegin(); block_it != local_meta_cache_ptr->rend(); ++block_it) {
        if (remaining_keys.empty()) break;
        const auto& [min_k, max_k, block_ptr] = *block_it;
        std::vector<std::string_view> next_remaining_keys;
        next_remaining_keys.reserve(remaining_keys.size());
        for (auto key_it = remaining_keys.begin(); key_it != remaining_keys.end(); ++key_it) {
            const auto& key = *key_it;
            if (key > max_k) {
                next_remaining_keys.insert(next_remaining_keys.end(), key_it, remaining_keys.end());
                break;
            }
            if (key >= min_k) {
                if (block_ptr->MayContain(key)) {
                    if (auto r = block_ptr->Get(key)) {
                        results.emplace(key, (r->type == RecordType::Put) ? GetResult(r->value) : std::nullopt);
                    } else {
                        next_remaining_keys.push_back(key);
                    }
                } else {
                    next_remaining_keys.push_back(key);
                }
            } else {
                next_remaining_keys.push_back(key);
            }
        }
        remaining_keys = std::move(next_remaining_keys);
    }
    return results;
}
inline void ColumnarMemTable::SealActiveBlockIfNeeded() {
    std::lock_guard<std::mutex> lock(seal_mutex_);
    auto b = std::atomic_load(&active_block_);
    if (b->size() < active_block_threshold_ || b->is_sealed()) return;

    b->Seal();
    auto new_b = std::make_shared<FlashActiveBlock>(active_block_threshold_);
    std::atomic_exchange(&active_block_, new_b);
    seal_sequence_.fetch_add(1, std::memory_order_release);

    std::shared_ptr<ColumnarBlock> cb;
    {
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);
        if (!columnar_block_pool_.empty()) {
            cb = std::move(columnar_block_pool_.back());
            columnar_block_pool_.pop_back();
        }
    }
    if (!cb) {
        cb = std::make_shared<ColumnarBlock>();
    }

    {
        ColumnarRecordArena& arena_to_copy = b->data_log_;

        std::vector<const ColumnarRecordArena::ThreadLocalData*> tls_data_snapshot;
        {
            std::lock_guard<std::mutex> arena_lock(arena_to_copy.registration_mutex_);
            for (const auto& tls_ptr : arena_to_copy.all_tls_data_) {
                tls_data_snapshot.push_back(tls_ptr.get());
            }
        }

        for (const auto* tls_data : tls_data_snapshot) {
            for (const auto& chunk_ptr : tls_data->chunks) {
                for (uint32_t i = 0; i < chunk_ptr->write_idx; ++i) {
                    const auto& rec = chunk_ptr->records[i].record;
                    cb->Add(rec.key, rec.value, rec.type);
                }
            }
        }
    }
    
     if (!cb->empty()) {
        std::lock_guard<std::mutex> ql(queue_mutex_);
        sealed_blocks_queue_.push_back(std::move(cb));
    } else {
        cb->Clear();
        std::lock_guard<std::mutex> pool_lock(pool_mutex_);
        columnar_block_pool_.push_back(std::move(cb));
    }
    queue_cond_.notify_one();
}

inline void ColumnarMemTable::WaitForBackgroundWork() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cond_.wait(lock, [this] { return sealed_blocks_queue_.empty() && !background_thread_processing_; });
}

inline std::unique_ptr<FlushIterator> ColumnarMemTable::NewRawFlushIterator() {
    WaitForBackgroundWork();
    std::vector<std::shared_ptr<const SortedColumnarBlock>> all_blocks;
    std::lock_guard<std::mutex> lock(seal_mutex_);

    auto s = std::atomic_load(&immutable_state_);
    auto ab = std::atomic_load(&active_block_);

    all_blocks = *s->blocks;

    if (ab->size() > 0) {
        auto tb = std::make_shared<ColumnarBlock>();
        for (const auto& r : *ab) {
            tb->Add(r.key, r.value, r.type);
        }
        if (!tb->empty()) {
            all_blocks.push_back(std::make_shared<const SortedColumnarBlock>(tb, *sorter_));
        }
    }

    return std::make_unique<FlushIterator>(std::move(all_blocks));
}

inline std::unique_ptr<CompactingIterator> ColumnarMemTable::NewCompactingIterator() {
    return std::make_unique<CompactingIterator>(NewRawFlushIterator());
}
inline void ColumnarMemTable::BackgroundWorkerLoop() {
    while (true) {
        std::vector<std::shared_ptr<ColumnarBlock>> blocks;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this] { return !sealed_blocks_queue_.empty() || stop_background_thread_; });
            if (stop_background_thread_ && sealed_blocks_queue_.empty()) return;
            blocks.swap(sealed_blocks_queue_);
            background_thread_processing_ = true;
        }
        ProcessBlocks(std::move(blocks));
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            background_thread_processing_ = false;
        }
        queue_cond_.notify_all();
    }
}

inline void ColumnarMemTable::ProcessBlocks(std::vector<std::shared_ptr<ColumnarBlock>> blocks) {
    if (blocks.empty()) return;

    auto old_s = std::atomic_load(&immutable_state_);
    auto new_list = std::make_shared<ImmutableState::SortedBlockList>();
    
    // --- Compaction Path ---
    if (enable_compaction_) {
        std::vector<std::shared_ptr<const SortedColumnarBlock>> to_merge;
        
        // 1. Add blocks from the previous state to the merge set
        if (old_s->blocks) {
            to_merge.insert(to_merge.end(), old_s->blocks->begin(), old_s->blocks->end());
        }

        // 2. Add newly sealed blocks to the merge set
        for (const auto& b : blocks) {
            if (b->empty()) continue;
            to_merge.push_back(std::make_shared<const SortedColumnarBlock>(b, *sorter_));
        }
        
        // 3. If there's anything to merge, run the compaction
        if (!to_merge.empty()) {
            auto fcb = std::make_shared<ColumnarBlock>(); // The new single compacted block
            CompactingIterator it(std::make_unique<FlushIterator>(std::move(to_merge)));
            while (it.IsValid()) {
                RecordRef r = it.Get();
                fcb->Add(r.key, r.value, r.type);
                it.Next();
            }

            // The new state will contain only the single, newly created compacted block
            if (!fcb->empty()) {
                new_list->push_back(std::make_shared<const SortedColumnarBlock>(fcb, *sorter_));
            }
        }
        
        // 4. Recycle the ColumnarBlocks that were just processed
        //    These are no longer needed because their data is now in `fcb`.
        {
            std::lock_guard<std::mutex> pool_lock(pool_mutex_);
            for (auto& b : blocks) {
                // No need to check use_count here, as their lifetime was tied to the local `to_merge`
                b->Clear();
                columnar_block_pool_.push_back(std::move(b));
            }
        }

    } 
    // --- Non-Compaction Path ---
    else {
        *new_list = *old_s->blocks;
        for (const auto& b : blocks) {
            if (b->empty()) {
                b->Clear();
                std::lock_guard<std::mutex> pool_lock(pool_mutex_);
                columnar_block_pool_.push_back(std::move(b));
                continue;
            }
            // The underlying ColumnarBlock `b` must live on as it backs the new SortedColumnarBlock.
            // No recycling is possible here. shared_ptr handles its lifetime.
            new_list->push_back(std::make_shared<const SortedColumnarBlock>(b, *sorter_));
        }
    }
    
    auto new_s = std::make_shared<ImmutableState>();
    new_s->blocks = std::move(new_list);
    auto meta_cache = std::make_shared<std::vector<ImmutableState::MetaInfo>>();
    if (new_s->blocks) {
        meta_cache->reserve(new_s->blocks->size());
        for (const auto& b : *new_s->blocks) {
            if (b && !b->empty()) {
                meta_cache->emplace_back(b->min_key(), b->max_key(), b);
            }
        }
    }
    new_s->read_meta_cache = std::move(meta_cache);
    std::atomic_store(&immutable_state_, std::shared_ptr<const ImmutableState>(std::move(new_s)));
}
#endif  // COLUMNAR_MEMTABLE_H