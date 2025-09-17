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
#include <utility>
#include <vector>

// Assumes xxhash.h is in your include path
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

// --- Core Utility Structures ---

struct XXHasher {
    std::size_t operator()(const std::string_view key) const noexcept { return XXH3_64bits(key.data(), key.size()); }
};

enum class RecordType { Put, Delete };

struct RecordRef {
    std::string_view key;
    std::string_view value;
    RecordType type;
};

// --- Bloom Filter Implementation (Unchanged) ---
class BloomFilter {
   public:
    explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01);
    void Add(std::string_view key);
    bool MayContain(std::string_view key) const;

    static std::array<uint64_t, 2> Hash(std::string_view key);
    std::vector<bool> bits_;
    int num_hashes_;
};

inline BloomFilter::BloomFilter(size_t num_entries, double false_positive_rate) {
    if (num_entries == 0) num_entries = 1;
    size_t bits = static_cast<size_t>(-1.44 * num_entries * std::log(false_positive_rate));
    bits_ = std::vector<bool>((bits + 7) & ~7, false);
    num_hashes_ = static_cast<int>(0.7 * (static_cast<double>(bits_.size()) / num_entries));
    if (num_hashes_ < 1) num_hashes_ = 1;
    if (num_hashes_ > 8) num_hashes_ = 8;
}

inline void BloomFilter::Add(std::string_view key) {
    std::array<uint64_t, 2> hash_values = Hash(key);
    for (int i = 0; i < num_hashes_; ++i) {
        uint64_t hash = hash_values[0] + i * hash_values[1];
        if (!bits_.empty()) {
            bits_[hash % bits_.size()] = true;
        }
    }
}

inline bool BloomFilter::MayContain(std::string_view key) const {
    if (bits_.empty()) return true;
    std::array<uint64_t, 2> hash_values = Hash(key);
    for (int i = 0; i < num_hashes_; ++i) {
        uint64_t hash = hash_values[0] + i * hash_values[1];
        if (!bits_[hash % bits_.size()]) {
            return false;
        }
    }
    return true;
}

inline std::array<uint64_t, 2> BloomFilter::Hash(std::string_view key) {
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;
    uint64_t h1 = 0xdeadbeefdeadbeef ^ (key.length() * m);
    const uint64_t* data = reinterpret_cast<const uint64_t*>(key.data());
    const int nblocks = key.length() / 8;
    for (int i = 0; i < nblocks; i++) {
        uint64_t k = data[i];
        k *= m;
        k ^= k >> r;
        k *= m;
        h1 ^= k;
        h1 *= m;
    }
    const unsigned char* data2 = reinterpret_cast<const unsigned char*>(key.data()) + nblocks * 8;
    switch (key.length() & 7) {
        case 7:
            h1 ^= uint64_t(data2[6]) << 48;
            [[fallthrough]];
        case 6:
            h1 ^= uint64_t(data2[5]) << 40;
            [[fallthrough]];
        case 5:
            h1 ^= uint64_t(data2[4]) << 32;
            [[fallthrough]];
        case 4:
            h1 ^= uint64_t(data2[3]) << 24;
            [[fallthrough]];
        case 3:
            h1 ^= uint64_t(data2[2]) << 16;
            [[fallthrough]];
        case 2:
            h1 ^= uint64_t(data2[1]) << 8;
            [[fallthrough]];
        case 1:
            h1 ^= uint64_t(data2[0]);
            h1 *= m;
    };
    h1 ^= h1 >> r;
    h1 *= m;
    h1 ^= h1 >> r;
    return {h1, h1 ^ m};
}

// --- Component 1: The Lock-Free Append-Only Data Store with Integrated Arena ---

struct StoredRecord {
    RecordRef record;
};

struct DataChunk {
    static constexpr size_t kRecordCapacity = 256;
    static constexpr size_t kBufferCapacity = 32 * 1024;

    // A single atomic variable to manage both counters.
    // Upper 32 bits: buffer_pos
    // Lower 32 bits: write_idx
    std::atomic<uint64_t> state_{0};

    std::array<StoredRecord, kRecordCapacity> records;
    alignas(16) char buffer[kBufferCapacity];
    std::atomic<DataChunk*> next_chunk{nullptr};
};

class LockFreeChunkList {
   public:
    class Iterator;
    LockFreeChunkList();
    ~LockFreeChunkList();
    // Changed return type to indicate success/failure, though not strictly needed with the higher-level retry
    // loop.
    const StoredRecord* AllocateAndAppend(std::string_view key, std::string_view value, RecordType type);
    size_t size() const { return size_.load(std::memory_order_relaxed); }
    Iterator begin() const;
    Iterator end() const;

   private:
    DataChunk* head_;
    std::atomic<DataChunk*> tail_;
    std::atomic<size_t> size_;
};

// --- Component 2: Corrected ConcurrentStringHashMap with Update-in-Place ---

class ConcurrentStringHashMap {
   public:
    static constexpr uint8_t EMPTY_TAG = 0xFF;
    static constexpr uint8_t LOCKED_TAG = 0xFE;

   private:
    struct alignas(16) Slot {
        std::atomic<uint8_t> tag;
        std::string_view key;
        std::atomic<const StoredRecord*> record;
    };
    std::unique_ptr<Slot[]> slots_;
    size_t capacity_;
    size_t capacity_mask_;
    XXHasher hasher_;

   public:
    ConcurrentStringHashMap(const ConcurrentStringHashMap&) = delete;
    ConcurrentStringHashMap& operator=(const ConcurrentStringHashMap&) = delete;
    static size_t calculate_power_of_2(size_t n) { return n == 0 ? 1 : 1UL << (64 - __builtin_clzll(n - 1)); }
    explicit ConcurrentStringHashMap(size_t build_size);
    void Insert(std::string_view key, const StoredRecord* new_record);
    const StoredRecord* Find(std::string_view key) const;
};

// --- LockFreeChunkList::Iterator (Defined before used by FlashActiveBlock) ---
class LockFreeChunkList::Iterator {
   public:
    Iterator(const DataChunk* chunk, uint32_t idx);
    
    const RecordRef& operator*() const { return current_chunk_->records[idx_].record; }
    
    // FIX: Corrected operator++ implementation
    Iterator& operator++();

    bool operator!=(const Iterator& other) const {
        return current_chunk_ != other.current_chunk_ || idx_ != other.idx_;
    }

   private:
    void advance_to_next_valid();
    const DataChunk* current_chunk_;
    uint32_t idx_;
};

// --- Component 3: The New Active Block ---
class FlashActiveBlock {
   public:
    explicit FlashActiveBlock(size_t capacity_in_records);
    bool TryAdd(std::string_view key, std::string_view value, RecordType type);
    std::optional<RecordRef> Get(std::string_view key) const;
    size_t size() const { return data_log_.size(); }
    LockFreeChunkList::Iterator begin() const;
    LockFreeChunkList::Iterator end() const;
    // Added methods to manage the sealed state.
    void Seal() { sealed_.store(true, std::memory_order_release); }
    bool is_sealed() const { return sealed_.load(std::memory_order_acquire); }

   private:
    LockFreeChunkList data_log_;
    ConcurrentStringHashMap index_;
    // Added atomic flag to prevent writes after sealing.
    std::atomic<bool> sealed_{false};
};

// --- Implementations for Classes Defined Above ---

// LockFreeChunkList implementations
inline LockFreeChunkList::LockFreeChunkList() : size_(0) {
    head_ = new DataChunk();
    tail_.store(head_, std::memory_order_relaxed);
}

inline LockFreeChunkList::~LockFreeChunkList() {
    DataChunk* current = head_;
    while (current) {
        DataChunk* next = current->next_chunk.load(std::memory_order_relaxed);
        delete current;
        current = next;
    }
}

inline const StoredRecord* LockFreeChunkList::AllocateAndAppend(std::string_view key, std::string_view value,
                                                                RecordType type) {
    size_t required_size = key.size() + value.size();
    while (true) {
        DataChunk* current_tail = tail_.load(std::memory_order_acquire);

        uint64_t old_state = current_tail->state_.load(std::memory_order_relaxed);
        while (true) {
            uint32_t buffer_pos = old_state >> 32;
            uint32_t write_idx = old_state & 0xFFFFFFFF;

            if (buffer_pos + required_size > DataChunk::kBufferCapacity || write_idx >= DataChunk::kRecordCapacity) {
                break;
            }

            uint64_t new_state = (static_cast<uint64_t>(buffer_pos + required_size) << 32) | (write_idx + 1);

            if (current_tail->state_.compare_exchange_strong(old_state, new_state, std::memory_order_relaxed)) {
                char* key_mem = current_tail->buffer + buffer_pos;
                memcpy(key_mem, key.data(), key.size());
                char* val_mem = key_mem + key.size();
                memcpy(val_mem, value.data(), value.size());
                StoredRecord& record = current_tail->records[write_idx];
                record.record = {{key_mem, key.size()}, {val_mem, value.size()}, type};
                size_.fetch_add(1, std::memory_order_relaxed);
                return &record;
            }
        }

        DataChunk* next = current_tail->next_chunk.load(std::memory_order_acquire);
        if (next == nullptr) {
            DataChunk* new_chunk = new DataChunk();
            DataChunk* expected = nullptr;
            if (current_tail->next_chunk.compare_exchange_strong(expected, new_chunk, std::memory_order_release, std::memory_order_relaxed)) {
                tail_.compare_exchange_strong(current_tail, new_chunk, std::memory_order_release, std::memory_order_relaxed);
            } else {
                delete new_chunk;
                tail_.compare_exchange_strong(current_tail, expected, std::memory_order_release, std::memory_order_relaxed);
            }
        } else {
            tail_.compare_exchange_strong(current_tail, next, std::memory_order_release, std::memory_order_relaxed);
        }
    }
}

inline LockFreeChunkList::Iterator::Iterator(const DataChunk* chunk, uint32_t idx) : current_chunk_(chunk), idx_(idx) {
    if (current_chunk_) {
        uint32_t write_idx = current_chunk_->state_.load(std::memory_order_acquire) & 0xFFFFFFFF;
        if (idx_ >= write_idx) {
            advance_to_next_valid();
        }
    }
}

inline LockFreeChunkList::Iterator& LockFreeChunkList::Iterator::operator++() {
    idx_++;
    if (current_chunk_) {
        // This is the line that was missed in the previous fix.
        uint32_t write_idx = current_chunk_->state_.load(std::memory_order_acquire) & 0xFFFFFFFF;
        if (idx_ >= write_idx) {
            advance_to_next_valid();
        }
    }
    return *this;
}

inline void LockFreeChunkList::Iterator::advance_to_next_valid() {
    current_chunk_ = current_chunk_->next_chunk.load(std::memory_order_acquire);
    idx_ = 0;
    if (current_chunk_) {
        uint32_t write_idx = current_chunk_->state_.load(std::memory_order_acquire) & 0xFFFFFFFF;
        if (idx_ >= write_idx) {
            current_chunk_ = nullptr;
        }
    }
}

inline LockFreeChunkList::Iterator LockFreeChunkList::begin() const { return Iterator(head_, 0); }
inline LockFreeChunkList::Iterator LockFreeChunkList::end() const { return Iterator(nullptr, 0); }

// ConcurrentStringHashMap implementations
inline ConcurrentStringHashMap::ConcurrentStringHashMap(size_t build_size) {
    size_t capacity = calculate_power_of_2(build_size * 1.5);
    capacity_ = capacity;
    capacity_mask_ = capacity - 1;
    slots_ = std::make_unique<Slot[]>(capacity_);
    for (size_t i = 0; i < capacity_; ++i) {
        slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
        slots_[i].record.store(nullptr, std::memory_order_relaxed);
    }
}

// --- Implementation ---
inline void ConcurrentStringHashMap::Insert(std::string_view key, const StoredRecord* new_record) {
    uint64_t hash = hasher_(key);
    uint8_t tag = (hash >> 56);
    if (tag == EMPTY_TAG || tag == LOCKED_TAG) tag = 0; // Avoid special tags

    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;

    while (true) {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);

        // Case 1: Found an existing key. Update in place.
        if (current_tag == tag && slots_[pos].key == key) {
            slots_[pos].record.store(new_record, std::memory_order_release);
            return;
        }

        // Case 2: Found an empty slot. Try to lock it.
        if (current_tag == EMPTY_TAG) {
            uint8_t expected_empty = EMPTY_TAG;
            // Phase 1: Acquire the slot by setting a lock tag
            if (slots_[pos].tag.compare_exchange_strong(expected_empty, LOCKED_TAG, std::memory_order_acq_rel)) {
                // We now own the slot exclusively.
                // Phase 2: Write data and commit.
                slots_[pos].key = key;
                slots_[pos].record.store(new_record, std::memory_order_relaxed);
                slots_[pos].tag.store(tag, std::memory_order_release); // Finalize with the real tag
                return;
            }
            // If CAS failed, another thread took the slot. Loop again to re-evaluate the slot.
            continue;
        }

        // Case 3: Slot is locked by another writer or is a collision. Probe to the next slot.
        // We can spin-wait briefly here if it's locked, but linear probing is simpler and often sufficient.
        pos = (pos + 1) & capacity_mask_;
        if (pos == initial_pos) {
            // Hash map is full.
            return;
        }
    }
}

inline const StoredRecord* ConcurrentStringHashMap::Find(std::string_view key) const {
    uint64_t hash = hasher_(key);
    uint8_t tag = (hash >> 56);
    if (tag == EMPTY_TAG) tag = 0;
    size_t pos = hash & capacity_mask_;
    const size_t initial_pos = pos;
    do {
        uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
        if (current_tag == EMPTY_TAG) return nullptr;
        if (current_tag == tag && slots_[pos].key == key) {
            return slots_[pos].record.load(std::memory_order_acquire);
        }
        pos = (pos + 1) & capacity_mask_;
    } while (pos != initial_pos);
    return nullptr;
}

// FlashActiveBlock implementations
inline FlashActiveBlock::FlashActiveBlock(size_t capacity_in_records) : index_(capacity_in_records) {}
// implemented the seal check.
inline bool FlashActiveBlock::TryAdd(std::string_view key, std::string_view value, RecordType type) {
    if (is_sealed()) {
        return false;  // Block is sealed, cannot add.
    }
    const StoredRecord* record_ptr = data_log_.AllocateAndAppend(key, value, type);
    if (record_ptr) {
        index_.Insert(record_ptr->record.key, record_ptr);
    }
    return record_ptr != nullptr;
}

inline std::optional<RecordRef> FlashActiveBlock::Get(std::string_view key) const {
    const StoredRecord* record_ptr = index_.Find(key);
    return record_ptr ? std::optional<RecordRef>(record_ptr->record) : std::nullopt;
}

inline LockFreeChunkList::Iterator FlashActiveBlock::begin() const { return data_log_.begin(); }
inline LockFreeChunkList::Iterator FlashActiveBlock::end() const { return data_log_.end(); }

// --- Sealed/Sorted Path Components (Unchanged) ---
class ColumnarBlock {
   public:
    class SimpleArena {
       public:
        SimpleArena() : current_block_idx_(-1) {}
        char* AllocateRaw(size_t bytes);
        std::string_view AllocateAndCopy(std::string_view data);

       private:
        struct Block {
            std::unique_ptr<char[]> data;
            size_t pos;
            size_t size;
            explicit Block(size_t s) : data(new char[s]), pos(0), size(s) {}
        };
        std::vector<Block> blocks_;
        int current_block_idx_;
    };
    SimpleArena arena;
    std::vector<std::string_view> keys;
    std::vector<std::string_view> values;
    std::vector<RecordType> types;
    void Add(std::string_view key_sv, std::string_view value_sv, RecordType type);
    size_t size() const { return keys.size(); }
    bool empty() const { return keys.empty(); }
};

inline char* ColumnarBlock::SimpleArena::AllocateRaw(size_t bytes) {
    if (current_block_idx_ < 0 || blocks_[current_block_idx_].pos + bytes > blocks_[current_block_idx_].size) {
        size_t block_size = std::max(bytes, static_cast<size_t>(4096));
        blocks_.emplace_back(block_size);
        current_block_idx_++;
    }
    Block& current_block = blocks_[current_block_idx_];
    char* result = current_block.data.get() + current_block.pos;
    current_block.pos += bytes;
    return result;
}

inline std::string_view ColumnarBlock::SimpleArena::AllocateAndCopy(std::string_view data) {
    char* mem = AllocateRaw(data.size());
    memcpy(mem, data.data(), data.size());
    return {mem, data.size()};
}

inline void ColumnarBlock::Add(std::string_view key_sv, std::string_view value_sv, RecordType type) {
    auto key = arena.AllocateAndCopy(key_sv);
    auto value = arena.AllocateAndCopy(value_sv);
    keys.push_back(key);
    values.push_back(value);
    types.push_back(type);
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

class SortedColumnarBlock {
   public:
    class Iterator;
    static constexpr size_t kSparseIndexSampleRate = 16;
    explicit SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block_to_sort, const Sorter& sorter);
    bool MayContain(std::string_view key) const;
    std::optional<RecordRef> Get(std::string_view key) const;
    std::string_view min_key() const { return min_key_; }
    std::string_view max_key() const { return max_key_; }
    Iterator begin() const;
    bool empty() const { return block_data_->empty(); }

   private:
    friend class Iterator;
    std::shared_ptr<ColumnarBlock> block_data_;
    std::vector<uint32_t> sorted_indices_;
    std::string_view min_key_;
    std::string_view max_key_;
    std::unique_ptr<BloomFilter> bloom_filter_;
    std::vector<std::pair<std::string_view, size_t>> sparse_index_;
};

inline SortedColumnarBlock::SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block_to_sort, const Sorter& sorter)
    : block_data_(std::move(block_to_sort)) {
    sorted_indices_ = sorter.Sort(*block_data_);
    if (sorted_indices_.empty()) return;
    min_key_ = block_data_->keys[sorted_indices_.front()];
    max_key_ = block_data_->keys[sorted_indices_.back()];
    bloom_filter_ = std::make_unique<BloomFilter>(block_data_->size());
    for (const auto& key : block_data_->keys) bloom_filter_->Add(key);
    sparse_index_.reserve(sorted_indices_.size() / kSparseIndexSampleRate + 1);
    for (size_t i = 0; i < sorted_indices_.size(); i += kSparseIndexSampleRate) {
        uint32_t original_index = sorted_indices_[i];
        sparse_index_.emplace_back(block_data_->keys[original_index], i);
    }
}

inline bool SortedColumnarBlock::MayContain(std::string_view key) const {
    if (sorted_indices_.empty() || key < min_key_ || key > max_key_) return false;
    return bloom_filter_->MayContain(key);
}

inline std::optional<RecordRef> SortedColumnarBlock::Get(std::string_view key) const {
    auto sparse_it =
        std::lower_bound(sparse_index_.begin(), sparse_index_.end(), key,
                         [](const std::pair<std::string_view, size_t>& a, std::string_view b) { return a.first < b; });
    auto start_it = sorted_indices_.begin();
    if (sparse_it != sparse_index_.begin()) start_it += (sparse_it - 1)->second;
    auto end_it = sorted_indices_.end();
    if (sparse_it != sparse_index_.end()) {
        end_it = sorted_indices_.begin() + sparse_it->second + kSparseIndexSampleRate;
        if (end_it > sorted_indices_.end()) end_it = sorted_indices_.end();
    }
    auto it = std::lower_bound(start_it, end_it, key,
                               [&](uint32_t index, std::string_view k) { return block_data_->keys[index] < k; });
    if (it != sorted_indices_.end() && *it < block_data_->keys.size() && block_data_->keys[*it] == key) {
        uint32_t index = *it;
        return RecordRef{block_data_->keys[index], block_data_->values[index], block_data_->types[index]};
    }
    return std::nullopt;
}

class SortedColumnarBlock::Iterator {
   public:
    Iterator(const SortedColumnarBlock* block, size_t pos) : block_(block), pos_(pos) {}
    RecordRef operator*() const {
        uint32_t index = block_->sorted_indices_[pos_];
        return {block_->block_data_->keys[index], block_->block_data_->values[index],
                block_->block_data_->types[index]};
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
        size_t source_index;
        bool operator>(const HeapNode& other) const { return record.key > other.record.key; }
    };
    std::vector<std::shared_ptr<const SortedColumnarBlock>> sources_;
    std::vector<SortedColumnarBlock::Iterator> iterators_;
    std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<HeapNode>> min_heap_;
};

inline FlushIterator::FlushIterator(std::vector<std::shared_ptr<const SortedColumnarBlock>> sources)
    : sources_(std::move(sources)) {
    for (size_t i = 0; i < sources_.size(); ++i) {
        if (!sources_[i] || !sources_[i]->begin().IsValid()) continue;
        iterators_.emplace_back(sources_[i]->begin());
        min_heap_.push({*iterators_.back(), i});
    }
}

inline void FlushIterator::Next() {
    if (!IsValid()) return;
    HeapNode node = min_heap_.top();
    min_heap_.pop();
    iterators_[node.source_index].Next();
    if (iterators_[node.source_index].IsValid()) min_heap_.push({*iterators_[node.source_index], node.source_index});
}

class CompactingIterator {
   public:
    template <typename IteratorType>
    explicit CompactingIterator(std::unique_ptr<IteratorType> source);
    bool IsValid() const { return is_valid_; }
    RecordRef Get() const { return current_record_; }
    void Next() { FindNext(); }

   private:
    struct IteratorConcept {
        virtual ~IteratorConcept() = default;
        virtual bool IsValid() const = 0;
        virtual RecordRef Get() const = 0;
        virtual void Next() = 0;
    };
    template <typename IteratorType>
    struct IteratorWrapper final : public IteratorConcept {
        explicit IteratorWrapper(std::unique_ptr<IteratorType> iter) : iter_(std::move(iter)) {}
        bool IsValid() const override { return iter_->IsValid(); }
        RecordRef Get() const override { return iter_->Get(); }
        void Next() override { iter_->Next(); }
        std::unique_ptr<IteratorType> iter_;
    };
    void FindNext();
    std::unique_ptr<IteratorConcept> source_;
    RecordRef current_record_;
    bool is_valid_ = false;
};

template <typename IteratorType>
inline CompactingIterator::CompactingIterator(std::unique_ptr<IteratorType> source)
    : source_(std::make_unique<IteratorWrapper<IteratorType>>(std::move(source))) {
    FindNext();
}

inline void CompactingIterator::FindNext() {
    is_valid_ = false;
    while (source_->IsValid()) {
        RecordRef latest_rec = source_->Get();
        std::string_view current_key = latest_rec.key;
        source_->Next();
        while (source_->IsValid() && source_->Get().key == current_key) {
            latest_rec = source_->Get();
            source_->Next();
        }
        if (latest_rec.type == RecordType::Put) {
            current_record_ = latest_rec;
            is_valid_ = true;
            return;
        }
    }
}

// --- The Final, High-Performance Columnar MemTable ---

class ColumnarMemTable {
   public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::map<std::string_view, GetResult, std::less<>>;
    using SortedBlockList = const std::vector<std::shared_ptr<const SortedColumnarBlock>>;
    explicit ColumnarMemTable(size_t active_block_size_bytes = 16 * 1024, bool enable_compaction = false,
                              std::shared_ptr<Sorter> sorter = std::make_shared<StdSorter>());
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
    std::unique_ptr<FlushIterator> NewRawFlushIterator();
    void Insert(std::string_view key, std::string_view value, RecordType type);
    void SealActiveBlockIfNeeded();
    void BackgroundWorkerLoop();
    void ProcessBlocks(std::vector<std::shared_ptr<ColumnarBlock>> blocks);
    const size_t active_block_threshold_;
    const bool enable_compaction_;
    std::shared_ptr<Sorter> sorter_;
    std::shared_ptr<FlashActiveBlock> active_block_;
    std::shared_ptr<SortedBlockList> sorted_blocks_;
    std::mutex seal_mutex_;
    std::vector<std::shared_ptr<ColumnarBlock>> sealed_blocks_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::thread background_thread_;
    std::atomic<bool> stop_background_thread_{false};
    bool background_thread_processing_{false};
};

inline ColumnarMemTable::ColumnarMemTable(size_t active_block_size_bytes, bool enable_compaction,
                                          std::shared_ptr<Sorter> sorter)
    : active_block_threshold_(std::max(static_cast<size_t>(1), active_block_size_bytes / 48)),
      enable_compaction_(enable_compaction),
      sorter_(std::move(sorter)) {
    active_block_ = std::make_shared<FlashActiveBlock>(active_block_threshold_);
    sorted_blocks_ = std::make_shared<SortedBlockList>();
    background_thread_ = std::thread(&ColumnarMemTable::BackgroundWorkerLoop, this);
}

inline ColumnarMemTable::~ColumnarMemTable() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_background_thread_ = true;
    }
    queue_cond_.notify_one();
    if (background_thread_.joinable()) background_thread_.join();
}

// Rewrote Insert to handle the case where a block is sealed during the add operation.
inline void ColumnarMemTable::Insert(std::string_view key, std::string_view value, RecordType type) {
    while (true) {
        auto current_block = std::atomic_load(&active_block_);
        if (current_block->TryAdd(key, value, type)) {
            if (current_block->size() >= active_block_threshold_) {
                SealActiveBlockIfNeeded();
            }
            return;  // Success
        }
        // If TryAdd failed, it means the block was sealed under us.
        // Loop again to get the new active block and retry.
    }
}

inline void ColumnarMemTable::Put(std::string_view key, std::string_view value) { Insert(key, value, RecordType::Put); }

inline void ColumnarMemTable::Delete(std::string_view key) { Insert(key, "", RecordType::Delete); }

inline ColumnarMemTable::GetResult ColumnarMemTable::Get(std::string_view key) const {
    // Step 1: Search the active block first. This is always the most recent data.
    auto current_active_block = std::atomic_load(&active_block_);
    auto result_in_active = current_active_block->Get(key);
    if (result_in_active.has_value()) {
        // If found in the active block, we have the latest version.
        // Return based on its type (Put or Delete).
        return (result_in_active->type == RecordType::Put) ? GetResult(result_in_active->value) : std::nullopt;
    }

    // Step 2: Access the immutable sorted blocks using a thread-local cache
    // to eliminate contention on the shared block list.

    // Grab a snapshot of the current global list of sorted blocks.
    // This atomic load is very fast.
    auto global_blocks_snapshot = std::atomic_load(&sorted_blocks_);

    // Define the thread-local cache structures. Each thread gets its own copy.
    // `last_seen_snapshot` tracks which version of the global list our cache corresponds to.
    thread_local std::shared_ptr<const SortedBlockList> last_seen_snapshot = nullptr;
    
    // `local_meta_cache` is the actual thread-private copy of the metadata.
    using MetaInfo = std::tuple<std::string_view, std::string_view, const SortedColumnarBlock*>;
    thread_local std::vector<MetaInfo> local_meta_cache;

    // Step 3: Check if our thread-local cache is stale. If it is, update it.
    // This check is the only point of interaction with shared state. It compares two pointers.
    if (last_seen_snapshot != global_blocks_snapshot) {
        // The global list has been updated by the background thread since we last looked.
        // We need to rebuild our private cache.
        
        local_meta_cache.clear();
        if (global_blocks_snapshot) { // Ensure the global list isn't null
            local_meta_cache.reserve(global_blocks_snapshot->size());
            
            // This is the copy operation. It briefly iterates over the shared global list
            // and copies the necessary metadata (min/max keys and a raw pointer)
            // into our private vector.
            for (const auto& block_ptr : *global_blocks_snapshot) {
                if (block_ptr && !block_ptr->empty()) {
                    local_meta_cache.emplace_back(block_ptr->min_key(), block_ptr->max_key(), block_ptr.get());
                }
            }
        }
        
        // After the copy is done, update our tracker to the new version.
        last_seen_snapshot = global_blocks_snapshot;
    }

    // Step 4: Perform the search on the thread-local cache.
    // This entire loop operates on `local_meta_cache`, which is private to this thread.
    // There is ZERO cross-core cache contention here.
    // We iterate in reverse because newer blocks are at the end of the original vector.
    for (auto it = local_meta_cache.rbegin(); it != local_meta_cache.rend(); ++it) {
        // Deconstruct the tuple for easy access.
        const auto& [min_key, max_key, block_raw_ptr] = *it;

        // First-level check: Use min/max keys for a quick range check.
        // This comparison is extremely fast (likely done in CPU registers).
        if (key < min_key || key > max_key) {
            continue; // The key is outside this block's range, skip to the next.
        }

        // Second-level check: Use the Bloom filter. Also very fast.
        if (block_raw_ptr->MayContain(key)) {
            // Only if the key is in range and might be in the block,
            // we perform the final, more expensive lookup.
            auto result = block_raw_ptr->Get(key);
            if (result.has_value()) {
                // Since we are iterating from newest to oldest, the first match
                // we find is the correct, most recent version of the record.
                return (result->type == RecordType::Put) ? GetResult(result->value) : std::nullopt;
            }
        }
    }

    // If the key was not found in the active block or any of the sorted blocks.
    return std::nullopt;
}

inline ColumnarMemTable::MultiGetResult ColumnarMemTable::MultiGet(const std::vector<std::string_view>& keys) const {
    MultiGetResult results;
    std::vector<std::string_view> remaining_keys;
    remaining_keys.reserve(keys.size());
    auto current_active = std::atomic_load(&active_block_);
    for (const auto& key : keys) {
        if (results.count(key)) continue;
        auto result = current_active->Get(key);
        if (result.has_value()) {
            results.emplace(key, (result->type == RecordType::Put) ? GetResult(result->value) : std::nullopt);
        } else {
            remaining_keys.push_back(key);
        }
    }
    if (remaining_keys.empty()) return results;
    auto sorted_blocks_snapshot = std::atomic_load(&sorted_blocks_);
    for (auto it = sorted_blocks_snapshot->rbegin(); it != sorted_blocks_snapshot->rend(); ++it) {
        remaining_keys.erase(std::remove_if(remaining_keys.begin(), remaining_keys.end(),
                                            [&](std::string_view key) {
                                                if (results.count(key)) return true;
                                                if ((*it)->MayContain(key)) {
                                                    auto result = (*it)->Get(key);
                                                    if (result.has_value()) {
                                                        results.emplace(key, (result->type == RecordType::Put)
                                                                                 ? GetResult(result->value)
                                                                                 : std::nullopt);
                                                        return true;
                                                    }
                                                }
                                                return false;
                                            }),
                             remaining_keys.end());
        if (remaining_keys.empty()) break;
    }
    return results;
}

inline void ColumnarMemTable::PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
    // A simple loop is okay here thanks to the retry logic in Insert.
    for (const auto& [key, value] : batch) {
        Insert(key, value, RecordType::Put);
    }
}

// Updated SealActiveBlockIfNeeded to use the new sealing protocol.
inline void ColumnarMemTable::SealActiveBlockIfNeeded() {
    std::lock_guard<std::mutex> lock(seal_mutex_);
    auto current_block = std::atomic_load(&active_block_);
    if (current_block->size() < active_block_threshold_ || current_block->is_sealed()) {
        return;
    }

    // Mark the current block as sealed to prevent any more writes.
    current_block->Seal();

    // Create a new active block and atomically swap it in.
    auto new_active_block = std::make_shared<FlashActiveBlock>(active_block_threshold_);
    std::atomic_exchange(&active_block_, new_active_block);

    // Now, the old block (current_block) is guaranteed to not receive new writes.
    // We can safely convert it and queue it for the background thread.
    auto columnar_block = std::make_shared<ColumnarBlock>();
    for (const auto& record_ref : *current_block) {
        columnar_block->Add(record_ref.key, record_ref.value, record_ref.type);
    }

    if (!columnar_block->empty()) {
        std::lock_guard<std::mutex> q_lock(queue_mutex_);
        sealed_blocks_queue_.push_back(std::move(columnar_block));
    }
    queue_cond_.notify_one();
}

inline void ColumnarMemTable::WaitForBackgroundWork() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cond_.wait(lock, [this] { return sealed_blocks_queue_.empty() && !background_thread_processing_; });
}

inline std::unique_ptr<FlushIterator> ColumnarMemTable::NewRawFlushIterator() {
    WaitForBackgroundWork();
    auto current_sorted_blocks = std::atomic_load(&sorted_blocks_);
    auto current_active_block = std::atomic_load(&active_block_);
    auto all_blocks_mutable = std::vector<std::shared_ptr<const SortedColumnarBlock>>(*current_sorted_blocks);
    if (current_active_block->size() > 0) {
        auto temp_block = std::make_shared<ColumnarBlock>();
        for (const auto& record_ref : *current_active_block) {
            temp_block->Add(record_ref.key, record_ref.value, record_ref.type);
        }
        all_blocks_mutable.push_back(std::make_shared<const SortedColumnarBlock>(temp_block, *sorter_));
    }
    return std::make_unique<FlushIterator>(std::move(all_blocks_mutable));
}

inline std::unique_ptr<CompactingIterator> ColumnarMemTable::NewCompactingIterator() {
    return std::make_unique<CompactingIterator>(NewRawFlushIterator());
}

inline void ColumnarMemTable::BackgroundWorkerLoop() {
    while (true) {
        std::vector<std::shared_ptr<ColumnarBlock>> blocks_to_process;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this] { return !sealed_blocks_queue_.empty() || stop_background_thread_; });
            if (stop_background_thread_ && sealed_blocks_queue_.empty()) return;
            blocks_to_process.swap(sealed_blocks_queue_);
            background_thread_processing_ = true;
        }
        ProcessBlocks(std::move(blocks_to_process));
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            background_thread_processing_ = false;
        }
        queue_cond_.notify_all();
    }
}

inline void ColumnarMemTable::ProcessBlocks(std::vector<std::shared_ptr<ColumnarBlock>> blocks) {
    if (blocks.empty()) return;
    auto old_list_ptr = std::atomic_load(&sorted_blocks_);
    auto new_list_mutable = std::make_shared<std::vector<std::shared_ptr<const SortedColumnarBlock>>>();
    std::vector<std::shared_ptr<const SortedColumnarBlock>> sources_to_merge;
    if (enable_compaction_) {
        sources_to_merge.insert(sources_to_merge.end(), old_list_ptr->begin(), old_list_ptr->end());
    } else {
        *new_list_mutable = *old_list_ptr;
    }
    for (const auto& block : blocks) {
        if (block->empty()) continue;
        auto sorted_block = std::make_shared<const SortedColumnarBlock>(block, *sorter_);
        if (enable_compaction_) {
            sources_to_merge.push_back(std::move(sorted_block));
        } else {
            new_list_mutable->push_back(std::move(sorted_block));
        }
    }
    if (enable_compaction_ && sources_to_merge.size() > 1) {
        auto final_compacted_block = std::make_shared<ColumnarBlock>();
        CompactingIterator iter(std::make_unique<FlushIterator>(std::move(sources_to_merge)));
        while (iter.IsValid()) {
            RecordRef rec = iter.Get();
            final_compacted_block->Add(rec.key, rec.value, rec.type);
            iter.Next();
        }
        new_list_mutable->clear();
        if (!final_compacted_block->empty()) {
            new_list_mutable->push_back(std::make_shared<const SortedColumnarBlock>(final_compacted_block, *sorter_));
        }
    } else if (enable_compaction_) {
        *new_list_mutable = sources_to_merge;
    }
    std::atomic_store(&sorted_blocks_, std::shared_ptr<SortedBlockList>(std::move(new_list_mutable)));
}

#endif  // COLUMNAR_MEMTABLE_H