#ifndef COLUMNAR_MEMTABLE_H
#define COLUMNAR_MEMTABLE_H

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <optional>
#include <functional>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <numeric>
#include <condition_variable>
#include <queue>
#include <cmath>
#include <array>
#include <cstring>

#define XXH_INLINE_ALL 
#include "xxhash.h"

// --- Forward Declarations ---
class SimpleArena;
enum class RecordType;
struct RecordRef;
class ColumnarBlock;
class Sorter;
class StdSorter;
class SortedColumnarBlock;
class FlushIterator;
class CompactingIterator;
class BloomFilter;
class IndexedUnsortedBlock; // Specialized class for the active block

struct XXHasher {
    std::size_t operator()(const std::string_view key) const noexcept {
        return XXH3_64bits(key.data(), key.size());
    }
};

// --- Simple Bloom Filter Implementation ---
class BloomFilter {
public:
    explicit BloomFilter(size_t num_entries, double false_positive_rate = 0.01) {
        if (num_entries == 0) num_entries = 1;
        size_t bits = static_cast<size_t>(-1.44 * num_entries * std::log(false_positive_rate));
        bits_ = std::vector<bool>((bits + 7) & ~7, false);
        num_hashes_ = static_cast<int>(0.7 * (static_cast<double>(bits_.size()) / num_entries));
        if (num_hashes_ < 1) num_hashes_ = 1;
        if (num_hashes_ > 8) num_hashes_ = 8;
    }

    void Add(std::string_view key) {
        std::array<uint64_t, 2> hash_values = Hash(key);
        for (int i = 0; i < num_hashes_; ++i) {
            uint64_t hash = hash_values[0] + i * hash_values[1];
            if (!bits_.empty()) {
                bits_[hash % bits_.size()] = true;
            }
        }
    }

    bool MayContain(std::string_view key) const {
        if (bits_.empty()) {
            return true;
        }
        std::array<uint64_t, 2> hash_values = Hash(key);
        for (int i = 0; i < num_hashes_; ++i) {
            uint64_t hash = hash_values[0] + i * hash_values[1];
            if (!bits_[hash % bits_.size()]) {
                return false;
            }
        }
        return true;
    }

private:
    static std::array<uint64_t, 2> Hash(std::string_view key);
    std::vector<bool> bits_;
    int num_hashes_;
};

// --- Core Data Structures (Arena, Record, Block) ---
class SimpleArena {
public:
    explicit SimpleArena(size_t block_size = 4096) : block_size_(block_size) {
        allocate_block(block_size_);
    }

    std::string_view AllocateAndCopy(std::string_view data) {
        char* mem = Allocate(data.size());
        if (!data.empty()) {
            memcpy(mem, data.data(), data.size());
        }
        return {mem, data.size()};
    }
private:
    char* Allocate(size_t bytes) {
        size_t aligned_bytes = (bytes + 7) & ~7;
        if (current_block_pos_ + aligned_bytes > block_size_) {
            allocate_block(std::max(block_size_, aligned_bytes));
        }
        char* result = blocks_.back().get() + current_block_pos_;
        current_block_pos_ += aligned_bytes;
        return result;
    }

    void allocate_block(size_t size) {
        blocks_.push_back(std::make_unique<char[]>(size));
        current_block_pos_ = 0;
        block_size_ = size;
    }

    std::vector<std::unique_ptr<char[]>> blocks_;
    size_t block_size_;
    size_t current_block_pos_ = 0;
};

enum class RecordType { Put, Delete };

struct RecordRef {
    std::string_view key;
    std::string_view value;
    RecordType type;
};

// A pure data container. It is used for sealed and sorted blocks.
class ColumnarBlock {
public:
    SimpleArena arena;
    std::shared_ptr<std::vector<std::string_view>> keys;
    std::shared_ptr<std::vector<std::string_view>> values;
    std::shared_ptr<std::vector<RecordType>> types;

    ColumnarBlock() : 
        keys(std::make_shared<std::vector<std::string_view>>()),
        values(std::make_shared<std::vector<std::string_view>>()),
        types(std::make_shared<std::vector<RecordType>>()) 
    {}
    
    // Simple add, used only by background compaction logic.
    void Add(std::string_view key_sv, std::string_view value_sv, RecordType type) {
        auto key = arena.AllocateAndCopy(key_sv);
        auto value = arena.AllocateAndCopy(value_sv);
        keys->push_back(key);
        values->push_back(value);
        types->push_back(type);
    }
    
    size_t size() const { return keys->size(); }
    bool empty() const { return keys->empty(); }
};

// Encapsulates the active, mutable block and its hash index.
class IndexedUnsortedBlock {
public:
    std::shared_ptr<ColumnarBlock> block; 
    std::shared_ptr<std::unordered_map<std::string_view, uint32_t, XXHasher>> key_index;

    explicit IndexedUnsortedBlock(size_t initial_capacity) :
        block(std::make_shared<ColumnarBlock>()),
        key_index(std::make_shared<std::unordered_map<std::string_view, uint32_t, XXHasher>>(initial_capacity))
    {
        if (initial_capacity > 0) {
            block->keys->reserve(initial_capacity);
            block->values->reserve(initial_capacity);
            block->types->reserve(initial_capacity);
            key_index->reserve(size_t(initial_capacity * 1.5));
        }
    }

    void Add(std::string_view key_sv, std::string_view value_sv, RecordType type) {
        auto key = block->arena.AllocateAndCopy(key_sv);
        auto value = block->arena.AllocateAndCopy(value_sv);

        bool vector_needs_resize = (block->keys->size() == block->keys->capacity());
        bool map_needs_rehash = (key_index->load_factor() > 0.75);

        if (vector_needs_resize || map_needs_rehash) {
            size_t new_capacity = vector_needs_resize ? std::max(static_cast<size_t>(16), block->keys->size() * 2) : block->keys->capacity();
            
            auto new_keys = std::make_shared<std::vector<std::string_view>>(*block->keys);
            auto new_values = std::make_shared<std::vector<std::string_view>>(*block->values);
            auto new_types = std::make_shared<std::vector<RecordType>>(*block->types);
            auto new_key_index = std::make_shared<std::unordered_map<std::string_view, uint32_t, XXHasher>>(
                key_index->begin(),
                key_index->end(),
                size_t(new_capacity * 1.5)
            );

            new_keys->reserve(new_capacity);
            new_values->reserve(new_capacity);
            new_types->reserve(new_capacity);

            (*new_key_index)[key] = new_keys->size();
            new_keys->push_back(key);
            new_values->push_back(value);
            new_types->push_back(type);
            
            std::atomic_store(&block->keys, new_keys);
            std::atomic_store(&block->values, new_values);
            std::atomic_store(&block->types, new_types);
            std::atomic_store(&key_index, new_key_index);
        } else {
            (*key_index)[key] = block->keys->size();
            block->keys->push_back(key);
            block->values->push_back(value);
            block->types->push_back(type);
        }
    }

    std::optional<std::string_view> Get(std::string_view key) const {
        auto index_ptr = std::atomic_load(&key_index);
        auto it = index_ptr->find(key);
        if (it != index_ptr->end()) {
            uint32_t index = it->second;
            auto values_ptr = std::atomic_load(&block->values);
            auto types_ptr = std::atomic_load(&block->types);
            if (index < types_ptr->size()) {
                 return ((*types_ptr)[index] == RecordType::Put) ? std::optional<std::string_view>((*values_ptr)[index]) : std::nullopt;
            }
        }
        return std::nullopt;
    }
    
    size_t size() const { return block->size(); }
};

// --- Sorter Abstraction ---
class Sorter {
public:
    virtual ~Sorter() = default;
    virtual std::vector<uint32_t> Sort(const ColumnarBlock& block) const = 0;
};

class StdSorter : public Sorter {
public:
    std::vector<uint32_t> Sort(const ColumnarBlock& block) const override {
        auto keys_ptr = std::atomic_load(&block.keys);
        if (keys_ptr->empty()) {
            return {};
        }
        std::vector<uint32_t> indices(keys_ptr->size());
        std::iota(indices.begin(), indices.end(), 0);
        std::stable_sort(indices.begin(), indices.end(),
            [&keys_ptr](uint32_t a, uint32_t b) {
                return (*keys_ptr)[a] < (*keys_ptr)[b];
            });
        return indices;
    }
};

// --- SortedColumnarBlock with Bloom Filter and Sparse Index ---
class SortedColumnarBlock {
public:
    class Iterator;
    explicit SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block_to_sort, const Sorter& sorter);

    bool MayContain(std::string_view key) const {
        if (sorted_indices_.empty() || key < min_key_ || key > max_key_) {
            return false;
        }
        return !bloom_filter_ || bloom_filter_->MayContain(key);
    }

    std::optional<RecordRef> Get(std::string_view key) const;
    Iterator begin() const;

private:
    std::shared_ptr<ColumnarBlock> block_data_;
    std::vector<uint32_t> sorted_indices_;
    std::unique_ptr<BloomFilter> bloom_filter_;
    std::string_view min_key_;
    std::string_view max_key_;
};

class SortedColumnarBlock::Iterator {
public:
    Iterator(const SortedColumnarBlock* block, size_t pos);
    RecordRef operator*() const;
    void Next();
    bool IsValid() const;

private:
    const SortedColumnarBlock* block_;
    size_t pos_;
    std::shared_ptr<const std::vector<std::string_view>> keys_;
    std::shared_ptr<const std::vector<std::string_view>> values_;
    std::shared_ptr<const std::vector<RecordType>> types_;
};

// --- The Flush Iterator (K-Way Merge) ---
class FlushIterator {
public:
    explicit FlushIterator(std::vector<std::shared_ptr<const SortedColumnarBlock>> sources);
    bool IsValid() const;
    RecordRef Get() const;
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

// --- The Compacting Iterator ---
class CompactingIterator {
public:
    explicit CompactingIterator(std::unique_ptr<FlushIterator> source);
    bool IsValid() const;
    RecordRef Get() const;
    void Next();
private:
    void FindNext();
    std::unique_ptr<FlushIterator> source_;
    RecordRef current_record_;
    bool is_valid_ = false;
};

// --- The Final Configurable Columnar MemTable ---
class ColumnarMemTable {
public:
    using GetResult = std::optional<std::string_view>;
    using MultiGetResult = std::map<std::string_view, GetResult, std::less<>>;
    using SortedBlockList = const std::vector<std::shared_ptr<const SortedColumnarBlock>>;

    explicit ColumnarMemTable(
        size_t unsorted_block_size_bytes = 16 * 1024,
        bool enable_compaction = false,
        std::shared_ptr<Sorter> sorter = std::make_shared<StdSorter>());
    ~ColumnarMemTable();

    ColumnarMemTable(const ColumnarMemTable&) = delete;
    ColumnarMemTable& operator=(const ColumnarMemTable&) = delete;

    void WaitForBackgroundWork();
    std::unique_ptr<CompactingIterator> NewCompactingIterator();
    void Put(std::string_view key, std::string_view value);
    void Delete(std::string_view key);
    GetResult Get(std::string_view key) const;
    void PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch);
    MultiGetResult MultiGet(const std::vector<std::string_view>& keys) const;
    size_t GetSortedBlockNum() const;

private:
    std::unique_ptr<FlushIterator> NewRawFlushIterator();
    void Insert(std::string_view key, std::string_view value, RecordType type);
    std::shared_ptr<ColumnarBlock> SealUnsortedBlock();
    void BackgroundWorkerLoop();
    void ProcessBlocks(std::vector<std::shared_ptr<ColumnarBlock>> blocks);
    
private:
    const size_t unsorted_block_size_bytes_;
    const bool enable_compaction_;
    std::shared_ptr<Sorter> sorter_;
    std::shared_ptr<IndexedUnsortedBlock> unsorted_block_;
    std::shared_ptr<SortedBlockList> sorted_blocks_;
    std::mutex write_mutex_;
    std::vector<std::shared_ptr<ColumnarBlock>> sealed_blocks_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::thread background_thread_;
    std::atomic<bool> stop_background_thread_{false};
    bool background_thread_processing_{false};
};

// ======================================================================================
// --- Complete Implementations ---
// ======================================================================================

inline std::array<uint64_t, 2> BloomFilter::Hash(std::string_view key) {
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;
    uint64_t h1 = 0xdeadbeefdeadbeef ^ (key.length() * m);
    const uint64_t* data = reinterpret_cast<const uint64_t*>(key.data());
    const int nblocks = key.length() / 8;
    for (int i = 0; i < nblocks; i++) {
        uint64_t k = data[i];
        k *= m; k ^= k >> r; k *= m;
        h1 ^= k; h1 *= m;
    }
    const unsigned char* data2 = reinterpret_cast<const unsigned char*>(key.data()) + nblocks * 8;
    switch (key.length() & 7) {
        case 7: h1 ^= uint64_t(data2[6]) << 48;
        case 6: h1 ^= uint64_t(data2[5]) << 40;
        case 5: h1 ^= uint64_t(data2[4]) << 32;
        case 4: h1 ^= uint64_t(data2[3]) << 24;
        case 3: h1 ^= uint64_t(data2[2]) << 16;
        case 2: h1 ^= uint64_t(data2[1]) << 8;
        case 1: h1 ^= uint64_t(data2[0]); h1 *= m;
    };
    h1 ^= h1 >> r; h1 *= m; h1 ^= h1 >> r;
    return {h1, h1 ^ m};
}

inline SortedColumnarBlock::SortedColumnarBlock(std::shared_ptr<ColumnarBlock> block_to_sort, const Sorter& sorter)
    : block_data_(std::move(block_to_sort)) {
    sorted_indices_ = sorter.Sort(*block_data_);
    if (!sorted_indices_.empty()) {
        auto keys_ptr = std::atomic_load(&block_data_->keys);
        min_key_ = (*keys_ptr)[sorted_indices_.front()];
        max_key_ = (*keys_ptr)[sorted_indices_.back()];
        bloom_filter_ = std::make_unique<BloomFilter>(block_data_->size());
        for (const auto& key : *keys_ptr) {
            bloom_filter_->Add(key);
        }
    }
}

inline std::optional<RecordRef> SortedColumnarBlock::Get(std::string_view key) const {
    auto keys_ptr = std::atomic_load(&block_data_->keys);
    auto values_ptr = std::atomic_load(&block_data_->values);
    auto types_ptr = std::atomic_load(&block_data_->types);
    auto it = std::lower_bound(sorted_indices_.begin(), sorted_indices_.end(), key,
        [&](uint32_t index, std::string_view k) {
            return (*keys_ptr)[index] < k;
        });
    if (it != sorted_indices_.end() && (*keys_ptr)[*it] == key) {
        uint32_t index = *it;
        return RecordRef{(*keys_ptr)[index], (*values_ptr)[index], (*types_ptr)[index]};
    }
    return std::nullopt;
}

inline SortedColumnarBlock::Iterator::Iterator(const SortedColumnarBlock* block, size_t pos) : 
    block_(block), 
    pos_(pos),
    keys_(std::atomic_load(&block->block_data_->keys)),
    values_(std::atomic_load(&block->block_data_->values)),
    types_(std::atomic_load(&block->block_data_->types)) {}

inline RecordRef SortedColumnarBlock::Iterator::operator*() const {
    uint32_t index = block_->sorted_indices_[pos_];
    return {(*keys_)[index], (*values_)[index], (*types_)[index]};
}

inline void SortedColumnarBlock::Iterator::Next() { ++pos_; }
inline bool SortedColumnarBlock::Iterator::IsValid() const { return pos_ < block_->sorted_indices_.size(); }
inline SortedColumnarBlock::Iterator SortedColumnarBlock::begin() const { return Iterator(this, 0); }

inline FlushIterator::FlushIterator(std::vector<std::shared_ptr<const SortedColumnarBlock>> sources)
    : sources_(std::move(sources)) {
    for (size_t i = 0; i < sources_.size(); ++i) {
        iterators_.emplace_back(sources_[i]->begin());
        if (iterators_.back().IsValid()) {
            min_heap_.push({ *iterators_.back(), i });
        }
    }
}

inline bool FlushIterator::IsValid() const { return !min_heap_.empty(); }
inline RecordRef FlushIterator::Get() const { return min_heap_.top().record; }

inline void FlushIterator::Next() {
    if (!IsValid()) {
        return;
    }
    HeapNode node = min_heap_.top();
    min_heap_.pop();
    iterators_[node.source_index].Next();
    if (iterators_[node.source_index].IsValid()) {
        min_heap_.push({ *iterators_[node.source_index], node.source_index });
    }
}

inline CompactingIterator::CompactingIterator(std::unique_ptr<FlushIterator> source) 
    : source_(std::move(source)) {
    FindNext();
}

inline bool CompactingIterator::IsValid() const { return is_valid_; }
inline RecordRef CompactingIterator::Get() const { return current_record_; }
inline void CompactingIterator::Next() { FindNext(); }

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

// ColumnarMemTable Implementations
inline ColumnarMemTable::ColumnarMemTable(size_t unsorted_block_size_bytes, bool enable_compaction, std::shared_ptr<Sorter> sorter)
    : unsorted_block_size_bytes_(unsorted_block_size_bytes),
      enable_compaction_(enable_compaction),
      sorter_(std::move(sorter)),
      stop_background_thread_(false),
      background_thread_processing_(false) {
    const size_t estimated_record_size = 32; 
    size_t initial_capacity = unsorted_block_size_bytes_ / estimated_record_size;
    if (initial_capacity == 0) {
        initial_capacity = 1;
    }
    unsorted_block_ = std::make_shared<IndexedUnsortedBlock>(initial_capacity);
    sorted_blocks_ = std::make_shared<SortedBlockList>();
    background_thread_ = std::thread(&ColumnarMemTable::BackgroundWorkerLoop, this);
}

inline ColumnarMemTable::~ColumnarMemTable() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_background_thread_ = true;
    }
    queue_cond_.notify_one();
    if (background_thread_.joinable()) {
        background_thread_.join();
    }
}

inline void ColumnarMemTable::WaitForBackgroundWork() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cond_.wait(lock, [this]{
        return sealed_blocks_queue_.empty() && !background_thread_processing_;
    });
}

inline std::unique_ptr<CompactingIterator> ColumnarMemTable::NewCompactingIterator() {
    return std::make_unique<CompactingIterator>(NewRawFlushIterator());
}

inline size_t ColumnarMemTable::GetSortedBlockNum() const {
    return std::atomic_load(&sorted_blocks_)->size();
}

inline void ColumnarMemTable::Put(std::string_view key, std::string_view value) {
    Insert(key, value, RecordType::Put);
}

inline void ColumnarMemTable::Delete(std::string_view key) {
    Insert(key, "", RecordType::Delete);
}

inline ColumnarMemTable::GetResult ColumnarMemTable::Get(std::string_view key) const {
    auto current_unsorted = std::atomic_load(&unsorted_block_);
    auto result_in_unsorted = current_unsorted->Get(key);
    if (result_in_unsorted.has_value()) {
        return result_in_unsorted;
    }

    auto sorted_blocks_snapshot = std::atomic_load(&sorted_blocks_);
    for (auto it_block = sorted_blocks_snapshot->rbegin(); it_block != sorted_blocks_snapshot->rend(); ++it_block) {
        if ((*it_block)->MayContain(key)) {
            auto sorted_result = (*it_block)->Get(key);
            if (sorted_result.has_value()) {
                return (sorted_result.value().type == RecordType::Put) ? GetResult(sorted_result.value().value) : std::nullopt;
            }
        }
    }
    return std::nullopt;
}

inline void ColumnarMemTable::Insert(std::string_view key, std::string_view value, RecordType type) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    auto current_block = std::atomic_load(&unsorted_block_);
    current_block->Add(key, value, type);
    
    const size_t estimated_record_size = 32; 
    size_t threshold_in_records = unsorted_block_size_bytes_ / estimated_record_size;
    if (threshold_in_records == 0) {
        threshold_in_records = 1;
    }
    
    if (current_block->size() >= threshold_in_records) {
        auto sealed_block = SealUnsortedBlock();
        {
            std::lock_guard<std::mutex> q_lock(queue_mutex_);
            sealed_blocks_queue_.push_back(std::move(sealed_block));
        }
        queue_cond_.notify_one();
    }
}

inline std::shared_ptr<ColumnarBlock> ColumnarMemTable::SealUnsortedBlock() {
    const size_t estimated_record_size = 32;
    size_t initial_capacity = unsorted_block_size_bytes_ / estimated_record_size;
    if (initial_capacity == 0) {
        initial_capacity = 1;
    }
    auto new_unsorted_block = std::make_shared<IndexedUnsortedBlock>(initial_capacity);
    auto old_indexed_block = std::atomic_exchange(&unsorted_block_, new_unsorted_block);
    return old_indexed_block->block;
}

inline void ColumnarMemTable::PutBatch(const std::vector<std::pair<std::string_view, std::string_view>>& batch) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    auto current_block = std::atomic_load(&unsorted_block_);
    for(const auto& [key, value] : batch) {
        current_block->Add(key, value, RecordType::Put);
    }
    
    const size_t estimated_record_size = 32;
    size_t threshold_in_records = unsorted_block_size_bytes_ / estimated_record_size;
    if (threshold_in_records == 0) {
        threshold_in_records = 1;
    }
    
    if (current_block->size() >= threshold_in_records) {
        auto sealed_block = SealUnsortedBlock();
        {
            std::lock_guard<std::mutex> q_lock(queue_mutex_);
            sealed_blocks_queue_.push_back(std::move(sealed_block));
        }
        queue_cond_.notify_one();
    }
}

inline ColumnarMemTable::MultiGetResult ColumnarMemTable::MultiGet(const std::vector<std::string_view>& keys) const {
    MultiGetResult results;
    std::vector<std::string_view> remaining_keys;
    remaining_keys.reserve(keys.size());

    auto current_unsorted = std::atomic_load(&unsorted_block_);
    for (const auto& key : keys) {
        auto result = current_unsorted->Get(key);
        if (result.has_value()) {
            results.emplace(key, *result);
        } else {
            remaining_keys.push_back(key);
        }
    }

    if (remaining_keys.empty()) {
        return results;
    }

    auto sorted_blocks_snapshot = std::atomic_load(&sorted_blocks_);
    for (auto it_block = sorted_blocks_snapshot->rbegin(); it_block != sorted_blocks_snapshot->rend(); ++it_block) {
        remaining_keys.erase(std::remove_if(remaining_keys.begin(), remaining_keys.end(), 
            [&](std::string_view key){
                if ((*it_block)->MayContain(key)) {
                    auto result = (*it_block)->Get(key);
                    if (result.has_value()) {
                        results.emplace(key, (result.value().type == RecordType::Put) ? GetResult(result.value().value) : std::nullopt);
                        return true;
                    }
                }
                return false;
            }), remaining_keys.end());
        if (remaining_keys.empty()) {
            break;
        }
    }
    return results;
}

inline std::unique_ptr<FlushIterator> ColumnarMemTable::NewRawFlushIterator() {
    WaitForBackgroundWork();
    auto current_sorted_blocks = std::atomic_load(&sorted_blocks_);
    auto current_unsorted_indexed_block = std::atomic_load(&unsorted_block_);
    auto current_unsorted_block = current_unsorted_indexed_block->block;
    
    auto all_blocks_mutable = std::vector<std::shared_ptr<const SortedColumnarBlock>>(*current_sorted_blocks);
    if (!current_unsorted_block->empty()) {
        auto compacted_block = std::make_shared<ColumnarBlock>();
        auto sorted_indices = sorter_->Sort(*current_unsorted_block);
        
        auto keys_ptr = std::atomic_load(&current_unsorted_block->keys);
        auto values_ptr = std::atomic_load(&current_unsorted_block->values);
        auto types_ptr = std::atomic_load(&current_unsorted_block->types);
        
        if (!sorted_indices.empty()) {
            for (size_t i = 0; i < sorted_indices.size();) {
                size_t j = i;
                while (j + 1 < sorted_indices.size() && (*keys_ptr)[sorted_indices[j+1]] == (*keys_ptr)[sorted_indices[i]]) {
                    j++;
                }
                uint32_t latest_index = sorted_indices[j];
                compacted_block->Add((*keys_ptr)[latest_index], (*values_ptr)[latest_index], (*types_ptr)[latest_index]);
                i = j + 1;
            }
        }
        if (!compacted_block->empty()){
            all_blocks_mutable.push_back(std::make_shared<const SortedColumnarBlock>(compacted_block, *sorter_));
        }
    }
    return std::make_unique<FlushIterator>(std::move(all_blocks_mutable));
}

inline void ColumnarMemTable::BackgroundWorkerLoop() {
    while (true) {
        std::vector<std::shared_ptr<ColumnarBlock>> blocks_to_process;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cond_.wait(lock, [this]{
                return !sealed_blocks_queue_.empty() || stop_background_thread_;
            });
            if (stop_background_thread_ && sealed_blocks_queue_.empty()) {
                return;
            }
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
    if (blocks.empty()) {
        return;
    }
    
    auto old_list_ptr = std::atomic_load(&sorted_blocks_);
    auto new_list_mutable = std::make_shared<std::vector<std::shared_ptr<const SortedColumnarBlock>>>();
    std::vector<std::shared_ptr<const SortedColumnarBlock>> sources_to_merge;
    
    if (enable_compaction_) {
        sources_to_merge.insert(sources_to_merge.end(), old_list_ptr->begin(), old_list_ptr->end());
    } else {
        *new_list_mutable = *old_list_ptr;
    }
    
    for (const auto& block : blocks) {
        if(block->empty()) {
            continue;
        }
        auto compacted_raw = std::make_shared<ColumnarBlock>();
        auto sorted_indices = sorter_->Sort(*block);
        
        auto keys_ptr = std::atomic_load(&block->keys);
        auto values_ptr = std::atomic_load(&block->values);
        auto types_ptr = std::atomic_load(&block->types);
        
        if (!sorted_indices.empty()) {
             for (size_t i = 0; i < sorted_indices.size();) {
                size_t j = i;
                while (j + 1 < sorted_indices.size() && (*keys_ptr)[sorted_indices[j+1]] == (*keys_ptr)[sorted_indices[i]]) {
                    j++;
                }
                uint32_t latest_index = sorted_indices[j];
                compacted_raw->Add((*keys_ptr)[latest_index], (*values_ptr)[latest_index], (*types_ptr)[latest_index]);
                i = j + 1;
            }
        }
        if (!compacted_raw->empty()){
            auto sorted_and_compacted = std::make_shared<const SortedColumnarBlock>(compacted_raw, *sorter_);
            if (enable_compaction_) {
                sources_to_merge.push_back(std::move(sorted_and_compacted));
            } else {
                new_list_mutable->push_back(std::move(sorted_and_compacted));
            }
        }
    }
    
    if (enable_compaction_ && sources_to_merge.size() > 1) {
        auto final_compacted_block = std::make_shared<ColumnarBlock>();
        CompactingIterator iter(std::make_unique<FlushIterator>(std::move(sources_to_merge)));
        while(iter.IsValid()) {
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

#endif // COLUMNAR_MEMTABLE_H