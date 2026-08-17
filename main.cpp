// This must be the first include to ensure it overrides the default allocator
#include <mimalloc.h>

// --- Standard Library Includes ---
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// --- Third-Party Includes ---
#include "benchmark/benchmark.h"

// --- Project Includes ---
#include "columnar_memtable.h"
#include "skiplist_memtable.h"

// =================================================================================================
// UTILITIES
// =================================================================================================
std::string generate_random_string(size_t length) {
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    thread_local std::mt19937 generator(0xC011A4U);
    std::uniform_int_distribution<size_t> distribution(0, max_index - 1);
    std::string random_string(length, '\0');
    for (size_t i = 0; i < length; ++i) {
        random_string[i] = charset[distribution(generator)];
    }
    return random_string;
}

// =================================================================================================
// FUNCTIONAL TESTS
// =================================================================================================
namespace FunctionalTests {
int total_tests_passed = 0;
int total_tests_failed = 0;
void RUN_TEST(const std::function<bool()>& test_func, const std::string& test_name) {
    std::cout << "[RUNNING] " << test_name << "..." << std::flush;
    if (test_func()) {
        total_tests_passed++;
        std::cout << "\r[  PASS ] " << test_name << std::endl;
    } else {
        total_tests_failed++;
        std::cout << "\r[  FAIL ] " << test_name << std::endl;
    }
}
template <typename T>
std::shared_ptr<T> create_memtable(size_t size, bool compaction, size_t batch_worker_count = 1) {
    if constexpr (std::is_same_v<T, SkipListMemTable>) {
        return std::shared_ptr<T>(new T(size, compaction, nullptr, batch_worker_count));
    } else {
        return T::Create(size, compaction, std::make_shared<StdSorter>(), 16, batch_worker_count);
    }
}
template <typename MemTableType>
bool TestBasicPutGetDelete() {
    auto memtable = create_memtable<MemTableType>(1024, false);
    memtable->Put("apple", "red");
    memtable->Put("banana", "yellow");
    auto val = memtable->Get("apple");
    if (!val.has_value() || val.value() != "red") return false;
    memtable->Delete("banana");
    val = memtable->Get("banana");
    if (val.has_value()) return false;
    memtable->WaitForBackgroundWork();
    return true;
}
template <typename MemTableType>
bool TestOverwrite() {
    auto memtable = create_memtable<MemTableType>(1024, false);
    memtable->Put("key1", "value1");
    memtable->Put("key1", "value2");
    auto val = memtable->Get("key1");
    memtable->WaitForBackgroundWork();
    return val.has_value() && val.value() == "value2";
}
template <typename MemTableType>
bool TestCompactingIteratorCorrectness() {
    auto memtable = create_memtable<MemTableType>(512, false);
    memtable->Put("c", "3");
    memtable->Put("a", "1");
    memtable->Put("e", "5");
    memtable->Put("b", "2");
    memtable->Put("d", "4");
    memtable->WaitForBackgroundWork();
    memtable->Put("c", "3_new");
    memtable->Delete("a");
    auto iter = memtable->NewCompactingIterator();
    std::vector<std::pair<std::string, std::string>> results;
    while (iter->IsValid()) {
        RecordRef rec = iter->Get();
        results.emplace_back(std::string(rec.key), std::string(rec.value));
        iter->Next();
    }
    std::vector<std::pair<std::string, std::string>> expected_results = {
        {"b", "2"}, {"c", "3_new"}, {"d", "4"}, {"e", "5"}};
    if (results != expected_results) {
        std::cerr << "\n  [DEBUG] Iterator Mismatch!\n"
                  << "  Expected: {b:2}, {c:3_new}, {d:4}, {e:5}\n"
                  << "  Actual:   ";
        for (const auto& p : results) std::cerr << "{" << p.first << ":" << p.second << "} ";
        std::cerr << std::endl;
        return false;
    }
    return true;
}
template <typename MemTableType>
bool TestBatchPut(size_t worker_count) {
    auto memtable = create_memtable<MemTableType>(116 * 64, false, worker_count);
    std::vector<std::pair<std::string, std::string>> owned;
    owned.reserve(2048);
    for (size_t i = 0; i < 2048; ++i) {
        owned.emplace_back("batch_key_" + std::to_string(i), "value_" + std::to_string(i));
    }
    std::vector<std::pair<std::string_view, std::string_view>> batch;
    batch.reserve(owned.size());
    for (const auto& [key, value] : owned) batch.emplace_back(key, value);
    memtable->PutBatch(batch);
    memtable->WaitForBackgroundWork();
    for (const auto& [key, value] : owned) {
        auto actual = memtable->Get(key);
        if (!actual || *actual != value) return false;
    }
    return true;
}
template <typename MemTableType>
bool TestMultiGet(size_t worker_count) {
    auto memtable = create_memtable<MemTableType>(116 * 8, false, worker_count);
    const std::vector<std::pair<std::string_view, std::string_view>> batch = {
        {"a", "1"}, {"b", "2"}, {"c", "3"}};
    memtable->PutBatch(batch);
    memtable->WaitForBackgroundWork();
    memtable->Delete("b");
    memtable->Put("d", "4");

    const std::vector<std::string_view> keys = {"a", "b", "c", "a", "d", "missing"};
    const auto results = memtable->MultiGet(keys);
    return results.size() == keys.size() && results[0] && *results[0] == "1" && !results[1] && results[2] &&
           *results[2] == "3" && results[3] && *results[3] == "1" && results[4] && *results[4] == "4" &&
           !results[5];
}
template <typename MemTableType>
bool TestSingleWorkerBatchOverwriteOrder() {
    auto memtable = create_memtable<MemTableType>(116 * 64, false, 1);
    const std::vector<std::pair<std::string_view, std::string_view>> batch = {
        {"same_key", "v1"}, {"other", "x"}, {"same_key", "v2"}, {"same_key", "v3"}};
    memtable->PutBatch(batch);

    auto value = memtable->Get("same_key");
    if (!value || *value != "v3") return false;

    auto iterator = memtable->NewCompactingIterator();
    while (iterator->IsValid()) {
        const RecordRef record = iterator->Get();
        if (record.key == "same_key") return record.value == "v3";
        iterator->Next();
    }
    return false;
}
bool TestColumnarInstanceIsolation() {
    {
        auto first = create_memtable<ColumnarMemTable>(116 * 64, false);
        first->Put("only_in_first", "old");
    }
    auto second = create_memtable<ColumnarMemTable>(116 * 64, false);
    if (second->Get("only_in_first").has_value()) return false;
    second->Put("only_in_second", "new");
    auto value = second->Get("only_in_second");
    return value && *value == "new";
}
template <typename MemTableType>
bool TestIteratorWithLongRandomKeys() {
    auto memtable = create_memtable<MemTableType>(116 * 128, false);
    std::mt19937_64 generator(0x123456789ULL);
    std::vector<std::pair<std::string, std::string>> expected;
    expected.reserve(1024);
    for (size_t i = 0; i < 1024; ++i) {
        std::string key(16, '\0');
        const uint64_t first = generator();
        const uint64_t second = generator();
        memcpy(key.data(), &first, sizeof(first));
        memcpy(key.data() + sizeof(first), &second, sizeof(second));
        expected.emplace_back(std::move(key), "v" + std::to_string(i));
        memtable->Put(expected.back().first, expected.back().second);
    }
    std::sort(expected.begin(), expected.end());
    auto iterator = memtable->NewCompactingIterator();
    size_t index = 0;
    while (iterator->IsValid()) {
        if (index >= expected.size()) return false;
        const auto record = iterator->Get();
        if (record.key != expected[index].first || record.value != expected[index].second) return false;
        ++index;
        iterator->Next();
    }
    return index == expected.size();
}
template <typename MemTableType>
void RunAllFor(const std::string& type_name) {
    std::cout << "\n--- Running Functional Tests for " << type_name << " ---" << std::endl;
    RUN_TEST(TestBasicPutGetDelete<MemTableType>, "Basic Put, Get, Delete");
    RUN_TEST(TestOverwrite<MemTableType>, "Key Overwrite");
    RUN_TEST(TestCompactingIteratorCorrectness<MemTableType>, "Compacting Iterator Correctness");
    RUN_TEST([] { return TestBatchPut<MemTableType>(1); }, "PutBatch Correctness (1 worker)");
    RUN_TEST([] { return TestBatchPut<MemTableType>(4); }, "PutBatch Correctness (4 workers)");
    RUN_TEST([] { return TestMultiGet<MemTableType>(1); }, "MultiGet Correctness (1 worker)");
    RUN_TEST([] { return TestMultiGet<MemTableType>(4); }, "MultiGet Correctness (4 workers)");
    RUN_TEST(TestSingleWorkerBatchOverwriteOrder<MemTableType>, "PutBatch overwrite order (1 worker)");
    RUN_TEST(TestIteratorWithLongRandomKeys<MemTableType>, "Iterator Ordering (long random keys)");
    if constexpr (std::is_same_v<MemTableType, ColumnarMemTable>)
        RUN_TEST(TestColumnarInstanceIsolation, "Instance-local TLS Cache");
}
}  // namespace FunctionalTests

// =================================================================================================
// BENCHMARKING INFRASTRUCTURE
// =================================================================================================
const int NUM_OPS = 500'000;
const size_t KEY_LEN = 16;
const size_t VAL_LEN = 100;
const size_t LOGICAL_RECORD_BYTES = KEY_LEN + VAL_LEN;
// Columnar interprets this setting per shard.  With 16 shards, 4 MiB each
// caps the aggregate active generation at roughly 64 MiB; the rest of the
// 1 GiB logical memtable resides in sealed/sorted blocks.
const size_t SHARD_ACTIVE_BLOCK_SIZE_BYTES = 4ULL * 1024 * 1024;

const size_t MEMTABLE_LOGICAL_MIB = 1024;
const size_t MEMTABLE_LOGICAL_BYTES = MEMTABLE_LOGICAL_MIB * 1024 * 1024;
const size_t SUSTAINED_GENERATIONS = 20;
const size_t RECORDS_PER_MEMTABLE = MEMTABLE_LOGICAL_BYTES / LOGICAL_RECORD_BYTES;
const size_t WRITE_CHUNK_RECORDS = 64 * 1024;
const size_t MAXIMUM_PENDING_LOGICAL_BYTES = 64ULL * 1024 * 1024;
const size_t BLOCK_SIZE_BYTES = SHARD_ACTIVE_BLOCK_SIZE_BYTES;

using StringPair = std::pair<std::string, std::string>;
std::vector<StringPair> write_data;
std::vector<std::string> read_keys;
std::vector<std::pair<std::string_view, std::string_view>> write_batch;

using FixedKey = std::array<char, KEY_LEN>;
std::vector<FixedKey> sustained_keys;
std::string sustained_value;
std::vector<std::vector<std::pair<std::string_view, std::string_view>>> sustained_write_batches;
std::once_flag sustained_data_once;

uint64_t Mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void PrepareSustainedData() {
    std::call_once(sustained_data_once, [] {
        std::cout << "--- Preparing " << RECORDS_PER_MEMTABLE
                  << " unique records for the sustained lifecycle benchmark ---" << std::endl;
        sustained_keys.resize(RECORDS_PER_MEMTABLE);
        sustained_value.assign(VAL_LEN, 'v');
        for (size_t i = 0; i < sustained_keys.size(); ++i) {
            // The randomized first half avoids a sorted-insertion advantage for
            // the skip list.  The second half makes every binary key unique.
            const uint64_t randomized = __builtin_bswap64(Mix64(i));
            const uint64_t unique = __builtin_bswap64(static_cast<uint64_t>(i));
            memcpy(sustained_keys[i].data(), &randomized, sizeof(randomized));
            memcpy(sustained_keys[i].data() + sizeof(randomized), &unique, sizeof(unique));
        }

        const size_t chunk_count =
            (sustained_keys.size() + WRITE_CHUNK_RECORDS - 1) / WRITE_CHUNK_RECORDS;
        sustained_write_batches.reserve(chunk_count);
        for (size_t begin = 0; begin < sustained_keys.size(); begin += WRITE_CHUNK_RECORDS) {
            const size_t end = std::min(sustained_keys.size(), begin + WRITE_CHUNK_RECORDS);
            auto& chunk = sustained_write_batches.emplace_back();
            chunk.reserve(end - begin);
            for (size_t i = begin; i < end; ++i) {
                chunk.emplace_back(std::string_view(sustained_keys[i].data(), sustained_keys[i].size()),
                                   std::string_view(sustained_value));
            }
        }

    });
}

void PrepareData(int num_ops) {
    std::cout << "--- Preparing " << num_ops << " key/value pairs for benchmarks ---" << std::endl;
    write_data.reserve(num_ops);
    read_keys.reserve(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        std::string key = generate_random_string(KEY_LEN);
        write_data.push_back({key, generate_random_string(VAL_LEN)});
        read_keys.push_back(key);
    }
    std::shuffle(read_keys.begin(), read_keys.end(), std::mt19937(0x5EEDU));
    write_batch.reserve(write_data.size());
    for (const auto& [key, value] : write_data) write_batch.emplace_back(key, value);
}

template <typename MemTableType>
class BenchmarkRunner {
   public:
    std::shared_ptr<MemTableType> memtable;
    void SetUp(size_t batch_worker_count, size_t block_size_bytes = BLOCK_SIZE_BYTES,
               bool async_background_processing = true, bool compaction = false) {
        if constexpr (std::is_same_v<MemTableType, SkipListMemTable>) {
            memtable = std::make_shared<MemTableType>(block_size_bytes, false, nullptr, batch_worker_count);
        } else {
            memtable = MemTableType::Create(block_size_bytes, compaction, std::make_shared<StdSorter>(), 16,
                                            batch_worker_count, async_background_processing);
        }
    }
    void Reset() { memtable.reset(); }
};

// The small diagnostics use one fresh memtable and 500k records.  The primary
// sustained benchmarks below rotate full logical memtables and include iterator
// flush traversal.  All benchmarks report wall time and process-wide CPU time.
template <typename MemTableType>
void BM_PutScalar1Worker(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(1);
    for (auto _ : state) {
        for (const auto& [key, value] : write_batch) runner.memtable->Put(key, value);
    }
    state.counters["memory_bytes"] = static_cast<double>(runner.memtable->ApproximateMemoryUsage());
    runner.Reset();
    state.SetItemsProcessed(state.iterations() * write_batch.size());
}

template <typename MemTableType>
void BM_PutBatch(benchmark::State& state) {
    const size_t worker_count = static_cast<size_t>(state.range(0));
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(worker_count);
    for (auto _ : state) runner.memtable->PutBatch(write_batch);
    state.counters["memory_bytes"] = static_cast<double>(runner.memtable->ApproximateMemoryUsage());
    runner.Reset();
    state.SetItemsProcessed(state.iterations() * write_batch.size());
}

template <typename MemTableType, bool MixedReads, bool BoundBackgroundBacklog = false,
          bool ScalarPut = false, size_t ActiveBlockSizeBytes = SHARD_ACTIVE_BLOCK_SIZE_BYTES,
          size_t Generations = SUSTAINED_GENERATIONS>
void BM_SustainedWriteFlush(benchmark::State& state) {
    PrepareSustainedData();
    const size_t worker_count = static_cast<size_t>(state.range(0));
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(worker_count, ActiveBlockSizeBytes, true, false);

    uint64_t write_records = 0;
    uint64_t point_reads = 0;
    uint64_t point_read_misses = 0;
    uint64_t flushed_records = 0;
    uint64_t flushed_logical_bytes = 0;
    uint64_t iterator_checksum = 0;
    uint64_t full_active_records = 0;
    uint64_t full_pending_records = 0;
    uint64_t full_sorted_records = 0;
    uint64_t full_pending_blocks = 0;
    uint64_t full_sorted_blocks = 0;
    uint64_t full_state_samples = 0;
    size_t peak_table_memory_bytes = 0;
    double put_seconds = 0.0;
    double point_read_seconds = 0.0;
    double flush_prepare_seconds = 0.0;
    double flush_iterate_seconds = 0.0;
    double rotate_seconds = 0.0;
    std::vector<std::string_view> point_read_batch;
    if constexpr (MixedReads) point_read_batch.reserve(WRITE_CHUNK_RECORDS / 4 + 1);

    for (auto _ : state) {
        for (size_t generation = 0; generation < Generations; ++generation) {
            size_t inserted_in_generation = 0;
            size_t read_ratio_remainder = 0;

            for (const auto& chunk : sustained_write_batches) {
                const auto put_start = std::chrono::steady_clock::now();
                if constexpr (ScalarPut) {
                    for (const auto& [key, value] : chunk) runner.memtable->Put(key, value);
                } else {
                    runner.memtable->PutBatch(chunk);
                }
                if constexpr (BoundBackgroundBacklog) {
                    constexpr size_t kMaximumPendingBlocks =
                        std::max<size_t>(1, MAXIMUM_PENDING_LOGICAL_BYTES / ActiveBlockSizeBytes);
                    runner.memtable->WaitForBackgroundBacklogAtMost(kMaximumPendingBlocks);
                }
                put_seconds +=
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - put_start).count();
                inserted_in_generation += chunk.size();
                write_records += chunk.size();
                peak_table_memory_bytes =
                    std::max(peak_table_memory_bytes, runner.memtable->ApproximateMemoryUsage());

                if constexpr (MixedReads) {
                    // One successful point lookup per four writes gives an
                    // exact 80% write / 20% read operation mix.
                    read_ratio_remainder += chunk.size();
                    const size_t reads_this_chunk = read_ratio_remainder / 4;
                    read_ratio_remainder %= 4;
                    const auto read_start = std::chrono::steady_clock::now();
                    point_read_batch.clear();
                    for (size_t read = 0; read < reads_this_chunk; ++read) {
                        const uint64_t selector =
                            Mix64(point_reads + read + generation * RECORDS_PER_MEMTABLE);
                        const size_t key_index = selector % inserted_in_generation;
                        point_read_batch.emplace_back(sustained_keys[key_index].data(),
                                                      sustained_keys[key_index].size());
                    }
                    const auto values = runner.memtable->MultiGet(point_read_batch);
                    for (const auto& value : values) point_read_misses += !value.has_value();
                    point_reads += reads_this_chunk;
                    point_read_seconds +=
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - read_start).count();
                }
            }

            full_active_records += runner.memtable->GetActiveRecordCount();
            full_pending_records += runner.memtable->GetPendingSealedRecordCount();
            full_sorted_records += runner.memtable->GetSortedRecordCount();
            full_pending_blocks += runner.memtable->GetPendingSealedBlockNum();
            full_sorted_blocks += runner.memtable->GetSortedBlockNum();
            ++full_state_samples;

            // This is the simulated flush.  For Columnar, iterator creation
            // necessarily seals the remaining active blocks and waits for their
            // sorting; that lifecycle cost stays inside the timed region.
            const auto flush_prepare_start = std::chrono::steady_clock::now();
            auto iterator = runner.memtable->NewCompactingIterator();
            flush_prepare_seconds += std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - flush_prepare_start)
                                         .count();
            peak_table_memory_bytes =
                std::max(peak_table_memory_bytes, runner.memtable->ApproximateMemoryUsage());
            uint64_t generation_records = 0;
            const auto flush_iterate_start = std::chrono::steady_clock::now();
            while (iterator->IsValid()) {
                const RecordRef record = iterator->Get();
                ++generation_records;
                ++flushed_records;
                flushed_logical_bytes += record.key.size() + record.value.size();
                iterator_checksum += static_cast<unsigned char>(record.key.front());
                iterator->Next();
            }
            flush_iterate_seconds += std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() - flush_iterate_start)
                                         .count();
            benchmark::DoNotOptimize(iterator_checksum);

            if (generation_records != RECORDS_PER_MEMTABLE) {
                state.SkipWithError("flush iterator returned an unexpected record count");
                return;
            }

            const auto rotate_start = std::chrono::steady_clock::now();
            iterator.reset();
            runner.Reset();
            if (generation + 1 < Generations) {
                runner.SetUp(worker_count, ActiveBlockSizeBytes, true, false);
            }
            rotate_seconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - rotate_start).count();
        }
    }

    if (point_read_misses != 0) {
        state.SkipWithError("mixed workload observed a missing point lookup");
        return;
    }

    const int64_t logical_write_bytes = static_cast<int64_t>(write_records * LOGICAL_RECORD_BYTES);
    state.counters["flushes"] = static_cast<double>(Generations * state.iterations());
    state.counters["flushed_records"] = static_cast<double>(flushed_records);
    state.counters["flushed_logical_bytes"] = static_cast<double>(flushed_logical_bytes);
    state.counters["full_active_records_avg"] =
        static_cast<double>(full_active_records) / full_state_samples;
    state.counters["full_pending_records_avg"] =
        static_cast<double>(full_pending_records) / full_state_samples;
    state.counters["full_sorted_records_avg"] =
        static_cast<double>(full_sorted_records) / full_state_samples;
    state.counters["full_pending_blocks_avg"] =
        static_cast<double>(full_pending_blocks) / full_state_samples;
    state.counters["full_sorted_blocks_avg"] =
        static_cast<double>(full_sorted_blocks) / full_state_samples;
    state.counters["peak_table_memory_bytes"] = static_cast<double>(peak_table_memory_bytes);
    state.counters["point_reads"] = static_cast<double>(point_reads);
    state.counters["phase_put_s"] = put_seconds;
    state.counters["phase_point_read_s"] = point_read_seconds;
    state.counters["phase_flush_prepare_s"] = flush_prepare_seconds;
    state.counters["phase_flush_iterate_s"] = flush_iterate_seconds;
    state.counters["phase_rotate_s"] = rotate_seconds;
    state.SetBytesProcessed(logical_write_bytes);
    state.SetItemsProcessed(static_cast<int64_t>(write_records + point_reads));
}

template <typename MemTableType>
void BM_GetAfterDrain1Worker(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(1);
    runner.memtable->PutBatch(write_batch);
    runner.memtable->WaitForBackgroundWork();
    for (auto _ : state) {
        for (const auto& key : read_keys) {
            auto value = runner.memtable->Get(key);
            benchmark::DoNotOptimize(value);
        }
    }
    state.counters["memory_bytes"] = static_cast<double>(runner.memtable->ApproximateMemoryUsage());
    runner.Reset();
    state.SetItemsProcessed(state.iterations() * read_keys.size());
}

// =================================================================================================
// FAIR BENCHMARK REGISTRATION
// =================================================================================================
template <typename MemTableType>
void RegisterBenchmarksForType(const std::string& type_name) {
    static std::vector<std::string> benchmark_names;
    constexpr int kRepetitions = 5;

    benchmark_names.push_back("BM_PutScalar1Worker<" + type_name + ">");
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_PutScalar1Worker<MemTableType>)
        ->Iterations(1)
        ->Repetitions(kRepetitions)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_PutBatchAdmission<" + type_name + ">");
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_PutBatch<MemTableType>)
        ->ArgName("workers")
        ->Arg(1)
        ->Arg(2)
        ->Arg(4)
        ->Arg(8)
        ->Arg(16)
        ->Iterations(1)
        ->Repetitions(kRepetitions)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    const std::string sustained_suffix = ">/memtable:" + std::to_string(MEMTABLE_LOGICAL_MIB) +
                                         "MiB/generations:" + std::to_string(SUSTAINED_GENERATIONS);

    benchmark_names.push_back("BM_SustainedWriteFlushBurst<" + type_name + sustained_suffix);
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(),
                                 &BM_SustainedWriteFlush<MemTableType, false>)
        ->ArgName("workers")
        ->Arg(1)
        ->Arg(4)
        ->Arg(16)
        ->Iterations(1)
        ->Repetitions(1)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_SustainedMixed80W20RFlushBurst<" + type_name + sustained_suffix);
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(),
                                 &BM_SustainedWriteFlush<MemTableType, true>)
        ->ArgName("workers")
        ->Arg(1)
        ->Arg(4)
        ->Arg(16)
        ->Iterations(1)
        ->Repetitions(1)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_SustainedWriteFlushBounded<" + type_name + sustained_suffix);
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(),
                                 &BM_SustainedWriteFlush<MemTableType, false, true>)
        ->ArgName("workers")
        ->Arg(1)
        ->Arg(4)
        ->Arg(16)
        ->Iterations(1)
        ->Repetitions(1)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_SustainedScalarWriteFlushBounded<" + type_name + sustained_suffix);
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(),
                                 &BM_SustainedWriteFlush<MemTableType, false, true, true>)
        ->ArgName("workers")
        ->Arg(1)
        ->Iterations(1)
        ->Repetitions(1)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_SustainedMixed80W20RFlushBounded<" + type_name + sustained_suffix);
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(),
                                 &BM_SustainedWriteFlush<MemTableType, true, true>)
        ->ArgName("workers")
        ->Arg(1)
        ->Arg(4)
        ->Arg(16)
        ->Iterations(1)
        ->Repetitions(1)
        ->MeasureProcessCPUTime()
        ->UseRealTime();

    benchmark_names.push_back("BM_GetAfterDrain1Worker<" + type_name + ">");
    benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_GetAfterDrain1Worker<MemTableType>)
        ->Iterations(1)
        ->Repetitions(kRepetitions)
        ->MeasureProcessCPUTime()
        ->UseRealTime();
}

// =================================================================================================
// MAIN
// =================================================================================================
int main(int argc, char** argv) {
    std::cout << "MemTable Implementation Test and Benchmark Suite\n" << "Allocator: mimalloc" << std::endl;
    FunctionalTests::RunAllFor<ColumnarMemTable>("ColumnarMemTable");
    FunctionalTests::RunAllFor<SkipListMemTable>("SkipListMemTable");
    std::cout << "\n--- Functional Test Summary ---\n"
              << "TOTAL PASSED: " << FunctionalTests::total_tests_passed
              << ", TOTAL FAILED: " << FunctionalTests::total_tests_failed << std::endl;
    if (FunctionalTests::total_tests_failed > 0) {
        std::cerr << "\n!!! SKIPPING PERFORMANCE TESTS due to functional test failures. !!!" << std::endl;
        return 1;
    }
    PrepareData(NUM_OPS);
    RegisterBenchmarksForType<ColumnarMemTable>("ColumnarMemTable");
    RegisterBenchmarksForType<SkipListMemTable>("SkipListMemTable");
    std::cout << "\n--- Running Performance Benchmarks ---" << std::endl;
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    std::cout << "\n--- Releasing benchmark data ---" << std::endl;
    write_batch.clear();
    write_batch.shrink_to_fit();
    sustained_write_batches.clear();
    sustained_write_batches.shrink_to_fit();
    sustained_keys.clear();
    sustained_keys.shrink_to_fit();
    sustained_value.clear();
    sustained_value.shrink_to_fit();
    write_data.clear();
    write_data.shrink_to_fit();
    read_keys.clear();
    read_keys.shrink_to_fit();
    std::cout << "\n--- mimalloc Final Stats ---" << std::endl;
    mi_stats_print(nullptr);
    return 0;
}
