// This must be the first include to ensure it overrides the default allocator
#include <mimalloc.h>

// --- Standard Library Includes ---
#include <algorithm>
#include <atomic>
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
// UTILITIES (Unchanged)
// =================================================================================================
std::string generate_random_string(size_t length) {
    const char charset[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<size_t> distribution(0, max_index - 1);
    std::string random_string(length, '\0');
    for (size_t i = 0; i < length; ++i) {
        random_string[i] = charset[distribution(generator)];
    }
    return random_string;
}

// =================================================================================================
// FUNCTIONAL TESTS (Unchanged)
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
std::shared_ptr<T> create_memtable(size_t size, bool compaction) {
    if constexpr (std::is_same_v<T, SkipListMemTable>) {
        return std::shared_ptr<T>(new T(size, compaction));
    } else {
        return std::make_shared<T>(size, compaction);
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
void RunAllFor(const std::string& type_name) {
    std::cout << "\n--- Running Functional Tests for " << type_name << " ---" << std::endl;
    RUN_TEST(TestBasicPutGetDelete<MemTableType>, "Basic Put, Get, Delete");
    RUN_TEST(TestOverwrite<MemTableType>, "Key Overwrite");
    if constexpr (std::is_same_v<MemTableType, ColumnarMemTable>) {
        RUN_TEST(TestCompactingIteratorCorrectness<MemTableType>, "Compacting Iterator Correctness");
    }
}
}  // namespace FunctionalTests

// =================================================================================================
// BENCHMARKING INFRASTRUCTURE
// =================================================================================================
const int NUM_OPS = 500'000;
const size_t KEY_LEN = 16;
const size_t VAL_LEN = 100;
const size_t BLOCK_SIZE_BYTES = 16 * 1024 * 116;
const unsigned int NUM_THREADS = std::thread::hardware_concurrency();

using StringPair = std::pair<std::string, std::string>;
std::vector<StringPair> write_data;
std::vector<std::string> read_keys;

void PrepareData(int num_ops) {
    std::cout << "--- Preparing " << num_ops << " key/value pairs for benchmarks ---" << std::endl;
    write_data.reserve(num_ops);
    read_keys.reserve(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        std::string key = generate_random_string(KEY_LEN);
        write_data.push_back({key, generate_random_string(VAL_LEN)});
        read_keys.push_back(key);
    }
    std::shuffle(read_keys.begin(), read_keys.end(), std::mt19937(std::random_device{}()));
}

template <typename MemTableType>
class BenchmarkRunner {
   public:
    std::shared_ptr<MemTableType> memtable;
    void SetUp(const benchmark::State& state) {
        bool compaction = state.range(0);
        size_t memtable_size = state.threads() > 1 ? (BLOCK_SIZE_BYTES * 4) : BLOCK_SIZE_BYTES;
        if constexpr (std::is_same_v<MemTableType, SkipListMemTable>) {
            memtable = std::shared_ptr<MemTableType>(new MemTableType(memtable_size, compaction));
        } else {
            memtable = std::make_shared<MemTableType>(memtable_size, compaction);
        }
    }
    void TearDown() {
        if (memtable) memtable->WaitForBackgroundWork();
        memtable.reset();
    }
};

// =================================================================================================
// BENCHMARK IMPLEMENTATION FUNCTIONS (Single-threaded are unchanged)
// =================================================================================================

template <typename MemTableType>
void BM_ScalarWrite(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(state);
    for (auto _ : state) {
        for (const auto& pair : write_data) runner.memtable->Put(pair.first, pair.second);
    }
    runner.TearDown();
    state.SetItemsProcessed(state.iterations() * write_data.size());
}

template <typename MemTableType>
void BM_ScalarRead(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(state);
    for (const auto& pair : write_data) runner.memtable->Put(pair.first, pair.second);
    runner.memtable->WaitForBackgroundWork();
    for (auto _ : state) {
        for (const auto& key : read_keys) {
            auto val = runner.memtable->Get(key);
            benchmark::DoNotOptimize(val);
        }
    }
    runner.TearDown();
    state.SetItemsProcessed(state.iterations() * read_keys.size());
}

template <typename MemTableType>
void BM_BatchWrite(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(state);
    std::vector<std::pair<std::string_view, std::string_view>> batch_view;
    batch_view.reserve(write_data.size());
    for (const auto& pair : write_data) batch_view.emplace_back(pair.first, pair.second);
    for (auto _ : state) runner.memtable->PutBatch(batch_view);
    runner.TearDown();
    state.SetItemsProcessed(state.iterations() * write_data.size());
}

template <typename MemTableType>
void BM_BatchRead(benchmark::State& state) {
    BenchmarkRunner<MemTableType> runner;
    runner.SetUp(state);
    for (const auto& pair : write_data) runner.memtable->Put(pair.first, pair.second);
    runner.memtable->WaitForBackgroundWork();
    std::vector<std::string_view> keys_view;
    keys_view.reserve(read_keys.size());
    for (const auto& key : read_keys) keys_view.emplace_back(key);
    for (auto _ : state) {
        auto result = runner.memtable->MultiGet(keys_view);
        benchmark::DoNotOptimize(result);
    }
    runner.TearDown();
    state.SetItemsProcessed(state.iterations() * read_keys.size());
}

// --- FIX: Concurrent benchmarks now use a stateful struct to manage the flag and pointer ---
template <typename MemTableType>
struct ConcurrentState {
    std::shared_ptr<MemTableType> memtable_sptr;
    std::once_flag flag;
};

template <typename MemTableType>
void BM_ConcurrentWrite(benchmark::State& state) {
    static auto* concurrent_state = new ConcurrentState<MemTableType>();

    std::call_once(concurrent_state->flag, [&]() {
        BenchmarkRunner<MemTableType> runner;
        runner.SetUp(state);
        concurrent_state->memtable_sptr = runner.memtable;
    });

    auto* memtable = concurrent_state->memtable_sptr.get();
    size_t items_per_thread = write_data.size() / state.threads();
    size_t start = state.thread_index() * items_per_thread;
    size_t end = (state.thread_index() == state.threads() - 1) ? write_data.size() : start + items_per_thread;

    for (auto _ : state) {
        for (size_t i = start; i < end; ++i) memtable->Put(write_data[i].first, write_data[i].second);
    }

    if (state.thread_index() == 0) {
        concurrent_state->memtable_sptr->WaitForBackgroundWork();
        delete concurrent_state;
        concurrent_state = new ConcurrentState<MemTableType>();
    }
    state.SetItemsProcessed(state.iterations() * (end - start));
}

template <typename MemTableType>
void BM_ConcurrentRead(benchmark::State& state) {
    static auto* concurrent_state = new ConcurrentState<MemTableType>();

    std::call_once(concurrent_state->flag, [&]() {
        BenchmarkRunner<MemTableType> runner;
        runner.SetUp(state);
        for (const auto& pair : write_data) runner.memtable->Put(pair.first, pair.second);
        runner.memtable->WaitForBackgroundWork();
        concurrent_state->memtable_sptr = runner.memtable;
    });

    auto* memtable = concurrent_state->memtable_sptr.get();
    size_t items_per_thread = read_keys.size() / state.threads();
    size_t start = state.thread_index() * items_per_thread;
    size_t end = (state.thread_index() == state.threads() - 1) ? read_keys.size() : start + items_per_thread;

    for (auto _ : state) {
        for (size_t i = start; i < end; ++i) {
            auto val = memtable->Get(read_keys[i]);
            benchmark::DoNotOptimize(val);
        }
    }

    if (state.thread_index() == 0) {
        concurrent_state->memtable_sptr->WaitForBackgroundWork();
        delete concurrent_state;
        concurrent_state = new ConcurrentState<MemTableType>();
    }
    state.SetItemsProcessed(state.iterations() * (end - start));
}

template <typename MemTableType>
void BM_ConcurrentMixed(benchmark::State& state) {
    static auto* concurrent_state = new ConcurrentState<MemTableType>();
    int read_percent = state.range(1);

    std::call_once(concurrent_state->flag, [&]() {
        BenchmarkRunner<MemTableType> runner;
        runner.SetUp(state);
        for (size_t i = write_data.size() / 2; i < write_data.size(); ++i)
            runner.memtable->Put(write_data[i].first, write_data[i].second);
        runner.memtable->WaitForBackgroundWork();
        concurrent_state->memtable_sptr = runner.memtable;
    });

    auto* memtable = concurrent_state->memtable_sptr.get();
    unsigned int write_threads = std::max(1u, (unsigned int)state.threads() * (100 - read_percent) / 100);

    size_t items_processed = 0;
    for (auto _ : state) {
        if (static_cast<unsigned int>(state.thread_index()) < write_threads) {
            size_t write_ops_total = write_data.size() / 2;
            size_t items_per_thread = write_ops_total / write_threads;
            size_t start = state.thread_index() * items_per_thread;
            size_t end = (static_cast<unsigned int>(state.thread_index()) == write_threads - 1)
                             ? write_ops_total
                             : start + items_per_thread;
            for (size_t i = start; i < end; ++i) memtable->Put(write_data[i].first, write_data[i].second);
            items_processed = end - start;
        } else {
            size_t read_threads_count = state.threads() - write_threads;
            if (read_threads_count > 0) {
                size_t thread_read_idx = state.thread_index() - write_threads;
                size_t items_per_thread = read_keys.size() / read_threads_count;
                size_t start = thread_read_idx * items_per_thread;
                size_t end = (thread_read_idx == read_threads_count - 1) ? read_keys.size() : start + items_per_thread;
                for (size_t i = start; i < end; ++i) {
                    auto val = memtable->Get(read_keys[i]);
                    benchmark::DoNotOptimize(val);
                }
                items_processed = end - start;
            }
        }
    }

    if (state.thread_index() == 0) {
        concurrent_state->memtable_sptr->WaitForBackgroundWork();
        delete concurrent_state;
        concurrent_state = new ConcurrentState<MemTableType>();
    }
    state.SetItemsProcessed(state.iterations() * items_processed);
}

// =================================================================================================
// BENCHMARK REGISTRATION (Unchanged)
// =================================================================================================
template <typename MemTableType>
void RegisterBenchmarksForType(const std::string& type_name) {
    static std::vector<std::string> benchmark_names;
    for (int compaction_arg : {0, 1}) {
        if (std::is_same_v<MemTableType, SkipListMemTable> && compaction_arg == 1) continue;
        std::string compaction_name = (compaction_arg == 0) ? "NoCompaction" : "Compaction";
        benchmark_names.push_back("BM_ScalarWrite<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_ScalarWrite<MemTableType>)
            ->Arg(compaction_arg);
        benchmark_names.push_back("BM_ScalarRead<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_ScalarRead<MemTableType>)->Arg(compaction_arg);
        benchmark_names.push_back("BM_BatchWrite<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_BatchWrite<MemTableType>)->Arg(compaction_arg);
        benchmark_names.push_back("BM_BatchRead<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_BatchRead<MemTableType>)->Arg(compaction_arg);
        benchmark_names.push_back("BM_ConcurrentWrite<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_ConcurrentWrite<MemTableType>)
            ->Arg(compaction_arg)
            ->Threads(NUM_THREADS)
            ->UseRealTime();
        benchmark_names.push_back("BM_ConcurrentRead<" + type_name + ">/" + compaction_name);
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_ConcurrentRead<MemTableType>)
            ->Arg(compaction_arg)
            ->Threads(NUM_THREADS)
            ->UseRealTime();
        benchmark_names.push_back("BM_ConcurrentMixed<" + type_name + ">/" + compaction_name + "_80R20W");
        benchmark::RegisterBenchmark(benchmark_names.back().c_str(), &BM_ConcurrentMixed<MemTableType>)
            ->Args({compaction_arg, 80})
            ->Threads(NUM_THREADS)
            ->UseRealTime();
    }
}

// =================================================================================================
// MAIN (Unchanged)
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
    write_data.clear();
    write_data.shrink_to_fit();
    read_keys.clear();
    read_keys.shrink_to_fit();
    std::cout << "\n--- mimalloc Final Stats ---" << std::endl;
    mi_stats_print(nullptr);
    return 0;
}