// This must be the first include to ensure it overrides the default allocator
#include <mimalloc.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <thread>
#include <functional>
#include <iomanip>
#include <numeric>
#include <atomic>
#include <algorithm> // for std::find_if

#include "columnar_memtable.h" // Use the final correct version

// =================================================================================================
// UTILITIES
// =================================================================================================

// High-precision timer
inline std::chrono::nanoseconds now_nanos() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
}

// Generate a random string of a given length
std::string generate_random_string(size_t length) {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
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
// FUNCTIONAL TESTING FRAMEWORK
// =================================================================================================

namespace FunctionalTests {

int tests_passed = 0;
int tests_failed = 0;

void RUN_TEST(const std::function<bool()>& test_func, const std::string& test_name) {
    std::cout << "[RUNNING] " << test_name << "..." << std::flush;
    bool success = test_func();
    if (success) {
        tests_passed++;
        std::cout << "\r[  PASS ] " << test_name << std::endl;
    } else {
        tests_failed++;
        std::cout << "\r[  FAIL ] " << test_name << std::endl;
    }
}

bool TestBasicPutGetDelete() {
    ColumnarMemTable memtable;
    memtable.Put("apple", "red");
    memtable.Put("banana", "yellow");
    auto val = memtable.Get("apple");
    if (!val.has_value() || val.value() != "red") return false;
    memtable.Delete("banana");
    val = memtable.Get("banana");
    return !val.has_value();
}

bool TestOverwrite() {
    ColumnarMemTable memtable;
    memtable.Put("key1", "value1");
    memtable.Put("key1", "value2");
    auto val = memtable.Get("key1");
    return val.has_value() && val.value() == "value2";
}

bool TestBackgroundSortCorrectness() {
    ColumnarMemTable memtable(512, false); // Small threshold, no compaction
    memtable.Put("apple", "red");
    memtable.Put("banana", "yellow");
    memtable.Put("apple", "green"); // Overwrite
    memtable.Put("orange", "orange");
    memtable.Put("grape", "purple"); // This will trigger sort
    memtable.WaitForBackgroundWork();
    
    memtable.Put("mango", "yellow"); // Goes into new unsorted block
    
    auto apple = memtable.Get("apple");
    return apple.has_value() && apple.value() == "green";
}

// ** MODIFIED TEST for CompactingIterator **
// This test now verifies the correctness of the high-level CompactingIterator.
bool TestCompactingIteratorCorrectness() {
    ColumnarMemTable memtable(512, false); // Small threshold
    memtable.Put("c", "3");
    memtable.Put("a", "1");
    memtable.Put("e", "5");
    memtable.Put("b", "2");

    memtable.Put("d", "4"); // Trigger sort
    memtable.WaitForBackgroundWork();
    
    memtable.Put("c", "3_new"); // Overwrite in unsorted
    memtable.Delete("a"); // Delete in unsorted

    // Use the new, high-level iterator that handles de-duplication.
    auto iter = memtable.NewCompactingIterator();
    
    std::vector<std::pair<std::string, std::string>> results;
    while(iter->IsValid()) {
        RecordRef rec = iter->Get();
        results.emplace_back(std::string(rec.key), std::string(rec.value));
        iter->Next();
    }

    std::vector<std::pair<std::string, std::string>> expected_results = {
        {"b", "2"}, {"c", "3_new"}, {"d", "4"}, {"e", "5"}
    };
    
    if (results != expected_results) {
        // Optional: Print for debugging
        std::cout << "\n[DEBUG] Mismatch in CompactingIterator output.\n";
        std::cout << "[DEBUG]   Expected: ";
        for(const auto& p : expected_results) std::cout << "{" << p.first << ":" << p.second << "} ";
        std::cout << "\n[DEBUG]   Actual:   ";
        for(const auto& p : results) std::cout << "{" << p.first << ":" << p.second << "} ";
        std::cout << std::endl;
        return false;
    }

    return true;
}

void RunAll() {
    std::cout << "--- Running Functional Tests ---" << std::endl;
    RUN_TEST(TestBasicPutGetDelete, "Basic Put, Get, Delete");
    RUN_TEST(TestOverwrite, "Key Overwrite");
    RUN_TEST(TestBackgroundSortCorrectness, "Background Sort Correctness");
    RUN_TEST(TestCompactingIteratorCorrectness, "Compacting Iterator Correctness");

    std::cout << "\n--- Test Summary ---" << std::endl;
    std::cout << "PASSED: " << tests_passed << ", FAILED: " << tests_failed << std::endl;
    if (tests_failed > 0) {
        std::cerr << "!!! SOME FUNCTIONAL TESTS FAILED !!!" << std::endl;
    }
}

} // namespace FunctionalTests


// =================================================================================================
// PERFORMANCE TESTING FRAMEWORK
// =================================================================================================
namespace PerformanceTests {

struct BenchmarkResult { long long operations; std::chrono::nanoseconds duration; };
class Benchmark {
public:
    Benchmark(std::string name, std::function<BenchmarkResult()> func) : name_(std::move(name)), func_(std::move(func)) {}
    void Run() { std::cout << "[RUNNING] " << name_ << "..." << std::flush; result_ = func_(); std::cout << "\r"; }
    void Report() const {
        double seconds = result_.duration.count() / 1e9;
        double ops_per_sec = result_.operations / seconds;
        std::cout << std::left << std::setw(25) << name_ << ": "
                  << std::right << std::setw(12) << std::fixed << std::setprecision(0) << ops_per_sec
                  << " ops/sec (" << result_.operations << " ops in "
                  << std::fixed << std::setprecision(3) << seconds << " s)" << std::endl;
    }
private:
    std::string name_; std::function<BenchmarkResult()> func_; BenchmarkResult result_;
};

const int NUM_OPS = 1'000'000;

// Benchmarks now accept a factory function to create configured memtables
using MemTableFactory = std::function<std::unique_ptr<ColumnarMemTable>()>;

BenchmarkResult BenchWriteRandom(int num_ops, const MemTableFactory& factory) {
    auto memtable = factory();
    std::vector<std::string> keys(num_ops);
    for (int i = 0; i < num_ops; ++i) { keys[i] = generate_random_string(16); }
    auto start = now_nanos();
    for (int i = 0; i < num_ops; ++i) { memtable->Put(keys[i], "v"); }
    memtable->WaitForBackgroundWork(); // Wait for bg thread to finish to get a full time
    auto end = now_nanos();
    return {num_ops, end - start};
}

BenchmarkResult BenchReadRandom(int num_ops, const MemTableFactory& factory) {
    auto memtable = factory();
    std::vector<std::string> keys(num_ops);
    for (int i = 0; i < num_ops; ++i) {
        keys[i] = generate_random_string(16);
        memtable->Put(keys[i], "value");
    }
    memtable->WaitForBackgroundWork();
    
    std::cout << "block num after put " << memtable->GetSortedBlockNum() << std::endl;
    std::shuffle(keys.begin(), keys.end(), std::mt19937(std::random_device{}()));
    auto start = now_nanos();
    for (int i = 0; i < num_ops; ++i) { (void)memtable->Get(keys[i]); }
    auto end = now_nanos();
    return {num_ops, end - start};
}

BenchmarkResult BenchReadWriteMixed(int num_ops, int num_threads, int read_percent, const MemTableFactory& factory) {
    auto memtable = factory();
    std::atomic<bool> done = false;
    int read_threads = std::max(1, static_cast<int>(num_threads * (read_percent / 100.0)));
    int write_threads = std::max(1, num_threads - read_threads);
    int write_ops_per_thread = num_ops / write_threads;
    std::atomic<long long> read_ops = 0;
    
    // Pre-populate with some data
    for (int i = 0; i < 10000; ++i) { memtable->Put(generate_random_string(16), "v"); }
    memtable->WaitForBackgroundWork();

    std::vector<std::thread> threads;
    auto start = now_nanos();
    for (int t = 0; t < write_threads; ++t) {
        threads.emplace_back([&, write_ops_per_thread]() {
            for (int i = 0; i < write_ops_per_thread; ++i) { memtable->Put(generate_random_string(16), "v"); }
        });
    }
    for (int t = 0; t < read_threads; ++t) {
        threads.emplace_back([&]() {
            long long local_read_ops = 0;
            while (!done.load(std::memory_order_acquire)) {
                (void)memtable->Get(generate_random_string(16));
                local_read_ops++;
            }
            read_ops.fetch_add(local_read_ops);
        });
    }
    for (int t = 0; t < write_threads; ++t) { threads[t].join(); }
    done.store(true, std::memory_order_release);
    for (size_t t = write_threads; t < threads.size(); ++t) { threads[t].join(); }
    auto end = now_nanos();
    return {num_ops + read_ops.load(), end - start};
}


void RunAll() {
    std::cout << "\n--- Running Performance Tests (1M ops each) ---" << std::endl;
    std::cout << "NOTE: Compile with -O3 or Release mode for accurate results." << std::endl;
    const unsigned int thread_count = std::thread::hardware_concurrency();
    std::cout << "Using " << thread_count << " threads for concurrent benchmarks." << std::endl;

    const size_t BLOCK_SIZE_BYTES = 16 * 1024 * 32;

    for (bool compaction_enabled : {false, true}) {
        std::cout << "\n--- Testing with Compaction " << (compaction_enabled ? "ENABLED" : "DISABLED") 
                  << " (Block Size: " << BLOCK_SIZE_BYTES / 1024 << "KB) ---" << std::endl;

        auto factory = [=]() {
            return std::make_unique<ColumnarMemTable>(BLOCK_SIZE_BYTES, compaction_enabled);
        };
        
        std::vector<Benchmark> benchmarks = {
            Benchmark("Write Random",     [&](){ return BenchWriteRandom(NUM_OPS, factory); }),
            Benchmark("Read Random",      [&](){ return BenchReadRandom(NUM_OPS, factory); }),
            Benchmark("80/20 Read/Write", [&](){ return BenchReadWriteMixed(NUM_OPS/10, thread_count, 80, factory); })
        };
        
        for (auto& bench : benchmarks) {
            bench.Run();
            bench.Report();
        }
    }
}

} // namespace PerformanceTests

// =================================================================================================
// MAIN
// =================================================================================================

int main() {
    std::cout << "ColumnarMemTable Test and Benchmark Suite" << std::endl;
    std::cout << "Allocator: mimalloc" << std::endl;

    FunctionalTests::RunAll();
    
    if (FunctionalTests::tests_failed == 0) {
        PerformanceTests::RunAll();
    } else {
        std::cerr << "\nSkipping performance tests due to functional test failures." << std::endl;
    }

    std::cout << "\n--- mimalloc Final Stats ---" << std::endl;
    mi_stats_print(nullptr);
    
    return FunctionalTests::tests_failed > 0 ? 1 : 0;
}