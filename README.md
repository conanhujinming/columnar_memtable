# Columnar MemTable: A High-Performance In-Memory Key-Value Store

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/conanhujinming/columnar_memtable)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Language](https://img.shields.io/badge/language-C%2B%2B17-purple.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)

**Columnar MemTable** is a high-performance, highly concurrent, sharded in-memory key-value store implemented in modern C++17. Engineered for write-intensive workloads, it leverages a suite of advanced techniques—including a columnar data layout, thread-local memory allocation, a lock-free-friendly hash index, and asynchronous background processing—to achieve exceptional write throughput and low latency.

In benchmarks, it significantly outperforms traditional Skip-List-based MemTable implementations. Under high-concurrency multi-core scenarios, it delivers **~3.5x higher write throughput** and **~4x better performance in mixed read/write workloads**.

## Core Features

-   🚀 **Blazing-Fast Writes**: Utilizes sharded, thread-local `ColumnarRecordArena`s, making write operations virtually lock-free and eliminating contention between threads.
-   ⚡️ **Efficient Concurrent Indexing**: Employs a `ConcurrentStringHashMap` based on linear probing and atomic operations for high-performance, non-blocking concurrent point lookups and updates.
-   🚄 **Optimized for Bulk Operations**: Features dedicated `PutBatch` and `MultiGet` APIs that are highly optimized to process bulk requests, maximizing CPU cache efficiency and instruction pipelining.
-   💎 **Columnar Storage Layout**: Stores data in a Structure-of-Arrays (SoA) format, which improves cache locality and paves the way for future SIMD optimizations for analytical queries.
-   🧠 **Asynchronous Flushing & Compaction**: When an active memory block is full, it is seamlessly switched out and handed over to a background thread for sorting, solidification, and optional compaction, minimizing impact on foreground write performance.
-   🎯 **High-Speed Parallel Sorting**: Features a built-in parallel radix sorter to rapidly sort keys during background processing.
-   🔧 **Modern C++ Design**: Heavily utilizes C++17 features like `std::string_view`, `std::atomic`, and `std::optional` to produce efficient and memory-safe code.
-   ⚙️ **Pluggable Architecture**: The design decouples key components like the memory allocator (Arena), index (HashMap), and sorter (Sorter), ensuring excellent extensibility.

## Performance Benchmarks

The following benchmarks were conducted on a Linux server with 128 logical cores. The test involved 500,000 operations with 16-byte keys and 100-byte values.

All metrics are **Operations per Second** (higher is better).

### Columnar MemTable Performance

| Benchmark (128 Threads)       | Compaction Off      | Compaction On       | Gain vs. SkipList |
| :---------------------------- | :------------------ | :------------------ | :---------------- |
| **Concurrent Writes**         | **17.15 M ops/s**   | **14.96 M ops/s**   | **~3.5x**         |
| **Concurrent Reads**          | **36.25 M ops/s**   | **49.81 M ops/s**   | **~0.9x - 1.2x**  |
| **Concurrent Mixed (80% Read)** | **61.19 M ops/s**   | **60.93 M ops/s**   | **~4.0x**         |
| **Single-Thread Bulk Write**    | **52.37 M ops/s**   | **47.90 M ops/s**   | **~50x**          |
| **Single-Thread Bulk Read**     | **726 k ops/s**     | **526 k ops/s**     | **~2.1x**         |

### SkipList MemTable (Baseline for Comparison)

| Benchmark (128 Threads)       | Performance (ops/s) |
| :---------------------------- | :------------------ |
| **Concurrent Writes**         | 4.42 M ops/s        |
| **Concurrent Reads**          | 39.90 M ops/s       |
| **Concurrent Mixed (80% Read)** | 15.08 M ops/s       |
| **Single-Thread Bulk Write**    | 981 k ops/s         |
| **Single-Thread Bulk Read**     | 334 k ops/s         |

*As the results show, Columnar MemTable demonstrates a commanding performance advantage, especially in write-heavy and mixed workloads.*

## Architecture Overview

The core design of `ColumnarMemTable` partitions the data lifecycle into several stages, each optimized with the most efficient data structures and concurrency strategies.

1.  **Write Path (`FlashActiveBlock`)**:
    -   All incoming writes are first routed to a specific **Shard** based on the key's hash.
    -   Within each shard, data is written to an active `FlashActiveBlock`.
    -   The `FlashActiveBlock` consists of two main parts:
        -   `ColumnarRecordArena`: A thread-local, append-only memory allocator. Each thread writes to its own memory chunk, completely avoiding write contention.
        -   `ConcurrentStringHashMap`: A highly concurrent hash map that serves as the index for the active block, providing fast point lookups.

2.  **Sealing (`Seal`)**:
    -   When a `FlashActiveBlock` reaches a size threshold, it is **atomically** marked as "sealed," and a new, empty `FlashActiveBlock` is created to handle subsequent writes.
    -   This switch is extremely fast and protected by a `SpinLock`, making it nearly transparent to client applications.

3.  **Background Processing (`BackgroundWorker`)**:
    -   Sealed blocks are enqueued and processed by a dedicated background thread.
    -   This thread transforms the data from the `FlashActiveBlock` into a columnar `ColumnarBlock` and then uses the parallel radix sorter to produce a `SortedColumnarBlock`.
    -   The `SortedColumnarBlock` includes a sparse index and a Bloom filter to accelerate future lookups.

4.  **Compaction**:
    -   (Optional) The background thread can merge multiple `SortedColumnarBlock`s into a single, larger block with unique keys, reducing memory fragmentation and read amplification.

5.  **Read Path**:
    -   A `Get` request traverses the levels in **reverse chronological order**: `Active Block` -> `Sealed Blocks` -> `Sorted Blocks`. This ensures that the most recent version of a key is always found first.

## Getting Started

### Prerequisites

-   A C++17 compatible compiler (e.g., GCC 7+ or Clang 5+)
-   CMake (version 3.10+)
-   [Google Benchmark](https://github.com/google/benchmark) (for performance testing)
-   [xxHash](https://github.com/Cyan4973/xxHash) (included as a header)
-   [mimalloc](https://github.com/microsoft/mimalloc) (linked for benchmarks)

### Building and Running

```bash
# Clone the repository
git clone https://github.com/conanhujinming/columnar_memtable.git
cd columnar_memtable

# Create a build directory
mkdir build && cd build

# Configure and build the project
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run functional tests and benchmarks
./memtable_benchmark
```

### API Usage

Using `ColumnarMemTable` is straightforward.

```cpp
#include <iostream>
#include <memory>
#include "columnar_memtable.h"

int main() {
    // Create an instance of ColumnarMemTable
    // Args: active block size, enable compaction, sorter, number of shards
    auto memtable = ColumnarMemTable::Create(
        16 * 1024 * 1024, // 16MB per active block
        true,             // Enable compaction
        std::make_shared<ParallelRadixSorter>(),
        16                // 16 shards
    );

    // Simple puts
    memtable->Put("apple", "red");
    memtable->Put("banana", "yellow");
    memtable->Put("grape", "purple");

    // Overwrite a key
    memtable->Put("apple", "green");

    // Retrieve a value
    auto value = memtable->Get("apple");
    if (value) {
        std::cout << "The color of apple is: " << *value << std::endl; // Prints: green
    }

    // Delete a key
    memtable->Delete("banana");
    auto deleted_value = memtable->Get("banana");
    if (!deleted_value) {
        std::cout << "Banana has been deleted." << std::endl;
    }

    // Bulk write
    std::vector<std::pair<std::string_view, std::string_view>> batch = {
        {"cherry", "red"},
        {"orange", "orange"}
    };
    memtable->PutBatch(batch);

    // Wait for all background work to complete (e.g., before exiting)
    memtable->WaitForBackgroundWork();

    // Use a compacting iterator to scan the final, sorted data
    std::unique_ptr<CompactingIterator> iter = memtable->NewCompactingIterator();
    std::cout << "Final contents:" << std::endl;
    while (iter->IsValid()) {
        RecordRef record = iter->Get();
        std::cout << "  " << record.key << ": " << record.value << std::endl;
        iter->Next();
    }

    return 0;
}
```

## Contributing

Contributions are welcome! If you have any questions, suggestions, or bug reports, please feel free to submit an Issue. If you'd like to improve the code, please fork the repository and submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).