# Columnar MemTable: A High-Performance In-Memory Key-Value Store

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/conanhujinming/columnar_memtable)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Language](https://img.shields.io/badge/language-C%2B%2B17-purple.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)

**Columnar MemTable** is a high-performance, highly concurrent, sharded in-memory key-value store implemented in modern C++17. Engineered for write-intensive workloads, it leverages a suite of advanced techniques—including a columnar data layout, thread-local memory allocation, a lock-free-friendly hash index, and asynchronous background processing—to achieve exceptional write throughput and low latency.

In the sustained lifecycle benchmark described below, Columnar MemTable reaches 4.40x, 3.03x, and 1.54x the logical write throughput of the optimized SkipList baseline with 1, 4, and 16 workers respectively. In the corrected 80%-write/20%-point-read workload it reaches 2.36x, 2.33x, and 1.47x the SkipList throughput. These measurements include bounded background sorting and a complete sorted-iterator traversal for every 1 GiB logical memtable.

## Core Features

-   🚀 **Blazing-Fast Writes**: Utilizes sharded, thread-local `ColumnarRecordArena`s, making write operations virtually lock-free and eliminating contention between threads.
-   ⚡️ **Efficient Concurrent Indexing**: Employs a `ConcurrentStringHashMap` based on linear probing and atomic operations for high-performance, non-blocking concurrent point lookups and updates.
-   🚄 **Optimized for Bulk Operations**: Features dedicated `PutBatch` and `MultiGet` APIs that are highly optimized to process bulk requests, maximizing CPU cache efficiency and instruction pipelining.
-   💎 **Columnar Storage Layout**: Stores data in a Structure-of-Arrays (SoA) format, which improves cache locality and paves the way for future SIMD optimizations for analytical queries.
-   🧠 **Asynchronous Flushing & Compaction**: When an active memory block is full, it is seamlessly switched out and handed over to a background thread for sorting, solidification, and optional compaction, minimizing impact on foreground write performance.
-   🎯 **Parallel Background Sorting**: Reuses the same bounded worker pool to sort independent sealed shard blocks concurrently.
-   🔧 **Modern C++ Design**: Heavily utilizes C++17 features like `std::string_view`, `std::atomic`, and `std::optional` to produce efficient and memory-safe code.
-   ⚙️ **Pluggable Architecture**: The design decouples key components like the memory allocator (Arena), index (HashMap), and sorter (Sorter), ensuring excellent extensibility.

## Performance Benchmarks

The old 500,000-operation admission-only results have been removed. They reused growing memtables, gave the two implementations different concurrency, and could leave Columnar sorting outside the measured interval. The results below use the same data, CPU budget, allocator, worker pool policy, and full memtable lifecycle for both implementations.

### Methodology

- Date and host: 2026-08-16, dual-socket AMD EPYC 7513, 128 logical CPUs; Linux CPU frequency scaling was enabled.
- Build: CMake `Release`, C++17, mimalloc, Google Benchmark process CPU time plus real wall time.
- CPU isolation: each worker configuration is run separately with `taskset -c 0`, `taskset -c 0-3`, or `taskset -c 0-15`. These are physical cores on the same NUMA node. The affinity applies to the entire process, so foreground work, the bounded pool, and background sorting cannot consume CPUs outside the stated budget.
- Records: unique randomized 16-byte binary keys and one 100-byte value, or 116 logical bytes per record. Randomized keys avoid giving SkipList a sorted-insertion advantage.
- Lifecycle: 9,256,395 records per logical memtable (1 GiB minus 4 bytes), 20 fresh memtables, 185,127,900 total records, and 20 GiB minus 80 bytes of logical writes.
- Columnar layout: 16 shards with a 4 MiB active-block limit per shard, so aggregate active capacity is about 64 MiB. Active, sealed, and sorted records together form the 1 GiB logical memtable.
- Batched submission: 65,536 records per `PutBatch`. Both `PutBatch` and `MultiGet` reuse the implementation's same bounded worker pool. Columnar sorts independent sealed shard blocks through that pool using `StdSorter`; it does not create an additional uncounted sorting pool.
- Bounded debt: after each write batch, writers block only if Columnar has more than 16 pending sealed blocks (about 64 MiB). The benchmark does not drain all background work after every batch.
- Simulated flush: when the logical memtable reaches 1 GiB, `NewCompactingIterator()` seals the remainder and makes a globally sorted view, then the benchmark traverses and validates every record. Iterator preparation and traversal both count toward wall throughput. The final sorted stream is consumed by one thread for both implementations, matching the usual one-memtable-to-one-SST flush model.
- Throughput is `logical key bytes + logical value bytes` divided by total wall time. It excludes allocator and index metadata bytes, but the same definition is used on both sides.

### Sustained `PutBatch` + Flush

| Workers | Columnar | SkipList | Columnar / SkipList | Columnar wall | SkipList wall |
|---:|---:|---:|---:|---:|---:|
| 1 | 280.90 MiB/s | 63.91 MiB/s | 4.40x | 72.91 s | 320.46 s |
| 4 | 447.71 MiB/s | 147.92 MiB/s | 3.03x | 45.74 s | 138.45 s |
| 16 | 486.60 MiB/s | 315.36 MiB/s | 1.54x | 42.09 s | 64.94 s |

The complete phase breakdown explains the scaling limit:

| Workers | Implementation | Put + backlog control | Flush preparation | Flush traversal | Process CPU | Peak table memory |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | Columnar | 39.25 s | 1.76 s | 31.72 s | 72.79 s | 1.55 GB |
| 1 | SkipList | 298.49 s | <0.01 s | 21.93 s | 319.91 s | 1.79 GB |
| 4 | Columnar | 13.02 s | 0.59 s | 31.89 s | 77.95 s | 1.54 GB |
| 4 | SkipList | 115.29 s | <0.01 s | 23.11 s | 365.85 s | 1.79 GB |
| 16 | Columnar | 8.93 s | 0.39 s | 32.48 s | 126.04 s | 1.55 GB |
| 16 | SkipList | 40.30 s | <0.01 s | 24.54 s | 543.40 s | 1.79 GB |

Columnar's foreground/background write phase scales from 39.25 seconds at one worker to 8.93 seconds at 16 workers. Total throughput then flattens because the final 20 globally ordered iterator traversals remain single-threaded. SkipList makes flush traversal cheaper by maintaining global order during every write, but its concurrent insertion consumes substantially more total CPU.

At the full-memtable samples, Columnar averaged 11.85, 8.00, and 8.55 pending sealed blocks for 1, 4, and 16 workers. The result is therefore not produced by allowing sorting debt to grow without bound or by timing writes only into active blocks.

### Single-Thread Scalar `Put` + Flush

This uses exactly the same 20 GiB lifecycle but calls `Put` once per record instead of `PutBatch`:

| Implementation | Scalar `Put` | `PutBatch` with 1 worker | Scalar wall | Scalar put phase | Scalar flush traversal |
|:---|---:|---:|---:|---:|---:|
| Columnar | 249.47 MiB/s | 280.90 MiB/s | 82.10 s | 50.06 s | 30.35 s |
| SkipList | 64.55 MiB/s | 63.91 MiB/s | 317.27 s | 295.55 s | 21.68 s |

Columnar is 3.86x faster than SkipList on scalar `Put`. Its one-worker `PutBatch` is 12.6% faster than scalar `Put` because it amortizes hashing, shard routing, and arena reservations. SkipList's two paths are effectively identical at one worker.

### Sustained 80% Write / 20% Point Read + Flush

The mixed workload uses the same 20 GiB lifecycle. After every 65,536-row `PutBatch`, it submits 16,384 successful point lookups chosen deterministically from all records inserted so far. This produces 185,127,900 writes and 46,281,975 actual lookups per run. Duplicate query keys retain separate result positions and are not silently coalesced.

With one worker, `MultiGet` executes serially. With 4 or 16 workers, both implementations execute the same query batch through their same bounded pool used by `PutBatch`; the process-wide CPU affinity still caps all foreground and background work to 1, 4, or 16 physical cores. All 46.282 million lookups succeeded in every run.

The write-bandwidth columns divide 20 GiB of logical key/value writes by total wall time, including reads and flush. Total operation throughput counts both writes and reads.

| Workers | Columnar write BW | SkipList write BW | Columnar / SkipList | Columnar total ops | SkipList total ops | Columnar wall | SkipList wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 139.57 MiB/s | 59.25 MiB/s | 2.36x | 1.577 Mops/s | 0.669 Mops/s | 146.74 s | 345.67 s |
| 4 | 296.81 MiB/s | 127.18 MiB/s | 2.33x | 3.354 Mops/s | 1.437 Mops/s | 69.00 s | 161.04 s |
| 16 | 405.28 MiB/s | 275.99 MiB/s | 1.47x | 4.579 Mops/s | 3.119 Mops/s | 50.53 s | 74.21 s |

| Workers | Implementation | Put + backlog control | `MultiGet` | Flush preparation | Flush traversal | Process CPU | Peak table memory |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | Columnar | 30.37 s | 77.74 s | 1.13 s | 36.84 s | 146.53 s | 1.53 GB |
| 1 | SkipList | 256.19 s | 68.69 s | <0.01 s | 20.62 s | 345.23 s | 1.79 GB |
| 4 | Columnar | 11.85 s | 24.74 s | 0.50 s | 31.62 s | 147.29 s | 1.54 GB |
| 4 | SkipList | 104.97 s | 32.61 s | <0.01 s | 23.29 s | 452.99 s | 1.79 GB |
| 16 | Columnar | 6.09 s | 12.12 s | 0.37 s | 31.66 s | 198.43 s | 1.54 GB |
| 16 | SkipList | 39.12 s | 10.41 s | <0.01 s | 24.58 s | 635.21 s | 1.79 GB |

Columnar's mixed throughput scales 2.13x from one to four cores and 2.90x from one to sixteen. At sixteen cores, iterator preparation plus the deliberately single-threaded traversal accounts for 63.4% of Columnar wall time, so the lifecycle result cannot scale with the parallel write/read phases indefinitely. SkipList is faster in the 16-core point-read phase and in final traversal, but Columnar's 6.09-second write phase versus 39.12 seconds for SkipList keeps its end-to-end throughput 1.47x higher while using much less total CPU.

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
    -   Sealed blocks are queued to one coordinator thread. It groups work by shard and, when more than one worker is configured, dispatches independent shard groups through the same bounded pool used by batched foreground operations.
    -   Each sealed `FlashActiveBlock` is materialized as a `ColumnarBlock` and ordered with `StdSorter` to produce a `SortedColumnarBlock`; there is no separate, uncounted sorting pool.
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
