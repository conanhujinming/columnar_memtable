#ifndef MEMTABLE_THREAD_POOL_H
#define MEMTABLE_THREAD_POOL_H

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

// A small fixed-size executor shared by both memtable implementations.  Pools
// are constructed before a benchmark starts, so thread creation is never hidden
// inside PutBatch latency.
class ThreadPool {
   public:
    explicit ThreadPool(size_t worker_count) {
        if (worker_count == 0) throw std::invalid_argument("ThreadPool needs at least one worker");
        workers_.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this] { WorkerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_available_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t worker_count() const noexcept { return workers_.size(); }

    template <typename Function>
    auto Submit(Function&& function) -> std::future<std::invoke_result_t<std::decay_t<Function>>> {
        using Result = std::invoke_result_t<std::decay_t<Function>>;
        auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Function>(function));
        std::future<Result> future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::runtime_error("cannot submit work to a stopped ThreadPool");
            tasks_.emplace([task] { (*task)(); });
        }
        work_available_.notify_one();
        return future;
    }

   private:
    void WorkerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_available_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable work_available_;
    bool stopping_ = false;
};

#endif  // MEMTABLE_THREAD_POOL_H
