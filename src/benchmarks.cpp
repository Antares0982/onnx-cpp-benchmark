//
// Created by antares on 3/27/23.
// MIT License
//
// Copyright (c) 2023 Antares
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "benchmarks.h"
#include "lockfree-threadpool/src/ThreadPool.h"
#include "model_wrapper.h"
#include "lockfree-threadpool/src/MemoryPool/src/MemoryPool.h"

namespace OnnxBenchmarks {
    struct Async {
        std::atomic<size_t> counter;
        std::promise<void> promise;

        void finish_one() {
            if (--counter == 0) {
                promise.set_value();
            }
        }

        void finish_n(size_t n) {
            size_t c = counter.fetch_sub(n);
            if (c == n) {
                promise.set_value();
            }
        }
    };

    using Clock = std::chrono::high_resolution_clock;

    auto DurationToMilliseconds(const Clock::duration &duration) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }

    inline auto &GetThreadPool() {
        static Antares::ThreadPool pool;
        return pool;
    }

    BenchMark::BenchMark(OnnxModel *inModel) : model(inModel) {
        model->RegisterBenchmark(this);
    }

    void BenchMark::Run_SingleThreadBenchmark() {
        size_t inArraySize = model->GetInputBufferSize();
        size_t outArraySize = model->GetOutputBufferSize();

        const bool isBatchSupported = model->IsBatchSupported();

        static constexpr size_t MaxRunRepeatTimes = 1000;

        auto testInBatch = [this, inArraySize, outArraySize](size_t batch) mutable {
            Logging("Testing batchNum = ", batch, "...");
            {
                auto testArray = Antares::MemoryPool::NewArray<float>(inArraySize * batch);
                for (size_t i = 0; i < inArraySize * batch; i++) {
                    testArray[i] = RandomNumber<float>(10000) / 5000.f - 1.f;
                }
                auto testOutArray = Antares::MemoryPool::NewArray<float>(outArraySize * batch);
                auto RunRepeatTimes = MaxRunRepeatTimes;
                while (batch * RunRepeatTimes > 10 * MaxRunRepeatTimes) {
                    RunRepeatTimes /= 2;
                }
                Clock::duration t_duration;
                {
                    ClockGuard guard(t_duration);

                    for (size_t i = 0; i < RunRepeatTimes; i++) {
                        model->Run(testArray, testOutArray, static_cast<int64_t>(batch));
                    }
                }
                auto elapsed = DurationToMilliseconds(t_duration);
                auto avgElapsed = static_cast<double>(elapsed) / static_cast<double>(RunRepeatTimes);
                auto avgEachInput = avgElapsed / static_cast<double>(batch);

                Logging("BatchNum = ", batch, " finished, repeated: ", RunRepeatTimes, " times, time elapsed: ",
                        elapsed, "ms, average time: ",
                        avgElapsed, "ms, average per input: ", avgEachInput, "ms");
            }
        };

        testInBatch(1);

        if (isBatchSupported) {
            testInBatch(2);
            testInBatch(4);
            testInBatch(8);
            testInBatch(16);
            testInBatch(32);
            testInBatch(64);
            testInBatch(128);
            testInBatch(256);
        }
    }

    void BenchMark::Run_MultiThreadBenchmark() {
        size_t inArraySize = model->GetInputBufferSize();
        size_t outArraySize = model->GetOutputBufferSize();

        const bool isBatchSupported = model->IsBatchSupported();

        static constexpr size_t RunRepeatTimes = 10;
        static const size_t MaxTotalTaskPerRun = 100 * std::thread::hardware_concurrency();

        auto testInBatch = [this, inArraySize, outArraySize](size_t batch) mutable {
            Logging("Testing batchNum = ", batch, "...");
            auto TotalTaskPerRun = MaxTotalTaskPerRun;
            while (batch * TotalTaskPerRun > 10 * MaxTotalTaskPerRun) {
                TotalTaskPerRun /= 2;
            }

            auto testArray = Antares::MemoryPool::NewArray<float>(inArraySize * batch * TotalTaskPerRun);
            for (size_t i = 0; i < inArraySize * batch * TotalTaskPerRun; i++) {
                testArray[i] = RandomNumber<float>(10000) / 5000.f - 1.f;
            }
            auto testOutArray = Antares::MemoryPool::NewArray<float>(outArraySize * batch * TotalTaskPerRun);

            Clock::duration t_duration;
            {
                ClockGuard guard(t_duration);

                for (size_t _ = 0; _ < RunRepeatTimes; _++) {
                    Async async;
                    async.counter = TotalTaskPerRun;
                    float *inArray = testArray;
                    float *outArray = testOutArray;

                    for (size_t i = 0; i < TotalTaskPerRun; i++) {
                        GetThreadPool().push_task(
                                [this, inArray, outArray, batch, &async]() {
                                    model->Run(inArray, outArray, static_cast<int64_t>(batch));
                                    async.finish_one();
                                });
                        inArray += inArraySize * batch;
                        outArray += outArraySize * batch;
                    }
                    async.promise.get_future().wait();
                }
            }

            auto elapsed = DurationToMilliseconds(t_duration);
            auto avgElapsed = static_cast<double>(elapsed) / RunRepeatTimes;
            auto avgEachTask = avgElapsed / static_cast<double>(TotalTaskPerRun);
            auto avgEachInput = avgEachTask / static_cast<double>(batch);

            Logging("BatchNum = ", batch, " finished, repeated: ", RunRepeatTimes, " times, job count: ",
                    TotalTaskPerRun, ", time elapsed: ",
                    elapsed, "ms, average time per loop: ",
                    avgElapsed, "ms, average per task: ", avgEachTask, "ms, average per input: ", avgEachInput, "ms");
        };

        testInBatch(1);

        if (isBatchSupported) {
            testInBatch(2);
            testInBatch(4);
            testInBatch(8);
            testInBatch(16);
            testInBatch(32);
            testInBatch(64);
            testInBatch(128);
            testInBatch(256);
        }
    }

    void BenchMark::PrintModelInfo() {
        Logging("Model info:");
        Logging("  Model input: ", model->GetInputNums(), " inputs: ", model->GetInputBufferSize(), " elements");
        Logging("  Model output: ", model->GetOutputNums(), " outputs: ", model->GetOutputBufferSize(), " elements");
        Logging("  Model batch supported: ", model->IsBatchSupported() ? "true" : "false");
        Logging("  Input names:");
        for (const auto &name: model->GetInputNames()) {
            Logging("    ", name);
        }
        Logging("  Output names:");
        for (const auto &name: model->GetOutputNames()) {
            Logging("    ", name);
        }
    }

    void BenchMark::WarmUp() {
        Logging("Warming up...");

        size_t inArraySize = model->GetInputBufferSize();
        size_t outArraySize = model->GetOutputBufferSize();

        auto testArray = Antares::MemoryPool::NewArray<float>(inArraySize);
        for (size_t i = 0; i < inArraySize; i++) {
            testArray[i] = RandomNumber<float>(10000) / 5000.f - 1.f;
        }
        auto testOutArray = Antares::MemoryPool::NewArray<float>(outArraySize);

        for (size_t i = 0; i < 1000; i++) {
            model->Run(testArray, testOutArray, 1);
        }

        Logging("Warm up finished");

        Antares::MemoryPool::DeleteArray(testArray, inArraySize);
        Antares::MemoryPool::DeleteArray(testOutArray, outArraySize);
    }

    void BenchMark::RunBenchmark() {
        if (model == nullptr) {
            throw std::runtime_error("Model is not initialized");
        }
        static_cast<void>(GetThreadPool());

        // testing single thread
        auto benchmark_runner = [this](auto task, const char *taskname)mutable {
            Clock::duration duration;
            {
                ClockGuard guard(duration);
                (this->*task)();
            }
            Logging(taskname, " finished, time elapsed: ", DurationToMilliseconds(duration), "ms");
        };

#define ONNX_BENCHMARK_RUN(task) benchmark_runner(&BenchMark::Run_##task, #task)

        PrintModelInfo();

        WarmUp();

        ONNX_BENCHMARK_RUN(SingleThreadBenchmark);
        ONNX_BENCHMARK_RUN(MultiThreadBenchmark);

#undef ONNX_BENCHMARK_RUN
    }
}
