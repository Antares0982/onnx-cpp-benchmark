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
    using Clock = std::chrono::high_resolution_clock;

    auto &GetThreadPool() {
        static Antares::ThreadPool pool;
        return pool;
    }

    void Run_SingleThreadBenchmark(OnnxBenchmarks::OnnxModel &session){
        size_t arraySize = session.GetInputBufferSize();

        const bool isBatchSupported = session.IsBatchSupported();

        {
            auto testArray = Antares::MemoryPool::NewArray<float>(arraySize);
            for (size_t i = 0; i < arraySize; i++) {
                testArray[i] = RandomNumber<float>(10000) / 5000.f - 1.f;
            }

            auto start = Clock::now();
            for (size_t i = 0; i < 1000; i++) {
                auto output = session.Run();
            }
        }
    }

    void RunBenchmark(OnnxBenchmarks::OnnxModel &session) {
        auto &pool = GetThreadPool();

        // testing single thread
        Run_SingleThreadBenchmark(session);
    }
}


