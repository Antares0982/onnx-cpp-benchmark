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

#ifndef TESTPROJECT_BENCHMARKS_H
#define TESTPROJECT_BENCHMARKS_H

#include <iostream>

namespace OnnxBenchmarks {
    class OnnxModel;

    template<typename ...T>
    void Logging(T &&... args) {
        (std::cout << ... << args) << std::endl;
    }

    template<typename ...T>
    void Warning(T &&... args) {
        (std::cerr << ... << args) << std::endl;
    }

    /// Generate a random number in [0, a)
    template<typename T = size_t>
    T RandomNumber(size_t a) {
        thread_local unsigned long long x = 123456789, y = 362436069, z = 521288629;

        unsigned long long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return static_cast<T>(z % a);
    }

    void RunBenchmark(OnnxModel &session);
}


#endif //TESTPROJECT_BENCHMARKS_H
