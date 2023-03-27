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

#ifndef TESTPROJECT_SESSION_H
#define TESTPROJECT_SESSION_H

#include "onnxruntime/onnxruntime_cxx_api.h"


namespace OnnxBenchmarks {
    class Session {
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        int64_t size[64];
        size_t sizeLen;

    public:
        Session() { // NOLINT(cppcoreguidelines-pro-type-member-init)
            env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
#ifdef CUDA_ENABLED
            OrtCUDAProviderOptions options;
        options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(options);
#endif
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session_options.EnableCpuMemArena();
            session_options.EnableMemPattern();
            auto maxThread = std::thread::hardware_concurrency();
            session_options.SetIntraOpNumThreads(static_cast<int>(maxThread));
            session_options.SetInterOpNumThreads(static_cast<int>(maxThread));
        }

        ~Session() = default;

        void Load(const char *modelName) {
            session = std::make_unique<Ort::Session>(env, modelName, session_options);
        }

        void Initialize(size_t argc, char **argv) {
            if (argc <= 2 || argc - 2 > sizeof size / sizeof(int64_t)) {
                throw std::length_error("Unexpected length");
            }
            sizeLen = argc - 2;
            for (size_t i = 0; i < sizeLen; ++i) {
                size[i] = std::stoll(argv[i + 2]);
            }
            Load(argv[1]);
        }

        Ort::Session &GetSession() {
            return *session;
        }

        size_t GetArraySize() {
            size_t arraySize = 1;
            for (size_t i = 0; i < sizeLen; ++i) {
                arraySize *= size[i];
            }
            return arraySize;
        }
    };
}
#endif //TESTPROJECT_SESSION_H
