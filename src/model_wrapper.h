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

#ifndef TESTPROJECT_MODEL_WRAPPER_H
#define TESTPROJECT_MODEL_WRAPPER_H

#include "onnxruntime/onnxruntime_cxx_api.h"


namespace OnnxBenchmarks {
    class OnnxModel {
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;
        size_t inputLen = 0;
        size_t outputLen = 0;
        std::vector<std::vector<int64_t >> inputDims;
        std::vector<std::vector<int64_t >> outputDims;
        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;
        std::vector<size_t> inBufferLenEachDim;
        std::vector<size_t> outBufferLenEachDim;

        const char **inNamePointers = nullptr;
        const char **outNamePointers = nullptr;

        bool batchSupported = false;

    public:
        OnnxModel();

        ~OnnxModel();

        void Initialize(size_t argc, char **argv);

        Ort::Session &GetSession() {
            return *session;
        }

        [[nodiscard]] const Ort::Session &GetSession() const {
            return *session;
        }

        [[nodiscard]] auto GetInputNums() const {
            return inputLen;
        }

        [[nodiscard]] auto GetOutputNums() const {
            return outputLen;
        }

        [[nodiscard]] size_t GetInputBufferSize() const;

        [[nodiscard]] bool isBatchSupported() const { return batchSupported; }

        void Run(float *inBuffer, float *outBuffer, int64_t batch);

    private:
        void _gen_name_pointer();

        Ort::Value *_create_in_values(float *inBuffer, size_t batchNum) const;

        Ort::Value *_create_out_values(float *outBuffer, size_t batchNum) const;
    };
}
#endif //TESTPROJECT_MODEL_WRAPPER_H
