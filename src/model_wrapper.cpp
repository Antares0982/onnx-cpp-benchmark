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

#include "model_wrapper.h"
#include <thread>
#include "lockfree-threadpool/src/MemoryPool/src/MemoryPool.h"
#include "benchmarks.h"

namespace OnnxBenchmarks {
    struct Defer {
        std::function<void()> func;

        Defer(std::function<void()> func) : func(std::move(func)) {} // NOLINT(google-explicit-constructor)
        ~Defer() {
            func();
        }
    };

    inline auto &GetMemoryInfo() {
        static Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        return memoryInfo;
    }

    OnnxModel::OnnxModel() { // NOLINT(cppcoreguidelines-pro-type-member-init)
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
#ifdef CUDA_ENABLED
        Logging("Note: CUDA enabled");
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

    OnnxModel::~OnnxModel() {
        delete[] inNamePointers;
        delete[] outNamePointers;
    }

    void OnnxModel::Initialize(size_t argc, char **argv) {
        if (argc <= 1) {
            throw std::length_error("Not enough arguments");
        }


        std::chrono::high_resolution_clock::duration duration;

        {
            std::unique_ptr<BenchMark::ClockGuard> clockGuard;
            if (benchMark) {
                clockGuard = std::make_unique<BenchMark::ClockGuard>(duration);
            }

            session = std::make_unique<Ort::Session>(env, argv[1], session_options);
        }

        if (benchMark) {
            Logging("Time used to load model ", argv[1], ": ",
                    std::chrono::duration_cast<std::chrono::milliseconds>(duration).count(), "ms");
        }


        inputLen = session->GetInputCount();
        outputLen = session->GetOutputCount();
        inputNames.reserve(inputLen);
        outputNames.reserve(outputLen);
        inputDims.resize(inputLen);
        outputDims.resize(outputLen);
        inBufferLenEachDim.reserve(inputLen);
        outBufferLenEachDim.reserve(outputLen);

        {
            auto allocator = Ort::AllocatorWithDefaultOptions();

            for (size_t i = 0; i < inputLen; ++i) {
                auto inputName = session->GetInputNameAllocated(i, allocator);
                inputNames.emplace_back(inputName.get());
            }
            for (size_t i = 0; i < outputLen; ++i) {
                auto outputName = session->GetOutputNameAllocated(i, allocator);
                outputNames.emplace_back(outputName.get());
            }
        }

        for (size_t i = 0; i < inputNames.size(); i++) {
            auto inputTypeInfo = session->GetInputTypeInfo(i);
            auto tensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            auto sizeLen = tensorInfo.GetDimensionsCount();
            auto &dim = inputDims[i];
            dim.reserve(sizeLen);
            size_t bufferLen = 1;

            if (i == 0) {
                batchSupported = tensorInfo.GetShape()[0] == -1;
            } else {
                batchSupported = batchSupported && tensorInfo.GetShape()[0] == -1;
            }

            for (size_t j = 0; j < sizeLen; ++j) {
                auto val = tensorInfo.GetShape()[j];
                dim.emplace_back(val);
                if (val > 0) {
                    bufferLen *= val;
                }
            }

            inBufferLenEachDim.emplace_back(bufferLen);
        }

        for (size_t i = 0; i < outputNames.size(); i++) {
            auto outputTypeInfo = session->GetOutputTypeInfo(i);
            auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            auto sizeLen = tensorInfo.GetDimensionsCount();
            auto &dim = outputDims[i];
            dim.reserve(sizeLen);
            size_t bufferLen = 1;

            for (size_t j = 0; j < sizeLen; ++j) {
                auto val = tensorInfo.GetShape()[j];
                dim.emplace_back(val);
                if (val > 0) {
                    bufferLen *= val;
                }
            }

            outBufferLenEachDim.emplace_back(bufferLen);
        }

        _gen_name_pointer();
    }

    size_t OnnxModel::GetInputBufferSize() const {
        size_t size = 0;
        for (const auto &value: inBufferLenEachDim) {
            size += value;
        }
        return size;
    }

    size_t OnnxModel::GetOutputBufferSize() const {
        size_t size = 0;
        for (const auto &value: outBufferLenEachDim) {
            size += value;
        }
        return size;
    }

    void OnnxModel::Run(float *inBuffer, float *outBuffer, int64_t batch) {
        auto inValues = _create_in_values(inBuffer, batch);
        auto outValues = _create_out_values(outBuffer, batch);
        session->Run(Ort::RunOptions{nullptr}, inNamePointers, inValues, inputLen, outNamePointers, outValues,
                     outputLen);
        Antares::MemoryPool::Free(inValues);
        Antares::MemoryPool::Free(outValues);
    }

    void OnnxModel::RunWithOutIndex(size_t index, float *inBuffer, float *outBuffer, int64_t batch) {
        auto inValues = _create_in_values(inBuffer, batch);
        auto outValue = _create_out_value_index(index, outBuffer, batch);
        session->Run(Ort::RunOptions{nullptr}, inNamePointers, inValues, inputLen, &outNamePointers[index], &outValue,
                     1);
        Antares::MemoryPool::Free(inValues);
    }

    void OnnxModel::RunWithOutIndexes(std::vector<size_t> indexes, float *inBuffer, float *outBuffer, int64_t batch) {
        auto inValues = _create_in_values(inBuffer, batch);
        auto outValues = (Ort::Value *) Antares::MemoryPool::MallocTemp(indexes.size() * sizeof(Ort::Value),
                                                                        alignof(Ort::Value));
        for (size_t i = 0; i < indexes.size(); ++i) {
            new(outValues + i) Ort::Value(_create_out_value_index(indexes[i], outBuffer, batch));
            outBuffer += batch * outBufferLenEachDim[indexes[i]];
        }
        auto names = _get_outnames_by_indexes(indexes);
        session->Run(Ort::RunOptions{nullptr}, inNamePointers, inValues, inputLen, names, outValues,
                     indexes.size());
        Antares::MemoryPool::Free(inValues);
        Antares::MemoryPool::Free(outValues);
        Antares::MemoryPool::Free(names);
    }

    void OnnxModel::_gen_name_pointer() {
        inNamePointers = new const char *[inputLen];
        outNamePointers = new const char *[outputLen];

        for (size_t i = 0; i < inputLen; ++i) {
            inNamePointers[i] = inputNames[i].c_str();
        }
        for (size_t i = 0; i < outputLen; ++i) {
            outNamePointers[i] = outputNames[i].c_str();
        }
    }

    const char **OnnxModel::_get_outnames_by_indexes(const std::vector<size_t> &indexes) {
        if (nullptr == outNamePointers) {
            _gen_name_pointer();
        }
        const char **answer = Antares::MemoryPool::NewTempArray<const char *>(indexes.size());
        for (size_t i = 0; i < indexes.size(); ++i) {
            answer[i] = outNamePointers[indexes[i]];
        }
        return answer;
    }

    Ort::Value *OnnxModel::_create_in_values(float *inBuffer, size_t batchNum) const {
        auto valueBuffer = (Ort::Value *) Antares::MemoryPool::MallocTemp(inputLen * sizeof(Ort::Value),
                                                                          alignof(Ort::Value));
        for (size_t i = 0; i < inputLen; ++i) {
            if (batchSupported) {
                auto dimsArray = Antares::MemoryPool::NewTempArray<int64_t>(inputDims[i].size());
                dimsArray[0] = static_cast<int64_t>(batchNum);
                for (size_t j = 1; j < inputDims[i].size(); ++j) {
                    dimsArray[j] = inputDims[i][j];
                }
                new(valueBuffer + i) Ort::Value(
                        Ort::Value::CreateTensor<float>(GetMemoryInfo(), inBuffer, batchNum * inBufferLenEachDim[i],
                                                        dimsArray, inputDims[i].size()));
                Antares::MemoryPool::DeleteArray(dimsArray, inputDims[i].size());
            } else {
                new(valueBuffer + i) Ort::Value(
                        Ort::Value::CreateTensor<float>(GetMemoryInfo(), inBuffer, batchNum * inBufferLenEachDim[i],
                                                        inputDims[i].data(), inputDims[i].size()));
            }
            inBuffer += batchNum * inBufferLenEachDim[i];
        }

        return valueBuffer;
    }

    Ort::Value *OnnxModel::_create_out_values(float *outBuffer, size_t batchNum) const {
        auto valueBuffer = (Ort::Value *) Antares::MemoryPool::MallocTemp(outputLen * sizeof(Ort::Value),
                                                                          alignof(Ort::Value));
        for (size_t i = 0; i < outputLen; ++i) {
            if (batchSupported) {
                auto dimsArray = Antares::MemoryPool::NewTempArray<int64_t>(outputDims[i].size());
                dimsArray[0] = static_cast<int64_t>(batchNum);
                for (size_t j = 1; j < outputDims[i].size(); ++j) {
                    dimsArray[j] = outputDims[i][j];
                }
                new(valueBuffer + i) Ort::Value(
                        Ort::Value::CreateTensor<float>(GetMemoryInfo(), outBuffer, batchNum * outBufferLenEachDim[i],
                                                        dimsArray, outputDims[i].size()));
                Antares::MemoryPool::DeleteArray(dimsArray, outputDims[i].size());
            } else {
                new(valueBuffer + i) Ort::Value(
                        Ort::Value::CreateTensor<float>(GetMemoryInfo(), outBuffer, batchNum * outBufferLenEachDim[i],
                                                        outputDims[i].data(), outputDims[i].size()));
            }
            outBuffer += batchNum * outBufferLenEachDim[i];
        }

        return valueBuffer;
    }

    Ort::Value OnnxModel::_create_out_value_index(size_t index, float *outBuffer, size_t batchNum) const {
        if (batchSupported) {
            auto dimsArray = Antares::MemoryPool::NewTempArray<int64_t>(outputDims[index].size());
            auto size = outputDims[index].size();
            Defer defer_run([dimsArray, size] {
                Antares::MemoryPool::DeleteArray(dimsArray, size);
            });
            dimsArray[0] = static_cast<int64_t>(batchNum);
            for (size_t j = 1; j < outputDims[index].size(); ++j) {
                dimsArray[j] = outputDims[index][j];
            }
            return Ort::Value::CreateTensor<float>(GetMemoryInfo(), outBuffer, batchNum * outBufferLenEachDim[index],
                                                   dimsArray, outputDims[index].size());
        } else {
            return Ort::Value::CreateTensor<float>(GetMemoryInfo(), outBuffer, batchNum * outBufferLenEachDim[index],
                                                   outputDims[index].data(), outputDims[index].size());
        }
    }


}
