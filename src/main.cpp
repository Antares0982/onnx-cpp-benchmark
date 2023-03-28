#include <iostream>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include <thread>
#include <chrono>
#include "lockfree-threadpool/src/ThreadPool.h"
#include "model_wrapper.h"
#include "benchmarks.h"


int main(int argc, char *argv[]) {
    using namespace OnnxBenchmarks;
    if (argc == 1) {
        std::cerr
                << "No argument, please specify the model path, e.g. ./onnxbenchmark ./model/model.onnx"
                << std::endl;
        exit(1);
    }

    OnnxModel model;
    BenchMark benchMark(&model);

    model.Initialize(argc, argv);

    benchMark.RunBenchmark();

    return 0;
}