#include <iostream>
#include "onnxruntime/onnxruntime_cxx_api.h"
#include <thread>
#include <chrono>
#include "lockfree-threadpool/src/ThreadPool.h"
#include "session.h"
#include "benchmarks.h"



int main(int argc, char *argv[]) {
    using namespace OnnxBenchmarks;
    if (argc == 1) {
        std::cerr
                << "No argument, please specify the size of input tensor, e.g. ./onnxbenchmark model.onnx 3 8 8 for 3*8*8 tensor"
                << std::endl;
        exit(1);
    }

    Session session;
    session.Initialize(argc, argv);

    run_benchmark(session);

    return 0;
}