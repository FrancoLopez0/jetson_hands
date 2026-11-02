// Small C ABI: independent of the Python version and its extension-module ABI.
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
    }
} logger;

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Tensor {
    std::string name;
    nvinfer1::Dims shape{};
    bool input = false;
    size_t count = 1;
    void* device = nullptr;
    float* host = nullptr;
    ~Tensor() {
        if (device) cudaFree(device);
        if (host) cudaFreeHost(host);
    }
};

struct Session {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<std::unique_ptr<Tensor>> tensors;
    Tensor* input = nullptr;
    cudaStream_t stream = nullptr;

    ~Session() {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }

    void initialize(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open engine file");
        auto length = file.tellg();
        if (length <= 0) throw std::runtime_error("Empty engine file");
        std::vector<char> bytes(static_cast<size_t>(length));
        file.seekg(0);
        if (!file.read(bytes.data(), length)) throw std::runtime_error("Cannot read engine file");
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) throw std::runtime_error("Cannot initialize TensorRT runtime");
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        if (!engine) throw std::runtime_error("Cannot deserialize engine for this TensorRT/GPU");
        context.reset(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Cannot create execution context");
        check(cudaStreamCreate(&stream));
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            auto tensor = std::make_unique<Tensor>();
            tensor->name = engine->getIOTensorName(i);
            tensor->shape = context->getTensorShape(tensor->name.c_str());
            tensor->input = engine->getTensorIOMode(tensor->name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
            if (engine->getTensorDataType(tensor->name.c_str()) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error("Engine I/O must be FP32 (internal layers may use FP16)");
            for (int j = 0; j < tensor->shape.nbDims; ++j) {
                int64_t dimension = tensor->shape.d[j];
                if (dimension <= 0 || dimension > 10000000 || tensor->count > 10000000ULL / dimension)
                    throw std::runtime_error("Only bounded, static tensor shapes are supported");
                tensor->count *= dimension;
            }
            size_t size = tensor->count * sizeof(float);
            check(cudaMalloc(&tensor->device, size));
            check(cudaMallocHost(reinterpret_cast<void**>(&tensor->host), size));
            if (!context->setTensorAddress(tensor->name.c_str(), tensor->device))
                throw std::runtime_error("Cannot bind tensor buffer");
            if (tensor->input) {
                if (input) throw std::runtime_error("Expected exactly one model input");
                input = tensor.get();
            }
            tensors.push_back(std::move(tensor));
        }
        if (!input) throw std::runtime_error("Model has no input");
    }
};
}  // namespace

extern "C" {
const char* hands_trt_error() { return last_error.c_str(); }

void* hands_trt_open(const char* path) {
    try {
        auto session = std::make_unique<Session>();
        session->initialize(path);
        return session.release();
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}

void hands_trt_close(void* handle) { delete static_cast<Session*>(handle); }
int hands_trt_count(void* handle) { return static_cast<int>(static_cast<Session*>(handle)->tensors.size()); }
const char* hands_trt_name(void* handle, int i) { return static_cast<Session*>(handle)->tensors.at(i)->name.c_str(); }
int hands_trt_input(void* handle, int i) { return static_cast<Session*>(handle)->tensors.at(i)->input; }
int hands_trt_shape(void* handle, int i, int64_t* dimensions) {
    auto& shape = static_cast<Session*>(handle)->tensors.at(i)->shape;
    for (int j = 0; j < shape.nbDims; ++j) dimensions[j] = shape.d[j];
    return shape.nbDims;
}
const float* hands_trt_output(void* handle, int i) { return static_cast<Session*>(handle)->tensors.at(i)->host; }

int hands_trt_run(void* handle, const float* source, size_t count) {
    try {
        auto& session = *static_cast<Session*>(handle);
        auto& input = *session.input;
        if (count != input.count) throw std::runtime_error("Input tensor size mismatch");
        // Stage through persistent pinned memory. No device allocation per frame.
        std::copy(source, source + count, input.host);
        check(cudaMemcpyAsync(input.device, input.host, count * sizeof(float), cudaMemcpyHostToDevice, session.stream));
        if (!session.context->enqueueV3(session.stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
        for (auto& tensor : session.tensors) {
            if (!tensor->input)
                check(cudaMemcpyAsync(tensor->host, tensor->device, tensor->count * sizeof(float),
                                      cudaMemcpyDeviceToHost, session.stream));
        }
        check(cudaStreamSynchronize(session.stream));
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
}
