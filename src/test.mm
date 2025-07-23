// testConv2d_benchmark.mm
// -----------------------------------------------
// Compile:  clang++ -ObjC++ testConv2d_benchmark.mm -framework Metal -framework Foundation
// -----------------------------------------------
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include "core/gpu/GpuEngine.h"
#include "core/tensor/Tensor.h"
#include <random>

static double ns2ms(uint64_t dt)
{
    static mach_timebase_info_data_t tb{0,0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)dt * tb.numer / tb.denom / 1.0e6;   // ms
}

void runBenchmark(const char* name, size_t batchSize, size_t height, size_t width, size_t inChannels, 
                 size_t outChannels, size_t kernelSize, size_t stride, size_t runs = 100)
{
    printf("\n=== %s ===\n", name);
    printf("Input: [%lu, %lu, %lu, %lu], Kernel: [%lu, %lu, %lu, %lu], Stride: %lu\n", 
           batchSize, height, width, inChannels, outChannels, kernelSize, kernelSize, inChannels, stride);
    
    @autoreleasepool
    {
        /* ---- Create tensors ------------------------------------------ */
        Tensor X({batchSize, height, width, inChannels});
        Tensor W({outChannels, kernelSize, kernelSize, inChannels});
        Tensor B({outChannels});
        
        // Calculate output dimensions
        size_t outHeight = (height - kernelSize) / stride + 1;
        size_t outWidth = (width - kernelSize) / stride + 1;
        Tensor Y({batchSize, outHeight, outWidth, outChannels});
        
        printf("Output: [%lu, %lu, %lu, %lu]\n", batchSize, outHeight, outWidth, outChannels);
        
        /* ---- Initialize with random data ----------------------------- */
        std::random_device rd;
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        // Input tensor
        auto& inputData = X.getFlat();
        inputData.resize(batchSize * height * width * inChannels);
        for (auto& val : inputData) val = dis(gen);
        
        // Weight tensor
        auto& weightData = W.getFlat();
        weightData.resize(outChannels * kernelSize * kernelSize * inChannels);
        for (auto& val : weightData) val = dis(gen);
        
        // Bias tensor
        auto& biasData = B.getFlat();
        biasData.resize(outChannels);
        for (auto& val : biasData) val = dis(gen);
        
        // Output tensor (zeros)
        auto& outputData = Y.getFlat();
        outputData.resize(batchSize * outHeight * outWidth * outChannels);
        std::fill(outputData.begin(), outputData.end(), 0.0f);
        
        /* ---- Upload to GPU ------------------------------------------- */
        X.uploadToGpu();  W.uploadToGpu();  B.uploadToGpu();  Y.uploadToGpu();
        
        /* ---- Warmup runs --------------------------------------------- */
        id<MTLCommandQueue> queue = GpuEngine::getCmdQueue();
        for (size_t i = 0; i < 5; ++i) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            X.conv2dForwardGpu(W, stride, Y, B, cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        /* ---- Benchmark runs ------------------------------------------ */
        double totalMs = 0.0;
        double minMs = 1e9, maxMs = 0.0;
        
        for (size_t run = 0; run < runs; ++run)
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            
            uint64_t t0 = mach_absolute_time();
            X.conv2dForwardGpu(W, stride, Y, B, cmd);
            [cmd commit];
            [cmd waitUntilCompleted];
            uint64_t t1 = mach_absolute_time();
            
            double ms = ns2ms(t1 - t0);
            totalMs += ms;
            minMs = std::min(minMs, ms);
            maxMs = std::max(maxMs, ms);
            
            if (run < 3 || run % 20 == 0) {
                printf("Run %3d: %.3f ms\n", run, ms);
            }
        }
        
        double avgMs = totalMs / runs;
        
        // Calculate theoretical operations
        long long ops = (long long)batchSize * outHeight * outWidth * outChannels * 
                       kernelSize * kernelSize * inChannels * 2; // 2 for MAC operation
        double gflops = ops / (avgMs * 1e6);
        
        // Calculate memory bandwidth
        long long memBytes = (long long)(batchSize * height * width * inChannels + 
                                        outChannels * kernelSize * kernelSize * inChannels +
                                        batchSize * outHeight * outWidth * outChannels) * sizeof(float);
        double gbps = memBytes / (avgMs * 1e6);
        
        printf("\nResults:\n");
        printf("  Average: %.3f ms\n", avgMs);
        printf("  Min:     %.3f ms\n", minMs);
        printf("  Max:     %.3f ms\n", maxMs);
        printf("  GFLOPS:  %.2f\n", gflops);
        printf("  GB/s:    %.2f\n", gbps);
        
        /* ---- Verify result (first few elements) --------------------- */
        Y.downloadFromGpu();
        auto& result = Y.getFlat();
        printf("First 4 output values: [%.3f, %.3f, %.3f, %.3f]\n", 
               result[0], result[1], result[2], result[3]);
    }
}

int main()
{
    @autoreleasepool
    {
        /* ---- Metal init ------------------------------------------ */
        GpuEngine::init();
        
        /* ---- Run various benchmarks ------------------------------ */
        
        // Small conv (similar to original)
        runBenchmark("Small Conv 3x3", 1, 32, 32, 16, 32, 3, 1, 50);
        
        // Medium conv
        runBenchmark("Medium Conv 3x3", 1, 64, 64, 32, 64, 3, 1, 50);
        
        // Large conv
        runBenchmark("Large Conv 3x3", 1, 128, 128, 64, 128, 3, 1, 30);
        
        // Batch processing
        runBenchmark("Batch Conv 3x3", 4, 64, 64, 32, 64, 3, 1, 30);
        
        // Different kernel sizes
        runBenchmark("5x5 Kernel", 1, 64, 64, 32, 64, 5, 1, 30);
        runBenchmark("7x7 Kernel", 1, 64, 64, 32, 64, 7, 1, 30);
        
        // Strided convolutions
        runBenchmark("Strided Conv 3x3", 1, 128, 128, 64, 128, 3, 2, 30);
        
        // Deep networks simulation
        runBenchmark("Deep Conv", 1, 56, 56, 128, 256, 3, 1, 20);
        
        printf("\n=== Benchmark Complete ===\n");
    }
    return 0;
}