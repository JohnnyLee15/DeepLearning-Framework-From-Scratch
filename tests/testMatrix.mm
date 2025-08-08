// // test_mm.mm  – custom Metal GEMM vs. MPSMatrixMultiplication
// // ---------------------------------------------------------------------------
// // Build: clang++ -std=c++17 -ObjC++ test_mm.mm                                \
// //               -framework Metal -framework Foundation                       \
// //               -framework MetalPerformanceShaders
// // ---------------------------------------------------------------------------
// #import <Metal/Metal.h>
// #import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// #import <Foundation/Foundation.h>
// #include <mach/mach_time.h>
// #include <random>
// #include "core/gpu/GpuEngine.h"
// #include "core/tensor/Tensor.h"
// #include "core/tensor/Matrix.h"          // contains Matrix::M().mmGpu

// static double ns2ms(uint64_t t)
// {
//     static mach_timebase_info_data_t tb{0,0};
//     if(tb.denom==0) mach_timebase_info(&tb);
//     return (double)t*tb.numer/tb.denom/1e6;
// }

// /* ------------------------------------------------------------------ */
// /*  Wrap a Tensor/Matrix buffer in an MPSMatrix                       */
// // replace makeMPSMatrix helper
// static MPSMatrix *makeMPSMatrix(id<MTLDevice> dev,
//                                 id<MTLBuffer> buf,
//                                 size_t rows, size_t cols)
// {
//     MPSMatrixDescriptor *desc =
//         [MPSMatrixDescriptor matrixDescriptorWithRows:rows
//                                               columns:cols
//                                               rowBytes:cols * sizeof(float)
//                                               dataType:MPSDataTypeFloat32];

//     return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
// }


// /* ------------------------------------------------------------------ */
// static void runMM(const char* tag,
//                   size_t M, size_t K, size_t N,
//                   size_t runs = 50)
// {
//     printf("\n=== %s  (%zux%zu · %zux%zu) ===\n", tag, M,K, K,N);

//     /* ---------------- tensors ---------------- */
//     Tensor A({M,K});
//     Tensor B({K,N});
//     Tensor C({M,N});              // product matrix for custom
//     Tensor Cref({M,N});           // to hold custom result
//     Tensor Cmps({M,N});           // to hold MPS result

//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> dis(-1.f,1.f);
//     for(float &v: A.getFlat()) v = dis(gen);
//     for(float &v: B.getFlat()) v = dis(gen);

//     A.uploadToGpu(); B.uploadToGpu(); 
//     C.uploadToGpu(); Cref.uploadToGpu(); Cmps.uploadToGpu();

//     id<MTLCommandQueue> q = GpuEngine::getCmdQueue();

//     // --- warm-up custom kernel ---
//     for(int i=0;i<3;i++){
//         auto cmd=[q commandBuffer];
//         A.M().mmGpu(B, C, cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     // --- timed runs: custom ---
//     double tot=0;
//     for(size_t r=0;r<runs;r++){
//         auto cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         A.M().mmGpu(B, C, cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         tot += ms;
//     }
//     double avg = tot / runs;
//     double flops = 2.0*M*N*K;
//     printf("-- Custom GEMM --  avg %.3f ms  (%.2f GFLOP/s)\n",
//            avg, flops/(avg*1e6));

//     // Download custom result
//     C.downloadFromGpu();
//     auto &outCustom = C.getFlat();
//     // copy into Cref (just to show you could keep it separate)
//     Cref.getFlat() = outCustom;  

//     /* ===========  MPS GEMM  ============ */
//     id<MTLDevice> dev = q.device;
//     auto *mA = makeMPSMatrix(dev, A.getGpuData(), M, K);
//     auto *mB = makeMPSMatrix(dev, B.getGpuData(), K, N);
//     auto *mC = makeMPSMatrix(dev, Cmps.getGpuData(), M, N);

//     MPSMatrixMultiplication *mps =
//         [[MPSMatrixMultiplication alloc] initWithDevice:dev
//                                         transposeLeft:false
//                                        transposeRight:false
//                                                 resultRows:M
//                                              resultColumns:N
//                                         interiorColumns:K
//                                                   alpha:1.0
//                                                    beta:0.0];

//     // warm-up MPS
//     for(int i=0;i<3;i++){
//         auto cmd=[q commandBuffer];
//         [mps encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     // timed runs: MPS
//     double mpsTot=0;
//     for(size_t r=0;r<runs;r++){
//         auto cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         [mps encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         mpsTot += ms;
//     }
//     double mpsAvg = mpsTot / runs;
//     printf("-- MPS GEMM  --    avg %.3f ms  (%.2f GFLOP/s, %.2fx faster)\n",
//            mpsAvg, flops/(mpsAvg*1e6), avg/mpsAvg);

//     // Download MPS result
//     Cmps.downloadFromGpu();
//     auto &outMPS = Cmps.getFlat();

//     // --- compute sum of absolute differences ---
//     double sumAbsDiff = 0.0;
//     for (size_t idx = 0, sz = outCustom.size(); idx < sz; ++idx) {
//         sumAbsDiff += fabs(outCustom[idx] - outMPS[idx]);
//     }
//     printf("-- Sum abs difference custom vs MPS: %.6f\n", sumAbsDiff);

//     // simple checksum printout
//     printf("First 4 custom: %.3f  %.3f  %.3f  %.3f\n",
//            outCustom[0], outCustom[1], outCustom[2], outCustom[3]);
//     printf("First 4   MPS : %.3f  %.3f  %.3f  %.3f\n",
//            outMPS[0],   outMPS[1],   outMPS[2],   outMPS[3]);
// }
// /* ------------------------------------------------------------------ */
// int main()
// {
//     @autoreleasepool {
//         GpuEngine::init();

//         runMM("Small 256",   256, 256, 256);
//         runMM("Medium 1024", 1024,1024,1024,20);
//         runMM("Big 4096",4096,4096,4096,5);

//         printf("\n=== 'Weird' Shapes to Test Boundary Conditions ===\n");
//         // The shape that caused the original Conv2D backprop error
//         // A(3200, 27)^T * B(3200, 16) -> C(27, 16)
//         runMM("Conv2D Fail Case", 3200, 27, 16, 20);

//         // Dimensions that are not multiples of tile sizes
//         runMM("Small Prime", 101, 53, 31);
//         runMM("Over Tile Size", 65, 33, 129);

//         // Highly non-square ("skinny" or "fat") matrices
//         runMM("Skinny K", 4096, 7, 1024);
//         runMM("Skinny N", 4096, 1024, 5);
//         runMM("Skinny M", 13, 1024, 1024);

//         printf("\n=== Matrix-multiply benchmark complete ===\n");
//     }
//     return 0;
// }
