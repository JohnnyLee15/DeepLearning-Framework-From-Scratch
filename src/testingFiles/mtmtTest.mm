// // test_mTmT.mm – Custom Metal GEMM (A^T * B^T) vs. MPSMatrixMultiplication
// // ---------------------------------------------------------------------------
// // Build: clang++ -std=c++17 -ObjC++ test_mTmT.mm                             \
// //               -framework Metal -framework Foundation                       \
// //               -framework MetalPerformanceShaders
// //
// // Note: This test assumes your class structure can dispatch the 'mTmT' kernel.
// // ---------------------------------------------------------------------------
// #import <Metal/Metal.h>
// #import <MetalPerformanceShaders/MetalPerformanceShaders.h>
// #import <Foundation/Foundation.h>
// #include <mach/mach_time.h>
// #include <random>
// #include "core/gpu/GpuEngine.h"
// #include "core/tensor/Tensor.h"
// #include "core/tensor/Matrix.h"
// #include "core/tensor/MatrixT.h"

// // Utility to convert mach time units to milliseconds
// static double ns2ms(uint64_t t)
// {
//     static mach_timebase_info_data_t tb{0,0};
//     if(tb.denom==0) mach_timebase_info(&tb);
//     return (double)t*tb.numer/tb.denom/1e6;
// }

// /* ------------------------------------------------------------------ */
// // Helper to wrap a Tensor/Matrix buffer in an MPSMatrix
// static MPSMatrix *makeMPSMatrix(id<MTLDevice> dev,
//                                 id<MTLBuffer> buf,
//                                 size_t rows, size_t cols)
// {
//     MPSMatrixDescriptor *desc =
//         [MPSMatrixDescriptor matrixDescriptorWithRows:rows
//                                               columns:cols
//                                              rowBytes:cols * sizeof(float)
//                                              dataType:MPSDataTypeFloat32];

//     return [[MPSMatrix alloc] initWithBuffer:buf descriptor:desc];
// }


// /* ------------------------------------------------------------------ */
// // Test runner for Transposed Matrix Multiplication (A^T * B^T)
// static void runMTMT(const char* tag,
//                     size_t M, size_t K, size_t N,
//                     size_t runs = 50)
// {
//     printf("\n=== %s  (A^T:%zux%zu · B^T:%zux%zu) ===\n", tag, M,K, K,N);

//     /* ---------------- tensors ---------------- */
//     // A is K x M, so its transpose A^T is M x K
//     Tensor A({K,M});
//     // B is N x K, so its transpose B^T is K x N
//     Tensor B({N,K});
//     // Result C is M x N
//     Tensor C({M,N});
//     Tensor Cref({M,N});           // to hold custom result
//     Tensor Cmps({M,N});           // to hold MPS result

//     // --- Initialize with random data ---
//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> dis(-1.f,1.f);
//     for(float &v: A.getFlat()) v = dis(gen);
//     for(float &v: B.getFlat()) v = dis(gen);

//     A.uploadToGpu(); B.uploadToGpu();
//     C.uploadToGpu(); Cref.uploadToGpu(); Cmps.uploadToGpu();

//     id<MTLCommandQueue> q = GpuEngine::getCmdQueue();

//     // --- warm-up custom kernel ---
//     // This assumes a call that can dispatch your new mTmT kernel.
//     // For example: A.M().T().mmTGpu(B.M().T(), C, cmd);
//     for(int i=0;i<3;i++){
//         auto cmd=[q commandBuffer];
//         A.M().T().mTmTGpu(B.M().T(), C, cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     // --- timed runs: custom ---
//     double tot=0;
//     for(size_t r=0;r<runs;r++){
//         auto cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         A.M().T().mTmTGpu(B.M().T(), C, cmd); // The call to your new transposed kernel
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         tot += ms;
//     }
//     double avg = tot / runs;
//     double flops = 2.0*M*N*K;
//     printf("-- Custom GEMM (A^T*B^T) -- avg %.3f ms  (%.2f GFLOP/s)\n",
//            avg, flops/(avg*1e6));

//     // Download custom result
//     C.downloadFromGpu();
//     auto &outCustom = C.getFlat();
//     Cref.getFlat() = outCustom;

//     /* ===========  MPS GEMM (A^T * B^T)  ============ */
//     id<MTLDevice> dev = q.device;
//     // For A^T, we pass the original A buffer (K x M) and tell MPS to transpose it.
//     auto *mA = makeMPSMatrix(dev, A.getGpuData(), K, M);
//     // For B^T, we pass the original B buffer (N x K) and tell MPS to transpose it.
//     auto *mB = makeMPSMatrix(dev, B.getGpuData(), N, K);
//     auto *mC = makeMPSMatrix(dev, Cmps.getGpuData(), M, N);

//     // CRITICAL CHANGE: Set both transposeLeft and transposeRight to 'true'
//     MPSMatrixMultiplication *mps =
//         [[MPSMatrixMultiplication alloc] initWithDevice:dev
//                                         transposeLeft:true // The key change for A^T
//                                        transposeRight:true // The key change for B^T
//                                            resultRows:M
//                                         resultColumns:N
//                                       interiorColumns:K
//                                                 alpha:1.0
//                                                  beta:0.0];

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
//     printf("-- MPS GEMM (A^T*B^T)  --    avg %.3f ms  (%.2f GFLOP/s, %.2fx faster)\n",
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

//         printf("\n=== Standard Shapes ===\n");
//         runMTMT("Small 256",   256, 256, 256);
//         runMTMT("Medium 1024", 1024,1024,1024,20);
//         runMTMT("Big 4096",4096,4096,4096,5);

//         printf("\n=== 'Weird' Shapes to Test Boundary Conditions ===\n");
//         // The shape that caused the original Conv2D backprop error
//         // A(3200, 27)^T * B(3200, 16) -> C(27, 16)
//         runMTMT("Conv2D Fail Case", 3200, 27, 16, 20);

//         // Dimensions that are not multiples of tile sizes
//         runMTMT("Small Prime", 101, 53, 31);
//         runMTMT("Over Tile Size", 65, 33, 129);

//         // Highly non-square ("skinny" or "fat") matrices
//         runMTMT("Skinny K", 4096, 7, 1024);
//         runMTMT("Skinny N", 4096, 1024, 5);
//         runMTMT("Skinny M", 13, 1024, 1024);

//         printf("\n=== Matrix-multiply (A^T*B^T) benchmark complete ===\n");
//     }
//     return 0;
// }