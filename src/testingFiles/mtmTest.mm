// // test_mTm.mm – Custom Metal GEMM (A^T * B) vs. MPSMatrixMultiplication
// // ---------------------------------------------------------------------------
// // Build: clang++ -std=c++17 -ObjC++ test_mTm.mm                              \
// //               -framework Metal -framework Foundation                       \
// //               -framework MetalPerformanceShaders
// //
// // Note: This test assumes your MatrixT class correctly dispatches to a method
// //       (e.g., mTmGpu) that encodes the custom 'mTm' kernel.
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
// // Test runner for Transposed Matrix Multiplication (A^T * B)
// static void runMTM(const char* tag,
//                    size_t M, size_t K, size_t N,
//                    size_t runs = 50)
// {
//     printf("\n=== %s  (A^T:%zux%zu · B:%zux%zu) ===\n", tag, M,K, K,N);

//     /* ---------------- tensors ---------------- */
//     // A is K x M, so its transpose A^T is M x K
//     Tensor A({K,M});
//     // B is K x N
//     Tensor B({K,N});
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
//     // This assumes A.M().T() returns a MatrixT that calls your mTmGpu method.
//     for(int i=0;i<3;i++){
//         auto cmd=[q commandBuffer];
//         A.M().T().mTmGpu(B, C, cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     // --- timed runs: custom ---
//     double tot=0;
//     for(size_t r=0;r<runs;r++){
//         auto cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         A.M().T().mTmGpu(B, C, cmd); // The call to your new transposed kernel
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         tot += ms;
//     }
//     double avg = tot / runs;
//     double flops = 2.0*M*N*K;
//     printf("-- Custom GEMM (A^T*B) -- avg %.3f ms  (%.2f GFLOP/s)\n",
//            avg, flops/(avg*1e6));

//     // Download custom result
//     C.downloadFromGpu();
//     auto &outCustom = C.getFlat();
//     Cref.getFlat() = outCustom;

//     /* ===========  MPS GEMM (A^T * B)  ============ */
//     id<MTLDevice> dev = q.device;
//     // For A^T, we pass the original A buffer (K x M) and tell MPS to transpose it.
//     auto *mA = makeMPSMatrix(dev, A.getGpuData(), K, M);
//     auto *mB = makeMPSMatrix(dev, B.getGpuData(), K, N);
//     auto *mC = makeMPSMatrix(dev, Cmps.getGpuData(), M, N);

//     // CRITICAL CHANGE: Set transposeLeft to 'true'
//     MPSMatrixMultiplication *mps =
//         [[MPSMatrixMultiplication alloc] initWithDevice:dev
//                                         transposeLeft:true // The key change for A^T * B
//                                        transposeRight:false
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
//     printf("-- MPS GEMM (A^T*B)  --    avg %.3f ms  (%.2f GFLOP/s, %.2fx faster)\n",
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

//         // Note: For A^T, the dimensions are (K, M). So a "Tall" A matrix
//         // (e.g., 1024x256) becomes a "Wide" A^T matrix (256x1024).
//         runMTM("Small 256",   256, 256, 256);
//         runMTM("Medium 1024", 1024,1024,1024,20);
//         runMTM("Wide A^T, Tall B",256,1024,256,20);  // A^T(256x1024) from A(1024x256) * B(1024x256)
//         runMTM("Tall A^T, Wide B",1024,256,1024,20); // A^T(1024x256) from A(256x1024) * B(256x1024)
//         runMTM("Big 4096",4096,4096,4096,5);

//         printf("\n=== Matrix-multiply (A^T*B) benchmark complete ===\n");
//     }
//     return 0;
// }
