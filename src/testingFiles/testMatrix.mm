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
//     Tensor C({M,N});              // product matrix as tensor

//     std::mt19937 gen(42);
//     std::uniform_real_distribution<float> dis(-1.f,1.f);
//     for(float &v: A.getFlat()) v = dis(gen);
//     for(float &v: B.getFlat()) v = dis(gen);

//     A.uploadToGpu(); B.uploadToGpu(); C.uploadToGpu();

//     id<MTLCommandQueue> q = GpuEngine::getCmdQueue();

//     /* -------- warm-up custom kernel ---------- */
//     for(int i=0;i<3;i++){
//         id<MTLCommandBuffer> cmd=[q commandBuffer];
//         A.M().mmGpu(B,C,cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     /* -------- timed runs: custom ------------- */
//     double tot=0,mn=1e9,mx=0;
//     for(size_t r=0;r<runs;r++){
//         id<MTLCommandBuffer> cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         A.M().mmGpu(B,C,cmd);
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         tot+=ms; mn=fmin(mn,ms); mx=fmax(mx,ms);
//         if(r<3||r%20==0) printf("Run %3zu: %.3f ms\n",r,ms);
//     }
//     double avg=tot/runs;

//     double flops = 2.0*M*N*K;
//     printf("-- Custom GEMM --  avg %.3f ms  (%.2f GFLOP/s)\n",
//            avg, flops/(avg*1e6));

//     /* ===========  MPS GEMM  ============ */
//     id<MTLDevice> dev = q.device;
//     auto *mA = makeMPSMatrix(dev,A.getGpuData(),M,K);
//     auto *mB = makeMPSMatrix(dev,B.getGpuData(),K,N);
//     auto *mC = makeMPSMatrix(dev,C.getGpuData(),M,N);

//     MPSMatrixMultiplication *mps =
//         [[MPSMatrixMultiplication alloc] initWithDevice:dev
//                                         transposeLeft:false
//                                        transposeRight:false
//                                                 resultRows:M
//                                              resultColumns:N
//                                         interiorColumns:K
//                                                   alpha:1.0
//                                                    beta:0.0];

//     // warm-up
//     for(int i=0;i<3;i++){
//         id<MTLCommandBuffer> cmd=[q commandBuffer];
//         [mps encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
//         [cmd commit]; [cmd waitUntilCompleted];
//     }

//     double mpsTot=0,mpsMn=1e9,mpsMx=0;
//     for(size_t r=0;r<runs;r++){
//         id<MTLCommandBuffer> cmd=[q commandBuffer];
//         uint64_t t0=mach_absolute_time();
//         [mps encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
//         [cmd commit]; [cmd waitUntilCompleted];
//         double ms=ns2ms(mach_absolute_time()-t0);
//         mpsTot+=ms; mpsMn=fmin(mpsMn,ms); mpsMx=fmax(mpsMx,ms);
//     }
//     double mpsAvg=mpsTot/runs;
//     printf("-- MPS GEMM  --    avg %.3f ms  (%.2f GFLOP/s, %.2fx faster)\n",
//            mpsAvg, flops/(mpsAvg*1e6), avg/mpsAvg);

//     /* ----- simple checksum --------------- */
//     C.downloadFromGpu();
//     auto &out=C.getFlat();
//     printf("First 4 results: %.3f  %.3f  %.3f  %.3f\n",
//            out[0],out[1],out[2],out[3]);
// }

// /* ------------------------------------------------------------------ */
// int main()
// {
//     @autoreleasepool {
//         GpuEngine::init();

//         runMM("Small 256",   256, 256, 256);
//         runMM("Medium 1024", 1024,1024,1024,20);
//         runMM("Tall  1024x256·256x1024",1024,256,1024,20);
//         runMM("Wide  256x1024·1024x256",256,1024,256,20);

//         printf("\n=== Matrix-multiply benchmark complete ===\n");
//     }
//     return 0;
// }
