// // tests/TestGlobalAveragePooling2D.cpp
// #include <cassert>
// #include <cmath>
// #include <iostream>
// #include <vector>

// #include "core/layers/GlobalAveragePooling2D.h"
// #include "core/tensor/Tensor.h"

// using std::vector;

// static inline bool almostEqual(float a, float b, float eps = 1e-5f) {
//     return std::fabs(a - b) <= eps * (1.0f + std::fabs(a) + std::fabs(b));
// }

// static void fillSequential(Tensor &t, float start = 0.0f, float step = 1.0f) {
//     auto &f = t.getFlat();
//     float v = start;
//     for (auto &x : f) { x = v; v += step; }
// }

// static void expectShape(const Tensor &t, const vector<size_t> &s) {
//     auto got = t.getShape();
//     assert(got.size() == s.size());
//     for (size_t i = 0; i < s.size(); ++i) assert(got[i] == s[i]);
// }

// static void test_forward_small() {
//     std::cout << "[GAP] forward small\n";
//     const size_t N=2, H=2, W=3, C=4;

//     Tensor x({N,H,W,C});        // allocate storage
//     fillSequential(x, 1.0f);

//     GlobalAveragePooling2D gap;
//     gap.build({N,H,W,C}, /*isInference*/ false);
//     gap.forward(x);

//     const Tensor &y = gap.getOutput();
//     expectShape(y, {N, C});

//     const auto &xf = x.getFlat();
//     const auto &yf = y.getFlat();
//     const float invHW = 1.0f / float(H*W);

//     for (size_t n=0; n<N; ++n) {
//         for (size_t d=0; d<C; ++d) {
//             double acc = 0.0;
//             for (size_t r=0; r<H; ++r)
//                 for (size_t c=0; c<W; ++c) {
//                     size_t idx = ((n*H + r)*W + c)*C + d;
//                     acc += xf[idx];
//                 }
//             float expv = float(acc * invHW);
//             float got  = yf[n*C + d];
//             assert(almostEqual(got, expv));
//         }
//     }
// }

// static void test_backward_small() {
//     std::cout << "[GAP] backward small\n";
//     const size_t N=2, H=3, W=2, C=3;

//     Tensor x({N,H,W,C});         // allocate storage
//     fillSequential(x, 0.5f);

//     GlobalAveragePooling2D gap;
//     gap.build({N,H,W,C}, /*isInference*/ false);
//     gap.forward(x);

//     Tensor g({N,C});             // upstream grad (N, C)
//     auto &gf = g.getFlat();
//     gf = {1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f};

//     gap.backprop(x, 0.0f, g, /*isFirstLayer*/ false);
//     const Tensor &dx = gap.getOutputGradient();

//     expectShape(dx, {N,H,W,C});
//     const auto &dxf = dx.getFlat();
//     const float invHW = 1.0f / float(H*W);

//     for (size_t n=0; n<N; ++n) {
//         for (size_t r=0; r<H; ++r) {
//             for (size_t c=0; c<W; ++c) {
//                 size_t base = ((n*H + r)*W + c)*C;
//                 for (size_t d=0; d<C; ++d) {
//                     float expv = gf[n*C + d] * invHW;
//                     float got  = dxf[base + d];
//                     assert(almostEqual(got, expv));
//                 }
//             }
//         }
//     }
// }

// static void test_1x1_identity() {
//     std::cout << "[GAP] 1x1 identity\n";
//     const size_t N=1, H=1, W=1, C=5;

//     Tensor x({N,H,W,C});         // allocate storage
//     auto &xf = x.getFlat();
//     for (int i=0;i<5;i++) xf[i] = float(i+1);

//     GlobalAveragePooling2D gap;
//     gap.build({N,H,W,C}, false);
//     gap.forward(x);

//     const Tensor &y = gap.getOutput();
//     expectShape(y, {N,C});
//     const auto &yf = y.getFlat();
//     for (int d=0; d<5; ++d) assert(almostEqual(yf[d], xf[d]));

//     Tensor g({N,C});             // allocate storage
//     auto &gf = g.getFlat();
//     for (int d=0; d<5; ++d) gf[d] = float(10+d);

//     gap.backprop(x, 0.0f, g, false);
//     const auto &dxf = gap.getOutputGradient().getFlat();
//     for (int d=0; d<5; ++d) assert(almostEqual(dxf[d], gf[d]));
// }

// static void test_multiple_calls_with_reshape() {
//     std::cout << "[GAP] multiple calls with reshape\n";
//     GlobalAveragePooling2D gap;

//     // First shape
//     {
//         const size_t N=2,H=4,W=4,C=2;
//         Tensor x({N,H,W,C});     // allocate storage
//         fillSequential(x, 0.0f);
//         gap.build({N,H,W,C}, false);
//         gap.forward(x);
//         expectShape(gap.getOutput(), {N,C});
//     }
//     // Second shape
//     {
//         const size_t N=2,H=4,W=4,C=2;
//         Tensor x({N,H,W,C});     // allocate storage
//         fillSequential(x, 0.0f);
//         gap.build({N,H,W,C}, false);
//         gap.forward(x);
//         expectShape(gap.getOutput(), {N,C});

//         Tensor g({N,C});         // allocate storage
//         fillSequential(g, 1.0f, 0.25f);
//         gap.backprop(x, 0.0f, g, false);
//         expectShape(gap.getOutputGradient(), {N,H,W,C});
//     }
// }

// int main() {
//     test_forward_small();
//     test_backward_small();
//     test_1x1_identity();
//     test_multiple_calls_with_reshape();
//     std::cout << "All GlobalAveragePooling2D tests passed\n";
//     return 0;
// }
