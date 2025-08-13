// // DenseCpuBackProp.cpp
// #include "core/layers/Dense.h"
// #include "core/tensor/Tensor.h"
// #include "core/activations/ReLU.h"   // If you have Identity, you can swap it in.

// #include <cassert>
// #include <cstdio>
// #include <cmath>
// #include <vector>
// #include <algorithm>

// using std::vector;

// static inline bool closef(float a, float b, float tol=1e-5f){ return std::fabs(a-b) <= tol; }

// static void assertSameShape(const Tensor& a, const Tensor& b, const char* name) {
//     const auto &sa = a.getShape(), &sb = b.getShape();
//     assert(sa.size() == sb.size() && "rank mismatch");
//     for (size_t i = 0; i < sa.size(); ++i) {
//         if (sa[i] != sb[i]) {
//             std::fprintf(stderr, "Shape mismatch in %s at dim %zu: %zu vs %zu\n",
//                          name, i, sa[i], sb[i]);
//             assert(false);
//         }
//     }
// }

// static void fillConst(Tensor &t, float v) {
//     auto &f = t.getFlat();
//     std::fill(f.begin(), f.end(), v);
// }

// static void fillArange(Tensor &t, float start=0.f, float step=0.01f) {
//     auto &f = t.getFlat();
//     for (size_t i = 0; i < f.size(); ++i) f[i] = start + step * float(i);
// }

// static Tensor tileRows(const Tensor &row, size_t times) {
//     const auto &sh = row.getShape(); // {1, F}
//     assert(sh.size()==2 && sh[0]==1);
//     Tensor out({times, sh[1]});
//     auto &of = out.getFlat();
//     const auto &rf = row.getFlat();
//     for (size_t r=0; r<times; ++r)
//         std::copy(rf.begin(), rf.end(), of.begin() + r*sh[1]);
//     return out;
// }

// static void snapshot(const Tensor &t, vector<float> &buf) {
//     const auto &f = t.getFlat();
//     buf.assign(f.begin(), f.end());
// }

// static float maxAbsDiff(const vector<float> &a, const vector<float> &b) {
//     assert(a.size()==b.size());
//     float m = 0.f;
//     for (size_t i=0;i<a.size();++i) m = std::max(m, std::fabs(a[i]-b[i]));
//     return m;
// }

// // --- Tests ---

// // 1) With zero upstream grad, L2 update matches: W_new = W_old - lr * 2λ * W_old (batch-size independent)
// static void test_l2_zero_data_grad_batch_invariance() {
//     const size_t F=8, M=4;
//     const float lr=0.01f, lambda=1e-3f;

//     // Two runs: B=1 and B=5 with identical initial weights (deterministic seeds in your Dense)
//     Dense d1(M, new ReLU(), lambda);
//     Dense d2(M, new ReLU(), lambda);
//     d1.build({1, F}, /*isInference=*/false);
//     d2.build({5, F}, /*isInference=*/false);

//     // Inputs (values don’t matter because upstream grad is zero)
//     Tensor x1({1, F}); fillConst(x1, 0.5f);
//     Tensor x5({5, F}); fillConst(x5, 0.5f);

//     // Forward to populate preActivations
//     d1.forward(x1);
//     d2.forward(x5);

//     // Save initial weights/biases
//     vector<float> W1_old = d1.getWeights().getFlat();
//     vector<float> W2_old = d2.getWeights().getFlat();
//     vector<float> B1_old = d1.getBiases().getFlat();
//     vector<float> B2_old = d2.getBiases().getFlat();

//     // Upstream grad = 0
//     Tensor g1({1, M}); fillConst(g1, 0.f);
//     Tensor g5({5, M}); fillConst(g5, 0.f);

//     // Backprop once
//     d1.backprop(x1, lr, g1, /*isFirstLayer=*/false);
//     d2.backprop(x5, lr, g5, /*isFirstLayer=*/false);

//     // Check: W_new = W_old - lr * 2λ * W_old  (same for both batch sizes)
//     const auto &W1_new = d1.getWeights().getFlat();
//     const auto &W2_new = d2.getWeights().getFlat();
//     assert(W1_old.size()==W1_new.size() && W2_old.size()==W2_new.size());

//     for (size_t i=0;i<W1_old.size();++i) {
//         float expected = W1_old[i] - lr * (2.f*lambda) * W1_old[i];
//         assert(closef(W1_new[i], expected, 1e-7f));
//         float expected2 = W2_old[i] - lr * (2.f*lambda) * W2_old[i];
//         assert(closef(W2_new[i], expected2, 1e-7f));
//     }

//     // Biases unchanged (no L2 and zero data grad)
//     const auto &B1_new = d1.getBiases().getFlat();
//     const auto &B2_new = d2.getBiases().getFlat();
//     for (size_t i=0;i<B1_old.size();++i) assert(closef(B1_new[i], B1_old[i], 1e-7f));
//     for (size_t i=0;i<B2_old.size();++i) assert(closef(B2_new[i], B2_old[i], 1e-7f));

//     std::puts("✅ test_l2_zero_data_grad_batch_invariance passed.");
// }

// // 2) Data gradient batch invariance (λ=0). Duplicate the SAME sample across the batch.
// static void test_data_grad_batch_invariance_lambda0() {
//     const size_t F=7, M=3;
//     const float lr=0.02f, lambda=0.0f;

//     Dense d1(M, new ReLU(), lambda);
//     Dense d4(M, new ReLU(), lambda);
//     d1.build({1, F}, /*isInference=*/false);
//     d4.build({4, F}, /*isInference=*/false);

//     // Row input then tile it so preActivations/masks are identical across rows
//     Tensor x_row({1, F}); fillArange(x_row, 0.0f, 0.05f);
//     Tensor x1 = x_row;
//     Tensor x4 = tileRows(x_row, 4);

//     d1.forward(x1);
//     d4.forward(x4);

//     // Upstream grad = ones (same across batch)
//     Tensor g1({1, M}); fillConst(g1, 1.f);
//     Tensor g4({4, M}); fillConst(g4, 1.f);

//     // Snapshot params
//     vector<float> W1_old = d1.getWeights().getFlat();
//     vector<float> B1_old = d1.getBiases().getFlat();
//     vector<float> W4_old = d4.getWeights().getFlat();
//     vector<float> B4_old = d4.getBiases().getFlat();

//     // Backprop
//     d1.backprop(x1, lr, g1, /*isFirstLayer=*/false);
//     d4.backprop(x4, lr, g4, /*isFirstLayer=*/false);

//     // Compare deltas: they should be equal despite different B
//     const auto &W1_new = d1.getWeights().getFlat();
//     const auto &W4_new = d4.getWeights().getFlat();
//     const auto &B1_new = d1.getBiases().getFlat();
//     const auto &B4_new = d4.getBiases().getFlat();

//     assert(W1_old.size()==W4_old.size());
//     for (size_t i=0;i<W1_old.size();++i) {
//         float d1_delta = W1_new[i] - W1_old[i];
//         float d4_delta = W4_new[i] - W4_old[i];
//         assert(closef(d1_delta, d4_delta, 1e-5f));
//     }
//     assert(B1_old.size()==B4_old.size());
//     for (size_t i=0;i<B1_old.size();++i) {
//         float d1_delta = B1_new[i] - B1_old[i];
//         float d4_delta = B4_new[i] - B4_old[i];
//         assert(closef(d1_delta, d4_delta, 1e-5f));
//     }

//     std::puts("✅ test_data_grad_batch_invariance_lambda0 passed.");
// }

// // 3) dX is zero when upstream grad is zero (any λ)
// static void test_dx_zero_when_upstream_zero() {
//     const size_t N=3, F=5, M=4;
//     Dense d(M, new ReLU(), /*lambda=*/1e-3f);
//     d.build({N, F}, /*isInference=*/false);

//     Tensor x({N, F}); fillArange(x);
//     Tensor g({N, M}); fillConst(g, 0.f);

//     d.forward(x);
//     d.backprop(x, /*lr=*/0.01f, g, /*isFirstLayer=*/false);

//     const auto &dx = d.getDeltaInputs().getFlat();
//     for (float v : dx) assert(closef(v, 0.f, 1e-7f));

//     std::puts("✅ test_dx_zero_when_upstream_zero passed.");
// }

// // 4) Basic shape sanity (forward/backward)
// static void test_shapes() {
//     const size_t N=6, F=10, M=7;
//     Dense d(M, new ReLU(), /*lambda=*/0.f);
//     d.build({N, F}, /*isInference=*/false);

//     Tensor x({N, F}); fillConst(x, 0.3f);
//     d.forward(x);

//     // Upstream grad same shape as output (N, M)
//     Tensor g({N, M}); fillConst(g, 1.f);

//     d.backprop(x, /*lr=*/0.005f, g, /*isFirstLayer=*/false);

//     // dX must be (N, F)
//     assertSameShape(d.getDeltaInputs(), x, "dense dX shape");

//     std::puts("✅ test_shapes passed.");
// }

// int main() {
//     // Keep OMP threads fixed if you want perfect determinism across runs:
//     // export OMP_NUM_THREADS=4

//     test_l2_zero_data_grad_batch_invariance();
//     test_data_grad_batch_invariance_lambda0();
//     test_dx_zero_when_upstream_zero();
//     test_shapes();

//     std::puts("🎉 All Dense CPU backprop tests passed.");
//     return 0;
// }
