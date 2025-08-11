// #include "core/layers/Dropout.h"
// #include "core/tensor/Tensor.h"

// #include <cassert>
// #include <cmath>
// #include <cstdio>
// #include <random>
// #include <vector>
// #include <algorithm>

// using std::vector;
// static inline bool closef(float a, float b, float tol=1e-5f){ return std::fabs(a-b) <= tol; }

// static void fillUniform(Tensor &t, uint32_t seed=123, float lo=-1.f, float hi=1.f) {
//     std::mt19937 rng(seed);
//     std::uniform_real_distribution<float> dist(lo, hi);
//     for (auto &v : t.getFlat()) v = dist(rng);
// }

// static void assertSameShape(const Tensor& a, const Tensor& b, const char* name) {
//     const auto &sa = a.getShape();
//     const auto &sb = b.getShape();
//     assert(sa.size() == sb.size() && "rank mismatch");
//     for (size_t i = 0; i < sa.size(); ++i) {
//         if (sa[i] != sb[i]) {
//             std::fprintf(stderr, "Shape mismatch in %s at dim %zu: %zu vs %zu\n",
//                          name, i, sa[i], sb[i]);
//             assert(false);
//         }
//     }
// }

// static void test_eval_identity(size_t N, size_t F, float p) {
//     vector<size_t> shape = {N, F};
//     Dropout d(p);
//     d.build(shape, /*isInference=*/true);

//     Tensor x(shape);
//     fillUniform(x, 7);
//     d.forward(x);
//     const auto &y = d.getOutput().getFlat();
//     const auto &xf = x.getFlat();

//     assertSameShape(d.getOutput(), x, "eval identity shape");
//     for (size_t i = 0; i < y.size(); ++i)
//         assert(closef(y[i], xf[i], 1e-6f));
// }

// static void test_train_forward_values_and_keep(size_t N, size_t F, float p) {
//     vector<size_t> shape = {N, F};
//     Dropout d(p);
//     d.build(shape, /*isInference=*/false);

//     // Use ones so output equals the mask (makes checks easy)
//     Tensor x(shape);
//     std::fill(x.getFlat().begin(), x.getFlat().end(), 1.0f);

//     d.forward(x);
//     const auto &y = d.getOutput().getFlat();

//     const float keep = 1.0f - p;
//     const float scale = 1.0f / keep;

//     size_t kept = 0;
//     for (size_t i = 0; i < y.size(); ++i) {
//         bool ok = closef(y[i], 0.0f) || closef(y[i], scale, 1e-5f);
//         if (!ok) {
//             std::fprintf(stderr, "Value set violation at %zu: got %.6f, expected {0, %.6f}\n",
//                          i, y[i], scale);
//             assert(false);
//         }
//         if (y[i] != 0.0f) kept++;
//     }
//     float frac_kept = float(kept) / float(y.size());
//     // Allow small tolerance for stochasticity; with your fixed-seed setup this will be deterministic
//     assert(std::fabs(frac_kept - keep) < 0.02f);
// }

// static void test_backward_matches_mask(size_t N, size_t F, float p) {
//     vector<size_t> shape = {N, F};
//     Dropout d(p);
//     d.build(shape, /*isInference=*/false);

//     // x = 1 → forward output y == mask
//     Tensor x(shape);
//     std::fill(x.getFlat().begin(), x.getFlat().end(), 1.0f);
//     d.forward(x);
//     const auto &y = d.getOutput().getFlat(); // == mask when x=1

//     // Upstream grad = 1 → dx should equal mask (== y)
//     Tensor grad(shape);
//     std::fill(grad.getFlat().begin(), grad.getFlat().end(), 1.0f);

//     d.backprop(x, /*lr*/0.0f, grad, /*isFirst*/false);

//     const auto &dx = d.getOutputGradient().getFlat();
//     assertSameShape(d.getOutputGradient(), grad, "backprop dX shape");
//     for (size_t i = 0; i < dx.size(); ++i)
//         assert(closef(dx[i], y[i], 1e-5f));
// }

// static void test_determinism_two_calls_same_mask(size_t N, size_t F, float p) {
//     // With your current seeds = thread_id design, two forwards produce identical masks.
//     vector<size_t> shape = {N, F};
//     Dropout d(p);
//     d.build(shape, /*isInference=*/false);

//     Tensor x(shape);
//     std::fill(x.getFlat().begin(), x.getFlat().end(), 1.0f);

//     d.forward(x);
//     vector<float> y1 = d.getOutput().getFlat(); // copy

//     d.forward(x);
//     const auto &y2 = d.getOutput().getFlat();

//     assert(y1.size() == y2.size());
//     for (size_t i = 0; i < y1.size(); ++i)
//         assert(closef(y1[i], y2[i]));
// }

// static void test_batch_reshape(size_t maxN, size_t F, size_t runN, float p) {
//     vector<size_t> buildShape = {maxN, F};
//     Dropout d(p);
//     d.build(buildShape, /*isInference=*/false);

//     // Run with smaller batch
//     vector<size_t> runShape = {runN, F};
//     Tensor x(runShape);
//     fillUniform(x, 99);

//     d.forward(x);
//     assertSameShape(d.getOutput(), x, "batch reshape output");
//     assertSameShape(d.getOutputGradient(), x, "batch reshape dX");
// }

// int main() {
//     // Keep threads fixed for reproducibility if you like:
//     // (set in env) OMP_NUM_THREADS=4

//     // Eval path
//     test_eval_identity(64, 128, 0.3f);

//     // Train forward/backward at a few ps
//     for (float p : {0.0f, 0.3f, 0.7f, 0.9f, 1.0f}) {
//         test_train_forward_values_and_keep(128, 256, p);
//         test_backward_matches_mask(128, 256, p);
//         test_determinism_two_calls_same_mask(128, 256, p);
//         test_batch_reshape(/*maxN=*/64, /*F=*/128, /*runN=*/32, p);
//     }

//     std::puts("✅ Dropout CPU tests passed.");
//     return 0;
// }
