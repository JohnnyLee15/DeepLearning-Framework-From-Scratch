// #include <cassert>
// #include <cmath>
// #include <iostream>
// #include <random>
// #include <algorithm>
// #include <numeric>

// #include "core/layers/Conv2D.h"
// #include "core/activations/Linear.h"
// #include "core/gpu/GpuEngine.h"

// // ---------- Small utilities ----------
// static inline size_t idxNHWC(size_t n,size_t h,size_t w,size_t c,
//                              size_t N,size_t H,size_t W,size_t C){
//     (void)N;
//     return ((n*H + h)*W + w)*C + c;
// }

// static void fillDeterministic(Tensor& t, float start=0.0f){
//     auto& f = const_cast<vector<float>&>(t.getFlat());
//     for(size_t i=0;i<f.size();++i) f[i] = start + 0.001f * float(i); // tiny ramp
// }

// static void fillRandom(Tensor& t, uint32_t seed=42, float lo=-1.f, float hi=1.f){
//     auto& f = const_cast<vector<float>&>(t.getFlat());
//     std::mt19937 rng(seed);
//     std::uniform_real_distribution<float> U(lo,hi);
//     for (auto& x: f) x = U(rng);
// }

// static void setAllZeros(Tensor& t){
//     auto& f = const_cast<vector<float>&>(t.getFlat());
//     std::fill(f.begin(), f.end(), 0.f);
// }

// // derive SAME pads from desired output Hout,Wout (matches TF rule when Hout is given)
// static void pads_from_out(int H,int W,int kH,int kW,int s,int Hout,int Wout,
//                           int& pt,int& pb,int& pl,int& pr){
//     int pad_h = max((Hout-1)*s + kH - H, 0);
//     int pad_w = max((Wout-1)*s + kW - W, 0);
//     pt = pad_h/2; pb = pad_h - pt;  // bottom/right get the extra when odd
//     pl = pad_w/2; pr = pad_w - pl;
// }

// // cross-correlation reference that MATCHES the layer's output shape
// static void conv2d_naive_matchOut(const Tensor& x, const Tensor& W, const Tensor& b,
//                                   int kH,int kW,int stride,
//                                   const vector<size_t>& yShape, Tensor& y_out){
//     auto xs=x.getShape();             // [N,H,W,Cin]
//     size_t N=xs[0], H=xs[1], WW=xs[2], Cin=xs[3];
//     size_t Hout=yShape[1], Wout=yShape[2], Cout=yShape[3];

//     int pt,pb,pl,pr; pads_from_out((int)H,(int)WW,kH,kW,stride,(int)Hout,(int)Wout,pt,pb,pl,pr);

//     y_out = Tensor({N,Hout,Wout,Cout});
//     auto& Y = const_cast<vector<float>&>(y_out.getFlat());
//     fill(Y.begin(), Y.end(), 0.f);

//     const auto& X = x.getFlat();
//     const auto& F = W.getFlat();
//     const auto& B = b.getFlat();

//     auto idxNHWC=[&](size_t n,size_t h,size_t w,size_t c){
//         return ((n*H + h)*WW + w)*Cin + c;
//     };
//     auto idxY=[&](size_t n,size_t h,size_t w,size_t c){
//         return ((n*Hout + h)*Wout + w)*Cout + c;
//     };

//     for (size_t n=0;n<N;++n)
//       for (size_t oh=0; oh<Hout; ++oh)
//         for (size_t ow=0; ow<Wout; ++ow)
//           for (size_t co=0; co<Cout; ++co){
//               float acc = B.empty()? 0.f : B[co];
//               int ih0 = (int)oh*stride - pt;
//               int iw0 = (int)ow*stride - pl;
//               for (int kh=0; kh<kH; ++kh){
//                   int ih = ih0 + kh; if (ih<0 || ih>=(int)H) continue;
//                   for (int kw=0; kw<kW; ++kw){
//                       int iw = iw0 + kw; if (iw<0 || iw>=(int)WW) continue;
//                       for (size_t ci=0; ci<Cin; ++ci){
//                           size_t xi = idxNHWC(n, ih, iw, ci);
//                           size_t wi = ((co*kH + kh)*kW + kw)*Cin + ci; // [Cout,kH,kW,Cin]
//                           acc += X[xi] * F[wi];
//                       }
//                   }
//               }
//               Y[idxY(n,oh,ow,co)] = acc;
//           }
// }


// // ---- helpers to pretty-print shapes ----
// static string s(const vector<size_t>& v){
//     string r="[";
//     for (size_t i=0;i<v.size();++i){ r+=to_string(v[i]); if(i+1<v.size()) r+=","; }
//     return r+="]";
// }

// // ---- stronger assert with shape logging ----
// static void assertAllClose(const Tensor& A, const Tensor& B, float atol=1e-5f, float rtol=1e-4f){
//     if (A.getFlat().size() != B.getFlat().size()){
//         cerr << "Size mismatch: A" << s(A.getShape()) << " vs B" << s(B.getShape()) << "\n";
//         assert(false);
//     }
//     const auto& a=A.getFlat(); const auto& b=B.getFlat();
//     for (size_t i=0;i<a.size();++i){
//         float diff=fabs(a[i]-b[i]);
//         float tol=atol+rtol*max(fabs(a[i]),fabs(b[i]));
//         if (diff>tol){
//             cerr<<"Mismatch idx "<<i<<": "<<a[i]<<" vs "<<b[i]<<" (tol "<<tol<<")\n";
//             assert(false);
//         }
//     }
// }

// // Finite-difference numeric gradient for input X
// static float loss_dot(const Tensor& Y, const Tensor& U){
//     const auto& y=Y.getFlat(); const auto& u=U.getFlat();
//     float s=0.f; for(size_t i=0;i<y.size();++i) s += y[i]*u[i]; return s;
// }

// static void numericGradInput(Conv2D& layer, Tensor X, const Tensor& W, const Tensor& B,
//                              int stride, bool use_same, int kH, int kW,
//                              const Tensor& U, Tensor& dX_num, float eps=1e-3f)
// {
//     // build an equivalent layer with Linear activation and the same params
//     (void)stride; (void)use_same; (void)kH; (void)kW; // (shape handled by layer)
//     dX_num = Tensor(X.getShape());
//     auto& dx = const_cast<vector<float>&>(dX_num.getFlat());
//     std::fill(dx.begin(), dx.end(), 0.f);

//     // Snapshot params (we’ll write directly into layer’s weights/bias)
//     auto& Wflat = const_cast<vector<float>&>(const_cast<Tensor&>(layer.getWeights()).getFlat());
//     auto Wsave = Wflat;
//     auto& Bflat = const_cast<vector<float>&>(const_cast<Tensor&>(layer.getBiases()).getFlat());
//     auto Bsave = Bflat;

//     // For each input element: central difference
//     auto& x = const_cast<vector<float>&>(X.getFlat());
//     for (size_t i=0;i<x.size();++i){
//         float old = x[i];

//         x[i] = old + eps; layer.forward(X); Tensor Yp = layer.getOutput(); float Lp = loss_dot(Yp, U);
//         x[i] = old - eps; layer.forward(X); Tensor Ym = layer.getOutput(); float Lm = loss_dot(Ym, U);
//         x[i] = old;

//         dx[i] = (Lp - Lm) / (2*eps);
//     }

//     // restore params
//     Wflat = Wsave; Bflat = Bsave;
// }

// // Recover dW/dB by reading parameter delta (since backprop applies the grad scaled)
// static void runBackpropAndRecoverDWDB(Conv2D& layer, const Tensor& X, const Tensor& U,
//                                       Tensor& dW_est, Tensor& dB_est,
//                                       float lr=1.0f)
// {
//     // forward
//     layer.forward(X);
//     Tensor grad = U; // dL/dY = U

//     // Snapshot params before update
//     auto& Wflat = const_cast<vector<float>&>(const_cast<Tensor&>(layer.getWeights()).getFlat());
//     auto Wbefore = Wflat;
//     auto& Bflat = const_cast<vector<float>&>(const_cast<Tensor&>(layer.getBiases()).getFlat());
//     auto Bbefore = Bflat;

//     // backprop; learningRate determines the applied update scale
//     // scaleFactor = -lr / batch_size inside backprop (bs=1 in tests)
//     layer.backprop(X, lr, grad, /*isFirstLayer=*/false);

//     // After update, delta = Wbefore - Wafter = dW (since scaleFactor = -1)
//     dW_est = Tensor({layer.getWeights().getShape()});
//     dB_est = Tensor({layer.getBiases().getShape()});
//     auto& dWf = const_cast<vector<float>&>(dW_est.getFlat());
//     auto& dBf = const_cast<vector<float>&>(dB_est.getFlat());
//     for(size_t i=0;i<dWf.size();++i) dWf[i] = Wbefore[i] - Wflat[i];
//     for(size_t i=0;i<dBf.size();++i) dBf[i] = Bbefore[i] - Bflat[i];
// }

// // ---------- Tests ----------
// static void test_forward_same_s1(){
//     GpuEngine::disableGpu(); // force CPU path
//     size_t N=1,H=4,W=4,Cin=1,Cout=1;
//     Conv2D conv(/*Cout*/Cout, /*kH*/3, /*kW*/3, /*stride*/1, "same", new Linear());
//     conv.build({N,H,W,Cin});

//     // set deterministic params
//     auto& WW = const_cast<vector<float>&>(const_cast<Tensor&>(conv.getWeights()).getFlat());
//     auto& BB = const_cast<vector<float>&>(const_cast<Tensor&>(conv.getBiases()).getFlat());
//     std::iota(WW.begin(), WW.end(), 1.0f); // 1,2,3,...  (asymmetric kernel reveals correlation vs convolution)
//     std::fill(BB.begin(), BB.end(), 0.f);

//     Tensor X({N,H,W,Cin}); fillDeterministic(X, 0.0f);
//     conv.forward(X);
//     Tensor Y = conv.getOutput();

//     Tensor Yref;
//     conv2d_naive_matchOut(X, const_cast<Tensor&>(conv.getWeights()), const_cast<Tensor&>(conv.getBiases()),
//                       /*kH*/3,/*kW*/3,/*s*/1, /*same*/Y.getShape(), Yref);
//     assertAllClose(Y, Yref);
//     std::cout << "[OK] forward SAME s=1\n";
// }

// static void test_forward_same_s2_asym(){
//     GpuEngine::disableGpu();
//     size_t N=1,H=5,W=5,Cin=2,Cout=2;
//     Conv2D conv(Cout, /*kH*/3, /*kW*/3, /*s*/2, "same", new Linear());
//     conv.build({N,H,W,Cin});

//     fillRandom(const_cast<Tensor&>(conv.getWeights()), 123);
//     setAllZeros(const_cast<Tensor&>(conv.getBiases()));

//     Tensor X({N,H,W,Cin}); fillRandom(X, 7);
//     conv.forward(X);
//     Tensor Y = conv.getOutput();

//     Tensor Yref;
//     conv2d_naive_matchOut(X, const_cast<Tensor&>(conv.getWeights()),
//                         const_cast<Tensor&>(conv.getBiases()),
//                         3,3,2, Y.getShape(), Yref);
//     assertAllClose(Y, Yref);
//     std::cout << "[OK] forward SAME s=2 (asymmetric pad)\n";
// }

// static void test_forward_valid_s1(){
//     GpuEngine::disableGpu();
//     size_t N=2,H=6,W=7,Cin=3,Cout=4;
//     Conv2D conv(Cout, /*kH*/3, /*kW*/3, /*s*/1, "none", new Linear());
//     conv.build({N,H,W,Cin});

//     fillRandom(const_cast<Tensor&>(conv.getWeights()), 99);
//     fillRandom(const_cast<Tensor&>(conv.getBiases()), 3);

//     Tensor X({N,H,W,Cin}); fillRandom(X, 5);
//     conv.forward(X);
//     Tensor Y = conv.getOutput();

//     Tensor Yref;
//     conv2d_naive_matchOut(X, const_cast<Tensor&>(conv.getWeights()), 
//                         const_cast<Tensor&>(conv.getBiases()),
//                       3,3,1, Y.getShape(), Yref);
//     assertAllClose(Y, Yref);
//     std::cout << "[OK] forward VALID s=1\n";
// }

// // Check dX with finite differences
// static void test_grad_input_numeric(){
//     GpuEngine::disableGpu();
//     size_t N=1,H=6,W=6,Cin=2,Cout=3;
//     Conv2D conv(Cout, /*kH*/3, /*kW*/3, /*s*/2, "same", new Linear());
//     conv.build({N,H,W,Cin});

//     // deterministic params
//     fillRandom(const_cast<Tensor&>(conv.getWeights()), 11, -0.5f, 0.5f);
//     setAllZeros(const_cast<Tensor&>(conv.getBiases()));

//     Tensor X({N,H,W,Cin}); fillRandom(X, 13, -0.5f, 0.5f);
//     conv.forward(X);
//     Tensor Y = conv.getOutput();
//     Tensor U(Y.getShape()); fillRandom(U, 17, -1.f, 1.f); // upstream grad / loss weights

//     // backprop (no weight update side-effects on dX)
//     Tensor grad = U;
//     conv.backprop(X, /*lr*/0.0f, grad, /*isFirst*/false);
//     Tensor dX_anal = conv.getOutputGradient();

//     // numeric dX
//     Tensor dX_num;
//     numericGradInput(conv, X, const_cast<Tensor&>(conv.getWeights()), const_cast<Tensor&>(conv.getBiases()),
//                      /*stride*/2, /*same*/true, /*kH*/3, /*kW*/3, U, dX_num);
//     assertAllClose(dX_anal, dX_num, 2e-3f, 1e-2f);
//     std::cout << "[OK] dX numeric check\n";
// }

// // Check dW & dB via parameter delta vs finite differences
// static void test_grad_params_numeric(){
//     GpuEngine::disableGpu();
//     size_t N=1,H=6,W=6,Cin=2,Cout=2;
//     Conv2D conv(Cout, /*kH*/3, /*kW*/3, /*s*/1, "same", new Linear());
//     conv.build({N,H,W,Cin});

//     fillRandom(const_cast<Tensor&>(conv.getWeights()), 21, -0.3f, 0.3f);
//     fillRandom(const_cast<Tensor&>(conv.getBiases()),   22, -0.1f, 0.1f);
//     Tensor X({N,H,W,Cin}); fillRandom(X, 23, -0.5f, 0.5f);

//     // Make U to map loss L = sum(Y*U)
//     conv.forward(X);
//     Tensor U(conv.getOutput().getShape());
//     fillRandom(U, 24, -1.f, 1.f);

//     // Recover analytic dW,dB via parameter delta (lr=1, bs=1 => W_after = W_before - dW)
//     Tensor dW_est, dB_est;
//     runBackpropAndRecoverDWDB(conv, X, U, dW_est, dB_est, /*lr=*/1.0f);

//     // Numeric dW/dB
//     const float eps = 1e-3f;
//     Tensor dW_num(dW_est.getShape()); setAllZeros(dW_num);
//     Tensor dB_num(dB_est.getShape()); setAllZeros(dB_num);

//     auto& Wflat = const_cast<vector<float>&>(const_cast<Tensor&>(conv.getWeights()).getFlat());
//     auto Wsave = Wflat;
//     auto& Bflat = const_cast<vector<float>&>(const_cast<Tensor&>(conv.getBiases()).getFlat());
//     auto Bsave = Bflat;

//     // dW numeric
//     for (size_t i=0;i<Wflat.size();++i){
//         float old = Wflat[i];
//         Wflat[i] = old + eps; conv.forward(X); float Lp = loss_dot(conv.getOutput(), U);
//         Wflat[i] = old - eps; conv.forward(X); float Lm = loss_dot(conv.getOutput(), U);
//         Wflat[i] = old;
//         const_cast<vector<float>&>(dW_num.getFlat())[i] = (Lp - Lm)/(2*eps);
//     }
//     // dB numeric
//     for (size_t i=0;i<Bflat.size();++i){
//         float old = Bflat[i];
//         Bflat[i] = old + eps; conv.forward(X); float Lp = loss_dot(conv.getOutput(), U);
//         Bflat[i] = old - eps; conv.forward(X); float Lm = loss_dot(conv.getOutput(), U);
//         Bflat[i] = old;
//         const_cast<vector<float>&>(dB_num.getFlat())[i] = (Lp - Lm)/(2*eps);
//     }

//     // restore
//     Wflat = Wsave; Bflat = Bsave;

//     assertAllClose(dW_est, dW_num, 1e-2f, 1e-2f);
//     assertAllClose(dB_est, dB_num, 1e-2f, 1e-2f);
//     std::cout << "[OK] dW/dB numeric check\n";
// }

// int main(){
//     try{
//         test_forward_same_s1();
//         test_forward_same_s2_asym();
//         test_forward_valid_s1();
//         test_grad_input_numeric();
//         test_grad_params_numeric();
//         std::cout << "✅ All Conv2D CPU tests passed.\n";
//     }catch(...){
//         std::cerr << "❌ Conv2D CPU tests failed.\n";
//         return 1;
//     }
//     return 0;
// }
