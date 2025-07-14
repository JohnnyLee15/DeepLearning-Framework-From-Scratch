#pragma once
#include <cstdint>
#include <vector>

class Tensor;
class Tensor;

using namespace std;

class Loss {
    public:
        // Methods
        virtual float calculateTotalLoss(const vector<float>&, const Tensor&) const = 0;    
        virtual Tensor calculateGradient(const vector<float>&, const Tensor&) const = 0;
        virtual ~Loss() = default;
        virtual float formatLoss(float) const;
        virtual uint32_t getEncoding() const = 0;

        enum Encodings : uint32_t {
            MSE,
            SoftmaxCrossEntropy
        };
};