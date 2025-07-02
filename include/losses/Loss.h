#pragma once
#include <cstdint>
#include <vector>

class Tensor;
class Tensor;

using namespace std;

class Loss {
    public:
        // Methods
        virtual double calculateTotalLoss(const vector<double>&, const Tensor&) const = 0;    
        virtual Tensor calculateGradient(const vector<double>&, const Tensor&) const = 0;
        virtual ~Loss() = default;
        virtual double formatLoss(double) const;
        virtual uint32_t getEncoding() const = 0;

        enum Encodings : uint32_t {
            MSE,
            SoftmaxCrossEntropy
        };
};