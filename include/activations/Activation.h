#pragma once
#include <cstdint>
#include <vector>

class Tensor;

using namespace std;

class Activation {
    public:
        // Methods
        virtual void activate(const Tensor&, Tensor&) const = 0;
        virtual void calculateGradient(const Tensor&, Tensor&) const = 0;
        virtual Tensor initBias(size_t) const = 0;
        virtual ~Activation() = default;
        virtual bool isFused() const;
        virtual uint32_t getEncoding() const = 0;

        // Enums
        enum Encodings : uint32_t {
            Linear,
            ReLU,
            Softmax
        };
};