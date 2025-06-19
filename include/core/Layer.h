#pragma once

class Matrix;

class Layer {
    public:
        virtual void calActivations(const Matrix&) = 0;
        virtual const Matrix getActivations() const = 0;
        virtual Matrix getOutputGradient() const = 0;
        virtual void backprop(const Matrix&, double, const Matrix&, bool) = 0;
        virtual ~Layer() = default;
};