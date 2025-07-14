#pragma once
#include <vector>
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Scalar {
    private:
        // Instance Variables
        bool fitted;

    public:
        // Methods
        virtual void fit(const Tensor&);
        virtual Tensor transform(const Tensor&) const = 0;
        virtual Tensor reverseTransform(const Tensor&) const = 0;

        virtual void fit(const vector<float>&);
        virtual vector<float> transform(const vector<float>&) const = 0;
        virtual vector<float> reverseTransform(const vector<float>&) const = 0;

        virtual void reset();

        virtual ~Scalar() = default;

        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&);
        
        virtual uint32_t getEncoding() const = 0;
        
        void checkFitted() const;

        // Enums
        enum Encodings : uint32_t {
            Greyscale,
            Minmax,
            None
        };
};