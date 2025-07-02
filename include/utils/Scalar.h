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

        // Methods
        void checkFitted();

    public:
        // Methods
        virtual void fit(const Tensor&);
        virtual void transform(Tensor&) = 0;
        virtual void reverseTransform(Tensor&) const = 0;

        virtual void fit(const vector<double>&);
        virtual void transform(vector<double>&) = 0;
        virtual void reverseTransform(vector<double>&) const = 0;

        virtual void resetToRaw();

        virtual ~Scalar() = default;

        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&);
        
        virtual uint32_t getEncoding() const = 0;

        // Enums
        enum Encodings : uint32_t {
            Greyscale,
            Minmax,
            None
        };
};