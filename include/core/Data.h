#pragma once
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Data {
    public:
        virtual const Tensor& getTrainFeatures() const = 0;
        virtual const Tensor& getTestFeatures() const = 0;
        virtual const vector<float>& getTrainTargets() const = 0;
        virtual const vector<float>& getTestTargets() const = 0;
        virtual size_t getNumTrainSamples() const = 0;
        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;
        virtual uint32_t getEncoding() const = 0;
        virtual ~Data() = default;

        enum Encodings : uint32_t {
            Tabular,
            Image2D,
            None
        };
};