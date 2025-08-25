#pragma once

#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Data {
    public:
        // Enums
        enum Encodings : uint32_t {
            Tabular,
            Image2D,
            None
        };

        // Virtual Destructor
        virtual ~Data() = default;

        // Methods
        virtual const vector<float>& getTrainTargets() const = 0;
        virtual const vector<float>& getTestTargets() const = 0;

        virtual void clearTrain() = 0;
        virtual void clearTest() = 0;

        virtual size_t getNumTrainSamples() const = 0;
        virtual Encodings getEncoding() const = 0;

        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;

        virtual Data* clone() const = 0;
};