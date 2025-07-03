#pragma once
#include <fstream>
#include <cstdint>

class Tensor;

using namespace std;

class Data {
    public:
        virtual const Tensor& getTrainFeatures() const = 0;
        virtual const Tensor& getTestFeatures() const = 0;
        virtual const vector<double>& getTrainTargets() const = 0;
        virtual const vector<double>& getTestTargets() const = 0;
        virtual void readTrain(string, size_t, bool header = false) = 0;
        virtual void readTest(string, size_t, bool header = false) = 0;
        virtual void readTrain(string, const string&) = 0;
        virtual void readTest(string, const string&) = 0;
        virtual size_t getNumTrainSamples() const = 0;
        virtual void writeBin(ofstream&) const;
        virtual void loadFromBin(ifstream&) = 0;
        virtual uint32_t getEncoding() const = 0;
        virtual ~Data() = default;

        virtual void headTrain(size_t numRows = 6) const = 0;
        virtual void headTest(size_t numRows = 6) const = 0;

        enum Encodings : uint32_t {
            Tabular,
            Image2D,
            None
        };
};