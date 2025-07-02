#pragma once
#include <fstream>
#include <cstdint>

class Task;
class Tensor;

using namespace std;

class Data {
    public:
        virtual const Tensor& getTrainFeatures() const = 0;
        virtual const Tensor& getTestFeatures() const = 0;
        virtual const vector<double>& getTrainTargets() const = 0;
        virtual const vector<double>& getTestTargets() const = 0;
        virtual size_t getNumTrainSamples() const = 0;
        virtual const Task* getTask() const = 0;
        virtual vector<size_t> generateShuffledIndices() const = 0;
        virtual void writeBin(ofstream&) const = 0;
        virtual void loadFromBin(ifstream&) = 0;
};