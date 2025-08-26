#pragma once

#include <string>
#include <vector>
#include <cstdint> 
#include <fstream>

using namespace std;

class Batch;
class Tensor;

class BinLoader {
    private:
        // Instance Variables
        mutable ifstream dataBin;
        vector<size_t> xShape;
        size_t elementsPerSample;
        uint64_t rowBytes;
        uint64_t targetOffBase;
        uint64_t featureOffBase;

        // Constants
        static const string DATA_EXTENSION;

        // Methods
        void writeToBin(const string&, const Tensor&, const vector<float>&) const;
        void writeShape(ofstream&, const Tensor&) const;
        void writeData(ofstream&, const Tensor&, const vector<float>&) const;
        void closeOutBin(ofstream&, const string&) const;
        ofstream openOut(const string&) const;
        void openData(const string&);
        void readShapeX();
        void calculateOffsets();

    public:
        // Constructors
        BinLoader(const string&, Tensor&, vector<float>&);
        BinLoader(const string&);
        BinLoader();

        // Methods
        const vector<size_t>& getShapeX() const;
        size_t sampleCount() const;
        size_t xSize() const;
        void fillBatch(Batch&) const;
        void fillInfBatch(Tensor&, size_t, size_t) const;
        vector<float> loadTargets() const;
};
