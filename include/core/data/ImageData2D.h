#pragma once

#include "core/data/Data.h"
#include "core/tensor/Tensor.h"
#include <unordered_map>
#include <vector>
#include <fstream>

struct RawImage {
    vector<unsigned char> pixels;
    size_t width;
    size_t height;
    size_t channels;
};

class ImageData2D : public Data {
    private:
        // Instance Variables
        vector<RawImage> trainFeatures;
        vector<float> trainTargets;

        vector<RawImage> testFeatures;
        vector<float> testTargets;

        unordered_map<string, int> labelMap;

        // Methods
        void read(vector<RawImage>&, vector<float>&, const string&, size_t);
        void scanDirectory(vector<string>&, vector<string>&, const string&) const;
        void extractImages(vector<RawImage>&, const vector<string>&, size_t) const;
        void extractLabels(vector<float>&, const vector<string>&);

    public:
        // Methods
        void readTrain(const string&, size_t);
        void readTest(const string&, size_t);
        void clearTrain();
        void clearTest();

        const vector<RawImage>& getTrainFeatures() const;
        const vector<RawImage>& getTestFeatures() const;
        const vector<float>& getTrainTargets() const override;;
        const vector<float>& getTestTargets() const override;;

        size_t getNumTrainSamples() const override;
        Data::Encodings getEncoding() const override;

        void writeBin(ofstream&) const override;
        void loadFromBin(ifstream&) override;

        Data* clone() const override;
};