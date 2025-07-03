#pragma once

#include "core/NeuralNet.h"
#include <string>
#include <fstream>

class Scalar;
class Data;

using namespace std;

class Pipeline {
    private:
        Data *data;
        Scalar *featureScalar;
        Scalar *targetScalar;
        NeuralNet *model;
        bool isLoadedPipeline;

        void checkIsLoadedPipeline(const string&) const;
        void loadData(ifstream&);
        void loadFeatureScalar(ifstream&);
        void loadTargetScalar(ifstream&);

    public:
        Pipeline();

        void setFeatureScalar(Scalar*);
        void setTargetScalar(Scalar*);
        void setModel(NeuralNet*);
        void setData(Data*);

        Data* getData() const;
        Scalar* getFeatureScalar() const;
        Scalar* getTargetScalar() const;
        NeuralNet* getModel() const;

        void saveToBin(const string&) const;
        void writeBin(ofstream&) const;
        static Pipeline loadFromBin(const string&);
        void loadComponents(ifstream&);

        ~Pipeline();
};