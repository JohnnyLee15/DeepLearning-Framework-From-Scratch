#pragma once

#include "core/model/NeuralNet.h"
#include <string>
#include <fstream>

class Scalar;
class Data;
class ImageTransform2D;

using namespace std;

class Pipeline {
    private:

        // Instance Variables
        Data *data;
        Scalar *featureScalar;
        Scalar *targetScalar;
        NeuralNet *model;
        ImageTransform2D *imageTransformer;

        bool isLoadedPipeline;

        // Methods
        void checkIsLoadedPipeline(const string&) const;

        void loadModel(ifstream&);
        void loadData(ifstream&);
        void loadFeatureScalar(ifstream&);
        void loadTargetScalar(ifstream&);
        void loadImageTransformer2D(ifstream&);

    public:
        // Constructor
        Pipeline();

        // Destructor
        ~Pipeline();

        // Methods
        void setFeatureScalar(Scalar*);
        void setTargetScalar(Scalar*);
        void setModel(NeuralNet*);
        void setData(Data*);
        void setImageTransformer2D(ImageTransform2D*);

        Data* getData() const;
        Scalar* getFeatureScalar() const;
        Scalar* getTargetScalar() const;
        NeuralNet* getModel() const;
        ImageTransform2D* getImageTransformer() const;

        void saveToBin(const string&) const;
        void writeBin(ofstream&) const;
        void loadComponents(ifstream&);

        // Static Methods
        static Pipeline loadFromBin(const string&);
};