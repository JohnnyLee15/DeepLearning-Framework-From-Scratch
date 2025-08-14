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

        string bestModelPath;

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
        Pipeline(const Pipeline&);

        // Destructor
        ~Pipeline();

        // Methods
        void setFeatureScalar(Scalar*);
        void setTargetScalar(Scalar*);
        void setModel(NeuralNet*);
        void setData(Data*);
        void setImageTransformer2D(ImageTransform2D*);
        void setBestModelPath(const string&);

        Data* getData() const;
        Scalar* getFeatureScalar() const;
        Scalar* getTargetScalar() const;
        NeuralNet* getModel() const;
        ImageTransform2D* getImageTransformer() const;
        const string& getBestModelPath() const;

        void saveToBin(const string&);
        void writeBin(ofstream&) const;
        void loadComponents(ifstream&);

        bool hasBestModel() const;
        void clearBestModelPath();

        // Static Methods
        static Pipeline loadFromBin(const string&);
};