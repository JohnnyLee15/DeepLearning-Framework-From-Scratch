#pragma once

#include <vector>
#include <fstream>
#include "core/data/ImageData2D.h"

using namespace std;

class ImageTransform2D {
    private:
        // Constants
        static const float MAX_COLOUR_VALUE;

        // Instance Variables
        int height;
        int width;
        int channels;

    public:
        // Constructors
        ImageTransform2D(int, int, int);
        ImageTransform2D();

        // Methods
        Tensor transform(const vector<RawImage>&) const;
        int getHeight() const;
        int getWidth() const;
        int getChannels() const;
        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);

        ImageTransform2D* clone() const;
};