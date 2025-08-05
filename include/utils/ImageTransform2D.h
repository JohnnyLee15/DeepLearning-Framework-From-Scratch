#pragma once

#include <vector>
#include <fstream>
#include "core/data/ImageData2D.h"

using namespace std;

class ImageTransform2D {
    private:
        int height;
        int width;
        int channels;

        static const float MAX_COLOUR_VALUE;

    public:
        ImageTransform2D(int, int, int);
        ImageTransform2D();
        Tensor transform(const vector<RawImage>&) const;
        int getHeight() const;
        int getWidth() const;
        int getChannels() const;
        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
};