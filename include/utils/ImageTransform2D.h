#pragma once

#include <vector>
#include <fstream>

using namespace std;

class ImageTransform2D {
    private:
        int height;
        int width;
        int channels;

        static const double MAX_COLOUR_VALUE;

    public:
        ImageTransform2D(int, int, int);
        ImageTransform2D();
        vector<double> transform(const unsigned char*, int, int, int) const;
        int getHeight() const;
        int getWidth() const;
        int getChannels() const;
        void writeBin(ofstream&) const;
        void loadFromBin(ifstream&);
};