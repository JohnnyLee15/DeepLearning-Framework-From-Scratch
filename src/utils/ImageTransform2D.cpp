#define STB_IMAGE_RESIZE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#pragma clang diagnostic ignored "-Wunused-variable"
#include <stb/stb_image_resize.h>
#pragma clang diagnostic pop

#include "utils/ImageTransform2D.h"
#include "utils/ConsoleUtils.h"

const float ImageTransform2D::MAX_COLOUR_VALUE = 255.0;

ImageTransform2D::ImageTransform2D(int height, int width, int channels) :
    height(height), width(width), channels(channels) {}

ImageTransform2D::ImageTransform2D() {}

vector<float> ImageTransform2D::transform(
    const unsigned char *input,
    int origHeight,
    int origWidth,
    int origChannels
) const {
    vector<unsigned char> resized(height * width * channels);
    stbir_resize_uint8(
        input, origWidth, origHeight, 0,
        resized.data(), width, height, 0,
        channels
    );

    vector<float> normalized(height * width * channels);
    size_t size = height * width * channels;
    for (size_t i = 0; i < size; i++) {
        normalized[i] = (float) resized[i] / MAX_COLOUR_VALUE;
    }

    return normalized;
}

int ImageTransform2D::getHeight() const {
    return height;
}

int ImageTransform2D::getWidth() const {
    return width;
}

int ImageTransform2D::getChannels() const {
    return channels;
}

void ImageTransform2D::writeBin(ofstream &modelBin) const {
    uint32_t heightWrite = height;
    uint32_t widthWrite = width;
    uint32_t channelsWrite = channels;

    modelBin.write((char*) &heightWrite, sizeof(uint32_t));
    modelBin.write((char*) &widthWrite, sizeof(uint32_t));
    modelBin.write((char*) &channelsWrite, sizeof(uint32_t));
}

void ImageTransform2D::loadFromBin(ifstream &modelBin) {
    uint32_t h,w,c;
    modelBin.read((char*) &h, sizeof(uint32_t));
    modelBin.read((char*) &w, sizeof(uint32_t));
    modelBin.read((char*) &c, sizeof(uint32_t));

    height = h;
    width = w;
    channels = c;
}