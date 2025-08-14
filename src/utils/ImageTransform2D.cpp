#define STB_IMAGE_RESIZE_IMPLEMENTATION
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#pragma clang diagnostic ignored "-Wunused-variable"
#include <stb/stb_image_resize.h>
#pragma clang diagnostic pop

#include "utils/ImageTransform2D.h"
#include "utils/ConsoleUtils.h"
#include <iostream>
#include <omp.h>

const float ImageTransform2D::MAX_COLOUR_VALUE = 255.0;

ImageTransform2D::ImageTransform2D(int height, int width, int channels) :
    height(height), width(width), channels(channels) {}

ImageTransform2D::ImageTransform2D() {}

Tensor ImageTransform2D::transform(const vector<RawImage> &rawImages) const {
    cout << endl << "🎨 Transforming " << rawImages.size() << " images." << endl;
    ConsoleUtils::loadMessage("Resizing & Normalizing images.");

    Tensor transformedImages = Tensor({
        rawImages.size(), (size_t) height, (size_t) width, (size_t) channels
    });

    vector<float> &imageFlat = transformedImages.getFlat();
    size_t numImages = rawImages.size();
    size_t size = height * width * channels;

    #pragma omp parallel for
    for (size_t n = 0; n < numImages; n++) {
        const RawImage &image = rawImages[n];
        if (channels != image.channels) {
            ConsoleUtils::fatalError(
                "Channel mismatch: requested " + std::to_string(channels) +
                " but file provides " + std::to_string(image.channels) +
                ". Choose a consistent channel count or pre-convert your data."
            ); 
        }

        vector<unsigned char> resized(height * width * channels);
        stbir_resize_uint8(
            image.pixels.data(), image.width, image.height, 0,
            resized.data(), width, height, 0,
            channels
        );

        size_t currIdx = n * size;
        for (size_t i = 0; i < size; i++) {
            imageFlat[i + currIdx] = (float) resized[i] / MAX_COLOUR_VALUE;
        }
    }
    
    ConsoleUtils::completeMessage();
    ConsoleUtils::printSepLine();
    return transformedImages;
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

ImageTransform2D* ImageTransform2D::clone() const {
    return new ImageTransform2D(*this);
}