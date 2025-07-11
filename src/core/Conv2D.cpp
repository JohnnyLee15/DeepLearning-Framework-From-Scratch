#include "core/Conv2D.h"
#include "activations/Activation.h"
#include <random>
#include <omp.h>
#include "utils/ConsoleUtils.h"
#include "utils/VectorUtils.h"

Conv2D::Conv2D(
    size_t numKernals, 
    size_t kRows, 
    size_t kCols, 
    size_t strideIn,
    const string &padIn, 
    Activation *activation
) : numKernals(numKernals), kRows(kRows), kCols(kCols), activation(activation) {
    initStride(strideIn);
    padding = Tensor::decodePadding(padIn);
}

void Conv2D::checkBuildSize(const vector<size_t> &inShape) const {
    if (inShape.size() != 4) {
        ConsoleUtils::fatalError(
            "Conv2D build error: Expected 4D input (batch_size, height, width, channels), "
            "but got tensor with " + to_string(inShape.size()) + " dimensions."
        );
    }
}

void Conv2D::build(const vector<size_t> &inShape) {
    checkBuildSize(inShape);
    size_t depth = inShape[3];
    kernals = Tensor({numKernals, kRows, kCols, depth});
    initKernals();
    biases = activation->initBias(numKernals);
}

vector<uint32_t> Conv2D::generateThreadSeeds() const {
    size_t numSeeds = omp_get_max_threads();
    vector<uint32_t> seeds(numSeeds);
    random_device rd;
    for (size_t i = 0; i < numSeeds; i++) {
        seeds[i] = rd();
    }

    return seeds;
}

void Conv2D::initKernals() {
    const vector<size_t> &kernalsShape = kernals.getShape();
    size_t size = kernals.getSize();
    double std = sqrt(2.0/(kernalsShape[1] * kernalsShape[2] * kernalsShape[3]));
    vector<double> &kernalsFlat = kernals.getFlat();
    vector<uint32_t> seeds = generateThreadSeeds();

    #pragma omp parallel
    {
        int thread = omp_get_thread_num();
        mt19937 generator(seeds[thread]);
        normal_distribution<double> distribution(0, std);

        #pragma omp for
        for (size_t i = 0; i < size; i++) {
            kernalsFlat[i] = distribution(generator);
        }
    }
}

void Conv2D::initStride(size_t strideIn) {
    if (strideIn == 0) {
        ConsoleUtils::fatalError(
            "Stride must be greater than zero for Conv2D layer configuration."
        );
    }
    stride = strideIn;
}

vector<size_t> Conv2D::getBuildOutShape(const vector<size_t> &inShape) const {
    checkBuildSize(inShape);
    Tensor dummy(inShape);
    WindowDims win = dummy.computeInputWindow(kRows, kCols, padding, stride);
    return {0, win.outRows, win.outCols, numKernals};
}

void Conv2D::forward(const Tensor &input) {
    WindowDims win = input.computeInputWindow(kRows, kCols, padding, stride);
    Tensor inputFwd = input.padIfNeeded(win, padding);

    preActivations = inputFwd.conv2dForward(kernals, win, stride, biases);
    activations = activation->activate(preActivations);
}

void Conv2D::backprop(
    const Tensor &prevActivations,
    double learningRate,
    const Tensor &outputGradients,
    bool isFirstLayer
) {
    (void) isFirstLayer;

    double scaleFactor = -learningRate / prevActivations.getShape()[0];
    WindowDims winInput = prevActivations.computeInputWindow(kRows, kCols, padding, stride);

    dZ = outputGradients;
    dZ *= activation->calculateGradient(preActivations);

    Tensor prevActProcessed = prevActivations.padIfNeeded(winInput, padding);
    Tensor dW = prevActProcessed.conv2dWeights(dZ, numKernals, kRows, kCols, stride);
    kernals += (dW *= scaleFactor);

    vector<double> dB = dZ.reduceSumBias();
    VectorUtils::scaleVecInplace(dB, scaleFactor);
    VectorUtils::addVecInplace(biases, dB);

    if (stride > 1) {
        dZ = dZ.gradUpsample(stride);
    }
    WindowDims winGrad = dZ.computeGradWindow(
        kRows, kCols, prevActProcessed.getShape()[1], 
        prevActProcessed.getShape()[2], stride, winInput
    );
    dZ = dZ.padWindowInput(winGrad);
    dZ = dZ.conv2dInput(kernals);
}



const Tensor& Conv2D::getOutput() const {
    return activations;
}

Tensor Conv2D::getOutputGradient() const {
    return dZ;
}

void Conv2D::writeBin(ofstream &modelBin) const {}
void Conv2D::loadFromBin(ifstream &modelBin) {}
uint32_t Conv2D::getEncoding() const {
    return Layer::Encodings::Conv2D;
}
