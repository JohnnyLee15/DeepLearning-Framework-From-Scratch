#include "core/tensor/Tensor.h"
#include "utils/ConsoleUtils.h"
#include "core/tensor/Matrix.h"
#include "utils/CsvUtils.h"
#include "core/gpu/GpuEngine.h"
#include <limits>
#include <iostream>

const string Tensor::PADDING_NONE = "none";
const string Tensor::PADDING_SAME = "same";

Tensor::Tensor(const vector<size_t> &shape) : shape(shape) {
    if (shape.size() > 0) {
        size_t size = 1;
        size_t dims = shape.size();
        for (size_t i = 0; i < dims; i++) {
            size *= shape[i];
        }

        data = vector<float>(size, 0.0f);
        ensureGpu();
    }
}

Tensor::Tensor(const vector<vector<float> > &mat) {
    size_t numRows = mat.size();
    size_t numCols = 0;
    if (numRows > 0) {
        numCols = mat[0].size();
    }

    shape = {numRows, numCols};
    data = vector<float>(numRows * numCols, 0.0);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            data[i * numCols + j] = mat[i][j];
        }
    }
    ensureGpu();
}

Tensor::Tensor(const vector<float> &data, const vector<size_t> &shape) :
    shape(shape), data(data) {
    ensureGpu();
}

Tensor::Tensor(const Tensor &other) :
    shape(other.shape), data(other.data), dataGpu(other.dataGpu) {}

Tensor::Tensor() {}

Tensor& Tensor::operator =(const Tensor &other) {
    if (this != &other) { 
        shape = other.shape;
        data = other.data;

        if (GpuEngine::isUsingGpu()) {
            #ifdef __APPLE__
                dataGpu = other.dataGpu;
            #endif 
        }
    }

    return *this;
}

void Tensor::ensureGpu() {
    if (GpuEngine::isUsingGpu()) {
        #ifdef __APPLE__
            initGpuTensor();
        #endif 
    }
}

void Tensor::print(const string &name) const {
    cout << name << ": shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        cout << shape[i];
        if (i < shape.size() - 1) cout << ", ";
    }
    cout << "]\n";
}

void Tensor::zero() {
    fill(data.begin(), data.end(), 0.0f);
    ensureGpu();
}

void Tensor::reShapeInPlace(const vector<size_t> &newShape) {
    shape = newShape;
}

const vector<float>& Tensor::getFlat() const {
    return data;
}

vector<float>& Tensor::getFlat() {
    return data;
}

const vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::getSize() const {
    if (shape.size() == 0 || shape[0] == 0) {
        return 0;
    }

    size_t dims = shape.size();
    size_t size = 1;
    for (size_t i = 0; i < dims; i++) {
        size *= shape[i];
    }

    return size;
}

size_t Tensor::getRank() const {
    if (shape.empty() || (shape.size() == 1 && shape[0] == 1)) {
        return 0;
    }

    return shape.size();
}

Matrix Tensor::M() const {
    if (getRank() != 2) {
        ConsoleUtils::fatalError(
            string("Matrix view requires a rank-2 tensor.\n") +
            "Received tensor with rank: " + to_string(getRank()) + "."
        );
    }

    return Matrix(const_cast<Tensor&>(*this));
}

Tensor::Paddings Tensor::decodePadding(const string &padIn) {
    string formattedPadding = CsvUtils::toLowerCase(CsvUtils::trim(padIn));
    if (formattedPadding == Tensor::PADDING_NONE) {
        return Paddings::NONE;
    } else if (formattedPadding == Tensor::PADDING_SAME) {
        return Paddings::SAME;
    } 

    ConsoleUtils::fatalError(
        "Invalid padding mode: \"" + padIn + "\".\n"
        "Valid options are: \"" + Tensor::PADDING_NONE + "\" or \"" + Tensor::PADDING_SAME + "\"."
    );
}

WindowDims Tensor::computeInputWindow(
    size_t kRows,
    size_t kCols,
    Tensor::Paddings padding,
    size_t stride
) const {
    WindowDims win;
    if (padding == Paddings::NONE) {

        if (shape[1] < kRows || shape[2] < kCols) {
            string errorMessage = "Convolution error (padding='none'): Kernel (" +
                to_string(kRows) + "x" + to_string(kCols) + ") cannot be larger than input image (" +
                to_string(shape[1]) + "x" + to_string(shape[2]) + ").";

            ConsoleUtils::fatalError(errorMessage);
        }

        win.outRows = (shape[1] - kRows)/stride + 1;
        win.outCols = (shape[2] - kCols)/stride + 1;
        win.padRows = 0;
        win.padCols = 0;
        win.padTop = 0;
        win.padLeft = 0;

    } else {
        win.outRows = (shape[1] + stride  - 1)/stride;
        win.outCols = (shape[2] + stride  - 1)/stride;

        long scaleRows = (long) stride * ((long) win.outRows - 1);
        long scaleCols = (long) stride * ((long) win.outCols - 1);
        long padRows = (long) kRows + scaleRows - (long) shape[1];
        long padCols = (long) kCols + scaleCols - (long) shape[2];

        win.padRows = (padRows < 0) ? 0 : padRows;
        win.padCols = (padCols < 0) ? 0 : padCols;

        win.padTop = win.padRows/2;
        win.padLeft = win.padCols/2;
    } 

    return win;
}

WindowDims Tensor::computeGradWindow(
    size_t kRows, 
    size_t kCols, 
    size_t inRows,
    size_t inCols,
    size_t stride,
    const WindowDims &inDims
) const {
    size_t outPadRow = (inRows - kRows) % stride;
    size_t outPadCol = (inCols - kCols) % stride;
    size_t inPadBottom = inDims.padRows - inDims.padTop;
    size_t inPadRight= inDims.padCols - inDims.padLeft;
    size_t gradPadBottom = kRows -1 - inPadBottom + outPadRow;
    size_t gradPadRight = kCols - 1 - inPadRight + outPadCol;

    WindowDims winGrad;
    winGrad.outRows = 0;
    winGrad.outCols = 0;
    winGrad.padTop = kRows - 1 - inDims.padTop;
    winGrad.padLeft = kCols - 1 - inDims.padLeft;
    winGrad.padRows = winGrad.padTop + gradPadBottom;
    winGrad.padCols = winGrad.padLeft + gradPadRight;

    return winGrad;
}

void Tensor::padWindowInput(
    Tensor &toPad,
    const WindowDims &win,
    float padVal
) const {
    // Add error checking

    size_t numSamples = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t depth = shape[3];

    size_t newRows = toPad.shape[1];
    size_t newCols = toPad.shape[2];

    vector<float> &padFlat = toPad.data;
    fill(padFlat.begin(), padFlat.end(), padVal);

    #pragma omp parallel for collapse(4)
    for (size_t n = 0; n < numSamples; n++) {
        for (size_t r = 0; r < inRows; r++) {
            for (size_t c = 0; c < inCols; c++) {
                for (size_t d = 0; d < depth; d++) {
                    size_t inIdx = (((n * inRows + r) * inCols + c) * depth) + d;
                    size_t padIdx = (((n * newRows + (r + win.padTop)) * newCols + (c + win.padLeft)) * depth)+ d;
                    padFlat[padIdx] = data[inIdx];
                }
            }
        }
    }
}

const Tensor& Tensor::padIfNeeded(
    Tensor &toPad,
    const WindowDims &win,
    Tensor::Paddings padding,
    float padVal
) const {
    if (padding == Paddings::NONE) {
        return *this;
    }

    padWindowInput(toPad, win, padVal);
    return toPad;
}

void Tensor::conv2dForward(
    const Tensor &kernals,
    size_t stride,
    Tensor &output,
    const Tensor &biases
) const {
    // Add error checking

    const vector<size_t> &kernalsShape = kernals.shape;
    size_t outDepth = kernalsShape[0];
    size_t kRows = kernalsShape[1];
    size_t kCols = kernalsShape[2];

    size_t numSamples =  shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t inDepth = shape[3];

    size_t outRows = output.shape[1];
    size_t outCols = output.shape[2];

    vector<float> &outFlat = output.data;
    const vector<float> &biasFlat = biases.data;
    const vector<float> &inFlat = data;
    const vector<float> &kernalsFlat = kernals.data;

    #pragma omp parallel for collapse(4)
    for (size_t n = 0; n < numSamples; n++) {
        for (size_t r = 0; r < outRows; r++) {
            for (size_t c = 0; c < outCols; c++) {
                for (size_t o = 0; o < outDepth; o++) {

                    float value = biasFlat[o];
                    for (size_t i = 0; i < kRows; i++) {
                        size_t inRow = r*stride + i;

                        for (size_t j = 0; j < kCols; j++) {
                            size_t inCol = c*stride + j;

                            for (size_t d = 0; d < inDepth; d++) {
                                size_t kIdx = (((o * kRows + i) * kCols + j) * inDepth) + d;
                                size_t inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + d;
                                value += inFlat[inIdx] * kernalsFlat[kIdx];
                            }
                        }
                    }

                    size_t outIdx = (((n * outRows + r) * outCols + c) * outDepth) + o;
                    outFlat[outIdx] = value;
                }
            }
        }
    }
}

void Tensor::maxPool2d(
    vector<size_t> &maxIndices,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor &pooledOutput,
    const WindowDims &winIn
) const {
    // Add error checking
    size_t batchSize = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t inDepth = shape[3];
    
    size_t origRows = inRows - winIn.padRows;
    size_t origCols = inCols - winIn.padCols;

    maxIndices.assign(pooledOutput.getSize(), SIZE_MAX);

    const vector<float> &inFlat = data;
    vector<float> &outFlat = pooledOutput.data;

    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batchSize; b++) {
        for (size_t r = 0; r < winIn.outRows; r++) {
            for (size_t c = 0; c < winIn.outCols; c++) {
                for (size_t d = 0; d < inDepth; d++) {

                    float maxVal = numeric_limits<float>::lowest();
                    size_t maxIdx = SIZE_MAX;
                    for (size_t i = 0; i < kRows; i++) {
                        size_t inRow = r * stride + i;
                        size_t origRow = inRow - winIn.padTop;

                        for (size_t j = 0; j < kCols; j++) {
                            size_t inCol = c * stride + j;
                            size_t origCol = inCol - winIn.padLeft;
                            size_t padIdx = (((b * inRows + inRow) * inCols + inCol) * inDepth) + d;
                            size_t idx = ((b * origRows + origRow) * origCols + origCol) * inDepth + d;

                            float val = inFlat[padIdx];
                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }

                    size_t outIdx = (((b * winIn.outRows + r) * winIn.outCols + c) * inDepth) + d;
                    outFlat[outIdx] = maxVal;
                    maxIndices[outIdx] = maxIdx;
                }
            }
        }
    }
}

void Tensor::conv2dWeights(
    const Tensor &grad,
    size_t numKernals,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor &dW
) const {

    // Add error checking
    const vector<size_t> &gradSize = grad.shape;
    size_t gradRows = gradSize[1];
    size_t gradCols = gradSize[2];

    size_t batchSize = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t inDepth = shape[3];

    const vector<float> &gradFlat = grad.data;
    vector<float> &dwFlat = dW.data;

    #pragma omp parallel for collapse(4)
    for (size_t k = 0; k < numKernals; k++) {
        for (size_t i = 0; i < kRows; i++) {
            for (size_t j = 0; j < kCols; j++) {
                for (size_t d = 0; d < inDepth; d++) {

                    float value = 0.0;
                    for (size_t n = 0; n < batchSize; n++) {
                        for (size_t r = 0; r < gradRows; r++) {
                            size_t inRow = r * stride + i;

                            for (size_t c = 0; c < gradCols; c++) {
                                size_t inCol = c * stride + j;
                                size_t inIdx = (((n * inRows + inRow) * inCols + inCol) * inDepth) + d;
                                size_t gradIdx = (((n * gradRows + r) * gradCols + c) * numKernals) + k;

                                value += (gradFlat[gradIdx] * data[inIdx]);
                            }
                        }
                    }

                    size_t dwIdx = (((k * kRows + i) * kCols + j) * inDepth) + d;
                    dwFlat[dwIdx] = value;
                }
            }
        }
    }
}

void Tensor::reduceSumBias(Tensor &dB) const {
    size_t batchSize = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];
    size_t numKernals = shape[3];

    vector<float> &dbFlat = dB.data;
    fill(dbFlat.begin(), dbFlat.end(), 0.0f);
    #pragma omp parallel
    {
        vector<float> localDB(numKernals);

        #pragma omp for collapse(4)
        for (size_t k = 0; k < numKernals; k++) {
            for (size_t b = 0; b < batchSize; b++) {
                for (size_t r = 0; r < gradRows; r++) {
                    for (size_t c = 0; c < gradCols; c++) {
                        size_t gradIdx = (((b * gradRows + r) * gradCols + c) * numKernals) + k;
                        localDB[k] += data[gradIdx];
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (size_t k = 0; k < numKernals; k++) {
                dbFlat[k] += localDB[k];
            }
        }
    }
}

void Tensor::padAndUpsampleGrad(
    Tensor &outGrad, 
    const WindowDims &winGrad, 
    size_t stride
) const{ 
    size_t batchSize = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];
    size_t numKernals = shape[3];

    size_t outRows = (stride > 1) ? stride * (gradRows - 1) + 1 : gradRows;
    size_t outCols = (stride > 1) ? stride * (gradCols - 1) + 1 : gradCols;
    outRows += winGrad.padRows;
    outCols += winGrad.padCols;

    outGrad.reShapeInPlace(
        {batchSize, outRows, outCols, numKernals}
    );

    vector<float> &outFlat = outGrad.data;
    fill(outFlat.begin(), outFlat.end(), 0.0f);

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < batchSize; n++) {
        for (size_t r = 0; r < gradRows; r++) {

            size_t upRow = r * stride + winGrad.padTop;
            for (size_t c = 0; c < gradCols; c++) {

                size_t upCol = c * stride + winGrad.padLeft;
                for (size_t d = 0; d < numKernals; d++) {
                    size_t inIdx = (((n * gradRows + r) * gradCols + c) * numKernals) + d;
                    size_t upIdx = (((n * outRows + upRow) * outCols + upCol) * numKernals) + d;
                    outFlat[upIdx] = data[inIdx];
                }
            }
        }
    }
}

void Tensor::conv2dInput(
    const Tensor &kernals,
    Tensor &dX
) const {
    const vector<size_t> &kShape = kernals.shape;
    size_t numkernals = shape[3];
    size_t kRows = kShape[1];
    size_t kCols = kShape[2];
    size_t inDepth = kShape[3];

    size_t numSamples = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];

    size_t dxRows = dX.shape[1];
    size_t dxCols = dX.shape[2];

    const vector<float> &gradFlat = data;
    const vector<float> &kFlat = kernals.data;
    vector<float> &dxFlat = dX.data;

    #pragma omp parallel for collapse(4)
    for (size_t n = 0; n < numSamples; n++) {
        for (size_t r = 0; r < dxRows; r++) {
            for (size_t c = 0; c < dxCols; c++) {
                for (size_t d = 0; d < inDepth; d++) {

                    float value = 0.0;
                    for (size_t i = 0; i < kRows; i++) {
                        size_t gradRow = r + i;
                        size_t flipI = kRows - 1 - i;

                        for (size_t j = 0; j < kCols; j++) {
                            size_t gradCol = c + j;
                            size_t flipJ = kCols - 1 - j;

                            for (size_t k = 0; k < numkernals; k++) {
                                size_t kIdx = (((k * kRows + flipI) * kCols + flipJ) * inDepth) + d;
                                size_t gradIdx = (((n * gradRows + gradRow) * gradCols + gradCol) * numkernals) + k;
                                value += (kFlat[kIdx] * gradFlat[gradIdx]);

                            }
                        }
                    }

                    size_t dxIdx = (((n * dxRows + r) * dxCols + c) * inDepth) + d;
                    dxFlat[dxIdx] = value;
                }
            }
        }
    }
}

void Tensor::maxPool2dGrad(
    const vector<size_t> &maxIndices,
    Tensor &dX
) const {
    vector<float> &dxFlat = dX.data;
    fill(dxFlat.begin(), dxFlat.end(), 0.0f);
    const vector<float> &gradFlat = data;
    
    size_t gradSize = getSize();

    #pragma omp parallel for
    for (size_t i = 0; i < gradSize; i++) {
        #pragma omp atomic
        dxFlat[maxIndices[i]] += gradFlat[i];
    }
}

void Tensor::hadamard(const Tensor &ten2) {
    // Add error checking
    size_t size = getSize();
    
    const vector<float> &ten2Flat = ten2.data;
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] *= ten2Flat[i];
    }
}

void Tensor::applyGrad(const Tensor &grad, float scaleFactor){
    size_t size = getSize();
    const vector<float> &gradFlat = grad.data;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] += (scaleFactor * gradFlat[i]);
    }
}

void Tensor::applyMask(const Tensor &mask, Tensor &output) const {
    const vector<float> &inFlat = data;
    const vector<float> &maskFlat = mask.data;
    vector<float> &outFlat = output.data;
    size_t size = getSize();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        outFlat[i] = maskFlat[i] * inFlat[i];
    }
}