#include "core/Tensor.h"
#include "utils/ConsoleUtils.h"
#include "core/Matrix.h"
#include "utils/VectorUtils.h"
#include "utils/CsvUtils.h"

const string Tensor::PADDING_NONE = "none";
const string Tensor::PADDING_SAME = "same";

Tensor::Tensor(const vector<size_t> &shape) : shape(shape) {
    if (shape.size() > 0) {
        size_t size = 1;
        size_t dims = shape.size();
        for (size_t i = 0; i < dims; i++) {
            size *= shape[i];
        }

        data = vector<double>(size, 0.0);
    }
}

Tensor::Tensor(const vector<vector<double> > &mat) {
    size_t numRows = mat.size();
    size_t numCols = 0;
    if (numRows > 0) {
        numCols = mat[0].size();
    }

    shape = {numRows, numCols};
    data = vector<double>(numRows * numCols, 0.0);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numRows; i++) {
        for (size_t j = 0; j < numCols; j++) {
            data[i * numCols + j] = mat[i][j];
        }
    }
}

Tensor::Tensor(const vector<double> &data, const vector<size_t> &shape) :
    shape(shape), data(data) {}

Tensor::Tensor() {}

Tensor Tensor::reShape(const vector<size_t> &newShape) const {
    Tensor reshaped = *this;
    reshaped.shape = newShape;
    return reshaped;
}

const vector<double>& Tensor::getFlat() const {
    return data;
}

vector<double>& Tensor::getFlat() {
    return data;
}

const vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::getSize() const {
    return data.size();
}

size_t Tensor::getRank() const {
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
    // Add error checking

    WindowDims window;
    if (padding == Paddings::NONE) {
        window.outRows = (shape[1] - kRows)/stride + 1;
        window.outCols = (shape[2] - kCols)/stride + 1;
        window.padRows = 0;
        window.padCols = 0;
        window.padTop = 0;
        window.padLeft = 0;

    } else {
        window.outRows = (shape[1] + stride  - 1)/stride;
        window.outCols = (shape[2] + stride  - 1)/stride;
        window.padRows = kRows + stride * (window.outRows - 1) - shape[1];
        window.padCols = kCols + stride * (window.outCols - 1) - shape[2];
        window.padTop = window.padRows/2;
        window.padLeft = window.padCols/2;
    } 

    return window;
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

Tensor Tensor::padWindowInput(
    const WindowDims &win
) const {
    // Add error checking

    size_t numSamples = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t depth = shape[3];

    size_t newRows = inRows + win.padRows;
    size_t newCols = inCols + win.padCols;

    Tensor padded({numSamples, newRows, newCols, depth});
    vector<double> &padFlat = padded.data;

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

    return padded;
}

Tensor Tensor::padIfNeeded(
    const WindowDims &win,
    Tensor::Paddings padding
) const {
    if (padding == Paddings::NONE) {
        return *this;
    }
    return padWindowInput(win);
}

Tensor Tensor::conv2dForward(
    const Tensor &kernals,
    const WindowDims &win,
    size_t stride,
    const vector<double> &biases
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

    Tensor output = Tensor({numSamples, win.outRows, win.outCols, outDepth});
    vector<double> &outFlat = output.data;
    const vector<double> &inFlat = data;
    const vector<double> &kernalsFlat = kernals.data;

    #pragma omp parallel for collapse(4)
    for (size_t n = 0; n < numSamples; n++) {
        for (size_t r = 0; r < win.outRows; r++) {
            for (size_t c = 0; c < win.outCols; c++) {
                for (size_t o = 0; o < outDepth; o++) {

                    double value = 0.0;
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

                    size_t outIdx = (((n * win.outRows + r) * win.outCols + c) * outDepth) + o;
                    outFlat[outIdx] = value + (biases.empty() ? 0.0 : biases[o]);
                }
            }
        }
    }

    return output;
}

Tensor Tensor::maxPool2d(
    const WindowDims &win,
    vector<size_t> &maxIndices,
    size_t kRows,
    size_t kCols,
    size_t stride,
    Tensor::Paddings padding
) const {
    // Add error checking
    size_t batchSize = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t inDepth = shape[3];

    vector<size_t> outShape = {batchSize, win.outRows, win.outCols, inDepth};
    Tensor pooled= Tensor(outShape);
    maxIndices.assign(pooled.getSize(), SIZE_MAX);
    const vector<double> &inFlat = data;
    vector<double> &outFlat = pooled.data;

    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batchSize; b++) {
        for (size_t r = 0; r < win.outRows; r++) {
            for (size_t c = 0; c < win.outCols; c++) {
                for (size_t d = 0; d < inDepth; d++) {

                    double maxVal = -VectorUtils::INF;
                    size_t maxIdx = SIZE_MAX;
                    size_t maxInRow = SIZE_MAX;
                    size_t maxInCol = SIZE_MAX;
                    for (size_t i = 0; i < kRows; i++) {
                        size_t inRow = r * stride + i;
                        for (size_t j = 0; j < kCols; j++) {
                            size_t inCol = c * stride + j;
                            size_t idx = (((b * inRows + inRow) * inCols + inCol) * inDepth) + d;

                            if (inFlat[idx] >= maxVal) {
                                maxVal = inFlat[idx];
                                maxIdx = idx;
                                maxInRow = inRow;
                                maxInCol = inCol;
                            }
                        }
                    }

                    size_t outIdx = (((b * win.outRows + r) * win.outCols + c) * inDepth) + d;
                    outFlat[outIdx] = maxVal;

                    bool isValidRow = true;
                    bool isValidCol = true;
                    if (padding == Tensor::Paddings::SAME) {
                        size_t origRows = inRows - win.padRows;
                        size_t origCols = inCols - win.padCols;
                        isValidRow = (maxInRow >= win.padTop) && (maxInRow < origRows + win.padTop);
                        isValidCol = (maxInCol >= win.padLeft) && (maxInCol < origCols + win.padLeft);
                    }

                    if (isValidRow && isValidCol) {
                        maxIndices[outIdx] = maxIdx;
                    }
                }
            }
        }
    }

    return pooled;
}

Tensor Tensor::conv2dWeights(
    const Tensor &grad,
    size_t numKernals,
    size_t kRows,
    size_t kCols,
    size_t stride
) const {
    const vector<size_t> &gradSize = grad.shape;
    size_t gradRows = gradSize[1];
    size_t gradCols = gradSize[2];

    size_t batchSize = shape[0];
    size_t inRows = shape[1];
    size_t inCols = shape[2];
    size_t inDepth = shape[3];

    Tensor dW = Tensor({numKernals, kRows, kCols, inDepth});

    const vector<double> &gradFlat = grad.data;
    vector<double> &dwFlat = dW.data;

    #pragma omp parallel for collapse(4)
    for (size_t k = 0; k < numKernals; k++) {
        for (size_t i = 0; i < kRows; i++) {
            for (size_t j = 0; j < kCols; j++) {
                for (size_t d = 0; d < inDepth; d++) {

                    double value = 0.0;
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

    return dW;
}

vector<double> Tensor::reduceSumBias() const {
    size_t batchSize = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];
    size_t numKernals = shape[3];

    vector<double> dB(numKernals);

    #pragma omp parallel
    {
        vector<double> localDB(numKernals);

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
                dB[k] += localDB[k];
            }
        }
    }

    return dB;
}

Tensor Tensor::gradUpsample(size_t stride) const{ 
    size_t batchSize = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];
    size_t numKernals = shape[3];

    size_t upRows = stride * (gradRows - 1) + 1;
    size_t upCols = stride * (gradCols - 1) + 1;

    Tensor up({batchSize, upRows, upCols, numKernals});
    vector<double> &upFlat = up.data;

    #pragma omp parallel for collapse(2)
    for (size_t n = 0; n < batchSize; n++) {
        for (size_t r = 0; r < gradRows; r++) {

            size_t upRow = r * stride;
            for (size_t c = 0; c < gradCols; c++) {

                size_t upCol = c * stride;
                for (size_t d = 0; d < numKernals; d++) {
                    size_t inIdx = (((n * gradRows + r) * gradCols + c) * numKernals) + d;
                    size_t upIdx = (((n * upRows + upRow) * upCols + upCol) * numKernals) + d;
                    upFlat[upIdx] = data[inIdx];
                }
            }
        }
    }

    return up;
}

Tensor Tensor::conv2dInput(
    const Tensor &kernals
) const {
    const vector<size_t> &kShape = kernals.shape;
    size_t numkernals = shape[3];
    size_t kRows = kShape[1];
    size_t kCols = kShape[2];
    size_t inDepth = kShape[3];

    size_t numSamples = shape[0];
    size_t gradRows = shape[1];
    size_t gradCols = shape[2];

    size_t dxRows = gradRows - kRows + 1;
    size_t dxCols = gradCols - kCols + 1;

    Tensor dX = Tensor({numSamples, dxRows, dxCols, inDepth});

    const vector<double> &gradFlat = data;
    const vector<double> &kFlat = kernals.data;
    vector<double> &dxFlat = dX.data;

    #pragma omp parallel for collapse(4)
    for (size_t n = 0; n < numSamples; n++) {
        for (size_t r = 0; r < dxRows; r++) {
            for (size_t c = 0; c < dxCols; c++) {
                for (size_t d = 0; d < inDepth; d++) {

                    double value = 0.0;
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

    return dX;
}

Tensor Tensor::maxPool2dGrad(
    const Tensor &grad,
    const vector<size_t> &maxIndices
) const {
    Tensor dZ = Tensor(shape);
    vector<double> &dzFlat = dZ.data;
    const vector<double> &gradFlat = grad.getFlat();
    
    size_t gradSize = grad.getSize();
    
    #pragma omp parallel for
    for (size_t i = 0; i < gradSize; i++) {
        size_t inIdx = maxIndices[i];
        if (inIdx != SIZE_MAX) {
            #pragma omp atomic
            dzFlat[inIdx] += gradFlat[i];
        }
    }

    return dZ;
}

Tensor& Tensor::operator *=(const Tensor &ten2) {
    // Add error checking
    size_t size = getSize();
    
    const vector<double> &ten2Flat = ten2.data;
    
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] *= ten2Flat[i];
    }

    return *this;
}

Tensor& Tensor::operator *=(double scaleFactor){
    size_t size = getSize();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] *= scaleFactor;
    }

    return *this;
}

Tensor& Tensor::operator +=(const Tensor &ten2) {
    // Add error checking

    size_t size = getSize();
    const vector<double> &ten2Flat = ten2.data;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        data[i] += ten2Flat[i];
    }

    return *this;
}


