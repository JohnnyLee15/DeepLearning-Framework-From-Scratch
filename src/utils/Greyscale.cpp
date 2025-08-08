#include "utils/Greyscale.h"
#include "core/tensor/Tensor.h"
#include <iostream>
#include "utils/ConsoleUtils.h"
#include <omp.h>

const float Greyscale::MAX_GREYSCALE_VALUE = 255.0;

Tensor Greyscale::transform(const Tensor &data) const {
    checkFitted();

    Tensor transformed(data.getShape());
    vector<float> &transformedFlat = transformed.getFlat();

    size_t size = data.getSize();
    const vector<float> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      transformedFlat[i] = (dataFlat[i] / MAX_GREYSCALE_VALUE);
    }

    return transformed;
}

Tensor Greyscale::reverseTransform(const Tensor &data) const {
    checkFitted();

    Tensor transformed(data.getShape());
    vector<float> &transformedFlat = transformed.getFlat();

    size_t size = data.getSize();
    const vector<float> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        transformedFlat[i] = (dataFlat[i] * MAX_GREYSCALE_VALUE);
    }

    return transformed;
}

void Greyscale::throwDataFormatError() const {
    ConsoleUtils::fatalError(
        "Greyscale only supports Tensor input.\n"
        "Vector input is not supported for this scalar."
    );
}

void Greyscale::fit(const vector<float> &data) {
    throwDataFormatError();
}

vector<float> Greyscale::transform(const vector<float> &data) const {
    throwDataFormatError();
    return {};
}

vector<float> Greyscale::reverseTransform(const vector<float> &data) const {
    throwDataFormatError();
    return {};
}

uint32_t Greyscale::getEncoding() const {
    return Scalar::Encodings::Greyscale;
}

Scalar* Greyscale::clone() const {
    return new Greyscale(*this);
}