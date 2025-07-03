#include "utils/Greyscale.h"
#include "core/Tensor.h"
#include <iostream>
#include "utils/ConsoleUtils.h"
#include <omp.h>

const double Greyscale::MAX_GREYSCALE_VALUE = 255.0;

Tensor Greyscale::transform(const Tensor &data) const {
    checkFitted();

    Tensor transformed(data.getShape());
    vector<double> &transformedFlat = transformed.getFlat();

    size_t size = data.getSize();
    const vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      transformedFlat[i] = (dataFlat[i] / MAX_GREYSCALE_VALUE);
    }

    return transformed;
}

Tensor Greyscale::reverseTransform(const Tensor &data) const {
    checkFitted();

    Tensor transformed(data.getShape());
    vector<double> &transformedFlat = transformed.getFlat();

    size_t size = data.getSize();
    const vector<double> &dataFlat = data.getFlat();

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

void Greyscale::fit(const vector<double> &data) {
    throwDataFormatError();
}

vector<double> Greyscale::transform(const vector<double> &data) const {
    throwDataFormatError();
    return {};
}

vector<double> Greyscale::reverseTransform(const vector<double> &data) const {
    throwDataFormatError();
    return {};
}

uint32_t Greyscale::getEncoding() const {
    return Scalar::Encodings::Greyscale;
}