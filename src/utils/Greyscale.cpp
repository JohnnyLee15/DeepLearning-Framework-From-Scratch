#include "utils/Greyscale.h"
#include "core/Tensor.h"
#include <iostream>
#include "utils/ConsoleUtils.h"
#include <omp.h>

const double Greyscale::MAX_GREYSCALE_VALUE = 255.0;

void Greyscale::transform(Tensor &data) {
    Scalar::transform(data);

    size_t size = data.getSize();

    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        dataFlat[i] /= MAX_GREYSCALE_VALUE;
    }
}

void Greyscale::reverseTransform(Tensor &data) const {
    size_t size = data.getSize();
    vector<double> &dataFlat = data.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        dataFlat[i] *= MAX_GREYSCALE_VALUE;
        
    }
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

void Greyscale::transform(vector<double> &data) {
    throwDataFormatError();
}

void Greyscale::reverseTransform(vector<double> &data) const {
    throwDataFormatError();
}

uint32_t Greyscale::getEncoding() const {
    return Scalar::Encodings::Greyscale;
}