#include "utils/VectorUtils.h"
#include <cassert>
#include <omp.h>

const double VectorUtils::INF = 1e308;

void VectorUtils::addVecInplace(
    vector<double> &vec1,
    const vector<double> &vec2
) {
    assert(vec1.size() == vec2.size());

    size_t size = vec1.size();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        vec1[i] += vec2[i];
    }
}

void VectorUtils::scaleVecInplace(
    vector<double> &vec,
    double scaleFactor
) {
    size_t size = vec.size();

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        vec[i] *= scaleFactor;
    }
}