#include <utils/FeatureEncoder.h>
#include <utils/ConsoleUtils.h>
#include "core/tensor/Tensor.h"
#include <stdexcept>

bool FeatureEncoder::getValueType(const string &value) {
    try {
        stod(value);
        return false;
    } catch(const invalid_argument &ia) {
        return true;
    } catch(const out_of_range &oof) {
        return true;
    }
}

vector<bool> FeatureEncoder::getCategoricalCols(
    const vector<vector<string> > &featuresRaw
) {
    size_t numRows = featuresRaw.size();
    size_t numCols = featuresRaw[0].size();
    vector<bool> isCategorical(numCols, false);

    for (size_t j = 0; j < numCols; j++)  {
        for (size_t i = 0; i < numRows && !isCategorical[j]; i++) {
            isCategorical[j] = getValueType(featuresRaw[i][j]);
        }
    }
    
    return isCategorical;
}

unordered_map<string, float> FeatureEncoder::encodeFeature(
    const vector<vector<string> > &featuresRaw,
    size_t colIdx
) {
    size_t numRows = featuresRaw.size();
    unordered_map<string, float> encoding;
    int nextIdx = 0;

    for (size_t i = 0; i < numRows; i++) {
        const string &val = featuresRaw[i][colIdx];
        if (encoding.find(val) == encoding.end()) {
            encoding[val] = (float) nextIdx++;
        }
    }

    return encoding;
}

vector<unordered_map<string, float> > FeatureEncoder::encodeFeatures(
    const vector<vector<string> > &featuresRaw,
    const vector<bool> &isCategorical
) {
    size_t numCols = featuresRaw[0].size();
    vector<unordered_map<string, float> > encodings(numCols);

    #pragma omp parallel for
    for (size_t j = 0; j < numCols; j++)  {
        if (isCategorical[j]) {
            encodings[j] = encodeFeature(featuresRaw, j);
        }
    }

    return encodings;
}

vector<size_t> FeatureEncoder::getOffsets(
    const vector<bool> &isCategorical,
    const vector<unordered_map<string, float> > &encodings,
    size_t numCols,
    size_t &totalCols
) {
    vector<size_t> offsets(numCols);
    for (size_t i = 0; i < numCols; i++) {
        offsets[i] = totalCols;
        if (isCategorical[i]) {
            totalCols += encodings[i].size();
        } else {
            totalCols++;
        }
    }

    return offsets;
}

Tensor FeatureEncoder::getFeatures(
    const vector<bool> &isCategorical,
    const vector<unordered_map<string, float> > &encodings,
    const vector<vector<string> > &featuresRaw
) {
    size_t numRows = featuresRaw.size();
    size_t numCols = featuresRaw[0].size();

    size_t totalCols = 0;
    vector<size_t> offsets = getOffsets(isCategorical, encodings, numCols, totalCols);
    Tensor features({numRows, totalCols});
    vector<float> &featuresFlat = features.getFlat();

    #pragma omp parallel for
    for (size_t i = 0; i < numRows; i++)  {
        for (size_t j = 0; j < numCols; j++) {
            size_t offset = offsets[j];
            if (isCategorical[j]) {
                unordered_map<string, float>::const_iterator it = encodings[j].find(featuresRaw[i][j]);
                if (it != encodings[j].end()) {
                    size_t catIdx = it->second;
                    featuresFlat[i * totalCols + offset + catIdx] = 1;
                }
            } else {
                featuresFlat[i * totalCols + offset] = stod(featuresRaw[i][j]);
            }
        }
    }

    return features;
}