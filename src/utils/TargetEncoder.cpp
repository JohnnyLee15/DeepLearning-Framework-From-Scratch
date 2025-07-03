#include "utils/TargetEncoder.h"
#include "utils/ConsoleUtils.h"

vector<double> TargetEncoder::getRegressionTarget(
    const vector<string> &targetRaw
) {
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        try {
            target[i] = stod(targetRaw[i]);
        } catch (const invalid_argument &ia) {
            ConsoleUtils::fatalError(
                "Cannot parse non-numeric target: \"" + targetRaw[i] + "\"."
            );
        }
        
    }
    return target;
}

unordered_map<string, int> TargetEncoder::createLabelMap(
    const vector<string> &targetRaw
) {
    size_t numSamples = targetRaw.size();
    int nextIdx = 0;

    unordered_map<string, int> labelMap;
    for (size_t i = 0; i < numSamples; i++) {
        const string &val = targetRaw[i];
        if (labelMap.find(val) == labelMap.end()) {
            labelMap[val] = nextIdx++;
        }
    }

    return labelMap;
}

vector<double> TargetEncoder::getClassificationTarget(
    const vector<string> &targetRaw,
    const unordered_map<string, int> &labelMap
) {
    size_t numSamples = targetRaw.size();
    vector<double> target(numSamples);

    #pragma omp parallel for
    for (size_t i = 0; i < numSamples; i++) {
        if (labelMap.find(targetRaw[i]) == labelMap.end()) {
            ConsoleUtils::fatalError(
                "Unknown label \"" + targetRaw[i] + "\' encountered."
            );
        }

        target[i] = (double) labelMap.at(targetRaw[i]);
    }

    return target;
}
