#pragma once

#include <vector>
#include <string>
#include <unordered_map>

class Tensor;

using namespace std;

class FeatureEncoder {
    private:
        // Methods
        static bool getValueType(const string&);
        static unordered_map<string, float> encodeFeature(const vector<vector<string> >&, size_t);

    public:
        // Methods
        static vector<bool> getCategoricalCols(const vector<vector<string> >&);
        static vector<size_t> getOffsets(const vector<bool>&, const vector<unordered_map<string, float> >&, size_t, size_t&);
        static vector<unordered_map<string, float> > encodeFeatures(const vector<vector<string> >&, const vector<bool>&);
        static Tensor getFeatures(
            const vector<bool>&, 
            const vector<unordered_map<string, float> >&, 
            const vector<vector<string> >&
        );
};