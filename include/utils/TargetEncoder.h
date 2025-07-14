#pragma once
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

class TargetEncoder {
    public:
        static vector<float> getClassificationTarget(const vector<string>&, const unordered_map<string, int>&);
        static vector<float> getRegressionTarget(const vector<string>&);
        static unordered_map<string, int> createLabelMap(const vector<string>&);
};