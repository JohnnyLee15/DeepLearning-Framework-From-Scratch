#pragma once
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

class TargetEncoder {
    public:
        static vector<double> getClassificationTarget(const vector<string>&, const unordered_map<string, int>&);
        static vector<double> getRegressionTarget(const vector<string>&);
        static unordered_map<string, int> createLabelMap(const vector<string>&);
};