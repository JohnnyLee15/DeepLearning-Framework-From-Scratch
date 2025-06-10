#pragma once

#include <vector>

using namespace std;

class VectorUtils {
    public:
        // Constants
        static const double INF;

        // Methods
        static void addVecInplace(vector<double>&, const vector<double>&);
        static void scaleVecInplace(vector<double>&, double);
    
};