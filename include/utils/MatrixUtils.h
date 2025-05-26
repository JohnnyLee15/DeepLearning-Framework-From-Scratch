#include <vector>

using namespace std;

class MatrixUtils {
    public:
        // CONSTANTS
        static const double INF;

        // Methods
        static vector<double> multMatVec(const vector<vector<double> >&, const vector<double>&);
        static double dot(const vector<double>&, const vector<double>&);
        static void addVecInplace(vector<double>&, const vector<double>&);
        static void addMatInplace(vector<vector<double> >&, const vector<vector<double> >&);
        static void scaleMatInplace(vector<vector<double> >&, double);
        static vector<vector<double> > multMatTMat(const vector<vector<double> >&, const vector<vector<double> >&);
        static vector<vector<double> > multMatMatT(const vector<vector<double> >&, const vector<vector<double> >&);
        static void scaleVecInplace(vector<double>&, double);
        static vector<double> colSums(const vector<vector<double> >&);
        static vector<vector<double> > multMatMat(const vector<vector<double> >&, const vector<vector<double> >&);
        static void hardamardInplace(vector<vector<double> >&, const vector<vector<double> >&);

};