#pragma once
#include <string>

class NeuralNet;
class Loss;
class Activation;
class Layer;

using namespace std;

class BinUtils {
    private:
        // Constants
        static const char CANCEL;
        static const char OVERRIDE;
        static const char RENAME;

        // Methods
        static bool fileExists(string, bool);
        static char getUserChoice();
        static void writeToBin(const NeuralNet&, const string&);
        static string getNewModelName();
        static void printOptions();
        static Loss* loadLoss(ifstream&);
        static Layer* loadLayer(ifstream&);
        static Activation* loadActivation(ifstream&);

    public:
        // Methods
        static void saveModel(const NeuralNet&, const string&);
        static NeuralNet loadModel(const string&);
};