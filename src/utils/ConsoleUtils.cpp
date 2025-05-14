#include "utils/ConsoleUtils.h"
#include <iostream>
#include <iomanip>

using namespace std;

const int ConsoleUtils::PROGRESS_BAR_LENGTH = 50;

void ConsoleUtils::reportEpochProgress(int epoch, int numEpochs, double avgLoss, double accuracy) {
    cout << endl << "Average Loss: " << avgLoss << endl;
    cout << "Accuracy: " << fixed << setprecision(2) << (100 * accuracy) << "%" << endl;
    cout << defaultfloat << setprecision(6);
}

void ConsoleUtils::printProgressBar(int currentSample, int totalSamples) {
    double progress = (double) currentSample / totalSamples;
    int progressChar = (int) (progress * PROGRESS_BAR_LENGTH);

    const string GREEN = "\033[32m";
    const string RESET = "\033[0m";

    cout << "|";
    for (int i = 0; i < PROGRESS_BAR_LENGTH; i++) {
        if (i <= progressChar) {
            cout << GREEN << "█" << RESET;
        } else {
            cout << " ";
        }
    }

    cout << "| " << fixed << setprecision(2) << (progress * 100) <<"%\r";
    cout << defaultfloat << setprecision(6);
    cout.flush();
}