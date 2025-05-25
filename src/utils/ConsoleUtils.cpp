#include "utils/ConsoleUtils.h"
#include <iostream>
#include <iomanip>

using namespace std;

const int ConsoleUtils::PROGRESS_BAR_LENGTH = 50;

void ConsoleUtils::printProgressBar(int currentSample, int totalSamples, double accuracy, double avgLoss, double timeElapsed) {
    double progress = (double) currentSample / totalSamples;
    int progressChar = (int) (progress * PROGRESS_BAR_LENGTH);
    int sampleWidth = to_string(totalSamples).length();

    const string GREEN = "\033[32m";
    const string RESET = "\033[0m";

    cout << setw(sampleWidth) << currentSample << "/" << totalSamples << " |";
    for (int i = 0; i < PROGRESS_BAR_LENGTH; i++) {
        if (i <= progressChar) {
            cout << GREEN << "█" << RESET;
        } else {
            cout << " ";
        }
    }

    cout << "| Accuracy: " << fixed << setprecision(2) << (accuracy) << "% | Avg Loss: " << avgLoss << " | ETA: " << timeElapsed  <<"s\r";
    cout << defaultfloat << setprecision(6);

    if (currentSample == totalSamples) {
        cout << endl;
    }

    cout.flush();
}