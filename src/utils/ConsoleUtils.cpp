#include "utils/ConsoleUtils.h"
#include <iostream>
#include <iomanip>

const int ConsoleUtils::PROGRESS_BAR_LENGTH = 50;
const string ConsoleUtils::GREEN = "\033[32m";
const string ConsoleUtils::RESET_COLOUR = "\033[0m";
const string ConsoleUtils::FILLED = "█";
const string ConsoleUtils::EMPTY = "░";

void ConsoleUtils::printProgressBar(int currentSample, int totalSamples, double accuracy, double avgLoss, double timeElapsed) {
    double progress = (double) currentSample / totalSamples;
    int progressChar = (int) (progress * PROGRESS_BAR_LENGTH);
    int sampleWidth = to_string(totalSamples).length();

    cout << setw(sampleWidth) << currentSample << "/" << totalSamples << " |";
    for (int i = 0; i < PROGRESS_BAR_LENGTH; i++) {
        if (i <= progressChar) {
            cout << GREEN << FILLED << RESET_COLOUR;
        } else {
            cout << EMPTY;
        }
    }

    cout << "| Accuracy: " << fixed << setprecision(2) << (accuracy) << "% | Avg Loss: " << avgLoss << " | ETA: " << timeElapsed  <<"s\r";
    cout << defaultfloat << setprecision(6);

    if (currentSample == totalSamples) {
        cout << endl;
    }

    cout.flush();
}