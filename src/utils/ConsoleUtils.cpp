#include "utils/ConsoleUtils.h"
#include <iostream>
#include <iomanip>
#include "core/ProgressMetric.h"

const int ConsoleUtils::PROGRESS_BAR_LENGTH = 50;
const string ConsoleUtils::GREEN = "\033[32m";
const string ConsoleUtils::CYAN = "\033[36m";
const string ConsoleUtils::YELLOW = "\033[33m";
const string ConsoleUtils::RED = "\033[31m";
const string ConsoleUtils::RESET_COLOUR = "\033[0m";
const string ConsoleUtils::FILLED = "█";
const string ConsoleUtils::EMPTY = "░";
const string ConsoleUtils::COMPLETE = "✔";
const string ConsoleUtils::CROSS = "✘";
const char ConsoleUtils::SPINNER_CHARS[] = {'\\', '|', '/', '-'};
const size_t ConsoleUtils::WIDTH = 60;
const string ConsoleUtils::TITLE_LINE = string(WIDTH, '=');
const string ConsoleUtils::SEP_LINE = string(WIDTH, '-');

string ConsoleUtils::currentLoadMessage = "";
atomic<bool> ConsoleUtils::spinnerRunning = false;
thread ConsoleUtils::spinnerThread;

void ConsoleUtils::printProgressBar(ProgressMetric &metric){
    double progress = (double) metric.getSamplesProcessed() / metric.getNumSamples();
    int progressChar = (int) (progress * PROGRESS_BAR_LENGTH);
    int sampleWidth = to_string(metric.getNumSamples()).length();

    cout << "\r\033[K" << setw(sampleWidth) << metric.getSamplesProcessed() << "/" << metric.getNumSamples() << " |";
    for (int i = 0; i < PROGRESS_BAR_LENGTH; i++) {
        if (i <= progressChar) {
            cout << GREEN << FILLED << RESET_COLOUR;
        } else {
            cout << EMPTY;
        }
    }

    cout << fixed << setprecision(2) << "| " << metric.getName() << ": " << metric.calculate() 
        << "%| Avg Loss: " << metric.getAvgLoss() << " | Elapsed: " << metric.getTimeElapsed() <<"s" 
        << defaultfloat << setprecision(6);

    if (metric.getSamplesProcessed()  == metric.getNumSamples()) {
        cout << endl;
    } else {
        cout << "\r";
    }

    cout.flush();
}

void ConsoleUtils::loadMessage(const string &message) {
    currentLoadMessage = message;
    spinnerRunning = true;
    spinnerThread = thread(runSpinner);
}

void ConsoleUtils::runSpinner() {
    int spinChar = 0;

    while(spinnerRunning) {
        cout << "\r\033[K" << "[" << CYAN << SPINNER_CHARS[spinChar] << RESET_COLOUR << "] " << currentLoadMessage << flush;
        spinChar = (spinChar + 1) % 4;

        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

void ConsoleUtils::completeMessage() {
    spinnerRunning = false;

    if (spinnerThread.joinable()) {
        spinnerThread.join();
    }

    cout << "\r\033[K" << "[" << GREEN << COMPLETE << RESET_COLOUR << "] " << currentLoadMessage << endl << flush;

    currentLoadMessage = "";
}

string ConsoleUtils::centerText(const string& text, size_t width) {
    if (text.length() >= width) {
        return text;
    } 

    size_t left = (width - text.length()) / 2;
    return string(left, ' ') + text;
}

void ConsoleUtils::printTitle() {
    cout << TITLE_LINE << endl;
    cout << centerText("🧠 MLP NEURAL NETWORK", WIDTH) << endl;
    cout << centerText("Lightweight C++ Neural Network", WIDTH) << endl;
    cout << TITLE_LINE << endl;
}

void ConsoleUtils::printSepLine() {
    cout << SEP_LINE << endl;
}

void ConsoleUtils::printSuccess(const string &message) {
    cout << "[" << GREEN << COMPLETE << RESET_COLOUR << "] " << message << endl;
}

void ConsoleUtils::printWarning(const string &message) {
    cout << "[" << YELLOW << "!" << RESET_COLOUR << "] " << message << endl;
}

void ConsoleUtils::printError(const string &message) {
    cout << "[" << RED << CROSS << RESET_COLOUR << "] " << message << endl;
}

void ConsoleUtils::fatalError(const string &message) {
    cerr << "Fatal Error: " << message << " Exiting." << endl;
    exit(1);
}