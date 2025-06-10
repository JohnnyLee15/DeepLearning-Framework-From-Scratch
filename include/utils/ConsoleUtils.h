#pragma once

#include <string>
#include <atomic>
#include <thread>
#include "utils/EpochStats.h"

using namespace std;

class ConsoleUtils {
    private:
        // Constants
        static const int PROGRESS_BAR_LENGTH;
        static const string GREEN;
        static const string CYAN ;
        static const string RESET_COLOUR;
        static const string FILLED;
        static const string EMPTY;
        static const string COMPLETE;
        static const char SPINNER_CHARS[];
        static const size_t WIDTH;
        static const string TITLE_LINE;
        static const string SEP_LINE;

        // Static Variables
        static atomic<bool> spinnerRunning; 
        static string currentLoadMessage;
        static thread spinnerThread;

        // Methods
        static void runSpinner();
        static string centerText(const string&, size_t);

    public:
        // Methods
        static void printProgressBar(EpochStats &stats);
        static void loadMessage(const string&);
        static void completeMessage();
        static void printTitle();
        static void printSepLine();
        
};