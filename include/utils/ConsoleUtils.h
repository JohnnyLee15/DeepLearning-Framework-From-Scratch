#pragma once

#include <string>
#include <atomic>
#include <thread>

class ProgressMetric;

using namespace std;

class ConsoleUtils {
    private:
        // Constants
        static const int PROGRESS_BAR_LENGTH;
        static const string GREEN;
        static const string CYAN;
        static const string YELLOW;
        static const string RED;
        static const string RESET_COLOUR;
        static const string FILLED;
        static const string EMPTY;
        static const string COMPLETE;
        static const string CROSS;
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
        static void printProgressBar(ProgressMetric&);
        static void printValidationMetrics(ProgressMetric&);
        static void loadMessage(const string&);
        static void completeMessage();
        static void printTitle();
        static void printSepLine();
        static void printSuccess(const string&, bool newLine = false);
        static void printWarning(const string&);
        static void printError(const string&);

        [[noreturn]]  
        static void fatalError(const string&);
        
};