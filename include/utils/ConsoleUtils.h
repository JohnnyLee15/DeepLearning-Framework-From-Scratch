#pragma once

#include <string>

using namespace std;

class ConsoleUtils {
    private:
        static const int PROGRESS_BAR_LENGTH;
        static const string GREEN;
        static const string RESET_COLOUR;
        static const string FILLED;
        static const string EMPTY;

    public:
        static void printProgressBar(int, int, double, double, double);
};