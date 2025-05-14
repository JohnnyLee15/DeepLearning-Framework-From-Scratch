#pragma once

class ConsoleUtils {
    private:
        static const int PROGRESS_BAR_LENGTH;

    public:
        static void reportEpochProgress(int, int, double, double);
        static void printProgressBar(int, int);
};