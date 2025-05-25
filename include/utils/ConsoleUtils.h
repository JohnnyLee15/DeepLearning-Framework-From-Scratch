#pragma once

class ConsoleUtils {
    private:
        static const int PROGRESS_BAR_LENGTH;

    public:
        static void printProgressBar(int, int, double, double, double);
};