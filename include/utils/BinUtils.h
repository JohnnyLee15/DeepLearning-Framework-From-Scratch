#pragma once
#include <string>

class Pipeline;

using namespace std;

class BinUtils {
    private:
        // Constants
        static const char CANCEL;
        static const char OVERRIDE;
        static const char RENAME;
        static const string MODEL_EXTENSION;
        
        // Methods
        static bool fileExists(string, bool);
        static char getUserChoice();
        static void writeToBin(const Pipeline&, const string&);
        static string getNewModelName();
        static void printOptions();
        static string addExtension(const string&);

    public:
        // Methods
        static void savePipeline(const Pipeline&, const string&);
        static Pipeline loadPipeline(const string&);
};