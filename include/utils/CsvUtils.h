#pragma once

#include <vector>
#include <string>

using namespace std;

class CsvUtils {
    private:
        // Methods
        static void parseLine(const string&, vector<string>&, string&, int);
        
    public:
        // Methods
        static vector<string> collectLines(const string&, bool);
        static void parseLines(const vector<string>&, vector<vector<string> >&, vector<string>&, int);
        static vector<string> readHeader(const string&);
        static string trim(const string&);
        static string toLowerCase(const string&);
        static void checkFile(const string&);
        static size_t countFirstCol(const string&);
};