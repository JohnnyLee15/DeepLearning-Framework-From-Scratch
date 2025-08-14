#include "utils/EarlyStop.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <ctime>
#include "utils/BinUtils.h"
#include <filesystem>
#include <limits>

using namespace std::chrono;
namespace fs = std::filesystem;

EarlyStop::EarlyStop(
    Pipeline *pipe, 
    size_t patience,
    float minDelta,
    size_t warmupEpochs
) : patience(patience), minDelta(minDelta), warmupEpochs(warmupEpochs),
    pipe(pipe), badEpochs(0), bestLoss(numeric_limits<float>::max()) 
{}

bool EarlyStop::shouldStop(float valLoss, size_t epoch) {
    if (epoch < warmupEpochs || pipe == nullptr) 
        return false;

    if (valLoss < bestLoss - minDelta) {
        bestLoss = valLoss;
        badEpochs = 0;
        saveBestPipe();
    } else {
        badEpochs++;
    }
    
    return badEpochs >= patience;
}

void EarlyStop::saveBestPipe() {
    if (!bestPipePath.empty() && fs::exists(bestPipePath)) {
        fs::remove(bestPipePath);
    }
        
    auto currTime = system_clock::now();
    auto currSeconds = system_clock::to_time_t(currTime);
    auto currMs = duration_cast<milliseconds>(currTime.time_since_epoch()) % 1000;

    tm tmCopy;
    localtime_r(&currSeconds, &tmCopy);

    ostringstream oss;
    oss << "./"
        << "best_weights_" << getpid() << "_" 
        << put_time(&tmCopy, "%Y%m%d_%H%M%S")
        << "_" << setfill('0') << setw(3) << currMs.count()
        << ".nn_tmp";
    
    bestPipePath = oss.str();
    BinUtils::writeToBin(*pipe, bestPipePath);
    pipe->setBestModelPath(bestPipePath);
}