#include "losses/Loss.h"

double Loss::formatLoss(double avgLoss) const {
    return avgLoss;
}

bool Loss::isFused() const {
    return false;
}