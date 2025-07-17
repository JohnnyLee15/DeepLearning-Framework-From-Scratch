#include "core/data/Data.h"

void Data::writeBin(ofstream &modelBin) const {
    uint32_t dataEncoding = getEncoding();
    modelBin.write((char*) &dataEncoding, sizeof(uint32_t));
}