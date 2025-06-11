#include "utils/Scalar.h"

class Greyscale : public Scalar {
    private:
        // Constants
        static const double MAX_GREYSCALE_VALUE;
        void throwDataFormatError() const;
    
    public:
        // Methods
        void transform(Matrix&) override;
        void reverseTransform(Matrix&) const override; 
        void fit(const vector<double>&) override;
        void transform(vector<double>&) override;
        void reverseTransform(vector<double>&) const override;
};

