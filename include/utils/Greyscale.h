#include "utils/Scalar.h"

class Greyscale : public Scalar {
    private:
        // Constants
        static const double MAX_GREYSCALE_VALUE;
        void throwDataFormatError() const;
    
    public:
        // Methods
        void fit(const vector<double>&) override;

        Tensor transform(const Tensor&) const override;
        Tensor reverseTransform(const Tensor&) const override; 

        vector<double> transform(const vector<double>&)const override;
        vector<double> reverseTransform(const vector<double>&) const override;

        uint32_t getEncoding() const override;
};

