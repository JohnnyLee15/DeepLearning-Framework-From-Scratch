#include "utils/Scalar.h"

class Greyscale : public Scalar {
    private:
        // Constants
        static const float MAX_GREYSCALE_VALUE;
        void throwDataFormatError() const;
    
    public:
        // Methods
        void fit(const vector<float>&) override;

        Tensor transform(const Tensor&) const override;
        Tensor reverseTransform(const Tensor&) const override; 

        vector<float> transform(const vector<float>&)const override;
        vector<float> reverseTransform(const vector<float>&) const override;

        uint32_t getEncoding() const override;

        Scalar* clone() const override;
};

