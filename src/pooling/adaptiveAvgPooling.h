#ifndef ADAPTIVE_AVG_POOLING_H
#define ADAPTIVE_AVG_POOLING_H

#include <vector>
#include "he/he.h"

class AdaptiveAvgPoolLayer {
public:
    AdaptiveAvgPoolLayer(CKKSPyfhel &he, std::pair<int, int> output_size);

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
    operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);

private:
    CKKSPyfhel &he_;
    std::pair<int, int> output_size_;

    std::vector<std::vector<seal::Ciphertext>>
    adaptive_avg(const std::vector<std::vector<seal::Ciphertext>> &image);
};

#endif // ADAPTIVE_AVG_POOLING_H
