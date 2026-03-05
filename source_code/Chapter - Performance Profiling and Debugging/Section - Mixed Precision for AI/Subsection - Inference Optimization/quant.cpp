#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

// Quantize FP32 weights to INT8 per output channel; returns quantized data and per-channel scales.
std::vector<int8_t> quantize_weights_per_channel(const std::vector<float>& weights,
                                                 int OC,
                                                 int IC,
                                                 int KH,
                                                 int KW,
                                                 std::vector<float>& scales) {
    const int channel_size = IC * KH * KW;
    scales.resize(OC);
    std::vector<int8_t> qweights(weights.size());

    for (int oc = 0; oc < OC; ++oc) {
        const float* chan_start = weights.data() + oc * channel_size;
        const float* chan_end   = chan_start + channel_size;

        float max_abs = *std::max_element(chan_start, chan_end,
                                          [](float a, float b) { return std::fabs(a) < std::fabs(b); });
        max_abs = std::fabs(max_abs);

        const float scale = (max_abs > 0.f) ? (127.f / max_abs) : 1.f;
        scales[oc]        = 1.f / scale;

        std::transform(chan_start, chan_end, &qweights[oc * channel_size],
                       [scale](float w) -> int8_t {
                           float q = std::round(w * scale);
                           q = std::max(-128.f, std::min(127.f, q));
                           return static_cast<int8_t>(q);
                       });
    }
    return qweights;
}

int main() {
    constexpr int OC = 64, IC = 3, KH = 3, KW = 3;
    std::vector<float> weights(OC * IC * KH * KW, 0.01f);
    std::vector<float> scales;
    auto q = quantize_weights_per_channel(weights, OC, IC, KH, KW, scales);
    std::cout << "Quantized " << q.size() << " weights, scales[0]=" << scales[0] << '\n';
}