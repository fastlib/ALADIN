#pragma once

#include "common.h"
#include "helpers.h"

using namespace std;

class PWaveProcessor {
public:
    explicit PWaveProcessor(bool debug = false) : debug_(debug) {
        tools = Tools();
    }

    /**
     * Top‑level entry – mirrors the Python signature.
     * The implementation is a literal line‑by‑line translation of the Python
     * reference, with vector<float> instead of NumPy ndarrays.
     */
    PWaveResult get_avg_p_wave(std::vector<std::shared_ptr<QRS>> &beats, std::vector<std::shared_ptr<P>> &pwaves, float fs, float qrs_median_range);

private:
    // === Basic helpers =====================================================
    static float mean(std::vector<float> &v) {
        return v.empty() ? 0.f : std::accumulate(v.begin(), v.end(), 0.f) / v.size();
    }

    static float l2norm(std::vector<float> &v) {
        float s = 0.f; for (auto x : v) s += x * x; return std::sqrt(s);
    }
    static float l2norm(std::shared_ptr<P> &v) {
        return l2norm(v->ecg);
    }

    static void pad_front(std::vector<float> &v, size_t count, float val = 0.f) {
        v.insert(v.begin(), count, val);
    }

    static void pad_back(std::vector<float> &v, size_t count, float val = 0.f) {
        v.insert(v.end(), count, val);
    }

    // Shift wave left (negative lag) or right (positive lag). Equivalent to the
    // Python shift_wave().
    static std::vector<float> shift_wave(std::vector<float> wave, int lag) {
        if (lag > 0) {
            pad_front(wave, static_cast<size_t>(lag));
        } else if (lag < 0) {
            size_t k = static_cast<size_t>(-lag);
            if (k >= wave.size()) {
                return {}; // all removed
            }
            wave.erase(wave.begin(), wave.begin() + k);
        }
        return wave;
    }
    static std::vector<float> shift_wave(std::shared_ptr<P> &wave, int lag) {
        return shift_wave(wave->ecg, lag);
    }

    // Match lengths by zero‑padding or truncation (Python make_equal_length()).
    static std::vector<float> make_equal_length(std::vector<float> src, std::vector<float> &dst) {
        if (src.size() < dst.size()) {
            pad_back(src, dst.size() - src.size());
        } else if (src.size() > dst.size()) {
            src.resize(dst.size());
        }
        return src;
    }
    static std::vector<float> make_equal_length(std::shared_ptr<P> &src, std::shared_ptr<P> &dst) {
        return make_equal_length(src->ecg, dst->ecg);
    }
    static std::vector<float> make_equal_length(std::shared_ptr<P> &src, std::vector<float> &dst) {
        return make_equal_length(src->ecg, dst);
    }
    static std::vector<float> make_equal_length(std::vector<float> &src, std::shared_ptr<P> &dst) {
        return make_equal_length(src, dst->ecg);
    }

    // // Element‑wise correlation coefficient for two equal‑length vectors.
    // static float corrcoef( std::vector<float> &a,  std::vector<float> &b) {
    //      size_t n = a.size();
    //     if (n == 0 || b.size() != n) return 0.f;

    //     float sum_a = 0.f, sum_b = 0.f, sum_aa = 0.f, sum_bb = 0.f, sum_ab = 0.f;
    //     for (size_t i = 0; i < n; ++i) {
    //         sum_a += a[i];
    //         sum_b += b[i];
    //         sum_aa += a[i] * a[i];
    //         sum_bb += b[i] * b[i];
    //         sum_ab += a[i] * b[i];
    //     }
    //     float num = n * sum_ab - sum_a * sum_b;
    //     float den = std::sqrt((n * sum_aa - sum_a * sum_a) * (n * sum_bb - sum_b * sum_b));
    //     return den == 0.f ? 0.f : num / den;
    // }

    static float minimum_mean_squared_error(const std::vector<float> &a, const std::vector<float> &b) {
        if (a.empty() || a.size() != b.size()) return std::numeric_limits<float>::infinity();
        std::vector<float> diff(a.size());
        std::transform(a.begin(), a.end(), b.begin(), diff.begin(), [](float x, float y) { return x - y; });
        float c = mean(diff); // optimal vertical shift
        float acc = 0.f;
        for (size_t i = 0; i < a.size(); ++i) {
            float d = a[i] - (b[i] + c);
            acc += d * d;
        }
        return std::sqrt(acc / a.size());
    }

    // Full cross‑correlation (same output length as NumPy correlate(..., "full")).
    static std::vector<float> cross_correlate( std::vector<float> &x,  std::vector<float> &y) {
         size_t nx = x.size(), ny = y.size();
         size_t n_out = nx + ny - 1;
        std::vector<float> out(n_out, 0.f);
        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                out[i + j] += x[i] * y[j];
            }
        }
        return out;
    }
    static std::vector<float> cross_correlate( std::shared_ptr<P> &x,  std::shared_ptr<P> &y) {
        return cross_correlate(x->ecg, y->ecg);
    }
    static std::vector<float> cross_correlate( std::shared_ptr<P> &x,  std::vector<float> &y) {
        return cross_correlate(x->ecg, y);
    }
    static std::vector<float> cross_correlate( std::vector<float> &x,  std::shared_ptr<P> &y) {
        return cross_correlate(x, y->ecg);
    }

    // Generate lag vector corresponding to cross_correlate full mode.
    static std::vector<int> correlation_lags(size_t len_x, size_t len_y) {
         size_t n_out = len_x + len_y - 1;
        std::vector<int> lags(n_out);
         int start = static_cast<int>(-static_cast<int>(len_y) + 1);
        for (size_t i = 0; i < n_out; ++i) lags[i] = start + static_cast<int>(i);
        return lags;
    }
    static std::vector<int> correlation_lags(std::shared_ptr<P> &x, std::shared_ptr<P> &y) {
        return correlation_lags(x->ecg.size(), y->ecg.size());
    }

    // RMSE with per‑signal normalisation used in original code.
    static float rmse_normalised( std::vector<float> &a,  std::vector<float> &b) {
         size_t n = a.size();
        if (n == 0 || n != b.size()) return std::numeric_limits<float>::infinity();
        float acc = 0.f;
        for (size_t i = 0; i < n; ++i) {
            float d = a[i] - b[i];
            acc += d * d;
        }
        return std::sqrt(acc / n);
    }
    static float rmse_normalised( std::shared_ptr<P> &a,  std::shared_ptr<P> &b) {
        return rmse_normalised(a->ecg, b->ecg);
    }
    static float rmse_normalised( std::shared_ptr<P> &a,  std::vector<float> &b) {
        return rmse_normalised(a->ecg, b);
    }
    static float rmse_normalised( std::vector<float> &a,  std::shared_ptr<P> &b) {
        return rmse_normalised(a, b->ecg);
    }

    // === Alignment / grouping helpers ======================================
    // These translate the recursive Python routines in a naïve fashion. They
    // are long but conceptually identical.

    // Find best pair among unaligned waves.
    std::tuple<std::pair<int, int>, int, float>
    find_best_pair(std::vector<std::shared_ptr<P>> &waves,
                    std::vector<int> &already_aligned) ;

    // Recursively align waves belonging to the same cluster (group).
    std::pair<std::vector<float>, std::vector<int>>
    recursively_align_waves(std::vector<std::shared_ptr<P>> &waves,
                            std::vector<int> &already_aligned,
                            std::vector<float> avg_wave,
                            int iteration,
                            int group) ;

    // Recursively split into groups and align each.
    void recursively_group_and_align_waves(std::vector<std::shared_ptr<P>> &waves,
                                           std::vector<int> &already_aligned,
                                           std::vector<std::vector<float>> &avg_waves,
                                           std::vector<std::vector<int>> &groups,
                                           int iteration) ;

    // Remove linear trend from a wave (simple detrend).
    static void detrend(std::vector<float> &wave) {
        if (wave.size() < 2) return;
        float m = (wave.back() - wave.front()) / static_cast<float>(wave.size() - 1);
        float b = wave.front();
        for (size_t i = 0; i < wave.size(); ++i) {
            wave[i] -= (m * static_cast<float>(i) + b);
        }
    }

    bool debug_;
    Tools tools;
};