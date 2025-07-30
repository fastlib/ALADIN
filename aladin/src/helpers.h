
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <deque>
#include <thread>
#include <cassert>
#include <complex.h>
#include <algorithm> // for std::sort
#include <numeric>
#include <stdexcept>
#include <random>

#include "iir.h"

using namespace std;

class ButterworthFilterBase {
    public:
        ButterworthFilterBase() = default;
        float filter(float input);
        void applyFilter(float* input_signal, int len);
        void filtfilt(float* input_signal, int len);
        void applyFilter(float* input_signal, int len, float* output_signal);
        void filtfilt(float* input_signal, int len, float* output_signal);

        std::vector<float> b;  // Numerator coefficients
        std::vector<float> a;  // Denominator coefficients
        std::vector<float> x_history;  // Input history (x[n], x[n-1], ...)
        std::vector<float> y_history;  // Output history (y[n], y[n-1], ...)
};

class LowPassFilter40Hz : public ButterworthFilterBase {
public:
    LowPassFilter40Hz(float fs) : ButterworthFilterBase() {

        calculateCoefficients(40.0, fs, 4);  // Initialize coefficients for 40 Hz low-pass filter

        // Initialize history buffers for the input and output
        x_history = std::vector<float>(b.size(), 0.0);  // Input history (x[n], x[n-1], ...)
        y_history = std::vector<float>(a.size(), 0.0);  // Output history (y[n], y[n-1], ...)
    }

    void calculateCoefficients(float cutoff, float fs, int order) {

        // float wc = std::tan(M_PI * cutoff / fs);
        // float k1 = std::sqrt(2.0);
        // float k2 = 1.0;

        // float norm = pow(wc, 4.0) + 2.0 * k1 * pow(wc, 3.0) + 2.0 * (k2 + 1.0) * pow(wc, 2.0) + 2.0 * k1 * wc + 1.0;

        // b = {pow(wc, 4.0) / norm, 4.0 * pow(wc, 4.0) / norm, 6.0 * pow(wc, 4.0) / norm, 4.0 * pow(wc, 4.0) / norm, pow(wc, 4.0) / norm};

        // a = {
        //     1,
        //     (4 * (pow(wc, 4) - 1)) / norm,
        //     (6 * (pow(wc, 4) - 2 * pow(wc, 2) + 1)) / norm,
        //     (4 * (pow(wc, 4) - 3 * pow(wc, 2) + 1)) / norm,
        //     (pow(wc, 4) - 4 * pow(wc, 2) + 6) / norm};

        // // Normalize to a[0] = 1
        // for (size_t i = 0; i < b.size(); ++i)
        //     b[i] /= a[0];
        // for (size_t i = 1; i < a.size(); ++i)
        //     a[i] /= a[0];
        // a[0] = 1.0;
    }
};

class LowPassFilter30Hz : public ButterworthFilterBase {
public:
    LowPassFilter30Hz(float fs) : ButterworthFilterBase() {
        calculateCoefficients(30.0, fs, 4);  // Initialize coefficients for 40 Hz low-pass filter

        // Initialize history buffers for the input and output
        x_history = std::vector<float>(b.size(), 0.0);  // Input history (x[n], x[n-1], ...)
        y_history = std::vector<float>(a.size(), 0.0);  // Output history (y[n], y[n-1], ...)
    }

    
    void calculateCoefficients(float cutoff, float fs, int order) {

        // float wc = std::tan(M_PI * cutoff / fs);
        // float k1 = std::sqrt(2.0);
        // float k2 = 1.0;

        // float norm = pow(wc, 4) + 2 * k1 * pow(wc, 3) + 2 * (k2 + 1) * pow(wc, 2) + 2 * k1 * wc + 1;

        // b = {pow(wc, 4) / norm, 4 * pow(wc, 4) / norm, 6 * pow(wc, 4) / norm, 4 * pow(wc, 4) / norm, pow(wc, 4) / norm};

        // a = {
        //     1,
        //     (4 * (pow(wc, 4) - 1)) / norm,
        //     (6 * (pow(wc, 4) - 2 * pow(wc, 2) + 1)) / norm,
        //     (4 * (pow(wc, 4) - 3 * pow(wc, 2) + 1)) / norm,
        //     (pow(wc, 4) - 4 * pow(wc, 2) + 6) / norm};

        // // Normalize to a[0] = 1
        // for (size_t i = 0; i < b.size(); ++i)
        //     b[i] /= a[0];
        // for (size_t i = 1; i < a.size(); ++i)
        //     a[i] /= a[0];
        // a[0] = 1.0;
    }

};

class HighPassFilter10Hz : public ButterworthFilterBase {
public:
    HighPassFilter10Hz(float fs) : ButterworthFilterBase() {
        calculateCoefficients(10.0, fs, 4);  // Initialize coefficients for 40 Hz low-pass filter

        // Initialize history buffers for the input and output
        x_history = std::vector<float>(b.size(), 0.0);  // Input history (x[n], x[n-1], ...)
        y_history = std::vector<float>(a.size(), 0.0);  // Output history (y[n], y[n-1], ...)
    }

    void calculateCoefficients(float cutoff, float fs, int order) {
        b = {0.92117099, -3.68468397, 5.52702596, -3.68468397, 0.92117099};
        a = {1.0, -3.83582554, 5.52081914, -3.53353522, 0.848556};
    }

};

class Tools {
    public:
        static float max(float *a);
        static float min(float *a);
        static float mean(float *a);
        static float median(float *a);

        static float max(float *a, int size);
        static float min(float *a, int size);
        static float mean(float *a, int size);
        static float median(float *a, int size);
        static float stdev(float *a, int size);
        static float rmssd(float *a, int size);

        static float absmax(float *a, int size);
        static float absmin(float *a, int size);

        static int argmin(float *a, int size);
        static int argmax(float *a, int size);

        static int max(int *a, int size);
        static int min(int *a, int size);
        static int mean(int *a, int size);
        static int median(int *a, int size);
        static int stdev(int *a, int size);

        static int sum(bool *a, int size);

        static int argmin(int *a, int size);
        static int argmax(int *a, int size);

        static int pageHinkleyDetect(float *x, int size, int start, int end, float reference, float delta, float thresh, bool reverse);

        void butterworth_low_pass_filter(float *x, int size, float cutoffFreq, float sampleRate);
        void butterworth_low_pass_filter(float *x, int size, float cutoffFreq, float sampleRate, float *y);

        void butterworth_high_pass_filter(float *x, int size, float cutoffFreq, float sampleRate);
        void butterworth_high_pass_filter(float *x, int size, float cutoffFreq, float sampleRate, float *y);

        void median_filter(float *x, int size, int window_size);
        void median_filter(float *x, int size, int window_size, float *y);
        void remove_baseline(float *x, int size, float fs);
        void normalize_zscore(float *x, int size);

        void iir_filter(float *x, int size, float *b, int b_size);
        void filtfilt_highpass(float *x, int size, float cutoffFreq, float sampleRate);
        void filtfilt_lowpass(float *x, int size, float cutoffFreq, float sampleRate);

        vector<int> find_initial_peaks(float *x, int size, float threshold);
        float compute_prominence(float *x, int size, int peak_idx);
        vector<int> find_peaks(float *x, int size, float threshold, int distance, float min_prominence);

        static float dtwPath(vector<float> &x, vector<float> &y, vector<pair<int, int>> &path, vector<float>& aligned_x, vector<float>& aligned_y);
        static float ddtwPath(vector<float> &x, vector<float> &y, vector<pair<int, int>> &path, vector<float>& aligned_x, vector<float>& aligned_y);

        void erosioncentered(float *x, int size, int kernel_size, float *y);
        void dilationcentered(float *x, int size, int kernel_size, float *y);
        void openingcentered(float *x, int size, int kernel_size, float *y);
        void closingcentered(float *x, int size, int kernel_size, float *y);

        void erosioncentered(bool *x, int size, int kernel_size, bool *y);
        void dilationcentered(bool *x, int size, int kernel_size, bool *y);
        void openingcentered(bool *x, int size, int kernel_size, bool *y);
        void closingcentered(bool *x, int size, int kernel_size, bool *y);

        void erosioncentered(std::vector<bool> &x, int kernel_size);
        void dilationcentered(std::vector<bool> &x, int kernel_size);
        void openingcentered(std::vector<bool> &x, int kernel_size);
        void closingcentered(std::vector<bool> &x, int kernel_size);

        static vector<tuple<int,int>> get_regions(bool *x, int size);
        static vector<tuple<int,int>> get_regions(std::vector<bool> &x);
        float pearson_correlation(const std::vector<float>& x, const std::vector<float>& y);
        
        // std::vector<float> Tools::interpolate(const std::vector<float>& old_indices, const std::vector<float>& new_indices, const std::vector<float>& signal);
        // std::vector<float> resize_signal(const std::vector<float>& signal, size_t new_length = 128);

        static float snr_log(std::vector<float> fore, std::vector<float> back);
        static float snr_lin(std::vector<float> fore, std::vector<float> back);

        static float cosen(std::vector<float>& ibi_series);
        static float cv(std::vector<float>& ibi_series);

        float check_hypothesis(float prediction, float actual, float std_dev, bool two_sided);

};

template <typename T>
class SlidingBuffer {
public:
    SlidingBuffer(size_t maxSize) : maxSize(maxSize) {}

    void push_back(const T& value) {
        if (buffer.size() >= maxSize) {
            buffer.pop_front();  // Remove the oldest element
        }
        buffer.push_back(value);  // Add the new element
    }

    void print() const {
        for (const auto& elem : buffer) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    T mean() const {
        if (buffer.empty()) return T();  // Return default value if empty
        T sum = std::accumulate(buffer.begin(), buffer.end(), T(0));
        return sum / static_cast<T>(buffer.size());
    }

    const std::deque<T>& getBuffer() const {
        return buffer;
    }

private:
    std::deque<T> buffer;
    size_t maxSize;
};