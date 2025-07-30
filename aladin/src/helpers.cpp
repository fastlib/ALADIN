
#include "helpers.h"


float ButterworthFilterBase::filter(float input) {
    // Shift the input history
    for (size_t i = b.size() - 1; i > 0; --i) {
        x_history[i] = x_history[i - 1];
    }
    x_history[0] = input;

    // Calculate the output
    float output = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        output += b[i] * x_history[i];
    }
    
    // Shift the output history
    for (size_t i = a.size() - 1; i > 0; --i) {
        y_history[i] = y_history[i - 1];
    }
    
    for (size_t i = 0; i < a.size() - 1; ++i) {
        output -= a[i + 1] * y_history[i];
    }

    y_history[0] = output;

    return output;
}

void ButterworthFilterBase::applyFilter(float* input_signal, int len) {
    len = 10;
    // Apply the filter on each sample of the input signal
    float *tmp = new float[len];
    for (int i = 0; i < len; ++i) {
        input_signal[i] = filter(input_signal[i]);  // Update the input signal with the filtered output
        std::cout << input_signal[i] << " ";  // Debug: print input signal

        for (int i=0; i < x_history.size(); i++) {
            std::cout << "x_history[" << i << "]: " << x_history[i] << ", y_history[" << i << "]: " << y_history[i] << std::endl;
        }
    }
    std::cout << std::endl;  // Debug: end of input signal
}

void ButterworthFilterBase::filtfilt(float* input_signal, int len) {

    for (int i=0; i < a.size(); i++) {
        std::cout << "b[" << i << "]: " << b[i] << ", a[" << i << "]: " << a[i] << std::endl;
    }

    // Apply filter forward
    applyFilter(input_signal, len);

    // Reverse the signal
    for (int i = 0; i < len / 2; ++i) {
        std::swap(input_signal[i], input_signal[len - i - 1]);
    }

    // Apply filter again on the reversed signal
    applyFilter(input_signal, len);

    // Reverse the signal again to get the final zero-phase filtered result
    for (int i = 0; i < len / 2; ++i) {
        std::swap(input_signal[i], input_signal[len - i - 1]);
    }
}

void ButterworthFilterBase::applyFilter(float* input_signal, int len, float* output_signal) {
    // Apply the filter on each sample of the input signal and store in output_signal
    for (int i = 0; i < len; ++i) {
        output_signal[i] = filter(input_signal[i]);  // Update the output signal with the filtered output
    }
}

void ButterworthFilterBase::filtfilt(float* input_signal, int len, float* output_signal) {
    // Apply filter forward
    applyFilter(input_signal, len, output_signal);

    // Reverse the output signal
    for (int i = 0; i < len / 2; ++i) {
        std::swap(output_signal[i], output_signal[len - i - 1]);
    }

    // Apply filter again on the reversed signal
    applyFilter(output_signal, len, output_signal);

    // Reverse the output signal again to get the final zero-phase filtered result
    for (int i = 0; i < len / 2; ++i) {
        std::swap(output_signal[i], output_signal[len - i - 1]);
    }
}

float Tools::max(float *a, int size) {
    float max_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
    }
    return max_val;
}
float Tools::min(float *a, int size) {
    float min_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] < min_val) {
            min_val = a[i];
        }
    }
    return min_val;
}
float Tools::mean(float *a, int size) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += a[i];
    }
    return sum / size;
}
float Tools::median(float *a, int size) {
    vector<float> tmp(size);
    for (int i = 0; i < size; ++i) {
        tmp[i] = a[i];
    }
    std::sort(tmp.begin(), tmp.end());
    if (size % 2 == 0) {
        float res = (tmp[size/2 - 1] + tmp[size/2]) / 2;
        return res;
    } else {
        float res = tmp[size/2];
        return res;
    }
}
float Tools::stdev(float *a, int size) {
    float mean_val = mean(a, size);
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += (a[i] - mean_val) * (a[i] - mean_val);
    }
    return sqrt(sum / size);
}

float Tools::rmssd(float *a, int size) {
    if (size < 2) return 0.0f; // RMSSD is not defined for less than 2 points
    float sum = 0;
    for (int i = 1; i < size; ++i) {
        float diff = a[i] - a[i - 1];
        sum += diff * diff;
    }
    return sqrt(sum / (size - 1));
}

float Tools::max(float *a) {
    return max(a, sizeof(a)/sizeof(a[0]));
}
float Tools::min(float *a) {
    return min(a, sizeof(a)/sizeof(a[0]));
}
float Tools::mean(float *a) {
    return mean(a, sizeof(a)/sizeof(a[0]));
}
float Tools::median(float *a) {
    return median(a, sizeof(a)/sizeof(a[0]));
}

int Tools::max(int *a, int size) {
    int max_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
    }
    return max_val;
}
int Tools::min(int *a, int size) {
    int min_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] < min_val) {
            min_val = a[i];
        }
    }
    return min_val;
}
int Tools::mean(int *a, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += a[i];
    }
    return sum / size;
}
int Tools::median(int *a, int size) {
    vector<int> tmp(size);
    for (int i = 0; i < size; ++i) {
        tmp[i] = a[i];
    }
    std::sort(tmp.begin(), tmp.end());
    if (size % 2 == 0) {
        int res = (tmp[size/2 - 1] + tmp[size/2]) / 2;
        return res;
    } else {
        int res = tmp[size/2];
        return res;
    }
}
int Tools::stdev(int *a, int size) {
    float mean_val = mean(a, size);
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += (a[i] - mean_val) * (a[i] - mean_val);
    }
    return sqrt(sum / size);
}

int Tools::sum(bool *a, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += a[i];
    }
    return sum;
}

int Tools::argmin(float *a, int size) {
    int min_index = 0;
    float min_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] < min_val) {
            min_val = a[i];
            min_index = i;
        }
    }
    return min_index;
}
int Tools::argmax(float *a, int size) {
    int max_index = 0;
    float max_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
            max_index = i;
        }
    }
    return max_index;
}
int Tools::argmin(int *a, int size) {
    int min_index = 0;
    int min_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] < min_val) {
            min_val = a[i];
            min_index = i;
        }
    }
    return min_index;
}
int Tools::argmax(int *a, int size) {
    int max_index = 0;
    int max_val = a[0];
    for (int i = 1; i < size; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
            max_index = i;
        }
    }
    return max_index;
}

float Tools::absmax(float *a, int size) {
    float max_val = std::abs(a[0]);
    for (int i = 1; i < size; ++i) {
        if (std::abs(a[i]) > max_val) {
            max_val = std::abs(a[i]);
        }
    }
    return max_val;
}
float Tools::absmin(float *a, int size) {
    float min_val = std::abs(a[0]);
    for (int i = 1; i < size; ++i) {
        if (std::abs(a[i]) < min_val) {
            min_val = std::abs(a[i]);
        }
    }
    return min_val;
}

int Tools::pageHinkleyDetect(
    float *x,
    int size,
    int start,                   // inclusive
    int end,                     // inclusive
    float reference,             // e.g. local baseline mean
    float delta,                 // small positive drift term
    float thresh,                // threshold λ
    bool reverse = false
){
    float ph = 0.0f, ph_min = 0.0f;
    assert (start > 0 && end < size);

    if (!reverse) {
        for (int i = start; i <= end; ++i) {
            ph += std::abs(x[i] - reference) - delta;
            ph_min = std::min(ph_min, ph);
            if (ph - ph_min > thresh)
                return i;
        }
    } else {
        for (int i = start; i >= end; --i) {
            ph += std::abs(x[i] - reference) - delta;
            ph_min = std::min(ph_min, ph);
            if (ph - ph_min > thresh)
                return i;
        }
    }
    // if no changepoint found, return the far end
    return reverse ? end : start;
}

void Tools::butterworth_low_pass_filter(float *x, int size, float cutoffFreq, float sampleRate, float *y) {
    // Butterworth low-pass filter implementation
    // This function should implement the logic to apply a Butterworth low-pass filter
    // to the input signal x with the specified cutoff frequency and sample rate.
    // The filtered signal should be stored in y.
    const int order = 4;
    Iir::Butterworth::LowPass<order> filter;
    filter.setup(sampleRate, cutoffFreq);
    std::vector<float> temp(size);
    for (int i = 0; i < size; ++i) {
        temp[i] = filter.filter(x[i]);
    }
    for (int i = 0; i < size; ++i) {
        y[i] = temp[i];
    }
    // filter.reset();
    // for (int i = size - 1; i >= 0; --i) {
    //     y[i] = filter.filter(temp[i]);
    // }
}
void Tools::butterworth_low_pass_filter(float *x, int size, float cutoffFreq, float sampleRate) {
    // Butterworth low-pass filter implementation
    // This function should implement the logic to apply a Butterworth low-pass filter
    // to the input signal x with the specified cutoff frequency and sample rate.
    // The filtered signal should be stored back in x.
  
    const int order = 4;
    Iir::Butterworth::LowPass<order> filter;
    filter.setup(sampleRate, cutoffFreq);
    std::vector<float> temp(size);
    for (int i = 0; i < size; ++i) {
        temp[i] = filter.filter(x[i]);
    }
    for (int i = 0; i < size; ++i) {
        x[i] = temp[i];
    }
    // filter.reset();
    // for (int i = size - 1; i >= 0; --i) {
    //     x[i] = filter.filter(temp[i]);
    // }
}

void Tools::butterworth_high_pass_filter(float *x, int size, float cutoffFreq, float sampleRate, float *y) {
    // Butterworth high-pass filter implementation
    // This function should implement the logic to apply a Butterworth high-pass filter
    // to the input signal x with the specified cutoff frequency and sample rate.
    // The filtered signal should be stored in y.
    const int order = 4;
    Iir::Butterworth::HighPass<order> filter;
    filter.setup(sampleRate, cutoffFreq);
    for (int i = 0; i < size; ++i) {
        y[i] = filter.filter(x[i]);
    }
}
void Tools::butterworth_high_pass_filter(float *x, int size, float cutoffFreq, float sampleRate) {
    // Butterworth high-pass filter implementation
    // This function should implement the logic to apply a Butterworth high-pass filter
    // to the input signal x with the specified cutoff frequency and sample rate.
    // The filtered signal should be stored back in x.

    const int order = 4;
    Iir::Butterworth::HighPass<order> filter;
    filter.setup(sampleRate, cutoffFreq);
    for (int i = 0; i < size; ++i) {
        x[i] = filter.filter(x[i]);
    }
}

void Tools::median_filter(float *x, int size, int window_size) {
    // Median filter implementation
    // This function should implement the logic to apply a median filter
    // to the input signal x with the specified window size.
    // The filtered signal should be stored back in x.

    std::vector<float> temp(size);
    for (int i = 0; i < size; ++i) {
        int start = std::max(0, i - window_size / 2);
        int end = std::min(size - 1, i + window_size / 2);
        std::vector<float> window(x + start, x + end + 1);
        std::sort(window.begin(), window.end());
        temp[i] = window[window.size() / 2];
    }
    std::copy(temp.begin(), temp.end(), x);
}

void Tools::median_filter(float *x, int size, int window_size, float *y) {
    // Median filter implementation
    // This function should implement the logic to apply a median filter
    // to the input signal x with the specified window size.
    // The filtered signal should be stored in y.

    std::vector<float> temp(size);
    for (int i = 0; i < size; ++i) {
        int start = std::max(0, i - window_size / 2);
        int end = std::min(size - 1, i + window_size / 2);
        std::vector<float> window(x + start, x + end + 1);
        std::sort(window.begin(), window.end());
        temp[i] = window[window.size() / 2];
    }
    std::copy(temp.begin(), temp.end(), y);
}

void Tools::iir_filter(float *x, int size, float *coeffs, int b_size) {
    // IIR filter implementation
    // This function should implement the logic to apply an IIR filter
    // to the input signal x with the specified coefficients b and a.
    // The filtered signal should be stored in y.
    const int half_window = b_size / 2;

    float *tmp = new float[size];

    for (size_t i = half_window; i < size - half_window; ++i) {
        double acc = 0.0;
        for (int j = -half_window; j <= half_window; ++j) {
            acc += coeffs[j + half_window] * x[i + j];
        }
        tmp[i] = acc;
    }

    // Copy boundary values unchanged (or handle with padding logic if needed)
    for (size_t i = 0; i < half_window; ++i) {
        tmp[i] = x[i];
        tmp[size - 1 - i] = x[size - 1 - i];
    }

    // Copy the filtered values back to the original array
    for (size_t i = 0; i < size; ++i) {
        x[i] = tmp[i];
    }
    delete[] tmp;
}

void Tools::filtfilt_highpass(float *x, int size, float cutoffFreq, float sampleRate) {

    const int order = 4; // 4th order (=2 biquads)
    Iir::Butterworth::HighPass<order> f;
    f.setup(sampleRate, cutoffFreq);  // Initialize coefficients for 40 Hz low-pass filter

    vector<float> y(size, 0.0);  // Output history (y[n], y[n-1], ...)
    for (int i = 0; i < size; ++i) {
        y[i] = f.filter(x[i]);  // Update the input signal with the filtered output
    }

    memcpy(x, y.data(), size * sizeof(float));  // Copy the filtered output back to x

    // // Reverse the signal
    // for (int i = 0; i < size / 2; ++i) {
    //     std::swap(y[i], y[size - i - 1]);
    // }
    // f.reset();

    // for (int i = 0; i < size; ++i) {
    //     x[i] = f.filter(y[i]);  // Update the input signal with the filtered output
    // }
}

void Tools::filtfilt_lowpass(float *x, int size, float cutoffFreq, float sampleRate) {

    const int order = 4; // 4th order (=2 biquads)
    Iir::Butterworth::LowPass<order> f2;
    f2.setup(sampleRate, cutoffFreq);  // Initialize coefficients for 40 Hz low-pass filter

    vector<float> y(size, 0.0);  // Output history (y[n], y[n-1], ...)
    for (int i = 0; i < size; ++i) {
        y[i] = f2.filter(x[i]);  // Update the input signal with the filtered output
    }
    memcpy(x, y.data(), size * sizeof(float));  // Copy the filtered output back to x

    // // Reverse the signal
    // for (int i = 0; i < size / 2; ++i) {
    //     std::swap(y[i], y[size - i - 1]);
    // }
    // f.reset();

    // for (int i = 0; i < size; ++i) {
    //     x[i] = f.filter(y[i]);  // Update the input signal with the filtered output
    // }
}


vector<int> Tools::find_initial_peaks(float *x, int size, float threshold) {
    std::vector<int> peaks;
    for (int i = 1; i < size - 1; ++i) {
        if (x[i] > x[i - 1] && x[i] > x[i + 1]) {
            if (x[i] > threshold) {
                peaks.push_back(i);
            }
        }
    }
    return peaks;
}

// Compute prominence of a peak
float Tools::compute_prominence(float *x, int size, int peak_idx) {
    float peak_height = x[peak_idx];

    // Look left
    float left_min = peak_height;
    for (int i = peak_idx - 1; i >= 0; --i) {
        if (x[i] > peak_height) break;
        left_min = std::min(left_min, x[i]);
    }

    // Look right
    float right_min = peak_height;
    for (int i = peak_idx + 1; i < size; ++i) {
        if (x[i] > peak_height) break;
        right_min = std::min(right_min, x[i]);
    }

    float reference_level = std::max(left_min, right_min);
    return peak_height - reference_level;
}

vector<int> Tools::find_peaks(float *x, int size, float threshold, int distance = 1, float min_prominence = 0.0) {
    std::vector<int> peaks = find_initial_peaks(x, size, threshold);
    std::vector<int> filtered;

    // Filter by prominence
    for (int idx : peaks) {
        float prom = compute_prominence(x, size, idx);
        if (prom >= min_prominence) {
            filtered.push_back(idx);
        }
    }

    // Enforce minimum distance
    if (distance > 1 && filtered.size() > 1) {
        std::vector<int> final_peaks;
        final_peaks.push_back(filtered[0]);

        for (size_t i = 1; i < filtered.size(); ++i) {
            if (filtered[i] - final_peaks.back() >= distance) {
                final_peaks.push_back(filtered[i]);
            }
        }
        return final_peaks;
    }
    return filtered;
}

float Tools::dtwPath(vector<float> &x, vector<float> &y, vector<pair<int,int>> &path, std::vector<float>& aligned_x, std::vector<float>& aligned_y) {

    const float INF = numeric_limits<float>::infinity();
    const int d = 20;      // Sakoe-Chiba band
    const int l = 2;      // max consecutive insert/delete
    int rows = x.size() + 1;
    int cols = y.size() + 1;

    //std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

    // Allocate cost matrix (rows x cols)
    float* D = (float*)std::malloc(rows * cols * sizeof(float));
    int* slopeX = (int*)std::malloc(rows * cols * sizeof(int));
    int* slopeY = (int*)std::malloc(rows * cols * sizeof(int));

    // Initialize base cases
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            D[i*cols + j] = INF;
            slopeX[i*cols + j] = slopeY[i*cols + j] = 0;
        }
    }
    D[0] = 0.0f;

    for (int i = 1; i <= x.size(); ++i) {
        int jmin = std::max(1, i - d);
        int jmax = std::min((int)y.size(), i + d);
        for (int j = jmin; j <= jmax; ++j) {
            float cost = std::fabs(x[i-1] - y[j-1]);
            // match (diag)
            float best = D[(i-1)*cols + (j-1)];
            int sx = 0, sy = 0;
            // insertion (vertical)
            if (slopeY[(i-1)*cols + j] < l) {
                float up = D[(i-1)*cols + j];
                if (up < best) {
                    best = up;
                    sx = 0;
                    sy = slopeY[(i-1)*cols + j] + 1;
                }
            }
            // deletion (horizontal)
            if (slopeX[i*cols + (j-1)] < l) {
                float left = D[i*cols + (j-1)];
                if (left < best) {
                    best = left;
                    sx = slopeX[i*cols + (j-1)] + 1;
                    sy = 0;
                }
            }
            D[i*cols + j] = cost + best;
            slopeX[i*cols + j] = sx;
            slopeY[i*cols + j] = sy;
        }
    }

    // Backtrack to find path
    path.clear();
    int i = x.size(), j = y.size();
    while (i > 0 || j > 0) {
        path.emplace_back(i-1, j-1);
        int idx = i*cols + j;
        // choose predecessor
        int pi = i-1, pj = j-1;
        float diag = (i>0 && j>0) ? D[(i-1)*cols + (j-1)] : INF;
        float up   = (i>0)        ? D[(i-1)*cols + j] : INF;
        float left = (j>0)        ? D[i*cols + (j-1)] : INF;
        if (diag <= up && diag <= left) { --i; --j; }
        else if (up < left) { --i; }
        else { --j; }
    }
    std::reverse(path.begin(), path.end());

    // Build aligned signals
    int K = path.size();
    aligned_x.resize(K);
    aligned_y.resize(K);
    for (int k = 0; k < K; ++k) {
        int ix = path[k].first;
        int iy = path[k].second;
        aligned_x[k] = x[ix];
        aligned_y[k] = y[iy];
    }

    float distance = D[x.size()*cols + y.size()];
    std::free(D);
    std::free(slopeX);
    std::free(slopeY);

    return distance;
}

float Tools::ddtwPath(vector<float> &x, vector<float> &y, vector<pair<int,int>> &path, std::vector<float>& aligned_x, std::vector<float>& aligned_y) {

    vector<float> dx(x.size());
    vector<float> dy(y.size());

    for (int i = 0; i < x.size()-1; ++i) {
        dx[i] = x[i+1] - x[i];
    }
    for (int i = 0; i < y.size()-1; ++i) {
        dy[i] = y[i+1] - y[i];
    }

    return dtwPath(dx, dy, path, aligned_x, aligned_y);
}

float Tools::check_hypothesis(float prediction, float actual, float std_dev, bool two_sided = false) {
    const int num_samples = 1000;
    std::default_random_engine generator(std::random_device{}());
    std::normal_distribution<float> distribution(0.0, std_dev);

    std::vector<float> gauss(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        gauss[i] = distribution(generator);
    }

    float delta = actual - prediction;

    if (two_sided) {
        delta = std::abs(delta);
        int count = 0;
        for (float g : gauss) {
            if (g > delta || g < -delta) {
                ++count;
            }
        }
        return static_cast<float>(count) / num_samples;
    }

    delta = prediction - actual;
    int count = 0;
    for (float g : gauss) {
        if (g > delta) {
            ++count;
        }
    }
    return static_cast<float>(count) / num_samples;
}

void Tools::erosioncentered(float *x, int size, int kernel_size, float *y) {
    // Erosion centered implementation
    // This function should implement the logic to apply a centered erosion
    // to the input signal x with the specified kernel size.
    // The eroded signal should be stored in y.

    int half_kernel = kernel_size / 2;
    for (int i = 0; i < size; ++i) {
        int start = std::max(0, i - half_kernel);
        int end = std::min(size - 1, i + half_kernel);
        y[i] = *std::min_element(x + start, x + end + 1);
    }
}

void Tools::dilationcentered(float *x, int size, int kernel_size, float *y) {
    // Dilation centered implementation
    // This function should implement the logic to apply a centered dilation
    // to the input signal x with the specified kernel size.
    // The dilated signal should be stored in y.

    int half_kernel = kernel_size / 2;
    for (int i = 0; i < size; ++i) {
        int start = std::max(0, i - half_kernel);
        int end = std::min(size - 1, i + half_kernel);
        y[i] = *std::max_element(x + start, x + end + 1);
    }
}

void Tools::openingcentered(float *x, int size, int kernel_size, float *y) {
    // Opening centered implementation
    // This function should implement the logic to apply a centered opening
    // to the input signal x with the specified kernel size.
    // The opened signal should be stored in y.

    float *temp = new float[size];
    this->erosioncentered(x, size, kernel_size, temp);
    this->dilationcentered(temp, size, kernel_size, y);
    delete[] temp;
}

void Tools::closingcentered(float *x, int size, int kernel_size, float *y) {
    // Closing centered implementation
    // This function should implement the logic to apply a centered closing
    // to the input signal x with the specified kernel size.
    // The closed signal should be stored in y.

    float *temp = new float[size];
    this->dilationcentered(x, size, kernel_size, temp);
    this->erosioncentered(temp, size, kernel_size, y);
    delete[] temp;
}

void Tools::erosioncentered(bool *x, int size, int kernel_size, bool *y) {
    int r = kernel_size / 2;
    std::memset(y, 0, size * sizeof(bool));
    for (int i = 0; i < size; ++i) {
        // Preserve border values
        if (i < r) {
            y[i] = x[0];
            continue;
        }
        if (i >= size - r) {
            y[i] = x[size - 1];
            continue;
        }
        bool keep = true;
        for (int offset = -r; offset <= r; ++offset) {
            int idx = i + offset;
            if (!x[idx]) {
                keep = false;
                break;
            }
        }
        y[i] = keep;
    }
}

void Tools::dilationcentered(bool *x, int size, int kernel_size, bool *y) {
    int r = kernel_size / 2;
    // Initialize output to false
    std::memset(y, 0, size * sizeof(bool));

    for (int i = 0; i < size; ++i) {
        bool found = false;
        // Scan neighborhood
        for (int offset = -r; offset <= r; ++offset) {
            int idx = i + offset;
            if (idx < 0 || idx >= size) continue;
            if (x[idx]) {
                found = true;
                break;
            }
        }
        y[i] = found;
    }
}

void Tools::openingcentered(bool *x, int size, int kernel_size, bool *y) {

    bool *temp = new bool[size];
    // 1. Erode input into temp
    erosioncentered(x, size, kernel_size, temp);
    // 2. Dilate temp into output
    dilationcentered(temp, size, kernel_size, y);
    delete[] temp;
}
void Tools::closingcentered(bool *x, int size, int kernel_size, bool *y) {

    bool *temp = new bool[size];
    // 1. Dilate input into temp
    dilationcentered(x, size, kernel_size, temp);
    // 2. Erode temp into output
    erosioncentered(temp, size, kernel_size, y);
    delete[] temp;
}

void Tools::erosioncentered(std::vector<bool> &x, int kernel_size) {

    bool *in = new bool[x.size()];
    bool *temp = new bool[x.size()];
    for (int i = 0; i < x.size(); ++i) {
        in[i] = x[i];
    }
    this->erosioncentered(in, x.size(), kernel_size, temp);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = temp[i];
    }
    delete[] in;
    delete[] temp;
}
void Tools::dilationcentered(std::vector<bool> &x, int kernel_size) {

    bool *in = new bool[x.size()];
    bool *temp = new bool[x.size()];
    for (int i = 0; i < x.size(); ++i) {
        in[i] = x[i];
    }
    this->dilationcentered(in, x.size(), kernel_size, temp);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = temp[i];
    }
    delete[] in;
    delete[] temp;
}

void Tools::openingcentered(std::vector<bool> &x, int kernel_size) {

    bool *in = new bool[x.size()];
    bool *temp = new bool[x.size()];
    for (int i = 0; i < x.size(); ++i) {
        in[i] = x[i];
    }
    this->openingcentered(in, x.size(), kernel_size, temp);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = temp[i];
    }
    delete[] in;
    delete[] temp;
}
void Tools::closingcentered(std::vector<bool> &x, int kernel_size) {

    bool *in = new bool[x.size()];
    bool *temp = new bool[x.size()];
    for (int i = 0; i < x.size(); ++i) {
        in[i] = x[i];
    }
    this->closingcentered(in, x.size(), kernel_size, temp);
    for (int i = 0; i < x.size(); ++i) {
        x[i] = temp[i];
    }
    delete[] in;
    delete[] temp;
}

vector<tuple<int,int>> Tools::get_regions(bool *x, int size) {

    vector<tuple<int,int>> regions;
    int start = -1;

    for (int i = 0; i < size; ++i) {
        if (x[i] && start == -1) {
            start = i;
        } else if (!x[i] && start != -1) {
            regions.push_back(make_tuple(start, i));
            start = -1;
        }
    }

    if (start != -1) {
        regions.push_back(make_tuple(start, size - 1));
    }

    return regions;
}
vector<tuple<int,int>> Tools::get_regions(std::vector<bool> &x) {

    vector<tuple<int,int>> regions;
    int start = -1;

    for (int i = 0; i < x.size(); ++i) {
        if (x[i] && start == -1) {
            start = i;
        } else if (!x[i] && start != -1) {
            regions.push_back(make_tuple(start, i));
            start = -1;
        }
    }

    if (start != -1) {
        regions.push_back(make_tuple(start, x.size() - 1));
    }

    return regions;
}

float Tools::pearson_correlation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must be same non‐zero length");
    }
    size_t n = x.size();

    // 1) compute means
    float mean_x = std::accumulate(x.begin(), x.end(), 0.0) / n;
    float mean_y = std::accumulate(y.begin(), y.end(), 0.0) / n;

    // 2) compute covariance and variances
    float cov_xy = 0.0;
    float var_x  = 0.0;
    float var_y  = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        cov_xy += dx * dy;
        var_x  += dx * dx;
        var_y  += dy * dy;
    }

    // 3) finish computing Pearson r
    float denom = std::sqrt(var_x * var_y);
    if (denom == 0.0) {
        throw std::runtime_error("Division by zero in correlation (constant vector?)");
    }
    return cov_xy / denom;
}

float Tools::snr_log(std::vector<float> fore, std::vector<float> back) {
    // Calculate the SNR (Signal-to-Noise Ratio)
    float background_power = 0.0f;
    for (int i = 0; i < back.size(); ++i) {
        background_power += back[i] * back[i];
    }
    background_power /= back.size();

    // Calculate the signal power
    float signal_power = 0.0f;
    for (int i = 0; i < fore.size(); ++i) {
        signal_power += fore[i] * fore[i];
    }
    signal_power /= fore.size();

    // Calculate SNR
    float snr = 10.0f * std::log10(signal_power / background_power);
    return snr;
}


float Tools::snr_lin(std::vector<float> fore, std::vector<float> back) {
    // Calculate the SNR (Signal-to-Noise Ratio)
    float background_power = 0.0f;
    for (int i = 0; i < back.size(); ++i) {
        background_power += abs(back[i]);
    }
    background_power /= back.size();

    // Calculate the signal power
    float signal_power = 0.0f;
    for (int i = 0; i < fore.size(); ++i) {
        signal_power += abs(fore[i]);
    }
    signal_power /= fore.size();

    // Calculate SNR
    float snr = signal_power / background_power;
    return snr;
}

float Tools::cosen(std::vector<float>& ibi_series) {
    int m = 1;
    const size_t N = ibi_series.size();
    if (N < static_cast<size_t>(m + 1)) {
        return std::numeric_limits<float>::infinity();
    }

    // Count matching pairs for embedding dim mm and tolerance rr
    auto count_matches = [&](int mm, float rr) -> long long {
        size_t L = (N >= static_cast<size_t>(mm)) ? (N - mm + 1) : 0;
        long long cnt = 0;
        for (size_t i = 0; i < L; ++i) {
            for (size_t j = 0; j < L; ++j) {
                if (i == j) continue;
                float d = 0.0f;
                for (int k = 0; k < mm; ++k) {
                    d = std::max(d, std::fabs(ibi_series[i + k] - ibi_series[j + k]));
                }
                if (d <= rr) {
                    ++cnt;
                }
            }
        }
        return cnt;
    };

    // Adaptive tolerance loop
    float r = 0.025f;
    long long A = 0, B = 0;
    while (A < 5 && r < 0.5f) {
        r += 0.005f;
        B = count_matches(m,     r);
        A = count_matches(m + 1, r);
    }

    if (B == 0) {
        return std::numeric_limits<float>::infinity();
    }

    // Compute mean (ignore NaNs)
    float sum = 0.0f;
    size_t valid = 0;
    for (float x : ibi_series) {
        if (!std::isnan(x)) {
            sum += x;
            ++valid;
        }
    }
    float mean = (valid > 0 ? sum / valid : NAN);

    // COSEn formula
    float res = 0.0f;
    if (A != 0) {
        res = -std::log(static_cast<float>(A) / static_cast<float>(B));
    }
    res += std::log(2.0f * r);
    res -= std::log(mean);

    return res;
}

float Tools::cv(std::vector<float>& ibi_series) {
    // Calculate the coefficient of variation (CV)
    const size_t N = ibi_series.size();
    if (N == 0) {
        return std::numeric_limits<float>::infinity();
    }

    // Calculate mean and standard deviation
    float mn = mean(ibi_series.data(), N);
    float std = stdev(ibi_series.data(), N);

    // Calculate CV
    float cv = std / mn;
    return cv;
}

void Tools::remove_baseline(float *x, int size, float fs) {
    // Remove baseline by subtracting the mean

    float *median_filtered = new float[size];
    median_filter(x, size, (int)(fs*0.2), median_filtered);
    median_filter(median_filtered, size, (int)(fs*0.6));

    for (int i = 0; i < size; ++i) {
        x[i] = x[i] - median_filtered[i];
    }
}

void Tools::normalize_zscore(float *x, int size) {
    // Normalize using z-score
    float mean_val = mean(x, size);
    float std_dev = stdev(x, size);

    if (std_dev == 0.0f) {
        std::fill(x, x + size, 0.0f); // Avoid division by zero
        return;
    }

    for (int i = 0; i < size; ++i) {
        x[i] = (x[i] - mean_val) / std_dev;
    }
}