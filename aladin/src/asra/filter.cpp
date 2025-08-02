//
//  filter.cpp
//  minHR
//
//  Created by Lukas Arts on 01/07/2019.
//  Copyright © 2019 Limstone Applications. All rights reserved.
//

#include "filter.h"

using namespace std;

/// Constructor, does nothing.
///@brief Constructor
Filter::~Filter() {
}

/// Calculates residue after substracting a rolling median filter from the original signal. Result is written to input array. Naming should be improved.
/// @brief Substracts median filtered signal from original signal.
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Median filter size
void Filter::median(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    float *temp = (float *)malloc(sizeof(float[size]));
    
    int halfsize = size/2;
    float *padded = (float *)malloc(sizeof(float)*(input_size+halfsize*2));
    
    memset(padded,input[0],sizeof(float)*halfsize);
    memcpy(&padded[halfsize],input,sizeof(float)*input_size);
    memset(&padded[halfsize+input_size],input[input_size-1],sizeof(float)*halfsize);
    
    for(int i=0; i<input_size; i++) {
        memcpy(temp,&padded[i],sizeof(float)*size);
        res[i] = findMedian(temp, size) - input[i];
    }
    memcpy(input, res, sizeof(float)*input_size);
    delete temp;
    delete res;
}
/// Performs a centered rolling median filter on the signal. Signal is padded with input[0] and input[input_size-1] values at both ends. Size should be odd. Result is written to input array.
/// @brief Centered rolling median filter
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Median filter size, should be odd
void Filter::rollmedian(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    float *temp = (float *)malloc(sizeof(float[size]));
    
    int halfsize = size/2;
    float *padded = (float *)malloc(sizeof(float)*(input_size+halfsize*2));
    
    memset(padded,input[0],sizeof(float)*halfsize);
    memcpy(&padded[halfsize],input,sizeof(float)*input_size);
    memset(&padded[halfsize+input_size],input[input_size-1],sizeof(float)*halfsize);
    
    for(int i=0; i<input_size; i++) {
        res[i] = findMedian(&padded[i], size);
    }
    memcpy(input, res, sizeof(float)*input_size);
    delete temp;
    delete res;
    delete padded;
}

/// Performs a right aligned rolling mean filter. The resulting signal is zeropadded at the end to match input_size. Result is written to input array.
/// @brief Right aligned rolling mean
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Filter size
void Filter::rollmean(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    
    int halfsize = size/2;
    float *padded = (float *)malloc(sizeof(float)*(input_size+halfsize*2));
    
    memset(padded,input[0],sizeof(float)*halfsize);
    memcpy(&padded[halfsize],input,sizeof(float)*input_size);
    memset(&padded[halfsize+input_size],input[input_size-1],sizeof(float)*halfsize);
    
    for(int i=0; i<(input_size); i++) {
        res[i] = findMean(&padded[i], size);
    }
    memcpy(input, res, sizeof(float)*input_size);
    delete res;
    delete padded;
}

void Filter::rollmeanpost(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    
    float *padded = (float *)malloc(sizeof(float)*(input_size+size));
    
    memset(padded,input[0],sizeof(float)*size);
    memcpy(&padded[size],input,sizeof(float)*input_size);
    
    for(int i=0; i<(input_size); i++) {
        res[i] = findMean(&padded[i], size);
    }
    memcpy(input, res, sizeof(float)*input_size);
    delete res;
    delete padded;
}

/// Performs a right aligned rolling summation filter of length 'size'. The resulting signal is zeropadded at the end to match input_size. Result is written to input array.
/// @brief Right aligned rolling summation
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Filter size
void Filter::rollsum(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    
    for(int i=0; i<(input_size-size); i++) {
        res[i] = findSum(&input[i], size);
    }
    memset(&res[input_size-size],0, sizeof(float)*size);
    memcpy(input, res, sizeof(float)*input_size);
    delete res;
}

/// Performs a right aligned rolling max function filter. The resulting signal is zeropadded at the end to match input_size. Result is written to input array.
/// @brief Right aligned rollling max function
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Filter size
void Filter::rollmax(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    
    for(int i=0; i<(input_size-size); i++) {
        res[i] = input[i] * (2 / findMaximum(&input[i], size, true));
    }
    memset(&res[input_size-size],0, sizeof(float)*size);
    memcpy(input, res, sizeof(float)*input_size);
    delete res;
}
/// Performs a right aligned rolling x/(max(x)-min(x)) normalisation operation.  The resulting signal is zeropadded at the end to match input_size. Result is written to input array.
/// @brief Right aligned normalisation operation
/// @param input Input signal
/// @param input_size Input signal size
/// @param size Filter size
void Filter::rollmaxmin(float *input, int input_size, int size) {
    float *res = (float *)malloc(sizeof(float[input_size]));
    
    int halfsize = size/2;
    float *padded = (float *)malloc(sizeof(float)*(input_size+halfsize*2));
    
    memset(padded,input[0],sizeof(float)*halfsize);
    memcpy(&padded[halfsize],input,sizeof(float)*input_size);
    memset(&padded[halfsize+input_size],input[input_size-1],sizeof(float)*halfsize);
    
    for(int i=0; i<(input_size); i++) {
        res[i] = input[i] * (2 / (findMaximum(&padded[i], size) - findMinimum(&padded[i], size)));
    }
    memcpy(input, res, sizeof(float)*input_size);
    delete res;
}

void Filter::derivative(float *input, int size, int fs) {
    float *res = (float *)calloc(size,sizeof(float));
    
    int halfsize = 2;
    float *padded = (float *)malloc(sizeof(float)*(size+halfsize*2));
    
    memset(padded,input[0],sizeof(float)*halfsize);
    memcpy(&padded[halfsize],input,sizeof(float)*size);
    memset(&padded[halfsize+size],input[size-1],sizeof(float)*halfsize);
    
    for(int i=0; i<(size-4); i++) {
        res[i] = (-1.0*input[i] - 2.0*input[i+1] + 2.0*input[i+3] + 1.0*input[i+4])/(8.0*(float)fs);
    }
    memcpy(input, res, sizeof(float)*size);
    delete res;
}


/// Performs wavelet transformation matrix max pooling compression. A (inputwidth/outputwidth)*(inputheight/outputheight) max pooling filter is evaluated in a convoluted fashion. The convolution is split into a horizontal and vertical action to speed up the process.  The resulting compressed matrix is also rotated 90 degrees to speed up column-wise peak detection as described in the Event class.
/// @brief Compresses a wavelet transformation matrix
/// @param input Pointer to input matrix
/// @param inputheight Height of input matrix
/// @param inputwidth Width of input matrix
/// @param output Pointer to output matrix
/// @param outputwidth Desired width of output matrix
/// @param outputheight  Desired height of output matrix
void Filter::compress(float *input, int inputheight, int inputwidth, float *output, int outputwidth, int outputheight) {
    
    int n_width = inputwidth/outputwidth;
    int n_height = inputheight/outputheight;
    //cout << "NWIDTH: " << n_width << " NHEIGHT:" << n_height << " W:" << w << " H:" << h << endl;
    
    float *half = (float*)malloc(sizeof(float)*n_width*inputheight);
    
    for(int y=0; y<inputheight; y++) {
        for(int x=0; x<(outputwidth*n_width); x+=outputwidth) {
            float *ip = &input[y*inputwidth + x];
            float max = findMaximum(ip, outputwidth);
            int xx = x/outputwidth;
            int yy = y;
            half[xx*inputheight + yy] = max;
        }
    }
    
    for(int x=0; x<n_width; x++) {
        for(int y=0; y<(outputheight*n_height); y+=outputheight) {
            float *ip = &half[x*inputheight + y];
            float max = findMaximum(ip, outputheight);
            
            int xx = x;
            int yy = y/outputheight;
            output[xx*n_height + yy] = max;
        }
    }
    delete half;
}

/// Calculates sample entropy, or chaosness, of an input signal. Used in the unused heuristic maximal ridgeline detection defined in the Event class.
/// @brief Calculates sample entropy
/// @param a Input signal
/// @param N Input signal size
/// @param r r parameter of sample entropy function
float Filter::getSampleEntropy(float a[], int N, float r)
{
    int Cm = 0, Cm1 = 0;
    float err = 0.0;
    int m = 2;
  
    float mean = 0;
    float var = 0.0;
    float sd = 0.0;
    for(int i=0;i<N;i++) {
        mean += a[i]/N;
    }
    for(int i=0;i<N;i++) {
        var += pow(a[i] - mean, 2);
        var=var/N;
        sd = sqrt(var);
    }
    
    err = sd * r;
  
    for (unsigned int i = 0; i < N - (m + 1) + 1; i++) {
        for (unsigned int j = i + 1; j < N - (m + 1) + 1; j++) {
            bool eq = true;
            //m - length series
            for (unsigned int k = 0; k < m; k++) {
                if (abs(a[i+k] - a[j+k]) > err) {
                    eq = false;
                    break;
                }
            }
            if (eq) Cm++;
      
            //m+1 - length series
            int k = m;
            if (eq && abs(a[i+k] - a[j+k]) <= err)
                Cm1++;
        }
    }
  
    if (Cm > 0 && Cm1 > 0)
        return log((float)Cm / (float)Cm1);
    else
        return 0.0;
}

float Filter::getNoiseLevel(float *a, int length, int fs) {
    
    int minl = max(fs*4,length);
    int offset = (minl-length)/2;
    float *tmp = (float*)calloc(minl,sizeof(float));
    memcpy(&tmp[offset],a,sizeof(float)*length);
    
    Buttersworth(tmp, minl, fs, 20, 40);
    
    for(int i=0; i<minl; i++) {
        tmp[i] = tmp[i]*tmp[i];
    }
    rollmean(&tmp[offset],length,fs*0.05);
    for(int i=0; i<minl; i++) {
        tmp[i] = sqrt(tmp[i]);
    }
    float nl = findMean(&tmp[offset], length);
    
    delete tmp;
    return nl;
}


float Filter::getNoiseLevel2(float *a, int length, int fs, float siglevel) {
    
    int minl = max(fs*4,length);
    int offset = (minl-length)/2;
    float *tmp = (float*)calloc(minl,sizeof(float));
    memcpy(&tmp[offset],a,sizeof(float)*length);
    
    Buttersworth(tmp, minl, fs, 20, 40);
    
    for(int i=0; i<minl; i++) {
        tmp[i] = tmp[i]*tmp[i];
    }
    rollmean(&tmp[offset],length,fs*0.05);
    
    int nsamples = 0;
    
    for(int i=0; i<minl; i++) {
        if(sqrt(tmp[i]) > siglevel) nsamples++;
    }
    
    delete tmp;
    float ratio = (float)nsamples/(float)length;
    return pow(ratio,1.0/5.0);
}

/// Returns median value of array.
/// @param a Input array
/// @param n Input array length
float Filter::findMedian(float a[], int n) {
    // First we sort the array
    sort(a, a + n);
    
    // check for even case
    if (n % 2 != 0)
        return a[n / 2];
    
    return (a[(n - 1) / 2] + a[n / 2]) / 2.0;
}

/// Returns mean value of array
/// @param a Input array
/// @param n Input array size
float Filter::findMean(float a[], int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    
    return sum/(float)n;
}

/// Returns standard deviation of array
/// @param a Input array
/// @param n Input array size
float Filter::findStd(float a[], int n) {
    float var = 0;
    float mean = findMean(a,n);
    for(int i = 0; i < n; i++ )
    {
      var += (a[i] - mean) * (a[i] - mean);
    }
    var /= (float)n;
    
    return sqrt(var);
}

/// Returns skewness of array
/// @param a Input array
/// @param n Input array size
float Filter::findSkew(float a[], int n) {
    float sum = 0;
    float mean = findMean(a, n);
    float std = findStd(a,n);
    
    for (int i = 0; i < n; i++)
        sum = (a[i] - mean) *
              (a[i] - mean) *
              (a[i] - mean);
    
    return sum / ((float)n * std * std * std * std);
}

/// Returns summation of array
/// @param a Input array
/// @param n Input array size
float Filter::findSum(float a[], int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i];
    
    return sum;
}

/// Returns maximum of array
/// @param a Input array
/// @param size Input array size
/// @param absolute If true, it uses | a[i] |
float Filter::findMaximum(float a[], int size, bool absolute) {
    
    float max = -MAXFLOAT;
    float aa;
    for (int i = 0; i < size; i++) {
        aa = a[i];
        if(absolute) aa = abs(aa);
        if(aa > max) max=aa;
    }
    
    return max;
}

/// Returns position of maximum in array
/// @param a Input array
/// @param size Input array size
/// @param absolute If true, it uses | a[i] |
int Filter::findPosMaximum(float a[], int size, bool absolute) {
    
    float max = -MAXFLOAT;
    float aa;
    int ind = 0;
    for (int i = 0; i < size; i++) {
        aa = a[i];
        if(absolute) aa = abs(aa);
        if(aa > max) { max=aa; ind=i;}
    }
    
    return ind;
}

/// Returns minimum of array
/// @param a Input array
/// @param size Input array size
float Filter::findMinimum(float a[], int size) {
    float min = MAXFLOAT;
    for (int i = 0; i < size; i++)
        if(a[i] < min) min=a[i];
    
    return min;
}

/// Return position of minimum in array
/// @param a Input array
/// @param size Input array size
int Filter::findPosMinimum(float a[], int size) {
    
    float min = MAXFLOAT;
    int ind = 0;
    for (int i = 0; i < size; i++) {
        if(a[i] < min) { min=a[i]; ind=i;}
    }
    
    return ind;
}

/// Perform 3rd order Butterworth Bandpass filter with cutoff frequencies f0 and f1 on a input signal of sample frequency fs.
/// @param input Input signal
/// @param size Input signal size
/// @param fs Sample frequency
/// @param f0 Low cutoff frequency
/// @param f1 High cutoff frequency
void Filter::Buttersworth(float *input, int size, int fs, float f0, float f1) {
    const int order = 2; // 4th order (=2 biquads)
    Iir::Butterworth::BandPass<order> f;
    const float samplingrate = fs; // Hz
    f.setup (samplingrate, (f0+f1)/2, f1-f0);
    
    float *temp = (float *)malloc(sizeof(float[size]));
    float *res = (float *)malloc(sizeof(float[size]));
    for(int i=0; i<size; i++) {
        temp[size-i-1] = f.filter(input[i]);
    }
    for(int i=0; i<size; i++) {
        res[size-i-1] = f.filter(temp[i]);
    }
    memcpy(input, res, sizeof(float)*size);
    delete res;
    delete temp;
}

void Filter::Notch(float *input, int size, int fs, float f0) {
    const int order = 2; // 4th order (=2 biquads)
    Iir::Butterworth::BandPass<order> f;
    const float samplingrate = fs; // Hz
    f.setup (samplingrate, f0, 1);
    
    float *temp = (float *)malloc(sizeof(float[size]));
    float *res = (float *)malloc(sizeof(float[size]));
    for(int i=0; i<size; i++) {
        temp[size-i-1] = f.filter(input[i]);
    }
    for(int i=0; i<size; i++) {
        res[size-i-1] = f.filter(temp[i]);
    }
    memcpy(input, res, sizeof(float)*size);
    delete res;
    delete temp;
}

void Filter::Highpass(float *input, int size, int fs, float f0) {
    const int order = 3;
    Iir::Butterworth::HighPass<order> f;
    const float samplingrate = fs;
    f.setup(samplingrate, f0);
    
    float *temp = (float *)malloc(sizeof(float[size]));
    float *res = (float *)malloc(sizeof(float[size]));
    for(int i=0; i<size; i++) {
        temp[size-i-1] = f.filter(input[i]);
    }
    for(int i=0; i<size; i++) {
        res[size-i-1] = f.filter(temp[i]);
    }
    memcpy(input, res, sizeof(float)*size);
    delete res;
    delete temp;
}

/// Perform 2nd order ChebyShev Bandpass filter with cutoff frequencies of 0.5 and 2Hz. Not used anymore. Implemented to test the performance difference between Butterworth and Chebyshev filtering.
/// @param input Input signal
/// @param size Input signal size
void Filter::ChebyshevII(float *input, int size) {
    int nzeros = 4;
    int npoles = 4;
    float gain = 5355.25;
    float xv[nzeros+1], yv[npoles+1];
    
    for(int i=0; i<(nzeros+1); i++) {
        xv[i] = 0; yv[i] = 0;
    }
    
    float *res = (float *)malloc(sizeof(float[size]));
    
    for (int i=0; i<size; i++) {
        
        xv[0] = xv[1]; xv[1] = xv[2]; xv[2] = xv[3]; xv[3] = xv[4];
        xv[4] = input[i] / gain;
        yv[0] = yv[1]; yv[1] = yv[2]; yv[2] = yv[3]; yv[3] = yv[4];
        yv[4] =   (xv[0] + xv[4]) + 4 * (xv[1] + xv[3]) + 6 * xv[2]
                     + ( -0.9746048354 * yv[0]) + (  3.7729985334 * yv[1])
                     + ( -5.6226345051 * yv[2]) + (  3.8212530867 * yv[3]);
        
        res[i] = yv[4];
    }
    memcpy(input, res, sizeof(float)*size);
    delete res;
}

void Filter::Gauss(float *a, int length, float std) {
    
    int mid = (length>>2) - 1;
    
    for(int i=0; i<length; i++) {
        
    }
}


