//
//  filter.hpp
//  minHR
//
//
/*!
@class Filter

@brief Helper class that facilitates filter functions.
 
Calculating the mean, standard deviation, the sum and extrema of an array is done throughout the application. This helper class facilitates these functions to improve readability and efficiency of other parts of the application. Not all functions are used in the final application due to the long development process.
 
*/

#pragma once

#include <string>
#include <vector>
#include <map>
#include <array>
#include <numeric>
#include <tuple>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdint> // <cstdint> requires c++11 support
#include <regex>
#include <time.h>
#include <memory>
#include <thread>
#include <sys/stat.h>

#include "../iir.h"

class Filter {
    
public:
    ~Filter();
    void median(float *input, int input_size, int size);
    void rollmedian(float *input, int input_size, int size);
    void rollmean(float *input, int input_size, int size);
    void rollmeanpost(float *input, int input_size, int size);
    void rollsum(float *input, int input_size, int size);
    void rollmax(float *input, int input_size, int size);
    void rollmaxmin(float *input, int input_size, int size);
    void Buttersworth(float *input, int size, int fs, float f0, float f1);
    void Highpass(float *input, int size, int fs, float f0);
    void Notch(float *input, int size, int fs, float f0);
    void ChebyshevII(float *input, int size);
    void derivative(float *input, int size, int fs);
    void compress(float *input, int inputheight, int inputwidth, float *output, int w, int h);
    void Gauss(float a[], int length, float std);
    
    float getSampleEntropy(float a[], int length, float r);
    float getNoiseLevel(float a[], int length, int scale);
    float getNoiseLevel2(float a[], int length, int scale, float siglevel);
    float findMedian(float a[], int n);
    float findMean(float a[], int n);
    float findStd(float a[], int n);
    float findSkew(float a[], int n);
    float findSum(float a[], int n);
    
    float findMaximum(float a[], int size, bool absolute=false);
    int findPosMaximum(float a[], int size, bool absolute);
    float findMinimum(float a[], int size);
    int findPosMinimum(float a[], int size);
    
private:
};



