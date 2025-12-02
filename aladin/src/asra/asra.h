
/**
@class ECGdetector

@brief Calculates HR from ECG signal
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

#include "filter.h"

using namespace std;

struct Annotation {
    int time;
    string type;
};

class ECGdetector {
public:
    ECGdetector(float* ecg, int size, int fs);
    ~ECGdetector();
    
    void preprocessingV2();
    void detectPeaksInWindow(int start);
    void detectPeaksV2();
    float getQuality();
    void loadmasks(string foldername, string filename);
    void railmask();
    void highfreqmask();
    void lowpowmask();
    void mergemasks();
    
    vector<int> getRPeaks();
    int findRawPeak(int time, int lowerbound);
    int* getTP() { return TP; };
    int* getFP() { return FP; };
    int* getFN() { return FN; };
    float* getSE() { return SE; };
    float* getPPV() { return PPV; };
    float* getACC() { return ACC; };
    
private:
    Filter* filter = new Filter();
    
    vector<int> rpeaks;
    vector<int> measurements;
    vector<float> entropies;
    vector<int> windowstart;
    vector<int> windowend;
    vector<int> estimations;
    vector<float> thresholds;
    vector<float> thresholds2;
    vector<int> lookbackpeaks;
    vector<int> filtonlypeaks;
    
    vector<tuple<int,int>> railmasks;
    vector<tuple<int,int>> highfreqmasks;
    vector<tuple<int,int>> lowpowmasks;
    vector<tuple<int,int>> masks;
    
    float* ecg;
    float* ecglow;
    float* ecgnoise;
    float* ecgfilt;
    float* ecgraw;
    float* gaussians;
    float* ecggauss;
    float* ecgref;
    int size;
    int fs;
    
    int TP[4] = {0,0,0,0};
    int FP[4] = {0,0,0,0};
    int FN[4] = {0,0,0,0};
    float SE[4] = {0.0,0.0,0.0,0.0};
    float PPV[4] = {0.0,0.0,0.0,0.0};
    float ACC[4] = {0.0,0.0,0.0,0.0};
};
