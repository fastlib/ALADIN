//
//  asra.cpp
//  ASRA
//
//  Created by Lukas Arts on 06/04/2022.
//

#include "asra.h"

/// Constructor of the ECGdetector class.
/// @brief Constructor
/// @param ecg Pointer to ECG signal
/// @param ecgref Pointer to optional 1Hz HR estimation signal based on R-Peak annotations. The conversion from R-Peak annotations to a 1Hz HR signal is done in the Data class for the MESA dataset.
/// @param size Number of samples
/// @param fs Sample frequency
ECGdetector::ECGdetector(float* ecg, int size, int fs) {
    this->fs = fs;
    this->ecg = ecg;
    this->size = size;
    
    ecglow = (float*)malloc(sizeof(float)*size);
    ecgnoise = (float*)malloc(sizeof(float)*size);
    ecgfilt = (float*)malloc(sizeof(float)*size);
    ecgraw = (float*)malloc(sizeof(float)*size);
    ecgref = (float*)malloc(sizeof(float)*size);
    gaussians = (float*)malloc(sizeof(float)*size);
    ecggauss = (float*)malloc(sizeof(float)*size);
    
    memset(gaussians,0,sizeof(float)*size);
    memset(ecgnoise,0,sizeof(float)*size);
    memset(ecggauss,0,sizeof(float)*size);
}

ECGdetector::~ECGdetector() {
    delete ecgfilt;
    rpeaks.clear();
    rpeaks.shrink_to_fit();
    
    entropies.clear();
    entropies.shrink_to_fit();
}

vector<int> ECGdetector::getRPeaks() {
    preprocessingV2();
    detectPeaksInWindow(0);
    return rpeaks;
}

void ECGdetector::preprocessingV2() {
    
    float mean = filter->findMean(ecg,size);
    
    for(int i=0; i<(size-1); i++) {
        ecgraw[i] = ecg[i];
    }
    
    //railmask();
    //highfreqmask();
    //lowpowmask();
    //mergemasks();
    
    for(int i=0; i<(size-1); i++) {
        ecg[i] = (ecg[i]-mean);
        ecglow[i] = ecg[i];
    }
    
    //Filter with bandpass filter
    filter->Buttersworth(ecg, size, fs, 8, 25);
    
    //Filter with lowpass filter
    filter->Buttersworth(ecglow, size, fs, 1, 25);
    
    for(int i=0; i<(size-1); i++) {
        //cout << ecg[i] << endl;
        ecgfilt[i] = ecg[i];
    }
    //Derivative filter
    filter->derivative(ecgfilt, size, fs);
    
    //Square it
    for(int i=0; i<(size-1); i++) {
        ecgfilt[i] = ecgfilt[i]*ecgfilt[i];
    }
    
    //Moving average
    filter->rollmeanpost(ecgfilt,size,fs*0.15);
    
    float max = filter->findMaximum(ecg, size-1, true);
    for(int i=0; i<size-1; i++) {
        ecg[i] = ecg[i]/max;
    }

    max = filter->findMaximum(ecgfilt, size-1, true);
    for(int i=0; i<size-1; i++) {
        ecgfilt[i] = ecgfilt[i]/max;
    }
    
    //filter->rollmaxmin(ecg, size-1, fs*2);
    //filter->rollmaxmin(ecgfilt, size-1, fs);
}

int ECGdetector::findRawPeak(int time, int lowerbound) {
    int wl = min(time,(int)((float)fs*0.15));
    
    float sum = 0;
    
    for(int i=0; i<wl; i++) {
        sum += ecglow[time-wl+i]*ecglow[time-wl+i]*ecglow[time-wl+i];
    }
    
    bool negativeQRS = false;
    if(sum < 0) {
        negativeQRS = true;
    }
    
    //cout << "t:" << time << ", " << sum << endl;
    return time-wl+filter->findPosMaximum(&ecg[time-wl], wl, negativeQRS);
}

bool compareFiducials(tuple<int,int,float,bool> fm1, tuple<int,int,float,bool> fm2)
{
    return (get<1>(fm1) < get<1>(fm2));
}

void ECGdetector::loadmasks(string foldername, string filename) {
    stringstream ff;
    ff << foldername << "/" << filename << ".mask.txt";
    
    cout << "Load masks: " << ff.str() << endl;
    
    ifstream file;
    file.open(ff.str());
    
    int n;
    file >> n;
    
    for(int i=0; i<n; i++) {
        int start;
        int ending;
        file >> start >> ending;
        cout << "mask " << i << "[" << start << "," << ending << "]" << endl;
        
        tuple<int,int> mask(start,ending);
        masks.push_back(mask);
    }
}

void ECGdetector::railmask() {
    
    float maxs = filter->findMaximum(ecgraw, size);
    float mins = filter->findMinimum(ecgraw, size);
    float range = maxs-mins;
    
    maxs = maxs-range*0.001;
    mins = mins+range*0.001;
    
    int nmax = 0;
    int nmin = 0;
    
    for(int i=0; i<size; i++) {
        if(ecgraw[i]>=maxs) nmax++;
        if(ecgraw[i]<=mins) nmin++;
    }
    
    if(nmax>2 || nmin>2) {
        //We have found a maximum and minimum rail voltage
        cout << "Found rail voltages: [" << nmin << "," << nmax << "]" << endl;
        
        int buildingmask = 0;
        int consecutive = 0;
        int winstart = 0;
        int winend = 0;
        for(int i=0; i<size; i++) {
            if(buildingmask>0) buildingmask--;
            
            if(nmax > 2 && ecgraw[i] > (maxs-range*0.05)) {
                consecutive++;
                if(consecutive>2) {
                    if(buildingmask==0) {
                        winstart = i-2;
                    }
                    buildingmask = fs/5;
                }
            } else if(nmin > 2 && ecgraw[i] < (mins+range*0.05)) {
                consecutive++;
                
                if(consecutive>2) {
                    if(buildingmask==0) {
                        winstart = i-2;
                    }
                    buildingmask = fs/5;
                }
            } else {
                consecutive = 0;
            }
            if(buildingmask==1) {
                winend = i-2;
                
                tuple<int,int> mask(winstart,winend);
                railmasks.push_back(mask);
                
                winstart = 0;
                winend = 0;
            }
        }
    }
    
    //extend masks to edges of signal if it is within the edges
    if(railmasks.size() > 1) {
        if(get<0>(railmasks[0]) < 3*fs) {
            get<0>(railmasks[0]) = 0;
        }
        if(get<0>(railmasks[railmasks.size()-1]) > size-3*fs-1) {
            get<0>(railmasks[railmasks.size()-1]) = size-1;
        }
    }
    
    for(int i=0; i<railmasks.size(); i++) {
        masks.push_back(railmasks[i]);
    }
}

void ECGdetector::highfreqmask() {
    
    float* tmp = (float*)malloc(sizeof(float)*size);
    memcpy(tmp,ecg,sizeof(float)*size);
    
    filter->Notch(tmp, size, fs, 50);
    filter->Highpass(tmp, size, fs, 40);
    
    for(int i=0; i<size; i++) {
        tmp[i] = tmp[i]*tmp[i];
    }
    
    filter->rollmean(tmp, size, fs*0.05);
    
    float mins = filter->findMinimum(ecg, size);
    float maxs = filter->findMaximum(ecg, size);
    float range = maxs-mins;
    
    int buildingmask = 0;
    int consecutive = 0;
    int winstart = 0;
    int winend = 0;
    for(int i=0; i<size; i++) {
        float val = sqrt(tmp[i]);
        ecgref[i] = val;
        if(buildingmask>0) buildingmask--;
        
        if(val > range*0.02) {
            consecutive++;
            if(consecutive>2) {
                if(buildingmask==0) {
                    winstart = i;
                }
                buildingmask = fs/5;
            }
        } else {
            consecutive = 0;
        }
        if(buildingmask==1) {
            winend = i+1;
            
            tuple<int,int> mask(winstart,winend);
            highfreqmasks.push_back(mask);
            
            winstart = 0;
            winend = 0;
        }
    }
    
    //extend masks to edges of signal if it is within the edges
    if(highfreqmasks.size() > 1) {
        if(get<0>(highfreqmasks[0]) < 3*fs) {
            get<0>(highfreqmasks[0]) = 0;
        }
        if(get<0>(highfreqmasks[highfreqmasks.size()-1]) > size-3*fs-1) {
            get<0>(highfreqmasks[highfreqmasks.size()-1]) = size-1;
        }
    }
    
    for(int i=0; i<highfreqmasks.size(); i++) {
        masks.push_back(highfreqmasks[i]);
    }
}

void ECGdetector::lowpowmask() {
    float* tmp = (float*)malloc(sizeof(float)*size);
    memcpy(tmp,ecg,sizeof(float)*size);
    
    filter->Buttersworth(tmp, size, fs, 8, 25);
    
    for(int i=0; i<size; i++) {
        tmp[i] = tmp[i]*tmp[i];
    }
    
    filter->rollmean(tmp, size, fs*0.05);
    
    float mins = filter->findMinimum(ecg, size);
    float maxs = filter->findMaximum(ecg, size);
    float range = maxs-mins;
    
    int buildingmask = 0;
    int consecutive = 0;
    int winstart = 0;
    int winend = 0;
    for(int i=0; i<size; i++) {
        float val = sqrt(tmp[i]);
        //ecgref[i] = val;
        if(buildingmask>0) buildingmask--;
        
        if(val < range*0.005) {
            consecutive++;
            if(consecutive>2) {
                if(buildingmask==0) {
                    winstart = i;
                }
                buildingmask = 2;
            }
        } else {
            consecutive = 0;
        }
        if(buildingmask==1) {
            winend = i+1;
            
            tuple<int,int> mask(winstart,winend);
            lowpowmasks.push_back(mask);
            
            winstart = 0;
            winend = 0;
        }
    }
    
    //extend masks to edges of signal if it is within the edges
    if(lowpowmasks.size() > 1) {
        if(get<0>(lowpowmasks[0]) < 3*fs) {
            get<0>(lowpowmasks[0]) = 0;
        }
        if(get<0>(lowpowmasks[lowpowmasks.size()-1]) > size-3*fs-1) {
            get<0>(lowpowmasks[lowpowmasks.size()-1]) = size-1;
        }
    }
    
    if(lowpowmasks.size()>0) {
        for(int i=lowpowmasks.size()-1; i>=0; i--) {
            int begin = get<0>(lowpowmasks[i]);
            int end = get<1>(lowpowmasks[i]);
            
            if(end-begin < 3*fs) {
                lowpowmasks.erase(lowpowmasks.begin()+i);
            }
        }
    }
    
    for(int i=0; i<lowpowmasks.size(); i++) {
        masks.push_back(lowpowmasks[i]);
    }
}

bool compareMasks(tuple<int,int> fm1, tuple<int,int> fm2)
{
    return (get<0>(fm1) < get<0>(fm2));
}

void ECGdetector::mergemasks() {
    
    //cout << "Masks size: " << masks.size() << endl;
    
    bool done = false;
    
    sort(masks.begin(),masks.end(),compareMasks);
    
    // for(int i=0; i<masks.size(); i++) {
    //     cout << "mask " << i << "[" << get<0>(masks[i]) << "," << get<1>(masks[i]) << "]" << endl;
    // }
    
    while(!done && masks.size()>1) {
        done = true;
        for(int i=0; i<masks.size()-1; i++) {
            int endmask1 = get<1>(masks[i]);
            int beginmask2 = get<0>(masks[i+1]);

            if(beginmask2 < endmask1) {
                done = false;
                if(get<1>(masks[i]) < get<1>(masks[i+1])) {
                    get<1>(masks[i]) = get<1>(masks[i+1]);
                }
                masks.erase(masks.begin()+i+1);
                break;
            }
            
            if(beginmask2-endmask1 < 5*fs) {
                done = false;
                get<1>(masks[i]) = get<1>(masks[i+1]);
                masks.erase(masks.begin()+i+1);
                break;
            }
        }
    }
    
    if(masks.size() > 1) {
        if(get<0>(masks[0]) < 3*fs) {
            get<0>(masks[0]) = 0;
        }
        if(get<0>(masks[masks.size()-1]) > size-3*fs-1) {
            get<0>(masks[masks.size()-1]) = size-1;
        }
    }
    
    if(masks.size()>0) {
        for(int i=masks.size()-1; i>=0; i--) {
            int begin = get<0>(masks[i]);
            int end = get<1>(masks[i]);
            
            if(end-begin < fs) {
                //masks.erase(masks.begin()+i);
            }
        }
    }
}

void ECGdetector::detectPeaksInWindow(int start) {
    
    float THRS_NOISE = filter->findMean(&ecgfilt[start], fs*2)*0.5; //0.2
    float THRS_SIG = filter->findMaximum(&ecgfilt[start], fs*2)*0.33; //1.5
    float NOISE_LVL = THRS_NOISE;
    float SIG_LVL = THRS_SIG;
    
    float THRS_NOISE1 = filter->findMean(&ecg[start], fs*2)*0.5; //0.2
    float THRS_SIG1 = filter->findMaximum(&ecg[start], fs*2)*0.33; //1.5
    float NOISE_LVL1 = THRS_NOISE1;
    float SIG_LVL1 = THRS_SIG1;
    int lookahead = fs/5;
    
    int lastpeak = -fs/5 + 1 + start;
    int curpeak = 0;
    float lastibi = fs;
    
    float* ibis = (float*)malloc(sizeof(float)*5);
    ibis[0] = ibis[1] = ibis[2] = ibis[3] = ibis[4] = lastibi;
    int ibismean = lastibi;
    int ibisstd = fs;
    int nwindow = 0;
    
    //cout << "masks size: " << masks.size() << endl;
    
    while(lastpeak+lookahead < size) {
        
        int estimation = lastpeak+ibismean;
        
        //Define clean and noisy intervals
        int window_clean_start = (int)min((float)(size-5),(float)(lastpeak+fs/5));
        int window_clean_end = (int)min((float)size,(float)lastpeak+ibismean*1.66f);
        int window_noise_start = (int)max((float)window_clean_start,(float)(estimation-ibisstd));
        int window_noise_end = (int)min((float)window_clean_end,(float)(estimation+ibisstd));
        
        //cout << "wcs:" << window_clean_start << " wce:" << window_clean_end << " wns:" << window_noise_start << " wne:" << window_noise_end << endl;
        
        //Calculate entropy for region
        //float entropy = filter->getSampleEntropy(&ecgraw[window_clean_start], window_clean_end-window_clean_start, 0.2);
        
        float entropy = filter->getNoiseLevel(&ecg[window_clean_start], window_clean_end-window_clean_start, fs);
        //float entropy = filter->getNoiseLevel2(&ecg[window_clean_start], window_clean_end-window_clean_start, fs, THRS_SIG);
        //cout << "noiselevel: " << entropy;
        //cout << " signal level: " << THRS_SIG;
        entropy = min(1.0f,entropy/THRS_SIG);
        //cout << " entropy: " << entropy << endl;
        
        if(rpeaks.size()<5) {
            entropy = 0.0;
        }
        
        //Calculate actual search window using linear interpolation between maxima
        int wstart = (int)((entropy)*(float)window_noise_start + (1.0f-entropy)*(float)window_clean_start);
        int wend = (int)((entropy)*(float)window_noise_end + (1.0f-entropy)*(float)window_clean_end);
        
        if(wstart > size-10 && wend > size-1) {
            //Window is too close to the end
            break;
        }
        wstart = (int)min((float)wstart,(float)(size-10));
        wend = (int)min((float)wend,(float)(size-1));
        int wlen = wend-wstart;
        
        //cout << " [" << ibis[0] << "," << ibis[1] << "," << ibis[2] << "," << ibis[3] << "," << ibis[4] << "] m:" << ibismean << endl;
        
        //cout << "s:" << entropy << " ";
        //cout << "[" << window_clean_start << " - " << wstart << " - " << window_noise_start << "] ";
        //cout << "[" << window_clean_end << " - " << wend << " - " << window_noise_end << "] " << endl;
        
        //cout << "lp:" << lastpeak << " ibismean:" << ibismean << " ibisstd:" << ibisstd << " ws:" << wstart << " we:" << wend << endl;
        
        bool insidemask = false;
        for(int i=0; i<masks.size(); i++) {
            int begin = get<0>(masks[i]);
            int end = get<1>(masks[i]);
            
            if((wstart > begin && wstart < end) && (wend > begin && wend < end)) {
                // << "Restart at " << end << endl;
                if(end != size) {
                    detectPeaksInWindow(end);
                }
                return;
            }
        }
        
        if(!insidemask) {
            windowstart.push_back(wstart);
            windowend.push_back(wend);
            estimations.push_back(estimation);
            
            float *win = (float*)malloc(sizeof(float)*(window_clean_end-window_clean_start));
            memcpy(win,&ecg[window_clean_start],sizeof(float)*(window_clean_end-window_clean_start));
            filter->Buttersworth(win, window_clean_end-window_clean_start, fs, 20, 40);
            
            for(int i=0; i<(window_clean_end-window_clean_start); i++) {
                win[i] = abs(win[i]);
            }
            filter->rollmean(win,window_clean_end-window_clean_start,fs*0.1);
            
            for(int i=window_clean_start; i<window_clean_end; i++) {
                float x = (float)(i-estimation);
                //gaussians[i] = win[i-window_clean_start];
                gaussians[i] = exp((-x*x)/(2.0*(wlen/1.5)*(wlen/1.5)));
                ecggauss[i] = ecgfilt[i]*gaussians[i];
                ecgnoise[i] = win[i-window_clean_start];
            }
            entropies.push_back(entropy);
            
            vector<tuple<int,int,float,bool>> fiducialmarks;
            for(int i=window_clean_start; i<window_clean_end; i++) {
                //Search for peak in signal
                if(ecggauss[i-1] < ecggauss[i] && ecggauss[i+1] < ecggauss[i]) {
                    fiducialmarks.push_back({i,abs(estimation-i),ecggauss[i],false});
                }
            }
            
            //cout << "fm size: " << fiducialmarks.size() << endl;
            
            bool done = false;
            while(!done) {
                //cout << "round" << endl;
                float maxpeak = 0;
                int maxpeakind = -1;
                int maxpeaki = 0;
                for(int i=0; i<fiducialmarks.size(); i++) {
                    float ind = get<0>(fiducialmarks[i]);
                    float h = get<2>(fiducialmarks[i]);
                    //cout << ind << "," << h << endl;
                    
                    if(h>maxpeak && !get<3>(fiducialmarks[i])) {
                        maxpeak = h;
                        maxpeakind = ind;
                        maxpeaki = i;
                    }
                }
                
                //cout << "max:" << maxpeakind << " ----" << endl;
                if(maxpeakind==-1) break;
                get<3>(fiducialmarks[maxpeaki]) = true;
                
                for(int i=fiducialmarks.size()-1; i>=0; i--) {
                    float ind = get<0>(fiducialmarks[i]);
                    if(abs(ind-maxpeakind) < 0.2*fs && ind!=maxpeakind) {
                        //cout << ind << " d:" << abs(ind-maxpeakind) << endl;
                        fiducialmarks.erase(fiducialmarks.begin()+i);
                    }
                }
                //cout << "fm size: " << fiducialmarks.size() << endl;
            }
            //cout << "fm size: " << fiducialmarks.size() << endl;
            
            bool qrsfound = false;
            bool qrssaved = false;
            for(int k=0; k<fiducialmarks.size(); k++) {
                    
                int i = get<0>(fiducialmarks[k]);
                int dist = get<1>(fiducialmarks[k]);
                
                //cout << "FM #" << k << ":" << i << " d:" << dist << endl;
                thresholds.push_back(THRS_SIG);
                
                int peakpos = findRawPeak(i,wstart);
                
                if(ecggauss[i] > THRS_SIG && !qrsfound) {
                    //Case 1, peak is larger than set threshold
                    qrsfound = true;
                    
                    curpeak = i;
                    lastibi = i-lastpeak;
                    //cout << "[" << rpeaks.size() << "] Found filt at " << (float)i/(float)fs << "s" << endl;
                    
                    //rpeaks.push_back(peakpos);
                    if(abs(ecg[peakpos]) > THRS_SIG1) {
                        rpeaks.push_back(peakpos);
                        qrssaved = true;
                        // cout << "✅ added normal:" << rpeaks.size() << ", " << lastibi << endl;
                        // cout << "[" << rpeaks.size() << "] Found filt at " << (float)curpeak/(float)fs << "s" << endl;
                        // cout << "Found raw at " << (float)peakpos/(float)fs << "s" << endl;
                        SIG_LVL1 = 0.125*ecg[peakpos] + 0.875*SIG_LVL1;
                    }
                    
                    SIG_LVL = 0.125*ecgfilt[i] + 0.875*SIG_LVL;
                    
                    break;
                    
                } else {
                    //Case 2, peak is smaller than signal thres, but larger than noise thres
                    
                    NOISE_LVL = 0.125*ecgfilt[i] + 0.875*NOISE_LVL;
                    NOISE_LVL1 = 0.125*ecg[peakpos] + 0.875*NOISE_LVL1;
                }
            }
            if(!qrsfound) {
                for(int k=0; k<fiducialmarks.size(); k++) {
                    
                    int i = get<0>(fiducialmarks[k]);
                        
                    if(ecggauss[i] > THRS_NOISE) {
                        //Found a peak with the lookback method
                        qrsfound = true;
                        
                        int peakpos = findRawPeak(i,wstart);
                        
                        curpeak = i;
                        lastibi = i-lastpeak;
                        //cout << "[" << rpeaks.size() << "] Found filt during lookback at " << (float)i/(float)fs << "s" << endl;
                        
                        
                        if(abs(ecg[peakpos]) > THRS_NOISE1) {
                            rpeaks.push_back(peakpos);
                            qrssaved = true;
                            // cout << "⚠️ added lookback:" << rpeaks.size() << ", " << lastibi << endl;
                            // cout << "Found raw during lookback at " << (float)peakpos/(float)fs << "s" << endl;
                            SIG_LVL1 = 0.25*ecg[peakpos] + 0.75*SIG_LVL1;
                        }
                        
                        //Adjust normalized signal threshold
                        SIG_LVL = 0.25*ecgfilt[i] + 0.75*SIG_LVL;
                        
                        break;
                    }
                }
            }
            
            if(!qrsfound) {
                
                float maxfilt = 0;
                int maxfiltpos = min(size-1,estimation);
                
                for(int i=wstart; i<wend; i++) {
                    if(ecggauss[i] > maxfilt) {
                        maxfilt = ecggauss[i];
                        maxfiltpos = i;
                    }
                }
                curpeak = findRawPeak(maxfiltpos,wstart);
                lastibi = curpeak-lastpeak;
                
                if(abs(ecg[curpeak]) > THRS_SIG1 && curpeak-lastpeak > fs/5) {
                    rpeaks.push_back(curpeak);
                    qrssaved = true;
                    lastibi = curpeak - rpeaks[rpeaks.size()-2];
                    // cout << "‼️ added last resort:" << curpeak-lastpeak << ", ibi:" << lastibi << endl;
                    // //cout << "maxfiltpos: " << maxfiltpos << " size:" << size << endl;
                    // cout << "window: [" << wstart-lastpeak << " - " << wend-lastpeak << "] l:" << wlen << endl;
                    // cout << "Found raw during lookback at " << (float)curpeak/(float)fs << "s" << endl;
                    SIG_LVL1 = 0.125*ecg[curpeak] + 0.875*SIG_LVL1;
                }
                
                //filtonlypeaks.push_back(curpeak);
                // cout << "found nothing at " << (float)estimation/(float)fs << "s" << endl;
            }
            
            if(NOISE_LVL!=0 && SIG_LVL!=0) {
                THRS_SIG = NOISE_LVL + 0.25*(abs(SIG_LVL - NOISE_LVL));
                THRS_NOISE = 0.5*THRS_SIG;
            }
            if(NOISE_LVL1!=0 && SIG_LVL1!=0) {
                THRS_SIG1 = NOISE_LVL1 + 0.25*(abs(SIG_LVL1 - NOISE_LVL1));
                THRS_NOISE1 = 0.5*THRS_SIG1;
            }
            
            if(qrssaved) {
                ibis[0] = ibis[1]; ibis[1] = ibis[2]; ibis[2] = ibis[3];
                ibis[3] = ibis[4]; ibis[4] = lastibi;
            } else {
                //filtonlypeaks.push_back(curpeak);
            }
            measurements.push_back(estimation);
        } else {
            curpeak = estimation;
        }
            
        ibismean = max(fs/5,(int)filter->findMean(ibis,5));
        ibisstd  = max(ibismean*0.1f,filter->findStd(ibis,5));
        
        lastpeak = curpeak;
        //cout << "======" << endl;
        
        nwindow++;
    }
           
}


float ECGdetector::getQuality() {
    int hrl = size/fs;
    float* hr = (float*)calloc((hrl),sizeof(float));
    float* residue = (float*)calloc((hrl),sizeof(float));
    int previndex = 0;
    
    for(int i=1; i<rpeaks.size(); i++) {
        int peak = rpeaks[i];
        int index = peak/fs;
        
        if(index < hrl) {
            if(index-previndex > 1) {
                for(int k=previndex; k<index; k++) {
                    hr[k+1] = (60*fs)/(float)(rpeaks[i] - rpeaks[i-1]);
                }
                previndex = index;
            } else {
                hr[index] = (60*fs)/(float)(rpeaks[i] - rpeaks[i-1]);
                previndex = index;
            }
        }
    }
    
    hr[0] = hr[1];
    hr[(hrl)-1] = filter->findMean(&hr[hrl-7],5);
    
    memcpy(residue,hr,sizeof(float)*hrl);
    
    filter->median(residue,hrl,5);
    
    return filter->findMean(residue, hrl);
}
