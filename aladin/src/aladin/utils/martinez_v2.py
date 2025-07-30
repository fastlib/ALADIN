#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated: 30/04/2018

This library contains all the functions required to perform wavelet-based
ECG delineation. The full delineation can be made with the function:
    
    P, QRS, T = signalDelineation(sig,fs):
        
Ideally, the function should be used with a maximum of 2^16 samples. This 
guarantees that the thresholds are properly updated and avoids errors due 
to transient, high magnitude, noise.

The delineation algorithm is explained in "A wavelet-based ECG delineator: 
evaluation on standard databases" by Martinez et al. (2003). The maximum modulus 
lines are found as explained in the paper: "Detection of ECG characteristic 
points using wavelet transforms." by Li et al. (1995). Users are invited to 
consult these papers in case there are doubts about the algorithms.

All the functions contained in this library were developed in the Multiscale
Cardiovascular Engineering Group at University College London (MUSE-UCL)
by Carlos Ledezma.

This work is protected by a Creative Commons Attribution-ShareAlike 4.0 
International license (https://creativecommons.org/licenses/by-sa/4.0/)
"""
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt
import wfdb
import os


def get_records_qt(folder):

    records = np.loadtxt(os.path.join(folder, "RECORDS_old"), dtype=str)

    objs = []

    for rec in records:
        #add leading zero

        sig = wfdb.rdrecord(os.path.join(folder, rec))
        ann = wfdb.rdann(os.path.join(folder, rec), 'q1c')
        ann.symbol = np.array(ann.symbol)
        qrsinds = np.where((ann.symbol == 'N') | (ann.symbol=='A') | (ann.symbol=='V') | (ann.symbol=='Q'))[0]
        qrspos = ann.sample[qrsinds]
        qrsonsets = [ann.sample[i-1] if ann.symbol[i-1]=='(' else np.nan for i in qrsinds]
        qrsoffsets = [ann.sample[i+1] if ann.symbol[i+1]==')' else np.nan for i in qrsinds]
        pinds = np.where(ann.symbol == 'p')[0]
        ppos = ann.sample[pinds]
        ponsets = [ann.sample[i-1] if ann.symbol[i-1]=='(' else np.nan for i in pinds]
        poffsets = [ann.sample[i+1] if ann.symbol[i+1]==')' else np.nan for i in pinds]
        tinds = np.where(ann.symbol == 't')[0]
        tpos = ann.sample[tinds]
        tonsets = [ann.sample[i-1] if ann.symbol[i-1]=='(' else np.nan for i in tinds]
        toffsets = [ann.sample[i+1] if ann.symbol[i+1]==')' else np.nan for i in tinds]
        fs = sig.fs

        #get signal
        signal = sig.p_signal
        signal1 = np.array(signal[:,0])
        signal2 = np.array(signal[:,1])
        obj = {"record": rec, 
               "signal1": signal1, 
               "signal2": signal2, 
               "fs":fs, 
               "p": [ponsets, ppos, poffsets],
               "qrs": [qrsonsets, qrspos, qrsoffsets], 
               "t": [tonsets, tpos, toffsets]
        }

        objs.append(obj)

    return objs

def waveletH(w):
    '''    
    H = waveletH(w)
    
    Constructs the low-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    
    Input:
        w (numpy array): contains the frequency points, in radians, that 
        will be used to construct the filter. w must be between 0 and 2pi.
    
    Output:
        H (numpoy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3. 
    '''
    return np.exp(1j*w/2) * np.cos(w/2) ** 3 
    
    
def waveletG(w):
    '''    
    G = waveletG(w)
    
    Constructs the high-pass filters required for the wavelet-based ECG delineator
    at a sampling frequency of 250 Hz.
    
    Input:
        w (numpy array): contains the frequency points, in radians, that 
        will be used to construct the filter. w must be between 0 and 2pi.
    
    Output:
        G (numpy array): contains the function H(w) = exp(1j*w/2) * cos(w/2)**3. 
    '''
    
    return 4j*np.exp(1j*w/2)*np.sin(w/2)

def waveletFilters(N,fs):
    '''    
    Q = waveletFilters(N,fs)
    
    Creates the filters required to make the wavelet decomposition using the
    algorithme-a-trous. This routine first creates the filters at 250 Hz and 
    resamples them to the required sampling frequency.
    
    Inputs:
        N (int): the number of samples of the signal that will be decomposed.
        
        fs (float): the sampling frequency of the signal that will be decomposed.
        
    Output:
        Q (list): contains five numpy arrays [Q1, Q2, Q3, Q4, Q5] that are the 
        five filters required to make the wavelet decomposition.
    '''
    
    # M is the number of samples at 250 Hz that will produce N samples after 
    # re-sampling the filters   
    M =  N* (250/fs)
    w = np.arange(0,2*np.pi, 2*np.pi/M) # Frequency axis in radians
    
    # Construct the filters at 250 Hz as specified in the paper
    Q = [waveletG(w)]    
    for k in range(2,6):
        G = waveletH(w)
        for l in range(1,k-1):
            G *= waveletH(2**l * w)
        Q += [waveletG(2 ** (k-1) * w) * G]
        
    # Resample the filters from 250 Hz to the desired sampling frequency
    for i in range(len(Q)):
        Q[i] = np.fft.fft(resample(np.fft.ifft(Q[i]),N))
    
    return Q    
    
def waveletDecomp(sig,Q):
    '''
    w = waveletDecomp(sig,Q)
    
    Performs the wavelet decomposition of a signal using the algorithme-a-trous. 
    
    Inputs:
        sig (numpy array): contains the signal to be decomposed.
        
        Q (list): contains the filters (numpy arrays) that will decompose the
        signal. It is recommended that Q is generated using the waveletFilters 
        function provided in this library.
        
    Output:
        
        w (list): numpy arrays [w1, w2, w3, w4, w5] containing the wavelet
        decomposition of the signal at scales 2^1..2^5. 
    '''
       
    w = []
    
    # Apply the filters in the frequency domain and return the result in the time domain
    for q in Q:
        w += [np.real(np.fft.ifft(np.fft.fft(sig) * q))]
        
    return w

def verifyMaximumModulusLine(w, prevns, fs, eps):
    
    n = []

    for prevn in prevns:
        win = int(0.120*fs)
        srch_idx_start = prevn-win
        srch_idx_end = prevn+win

        cand = findMaximumModulusLines(w[srch_idx_start:srch_idx_end],fs,eps)
        cand = np.array([i + srch_idx_start for i in cand if i + srch_idx_start not in n])
        #print(prevn, cand)

        if len(cand) == 0:
            continue
        
        if len(cand) == 1:
            n += [cand[0]]
            continue
        
        sqw = np.abs(w)
        max_idx = np.argmax(sqw[cand])
        maximum = sqw[cand[max_idx]]

        islarger = True
        for i in cand:
            if i != max_idx and sqw[i]*1.2 > maximum:
                islarger = False
                break
        
        #maximum is at least 1.2 times larger than the other candidates
        if islarger:
            #print("larger: ", cand[max_idx])
            n += [cand[max_idx]]
        else:
            #choose closest 
            dist = np.abs(cand - prevn)
            closest_val = np.min(dist)
            closest_set = np.where(dist == closest_val)[0]
            max_idx = np.argmax(sqw[cand[closest_set]])
            n += [cand[closest_set[max_idx]]]

    return n

def findMaximumModulusLines(w, fs, eps):

    sqw = np.abs(w)
    n = []
    for i in range(1,len(sqw)-1):

        if sqw[i] > eps and sqw[i] > sqw[i-1] and sqw[i] > sqw[i+1]:
            n += [i]

    return n

def removeIsolationLines(n1,fs):
    thres = int(0.120*fs)

    todel = []
    for i in range(len(n1)):
        dist = 0
        if i == 0:
            dist = n1[i+1] - n1[i]
        elif i == len(n1)-1:
            dist = n1[i] - n1[i-1]
        else:
            dist = min(n1[i+1] - n1[i], n1[i] - n1[i-1])

        if dist > thres:
            todel += [i]

    n1 = np.delete(n1,todel)

    return n1

def removeRedundantLines(w, n3,fs):
    win = int(0.120*fs)
    n3 = np.array(n3)

    done = False
    while done == False:
        for i in range(len(n3)):
            srch_idx_start = n3[i]-win
            srch_idx_end = n3[i]+win

            neighbours = n3[np.where((n3 > srch_idx_start) & (n3 < srch_idx_end))]
            neighbours = neighbours[neighbours != n3[i]]
            #sort by distance
            #neighbours = np.array(sorted(neighbours, key=lambda x: abs(x-n3[i])))

            if len(neighbours) > 1:
                #if we have a positive maximum
                if w[n3[i]] > 0:
                    #find all negative maxima
                    minima = np.where(w[neighbours] < 0)[0]
                    if len(minima) > 1:
                        A1 = w[neighbours[minima[0]]]
                        A2 = w[neighbours[minima[1]]]
                        L1 = neighbours[minima[0]] - n3[i]
                        L2 = neighbours[minima[1]] - n3[i]
                        if A1/abs(L1) > 1.2*(A2/abs(L2)):
                            todel = np.where(n3 == neighbours[minima[1]])[0]
                        elif A2/abs(L2) > 1.2*(A1/abs(L1)):
                            todel = np.where(n3 == neighbours[minima[0]])[0]
                        else:
                            if np.sign(L1) == np.sign(L2): #on the same side
                                if abs(L2) > abs(L1):
                                    todel = np.where(n3 == neighbours[minima[1]])[0]
                                else:
                                    todel = np.where(n3 == neighbours[minima[0]])[0]
                            else: #on different sides
                                todel = np.where(n3 == neighbours[minima[0]])[0]

                        n3 = np.delete(n3,todel)
                        break

                else:

                    #find all positive maxima
                    maxima = np.where(w[neighbours] > 0)[0]
                    if len(maxima) > 1:
                        A1 = w[neighbours[maxima[0]]]
                        A2 = w[neighbours[maxima[1]]]
                        L1 = neighbours[maxima[0]] - n3[i]
                        L2 = neighbours[maxima[1]] - n3[i]
                        if A1/abs(L1) > 1.2*(A2/abs(L2)):
                            todel = np.where(n3 == neighbours[maxima[1]])[0]
                        elif A2/abs(L2) > 1.2*(A1/abs(L1)):
                            todel = np.where(n3 == neighbours[maxima[0]])[0]
                        else:
                            if np.sign(L1) == np.sign(L2): #on the same side
                                if abs(L2) > abs(L1):
                                    todel = np.where(n3 == neighbours[maxima[1]])[0]
                                else:
                                    todel = np.where(n3 == neighbours[maxima[0]])[0]
                            else: #on different sides
                                todel = np.where(n3 == neighbours[maxima[0]])[0]

                        n3 = np.delete(n3,todel)
                        break
            
            if i == len(n3)-1:
                done = True
    return n3

def findZeroCrossings(w, n, fs):
    r = []
    win = int(0.120*fs)
    #sort n in ascending order
    n = np.sort(n)

    for i in range(len(n)-1):
        if w[n[i]] > 0 and w[n[i+1]] < 0 and n[i+1]-n[i] < win:
            r += [np.argmin(abs(w[n[i]:n[i+1]])) + n[i]]

    for i in range(len(n)-1):
        if w[n[i]] < 0 and w[n[i+1]] > 0 and n[i+1]-n[i] < win:
            r += [np.argmin(abs(w[n[i]:n[i+1]])) + n[i]]
    
    r = np.sort(r)
    return r

def Rdetection(w,fs):

    eps = []
    for i in range(len(w)):
        eps += [np.sqrt(np.mean(w[i]**2))]
    eps[3] = 0.5 * eps[3]

    n4 = findMaximumModulusLines(w[3],fs,eps[3])
    n3 = verifyMaximumModulusLine(w[2],n4,fs,eps[2])
    n3 = removeRedundantLines(w[2],n3,fs)
    n2 = verifyMaximumModulusLine(w[1],n3,fs,eps[1])
    n1 = verifyMaximumModulusLine(w[0],n2,fs,eps[0])

    n1 = removeIsolationLines(n1,fs)

    r = findZeroCrossings(w[0],n1,fs)
    if len(r) > 1:
        r = blanking(r,fs)

    return r, [n1,n2,n3,n4]

def blanking(r, fs):
    win = int(0.200*fs)
    done = False

    while done == False:
        for i in range(len(r)-1):
            if (r[i+1] - r[i]) < win:
                r = np.delete(r,i+1)
                break
            if i == len(r)-2:
                done = True

    return r

def QRSdelineation(sig, fs):

    N = sig.shape[0] # Number of samples in the signal
    Q = waveletFilters(N,fs) # Create the filters to apply the algorithme-a-trous

    w = waveletDecomp(sig,Q) # Perform signal decomposition

    r, ns = Rdetection(w,fs) # Detect the R peaks

    return r, ns

def signalDelineation(sig,fs):

    #sig = sig[0:20*fs]
        
    N = sig.shape[0] # Number of samples in the signal
    Q = waveletFilters(N,fs) # Create the filters to apply the algorithme-a-trous

    w = waveletDecomp(sig,Q) # Perform signal decomposition

    r, ns = Rdetection(w,fs) # Detect the R peaks
    
    fig, ax = plt.subplots(len(w), 1, figsize=(12, 8), dpi=200, sharex=True)
    ax[0].plot(sig, lw=0.5)
    for i in range(len(w)-1):
        ax[i+1].plot(w[i], lw=0.5)
        ax[i+1].set_title('Wavelet scale: 2^{}'.format(i+1))
        eps = np.sqrt(np.mean(w[i]**2))
        if i == 3:
            eps = 0.5 * eps
        ax[i+1].axhline(eps, color='r', lw=0.5)
        for n in ns[i]:
            ax[i+1].axvline(n, color='g', lw=0.5)

    for i in range(len(r)):
        ax[0].axvline(r[i], color='r', lw=0.5)

    plt.tight_layout()
    plt.show()
    # R, n = Rdetection(w,fs) # Detect the R peaks
    
    # Q,S,QRSon,QRSend = QRSdelineation(R,w,n,fs) # Delineate QRS complex
    
    # T1,T2,Ton,Tend = Tdelineation(QRSend,w,fs) # Detect and delineate the T wave
    
    # P1,P2,Pon,Pend = Pdelineation(QRSon,w,fs) # Detect and delineate the P wave
    
    # # Create output arrays
    # QRS = np.array([QRSon,Q,R,S,QRSend]).T
    # Twav = np.array([Ton,T1,T2,Tend]).T
    # Pwav = np.array([Pon,P1,P2,Pend]).T
    
    #return w, n, Pwav, QRS, Twav

def test():
    recs = get_records_qt("../../Datasets/QT")

    rec = None
    for r in recs:
        if r["record"] == "sel15814":
            rec = r
            break
    signalDelineation(rec["signal1"], rec["fs"])


if __name__ == "__main__":
    test()