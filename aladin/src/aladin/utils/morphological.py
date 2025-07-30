import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqs
from scipy.ndimage.filters import generic_filter1d,maximum_filter1d,minimum_filter1d


def dilation_fast(inp, elem, axis =1):
    a = np.pad(inp,(elem.shape[-1]-1,0),'edge')
    shape = a.shape[:-1] + (a.shape[-1] - elem.shape[-1] + 1, elem.shape[-1])
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    rolling = rolling+elem
    return np.max(rolling,axis=axis)

def dilation_fast_multiscale(inp, N, *elems):
    a = inp
    for i in range(N):
        a = np.pad(a,(elems[i].shape[-1]-1,0),'edge')
        shape = a.shape[:-1] + (a.shape[-1] - elems[i].shape[-1] + 1, elems[i].shape[-1])
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        rolling = rolling+elems[i]
        a = np.max(rolling, axis=1)[elems[i].shape[-1]:]
    return a

def erosion_fast(inp, elem, axis =1):
    a = np.pad(inp,(0,elem.shape[-1]-1),'edge')
    shape = a.shape[:-1] + (a.shape[-1] - elem.shape[-1] + 1, elem.shape[-1])
    strides = a.strides + (a.strides[-1],)
    rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    rolling = rolling-elem
    return np.min(rolling,axis=axis)

def erosion_fast_multiscale(inp, N, *elems):
    a = inp
    for i in range(N):
        a = np.pad(a,(0,elems[i].shape[-1]-1),'edge')
        shape = a.shape[:-1] + (a.shape[-1] - elems[i].shape[-1] + 1, elems[i].shape[-1])
        strides = a.strides + (a.strides[-1],)
        rolling = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
        rolling = rolling-elems[i]
        a = np.min(rolling, axis=1)[0:-elems[i].shape[-1]]
    return a

def dilation(input, elem):
    
    l = int(elem.shape[0])
    output = np.zeros(input.shape[0]+l,dtype='float')
    tmp = np.pad(input,(l,0),'edge')

    for n in range(l,input.shape[0]+l):
        output[n] = np.max(tmp[(n-l+1):(n+1)])

    return output[l:input.shape[0]+l]

def erosion(input, elem):

    l = int(elem.shape[0])
    output = np.zeros(input.shape[0]+l,dtype='float')
    tmp = np.pad(input,(0,l),'edge')
    for n in range(0,input.shape[0]):
        output[n] = np.min(tmp[(n):(n+l)])

    return output[0:input.shape[0]]

def dilationcentered(input, elem):
    
    l = int(elem.shape[0]/2)
    output = np.zeros(input.shape[0]+l*2,dtype='float')
    tmp = np.pad(input,(l,l),'edge')

    for n in range(l,input.shape[0]+l):
        output[n] = np.max(tmp[(n-l):(n+l)])

    return output[l:-l]

def erosioncentered(input, elem):

    l = int(elem.shape[0]/2)
    output = np.zeros(input.shape[0]+l*2,dtype='float')
    tmp = np.pad(input,(l,l),'edge')

    for n in range(l,input.shape[0]+l):
        output[n] = np.min(tmp[(n-l):(n+l)])

    return output[l:-l]


def opening(input, elem):
    tmp = erosion_fast(input,elem)
    return dilation_fast(tmp,elem)  

def opening_multiscale(input, N, *elems):
    tmp = input

    for i in range(N):
        tmp = erosion_fast(tmp, elems[i])
        tmp = dilation_fast(tmp, elems[i])

    return tmp
    
def closing(input, elem):
    tmp = dilation_fast(input,elem)
    return erosion_fast(tmp,elem)  

def closing_multiscale(input, N, *elems):
    tmp = input

    for i in range(N):
        tmp = dilation_fast(tmp, elems[i])
        tmp = erosion_fast(tmp, elems[i])

    return tmp

def openingcentered(input, elem):
    tmp = erosioncentered(input,elem)
    return dilationcentered(tmp,elem)  
    
def closingcentered(input, elem):
    tmp = dilationcentered(input,elem)
    return erosioncentered(tmp,elem) 

def top_hat(input, elem):
    return input - opening(input,elem) 

def bottom_hat(input, elem):
    return input - closing(input,elem) 

def topbottom_average(input, elem):
    return input - (opening(input, elem) + closing(input, elem)) / 2

def topbottom_average_centered(input, elem):
    return input - (openingcentered(input, elem) + closingcentered(input, elem)) / 2

def filtering(input,elem):
    return (closing(opening(input,elem),elem) + opening(closing(input,elem),elem)) / 2

def PVE(input, elem):
    return input - closing(opening(input,elem),elem)

def filter3M(input,N,*elems):
    inp1 = input
    inp2 = input
    tmp = np.zeros(input.shape)

    assert len(elems)==N, f'number of structure elements does not match with N'

    for i in range(N):
        inp1 = opening_multiscale(inp1,i+1,*elems[0:i+1])
        inp2 = closing_multiscale(inp2,i+1,*elems[0:i+1])

        tmp += (inp1-inp2)*2**(-(N+1-i))

    tmp += inp2

    return tmp

def update_fsp(oldfsp,newfsp,alpha):
    
    return (1-alpha)*oldfsp + alpha*newfsp


def get_default_fsp(input, fs):
    initmin = np.min(input)
    initmax = np.max(input)

    # fps = np.zeros((5,2))
    # fps[0][0] = 0
    # fps[0][1] = 0
    # fps[1][0] = 0
    # fps[1][1] = int(fs*0.09*0.25)
    # fps[2][0] = initmax
    # fps[2][1] = int(fs*0.09*0.5)
    # fps[3][0] = 0
    # fps[3][1] = int(fs*0.09*0.75)
    # fps[4][0] = 0
    # fps[4][1] = int(fs*0.09)
    fps = np.zeros((5,2))
    fps[0][0] = 0
    fps[0][1] = 0
    fps[1][0] = 0
    fps[1][1] = 1
    fps[2][0] = initmax
    fps[2][1] = int(fs*0.09*0.5)
    fps[3][0] = 0
    fps[3][1] = int(fs*0.09)-1
    fps[4][0] = 0
    fps[4][1] = int(fs*0.09)

    return fps

#def update_fps(fps,newfps):


def generate_elem(fps):

    if(fps[4][1]%2==0):
        fps[4][1]-=1

    oq = np.linspace(fps[0][0],fps[1][0],int(fps[1][1]))
    qr = np.linspace(fps[1][0],fps[2][0],int(fps[2][1]-fps[1][1])+2)
    rs = np.linspace(fps[2][0],fps[3][0],int(fps[3][1]-fps[2][1])+2)
    se = np.linspace(fps[3][0],fps[4][0],int(fps[4][1]-fps[3][1]))
    
    elem = np.concatenate(([fps[0][0]],qr[1:qr.shape[0]-1],[fps[2][0]],rs[1:rs.shape[0]-1]))

    #elem = np.zeros((int(fps[4][1])))
    return elem

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def preprocessing(input,fs):
    order = 4
    cutoff = 50
    return butter_lowpass_filter(input, cutoff, fs, order)

def extract_active_periods(input,fs):
    ee = 0.0001
    input = input[int(fs*0.5):int(fs*1.5)]
    input[abs(input)<ee] = 0
    tmp = (input!=0)*1

    periods = []
    curper = 0
    start = -1
    
    for i in range(1,tmp.shape[0]):
        if((tmp[i])-tmp[i-1]==1):
            start = i

        if((tmp[i])-tmp[i-1]==-1):
            if(start>0 and i-1-start > fs*0.07):
                periods.append([start,i-1,i-1-start])
                start = -1

    return periods
    
def extract_characteristics(input,period):

    R1 = np.max(input[period[0]:period[1]])
    R1pos = period[0]+np.argmax(input[period[0]:period[1]])

    R2 = np.min(input[period[0]:period[1]])
    R2pos = period[0]+np.argmin(input[period[0]:period[1]])

    Q1 = np.min(input[period[0]:R1pos])
    Q1pos = np.argmin(input[period[0]:R1pos])

    Q2 = np.max(input[period[0]:R2pos])
    Q2pos = np.argmax(input[period[0]:R2pos])

    S1 = np.min(input[R1pos:period[1]])
    S1pos = np.argmin(input[R1pos:period[1]])

    S2 = np.max(input[R2pos:period[1]])
    S2pos = np.argmax(input[R2pos:period[1]])

    pos_peak = -Q1+(R1-Q1)+(R1-S1)-S1
    neg_peak = Q2+(Q2-R2)+(S2-R2)+S2

    if(pos_peak > neg_peak):

        # plt.plot(input[period[0]:period[1]])
        # plt.plot([R1pos-period[0]],[R1],marker="o",color='red')
        # plt.plot([Q1pos],[Q1],marker="o",color='blue')
        # plt.plot([S1pos+R1pos-period[0]],[S1],marker="o",color='green')
        # plt.show()
        print("POSITIVE")
        fps = np.zeros((5,2))
        fps[0][0] = 0
        fps[0][1] = 0
        fps[1][0] = Q1
        fps[1][1] = Q1pos
        fps[2][0] = R1
        fps[2][1] = R1pos-period[0]
        fps[3][0] = S1
        fps[3][1] = S1pos+R1pos-period[0]
        fps[4][0] = 0
        fps[4][1] = period[1]-period[0]

        # fps = np.zeros((5,2))
        # fps[0][0] = 0
        # fps[0][1] = 0
        # fps[1][0] = 0
        # fps[1][1] = 1
        # fps[2][0] = R1
        # fps[2][1] = int(360*0.09*0.5)
        # fps[3][0] = 0
        # fps[3][1] = int(360*0.09)-1
        # fps[4][0] = 0
        # fps[4][1] = int(360*0.09)
        return fps
    else:
        # plt.plot(input[period[0]:period[1]])
        # plt.plot([R2pos-period[0]],[R2],marker="o")
        # plt.plot([Q2pos],[Q2],marker="o")
        # plt.plot([S2pos+R1pos-period[0]],[S2],marker="o")
        # plt.show()
        print("NEGATIVE")
        fps = np.zeros((5,2))
        fps[0][0] = 0
        fps[0][1] = 0
        fps[1][0] = -Q2
        fps[1][1] = Q2pos
        fps[2][0] = -R2
        fps[2][1] = R2pos-period[0]
        fps[3][0] = -S2
        fps[3][1] = S2pos+R2pos-period[0]
        fps[4][0] = 0
        fps[4][1] = period[1]-period[0]
        return fps 