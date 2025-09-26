import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter
def EEGSignal_Preprocessing():
    mat_data = sio.loadmat('Input_Signal.mat')
    eeg_data = mat_data['o']  
    lowcut = 0.5  
    highcut = 30.0  
    fs = 1000.0  
    notch_frequency = 50.0  
    Q = 30.0  
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, fs, order=6)
    def notch_filter(data, fs, notch_frequency, Q):
        nyquist = 0.5 * fs
        f0 = notch_frequency / nyquist
        b, a = butter(1, [f0 - 1.0 / (2 * Q), f0 + 1.0 / (2 * Q)], 'stop')
        y = lfilter(b, a, data)
        return y
    filtered_data = notch_filter(filtered_data, fs, notch_frequency, Q)
    sio.savemat('Preprocessed_EEG_Signal.mat', {'preprocessed_data': filtered_data})
    import matplotlib.pyplot as plt
    mat_data = sio.loadmat('Preprocessed_EEG_Signal.mat')
    preprocessed_data = mat_data['preprocessed_data']  
    plt.plot(preprocessed_data[0, :500])
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title('Preprocessed EEG Signal')
    plt.grid(True)
    plt.show()
