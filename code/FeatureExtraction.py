import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pywt
import scipy.io
import numpy as np
import scipy.signal
from scipy.stats import skew, kurtosis
def hw_stft(signal, wavelet='db4', level=4, nperseg=256):
    _, _, stft_matrix = stft(signal, fs=1.0, nperseg=nperseg)
    wavelet_object = pywt.Wavelet(wavelet)
    coeffs = pywt.wavedec(signal, wavelet_object, level=level)
    feature_matrices = [np.abs(stft_matrix)]
    for i in range(level + 1):
        rows, cols = coeffs[i].shape[0], stft_matrix.shape[1]
        coeff_matrix = np.zeros((rows, cols), dtype=np.complex_)
        coeff_matrix[:rows, :] = coeffs[i][:, np.newaxis]
        feature_matrices.append(np.abs(coeff_matrix))
    return feature_matrices
    import numpy as np
    import scipy.signal
    import pywt
    signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  
    window_size = 256  
    wavelet_name = 'db4'  
    levels = 5  
    def hybrid_wavelet_stft(signal, window_size, wavelet_name, levels):
        stft = []
        wavelet_transformed = []
        for i in range(0, len(signal) - window_size, window_size):
            segment = signal[i:i+window_size]
            f, t, Zxx = scipy.signal.stft(segment, nperseg=window_size)
            stft.append(Zxx)
            coeffs = pywt.wavedec(segment, wavelet_name, level=levels)
            wavelet_transformed.append(coeffs)
        return stft, wavelet_transformed
    stft_result, wavelet_result = hybrid_wavelet_stft(signal, window_size, wavelet_name, levels)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.pcolormesh(stft_result[0])
    plt.title('STFT Magnitude')
    plt.colorbar()
    plt.show()
def main():
    segmented_data = scipy.io.loadmat('Segmented_EEG_Signal.mat')['segmented_data']
    def extract_mean(signal):
        return np.mean(signal, axis=1)
    def extract_variance(signal):
        return np.var(signal, axis=1)
    def extract_skewness(signal):
        return skew(signal, axis=1)
    def extract_kurtosis(signal):
        return kurtosis(signal, axis=1)
    def extract_spectral_features(signal, fs):
        f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=256)
        peak_frequency = f[np.argmax(Pxx, axis=1)]
        spectral_entropy = -np.sum(Pxx * np.log2(Pxx + np.finfo(float).eps), axis=1)
        return peak_frequency, spectral_entropy
    fs = 1000  
    features = []
    for segment in segmented_data:
        mean_values = extract_mean(segment)
        variance_values = extract_variance(segment)
        skewness_values = extract_skewness(segment)
        kurtosis_values = extract_kurtosis(segment)
        peak_frequencies, spectral_entropies = extract_spectral_features(segment, fs)
        segment_features = np.hstack([mean_values, variance_values, skewness_values, kurtosis_values, peak_frequencies, spectral_entropies])
        features.append(segment_features)
    feature_matrix = np.array(features)
    print("Feature matrix shape:", feature_matrix.shape)
    print("Extracted features for the segmented EEG data:")
    print(feature_matrix)
    scipy.io.savemat('Segmented_EEG_Features.mat', {'segmented_eeg_features': feature_matrix})
