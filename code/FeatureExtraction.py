import numpy as np
import scipy.io
import scipy.signal
import pywt
from scipy.stats import skew, kurtosis

def extract_statistical_features(signal):
    mean_val = np.mean(signal, axis=1)
    var_val = np.var(signal, axis=1)
    skew_val = skew(signal, axis=1)
    kurt_val = kurtosis(signal, axis=1)
    return mean_val, var_val, skew_val, kurt_val

def extract_spectral_features(signal, fs=1000):
    f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=256)
    peak_freq = f[np.argmax(Pxx, axis=1)]
    spectral_entropy = -np.sum(Pxx * np.log2(Pxx + np.finfo(float).eps), axis=1)
    return peak_freq, spectral_entropy

def hybrid_wavelet_stft(signal, wavelet='db4', level=4, nperseg=256):
    _, _, stft_matrix = scipy.signal.stft(signal, fs=1.0, nperseg=nperseg)
    feature_matrices = [np.abs(stft_matrix)]
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    for coeff in coeffs:
        coeff_matrix = np.expand_dims(coeff, axis=0)
        feature_matrices.append(np.abs(coeff_matrix))
    return feature_matrices

def main():
    mat = scipy.io.loadmat('Segmented_EEG_Signal.mat')
    segmented_data = mat['segmented_data']
    fuzzy_features = mat['fuzzy_features']
    labels = mat['labels'].ravel()

    all_features = []
    fs = 1000
    for segment in segmented_data:
        mean_vals, var_vals, skew_vals, kurt_vals = extract_statistical_features(segment)
        peak_freqs, spec_entropy = extract_spectral_features(segment, fs=fs)
        combined_features = np.hstack([mean_vals, var_vals, skew_vals, kurt_vals, peak_freqs, spec_entropy])
        all_features.append(combined_features)

    all_features = np.array(all_features)
    full_feature_matrix = np.hstack([all_features, fuzzy_features])

    scipy.io.savemat('Segmented_EEG_Features.mat', {
        'segmented_eeg_features': full_feature_matrix,
        'labels': labels
    })

    print("Feature extraction completed. Final feature matrix:", full_feature_matrix.shape)
