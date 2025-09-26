import scipy.io
def RLASSO():
    mat = scipy.io.loadmat('Segmented_EEG_Features.mat')
    segments = mat['segmented_eeg_features']
    print(segments)
