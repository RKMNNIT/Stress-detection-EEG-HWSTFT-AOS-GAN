import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
def FA_FCN():
    class FCN(nn.Module):
        def __init__(self, n_class):
            super(FCN, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(64*4*4, n_class)
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out
    eeg_data = scipy.io.loadmat('Preprocessed_EEG_Signal.mat')['preprocessed_data']
    segment_length = 1000  
    overlap = 500  
    segments = []
    for i in range(0, 50):
        segment = eeg_data[:, i:i + segment_length]
        segments.append(segment)
    segmented_data = np.array(segments)
    scipy.io.savemat('Segmented_EEG_Signal.mat', {'segmented_data': segmented_data})
    for i, segment in enumerate(segmented_data):
        plt.figure()
        plt.title(f'Segment {i+1}')
        plt.plot(segment.T)
        plt.xlabel('Sample Index')
        plt.ylabel('EEG Signal Value')
        plt.show()
