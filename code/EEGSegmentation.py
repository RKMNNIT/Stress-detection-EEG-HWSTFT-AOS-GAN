import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class FuzzyAttention(nn.Module):
    def __init__(self, input_dim):
        super(FuzzyAttention, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        scores = torch.tanh(torch.matmul(x, self.weight) + self.bias)
        weights = torch.softmax(scores, dim=1)
        attended = x * weights
        return attended.sum(dim=1)

class FA_FCN_Model(nn.Module):
    def __init__(self, n_classes=2):
        super(FA_FCN_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.attention = FuzzyAttention(64)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x = self.fc(x)
        return x

def FA_FCN(train_epochs=20, batch_size=32, segment_length=1000, overlap=500):
    eeg_data = scipy.io.loadmat('Preprocessed_EEG_Signal.mat')['preprocessed_data']
    n_samples = eeg_data.shape[1]

    segments, labels = [], []
    for start in range(0, n_samples - segment_length, segment_length - overlap):
        end = start + segment_length
        segment = eeg_data[:, start:end]
        segments.append(segment)
        labels.append(1 if np.mean(segment) > 0 else 0)

    segments = np.array(segments)
    labels = np.array(labels)

    X = torch.tensor(segments, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FA_FCN_Model(n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        features = []
        for batch_X, _ in loader:
            x = model.pool(F.relu(model.conv1(batch_X)))
            x = model.pool(F.relu(model.conv2(x)))
            x = model.pool(F.relu(model.conv3(x)))
            x = x.permute(0, 2, 1)
            attended = model.attention(x)
            features.append(attended.numpy())
        features = np.vstack(features)

    scipy.io.savemat('Segmented_EEG_Signal.mat', {
        'segmented_data': segments,
        'labels': labels,
        'fuzzy_features': features
    })

    print("Segmentation completed. Segments:", segments.shape, "Features:", features.shape)
