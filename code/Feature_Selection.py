import scipy.io
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

def RLASSO(alpha_values=None, cv=5):
    if alpha_values is None:
        alpha_values = np.logspace(-3, 1, 50)

    mat = scipy.io.loadmat('Segmented_EEG_Features.mat')
    X = mat['segmented_eeg_features']
    y = mat['labels'].ravel()

    print("Original feature matrix:", X.shape)

    lasso = LassoCV(alphas=alpha_values, cv=cv, max_iter=5000).fit(X, y)
    model = SelectFromModel(lasso, prefit=True)
    X_selected = model.transform(X)

    print("Selected features matrix:", X_selected.shape)

    scipy.io.savemat('Selected_Features.mat', {'selected_features': X_selected, 'labels': y})
    return X_selected, y
