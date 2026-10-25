import numpy as np
import kagglehub

def _load_images(path, name):
    pixels = np.fromfile(f"{path}/{name}/{name}", dtype=np.uint8, offset=16)
    return pixels.reshape(-1, 784) / 255.0

def _load_labels(path, name):
    return np.fromfile(f"{path}/{name}/{name}", dtype=np.uint8, offset=8)

def load_mnist():
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")
    X_train = _load_images(path, "train-images-idx3-ubyte")
    y_train = _load_labels(path, "train-labels-idx1-ubyte")
    X_test = _load_images(path, "t10k-images-idx3-ubyte")
    y_test = _load_labels(path, "t10k-labels-idx1-ubyte")
    return X_train, y_train, X_test, y_test
