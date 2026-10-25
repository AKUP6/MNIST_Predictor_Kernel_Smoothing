import numpy as np
from helpers.image_processing import load_mnist
from helpers.kernel import gaussian_kernel
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = load_mnist()  
X_train_mean = X_train.mean(axis=0)

def predict_image(image, h=1.0):
    x0 = image.reshape(1, -1)
    top = np.zeros(10)
    bottom = 0.0
    for xi, yi in zip(X_train[:], y_train[:]):
        k = gaussian_kernel(x0, xi.reshape(1, -1), h=h).item()
        top[yi] += k
        bottom += k
    scores = top / bottom
    return np.argmax(scores), scores

def run(h=1.0):
    for image, label in zip(X_test[:], y_test[:]):
        pred, scores = predict_image(image, h=h)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {pred}, True: {label}")
        plt.show()

if __name__ == "__main__":
    run()