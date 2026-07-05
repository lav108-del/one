"""
Pure NumPy inference engine for the digit recognizer CNN.
------------------------------------------------------------
Re-implements the trained model's forward pass using only NumPy,
so the web app doesn't depend on TensorFlow at all (avoids
platform/Python-version compatibility issues on hosting services).

The weights themselves were extracted from the TensorFlow-trained
model (see extract_weights.py) into model_weights.npz.
"""

import numpy as np


def conv2d(x, w, b, stride=1):
    """
    x: (H, W, C_in)
    w: (kh, kw, C_in, C_out)
    b: (C_out,)
    Valid (no padding) convolution, matching Keras default Conv2D padding="valid".
    """
    H, W, C_in = x.shape
    kh, kw, _, C_out = w.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1

    # im2col for speed
    patches = np.zeros((out_h, out_w, kh, kw, C_in), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patches[i, j] = x[i * stride:i * stride + kh, j * stride:j * stride + kw, :]

    patches = patches.reshape(out_h * out_w, kh * kw * C_in)
    w_flat = w.reshape(kh * kw * C_in, C_out)
    out = patches @ w_flat + b
    return out.reshape(out_h, out_w, C_out)


def batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def relu(x):
    return np.maximum(0, x)


def max_pool2d(x, size=2, stride=2):
    H, W, C = x.shape
    out_h, out_w = H // stride, W // stride
    out = np.zeros((out_h, out_w, C), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = x[i * stride:i * stride + size, j * stride:j * stride + size, :].max(axis=(0, 1))
    return out


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class DigitCNN:
    """Loads extracted weights and runs the same forward pass as the
    trained Keras model, matching layer-for-layer:
    Conv-BN-Conv-Pool-Conv-BN-Pool-Flatten-Dense-Dense."""

    def __init__(self, weights_path="model_weights.npz"):
        w = np.load(weights_path)
        self.w = {k: w[k] for k in w.files}

    def predict(self, img):
        """img: (28, 28) or (28, 28, 1) float32 array normalized to [0,1]."""
        w = self.w
        x = img.reshape(28, 28, 1).astype(np.float32)

        # Conv2D(32, activation="relu") -> BatchNorm
        x = conv2d(x, w["conv1_w"], w["conv1_b"])
        x = relu(x)
        x = batch_norm(x, w["bn1_gamma"], w["bn1_beta"], w["bn1_mean"], w["bn1_var"])

        # Conv2D(32, activation="relu") -> MaxPool
        x = conv2d(x, w["conv2_w"], w["conv2_b"])
        x = relu(x)
        x = max_pool2d(x)

        # Conv2D(64, activation="relu") -> BatchNorm -> MaxPool
        x = conv2d(x, w["conv3_w"], w["conv3_b"])
        x = relu(x)
        x = batch_norm(x, w["bn2_gamma"], w["bn2_beta"], w["bn2_mean"], w["bn2_var"])
        x = max_pool2d(x)

        # Flatten -> Dense(128, relu) -> Dense(10, softmax)
        x = x.flatten()
        x = x @ w["dense1_w"] + w["dense1_b"]
        x = relu(x)
        x = x @ w["dense2_w"] + w["dense2_b"]

        return softmax(x)
