# Handwritten Digit Recognition — Neural Network Prototype

A complete, working ML prototype that recognizes handwritten digits (0–9)
using a Convolutional Neural Network (CNN) trained on the real **MNIST**
dataset (70,000 handwritten digit images) — plus an interactive GUI where
you draw a digit with your mouse and watch the model predict it live.

**Result: 99.10% accuracy on the held-out test set.**

## What's included

| File | Purpose |
|---|---|
| `mnist.npz` | The real MNIST dataset (60k train / 10k test), bundled locally so training works offline |
| `train_model.py` | Builds and trains the CNN with TensorFlow, saves `digit_model.keras`, and produces accuracy/loss plots |
| `digit_model.keras` | The trained TensorFlow model (already trained for you, ready to use) |
| `numpy_model.py` | A pure-NumPy re-implementation of the trained model's forward pass — used by `app.py` so the web app has no TensorFlow dependency |
| `model_weights.npz` | The trained weights extracted from `digit_model.keras`, in NumPy format |
| `predict_gui.py` | Tkinter desktop app — draw a digit, see the prediction + confidence bars (uses TensorFlow) |
| `app.py` | Streamlit web app — same idea, runs in a browser and deploys online (uses `numpy_model.py`, no TensorFlow) |
| `training_history.png` | Accuracy/loss curves from training |
| `sample_predictions.png` | 10 random test images with predicted vs. true labels |
| `requirements.txt` | Dependencies for the web app (`app.py`) — lightweight, no TensorFlow |
| `requirements-train.txt` | Dependencies for training/`predict_gui.py` — includes TensorFlow |

## Why two versions of the model?

TensorFlow doesn't yet ship installable packages for the newest Python
versions, which some cloud hosts (like Streamlit Community Cloud) can
default to — causing deployment failures that have nothing to do with
your code. To make the web app deploy reliably anywhere, `app.py` uses
`numpy_model.py`, a hand-written forward pass using only NumPy, loaded
from weights extracted out of the real trained model. It was verified
to produce numerically identical predictions to the original TensorFlow
model (max difference ~0.000001, purely floating-point rounding).

The desktop app (`predict_gui.py`) still uses TensorFlow directly since
it runs on your own machine, where you control the Python version.

## How it works

**Architecture** (CNN, ~235K parameters):
```
Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool → Dropout
Conv2D(64) → BatchNorm → MaxPool → Dropout
Flatten → Dense(128) → Dropout → Dense(10, softmax)
```

**Data**: 28×28 grayscale images of handwritten digits, pixel values
normalized to [0, 1]. 60,000 images for training (10% held out for
validation), 10,000 for final testing.

**Training**: Adam optimizer, sparse categorical cross-entropy loss,
early stopping on validation loss, 8 epochs (~10 min on CPU).

## Quick start (after cloning from GitHub)

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

The trained model weights (`model_weights.npz`) are included in the
repo, so this works immediately — no training or downloads required.

## Deploying on the web (free)

`predict_gui.py` (Tkinter) only runs on your own computer — desktop apps
can't run inside a browser. `app.py` (Streamlit) is the web version, and
can be deployed for free in a few clicks:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
3. Click **"New app"**, select this repository, branch `main`, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit installs everything from `requirements.txt`
   automatically and gives you a public URL like
   `https://your-app-name.streamlit.app`.

That's it — no server setup, no separate hosting account needed. Since
`app.py` doesn't depend on TensorFlow, this deploys reliably regardless
of which Python version the host uses.

## Setup

**For the web app (`app.py`):**
```bash
pip install -r requirements.txt
```

**For training or the desktop GUI (`train_model.py` / `predict_gui.py`), which need TensorFlow:**
```bash
pip install -r requirements-train.txt
```

> **Note on Tkinter (for the desktop GUI):** Tkinter ships with most
> Python installations, but on some Linux systems you may need to
> install it separately:
> - Ubuntu/Debian: `sudo apt install python3-tk`
> - macOS/Windows: already included with standard Python installers

## Usage

### 1. Train the model (optional — a trained model is already included)

```bash
python train_model.py
```

This loads `mnist.npz`, trains the CNN, prints test accuracy, and saves:
- `digit_model.keras` — the trained model
- `training_history.png` — accuracy/loss curves
- `sample_predictions.png` — visual sanity check on random test images

### 2. Run the interactive GUI

```bash
python predict_gui.py
```

- Draw a digit anywhere on the black canvas with your mouse
- Release the mouse button (or click "Predict") to see the model's guess
- The confidence bar for each digit (0–9) shows how sure the model is
- Click "Clear" to try another digit

## Tips for best results in the GUI

- Draw the digit fairly large and centered, filling most of the canvas
- Use a thick stroke (the brush is already sized for this)
- MNIST digits are white-on-black, which is exactly how the canvas works

## Extending this project

Ideas if you want to take it further for your submission:
- Report a **confusion matrix** to show which digits get confused with each other
- Try data augmentation (rotation/shift) to make it more robust to messy handwriting
- Swap the CNN for a plain fully-connected network and compare accuracy — a good way to demonstrate *why* CNNs work better for images
- Add a "prediction history" panel to the GUI
- Deploy it as a simple Flask/Streamlit web app instead of Tkinter
