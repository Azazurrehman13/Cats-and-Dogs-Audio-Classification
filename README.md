**Cats and Dogs Audio Classification**

This repository implements a deep learning model to classify cat and dog sounds using mel spectrograms, built with PyTorch. The project addresses challenges like class imbalance, input shape mismatches, and poor initial performance (e.g., 55% accuracy, 0.20 dog recall) through a robust pipeline with a deeper CNN, data augmentation, and balanced class weights.

Dataset

**Source:** [Kaggle Cats and Dogs Audio Dataset](https://www.kaggle.com/datasets/mmoreaux/audio-cats-and-dogs)

**Structure:** WAV files for cat and dog sounds, split into training and test sets via train_test_split.csv.

Details: ~942s of cat audio, ~317s of dog audio (training), with a ~3:1 imbalance.

**Challenges Addressed**

- Channel Mismatch Error: Initial training failed due to incorrect mel spectrogram shapes ([20, 40, 32, 1] vs. expected [20, 1, 40, 32]). Fixed by correcting utils.py to output proper shapes.

- Poor Performance: Earlier runs had 55% accuracy and 0.20 dog recall due to:

- Shallow model (3 conv layers).


- Class imbalance (no or incorrect weights).


- Raw audio input instead of mel spectrograms.


- Weak regularization and augmentation. Solution: Deeper CNN with residual connections, class weights [1.0, 2.0], dropout (0.4), weight decay (1e-2), and augmentations (pitch shift, time stretch, noise).


- Setup Misalignment: Model and data pipeline were misaligned (e.g., raw audio vs. mel spectrograms). Fixed with consistent mel spectrogram input and debug prints.

**Requirements**

Python 3.11

See requirements.txt for dependencies.

Setup

**Clone the Repository:**

git clone https://github.com/your-username/cats-dogs-audio-classification.git
cd cats-dogs-audio-classification

**Install Dependencies:**

pip install -r requirements.txt

**Usage**

Run the main script to load data, train the model, convert to ONNX, and compare inference times:

python main.py

**Output:**

1. Training/validation loss and accuracy per epoch.

2. Final confusion matrix and classification report (target: >70% accuracy, >0.6 dog recall).

3. ONNX model saved to ./cats_dogs_model.onnx.

4. Inference time comparison (PyTorch vs. ONNX).

**Files**

main.py: Loads dataset, defines model, trains, converts to ONNX, and compares inference.

utils.py: Handles data loading, augmentation, and mel spectrogram generation.

requirements.txt: Lists dependencies.

.gitignore: Ignores unnecessary files.

**Notes**


**Mel Spectrograms:** Used for better feature extraction (shape: (20, 1, 40, 32)).

**Raw Audio Alternative:** Modify utils.py and main.py to use raw audio ((20, 1, 16000)) with a 1D CNN if preferred, but expect lower performance.

**Debugging:** Check utils.py debug prints for input shapes.

**Performance Tuning:**

Adjust class_weights (e.g., [1.0, 3.0]) if dog recall is low.

Increase sample_augmentation (e.g., 6) for more robustness.

Reduce batch size to (10, 16000) if memory issues occur.

**Expected Results**

Validation accuracy: >70%.

Dog recall: >0.6.

Balanced confusion matrix.

**Contributing**

Feel free to open issues or submit pull requests for improvements.
