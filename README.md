# Music Source Separation Project

This project implements multiple approaches for music source separation, aiming to isolate individual components (e.g., bass, vocals, drums) from a mixed audio track. It includes four different methods: frequency thresholding, spectrogram masking with NMF, classical machine learning with Random Forest, and a deep learning approach using a U-Net architecture. The project uses the MUSDB18 dataset for training and evaluation.

## Project Structure

- **stage1_frequency_thresholding.py**: Implements a simple frequency-based separation using band-pass filters to isolate bass, vocals, and drums based on predefined frequency ranges.
- **stage2_spectrogram_masking.py**: Uses Non-negative Matrix Factorization (NMF) to decompose the spectrogram and assign components to sources based on frequency dominance.
- **stage3_classical_ml.py**: Employs a Random Forest classifier to predict source labels for time-frequency bins, creating masks for source separation.
- **U-Net.py**: Implements a deep learning-based approach using a U-Net architecture with residual blocks for spectrogram-based source separation.

## Features

- **Dataset**: Utilizes the MUSDB18 dataset, which contains stereo audio tracks with separated stems (bass, vocals, drums, other, accompaniment).
- **Evaluation**: Uses the `museval` library to compute Signal-to-Distortion Ratio (SDR) scores for evaluating separation quality.
- **Deep Learning**: The U-Net model includes residual blocks, attention mechanisms, and a combined loss function (MSE + spectral convergence).
- **Audio Processing**: Leverages `librosa`, `torchaudio`, and `soundfile` for audio processing and spectrogram computation.
- **Preprocessing**: Includes data augmentation, spectrogram normalization, and chunking with overlap for handling long audio files.

## Prerequisites

To run this project, ensure you have the following:

- **MUSDB18 Dataset**: Download and place the dataset in the `/path/to/musdb18` directory. Update the `MUSDB_PATH` variable in the scripts accordingly.
- **FFmpeg**: Required for stem extraction in the U-Net implementation.
- **Python**: Version 3.8 or higher.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg:
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On macOS: `brew install ffmpeg`
   - On Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH.

4. Set up the MUSDB18 dataset:
   - Download the dataset from [MUSDB18](https://zenodo.org/record/3338373).
   - Update the `MUSDB_PATH` variable in each script to point to the dataset directory.

## Usage

### Stage 1: Frequency Thresholding
Run the frequency-based separation:
```bash
python stage1_frequency_thresholding.py
```
Outputs are saved to the `stage1_outputs` directory.

### Stage 2: Spectrogram Masking with NMF
Run the NMF-based separation:
```bash
python stage2_spectrogram_masking.py
```
Outputs are saved to the `stage2_outputs` directory.

### Stage 3: Classical Machine Learning
Run the Random Forest-based separation:
```bash
python stage3_classical_ml.py
```
Outputs are saved to the `stage3_outputs` directory.

### Stage 4: U-Net Deep Learning
Train and evaluate the U-Net model:
```bash
python U-Net.py
```
- Outputs are saved to the `outputs` directory during evaluation.
- For inference on a custom audio file, update the `input_audio` path in `U-Net.py` and ensure the file exists. Outputs are saved to the `inferred_outputs` directory.

## Notes

- **U-Net Training**: The U-Net model requires a GPU for efficient training. Adjust `BATCH_SIZE` and `ACCUM_STEPS` based on your hardware.
- **Dataset Path**: Ensure the MUSDB18 dataset path is correctly set in all scripts.
- **Evaluation**: SDR scores are printed for each source (bass, vocals, drums) after processing.
- **Inference**: The U-Net script supports inference on new audio files, producing separated stems in the `inferred_outputs` directory.

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **MUSDB18**: For providing the dataset used in this project.
- **Librosa, Torchaudio, and Scikit-learn**: For audio processing and machine learning utilities.
- **PyTorch**: For the deep learning framework used in the U-Net implementation.
