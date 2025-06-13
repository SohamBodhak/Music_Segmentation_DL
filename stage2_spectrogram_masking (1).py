import musdb
import librosa
import numpy as np
import soundfile as sf
import museval
import os
from sklearn.decomposition import NMF

# Set paths
MUSDB_PATH = "/path/to/musdb18"  # Root directory with train/ and test/ folders
OUTPUT_DIR = "stage2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MUSDB18 dataset
mus = musdb.DB(root=MUSDB_PATH, subsets="test")

# NMF-based separation
def nmf_separation(mixture, n_components=12, sr=44100, n_fft=2048, hop_length=512):
    # Compute magnitude spectrogram
    mixture_mono = np.mean(mixture, axis=1)
    S = np.abs(librosa.stft(mixture_mono, n_fft=n_fft, hop_length=hop_length))
    
    # Apply NMF
    nmf = NMF(n_components=n_components, init='random', random_state=42)
    W = nmf.fit_transform(S)  # Basis (frequency components)
    H = nmf.components_  # Activations (time components)
    
    # Manually assign components to sources (heuristic: bass=low freq, vocals=mid, drums=high)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bass_idx = np.where(freqs < 250)[0]
    vocal_idx = np.where((freqs >= 300) & (freqs <= 3400))[0]
    drum_idx = np.where(freqs > 5000)[0]
    
    # Create masks
    bass_mask = np.zeros_like(S)
    vocal_mask = np.zeros_like(S)
    drum_mask = np.zeros_like(S)
    
    # Assign components based on frequency dominance
    for i in range(n_components):
        w = W[:, i].reshape(-1, 1)
        h = H[i, :].reshape(1, -1)
        component = w @ h
        freq_energy = np.sum(w[:len(freqs)], axis=1)
        if np.argmax(freq_energy) in bass_idx:
            bass_mask += component
        elif np.argmax(freq_energy) in vocal_idx:
            vocal_mask += component
        else:
            drum_mask += component
    
    # Normalize masks
    total = bass_mask + vocal_mask + drum_mask + 1e-10
    bass_mask /= total
    vocal_mask /= total
    drum_mask /= total
    
    # Apply masks
    stft_mixture = librosa.stft(mixture_mono, n_fft=n_fft, hop_length=hop_length)
    bass = librosa.istft(stft_mixture * bass_mask, hop_length=hop_length)
    vocals = librosa.istft(stft_mixture * vocal_mask, hop_length=hop_length)
    drums = librosa.istft(stft_mixture * drum_mask, hop_length=hop_length)
    
    return bass, vocals, drums

# Process one track
def process_track(track, sr=44100):
    mixture = track.audio
    bass, vocals, drums = nmf_separation(mixture, n_components=12, sr=sr)
    
    # Save separated sources (mono to stereo)
    sources = {
        'bass': np.stack([bass, bass], axis=1),
        'vocals': np.stack([vocals, vocals], axis=1),
        'drums': np.stack([drums, drums], axis=1)
    }
    
    # Save outputs
    track_name = track.name.replace(" ", "_")
    for source_name, source_audio in sources.items():
        sf.write(f"{OUTPUT_DIR}/{track_name}_{source_name}.wav", source_audio, sr)
    
    return sources, track

# Evaluate SDR
def evaluate_separation(track, estimated_sources):
    references = {
        'bass': track.targets['bass'].audio,
        'vocals': track.targets['vocals'].audio,
        'drums': track.targets['drums'].audio
    }
    sdr, _, _, _ = museval.evaluate(references, estimated_sources)
    return sdr

# Main loop
track = mus.tracks[0]
estimated_sources, track = process_track(track)
sdr_scores = evaluate_separation(track, estimated_sources)

# Print SDR
for source, sdr in sdr_scores.items():
    print(f"SDR for {source}: {np.nanmean(sdr):.2f} dB")