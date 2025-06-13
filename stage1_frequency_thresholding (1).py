import musdb
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import museval
import os

# Set paths
MUSDB_PATH = "/path/to/musdb18"  # Root directory with train/ and test/ folders
OUTPUT_DIR = "stage1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MUSDB18 dataset
mus = musdb.DB(root=MUSDB_PATH, subsets="test")  # Use test set for evaluation

# Band-pass filter design
def bandpass_filter(signal, lowcut, highcut, sr, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.lfilter(b, a, signal)

# Process one track
def process_track(track, sr=44100):
    # Load mixture (stereo)
    mixture = track.audio  # Shape: (samples, 2)
    mixture_mono = np.mean(mixture, axis=1)  # Convert to mono for simplicity

    # Frequency ranges
    bass_low, bass_high = 20, 250
    vocal_low, vocal_high = 300, 3400
    drum_high = 5000

    # Apply band-pass filters
    bass = bandpass_filter(mixture_mono, bass_low, bass_high, sr)
    vocals = bandpass_filter(mixture_mono, vocal_low, vocal_high, sr)
    
    # High-pass for drums
    b, a = scipy.signal.butter(5, drum_high / (0.5 * sr), btype='high')
    drums = scipy.signal.lfilter(b, a, mixture_mono)

    # Save separated sources (repeat mono to stereo)
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

# Main loop (process one track for brevity)
track = mus.tracks[0]  # First test track
estimated_sources, track = process_track(track)
sdr_scores = evaluate_separation(track, estimated_sources)

# Print SDR
for source, sdr in sdr_scores.items():
    print(f"SDR for {source}: {np.nanmean(sdr):.2f} dB")