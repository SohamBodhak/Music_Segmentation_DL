import musdb
import librosa
import numpy as np
import soundfile as sf
import museval
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set paths
MUSDB_PATH = "/path/to/musdb18"  # Root directory with train/ and test/ folders
OUTPUT_DIR = "stage3_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load MUSDB18 dataset
mus = musdb.DB(root=MUSDB_PATH, subsets=["train", "test"])
train_tracks = [t for t in mus.tracks if t.subset == "train"][:10]  # 10 train tracks
test_tracks = [t for t in mus.tracks if t.subset == "test"][:1]    # 1 test track

# Feature extraction
def extract_features(audio, sr=44100, n_fft=2048, hop_length=512, n_mfcc=13):
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
    features = np.vstack([mfcc, centroid, zcr]).T  # Shape: (time_frames, n_features)
    return features, S

# Train classifier
def train_classifier(train_tracks, sr=44100, n_fft=2048, hop_length=512):
    X, y = [], []
    for track in train_tracks:
        mixture = np.mean(track.audio, axis=1)  # Mono
        bass = np.mean(track.targets['bass'].audio, axis=1)
        vocals = np.mean(track.targets['vocals'].audio, axis=1)
        drums = np.mean(track.targets['drums'].audio, axis=1)
        
        # Extract features
        feat_mixture, S_mixture = extract_features(mixture, sr, n_fft, hop_length)
        _, S_bass = extract_features(bass, sr, n_fft, hop_length)
        _, S_vocals = extract_features(vocals, sr, n_fft, hop_length)
        _, S_drums = extract_features(drums, sr, n_fft, hop_length)
        
        # Create labels (1=bass, 2=vocals, 3=drums, 0=other)
        labels = np.zeros(S_mixture.shape[1], dtype=int)
        for t in range(S_mixture.shape[1]):
            energies = [np.sum(S_bass[:, t]), np.sum(S_vocals[:, t]), np.sum(S_drums[:, t])]
            if max(energies) < 1e-6:  # Silence
                labels[t] = 0
            else:
                labels[t] = np.argmax(energies) + 1
        
        X.append(feat_mixture)
        y.append(labels)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    # Train Random Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)
    return clf, scaler

# Separate sources
def separate_sources(mixture, clf, scaler, sr=44100, n_fft=2048, hop_length=512):
    mixture_mono = np.mean(mixture, axis=1)
    feat, S = extract_features(mixture_mono, sr, n_fft, hop_length)
    feat_scaled = scaler.transform(feat)
    
    # Predict labels
    labels = clf.predict(feat_scaled)  # 0=other, 1=bass, 2=vocals, 3=drums
    
    # Create masks
    bass_mask = (labels == 1).astype(float).reshape(1, -1)
    vocal_mask = (labels == 2).astype(float).reshape(1, -1)
    drum_mask = (labels == 3).astype(float).reshape(1, -1)
    
    # Apply masks
    stft_mixture = librosa.stft(mixture_mono, n_fft=n_fft, hop_length=hop_length)
    bass = librosa.istft(stft_mixture * bass_mask, hop_length=hop_length)
    vocals = librosa.istft(stft_mixture * vocal_mask, hop_length=hop_length)
    drums = librosa.istft(stft_mixture * drum_mask, hop_length=hop_length)
    
    return bass, vocals, drums

# Process test track
def process_track(track, clf, scaler, sr=44100):
    mixture = track.audio
    bass, vocals, drums = separate_sources(mixture, clf, scaler, sr)
    
    # Save separated sources
    sources = {
        'bass': np.stack([bass, bass], axis=1),
        'vocals': np.stack([vocals, vocals], axis=1),
        'drums': np.stack([drums, drums], axis=1)
    }
    
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

# Main
clf, scaler = train_classifier(train_tracks)
track = test_tracks[0]
estimated_sources, track = process_track(track, clf, scaler)
sdr_scores = evaluate_separation(track, estimated_sources)

# Print SDR
for source, sdr in sdr_scores.items():
    print(f"SDR for {source}: {np.nanmean(sdr):.2f} dB")