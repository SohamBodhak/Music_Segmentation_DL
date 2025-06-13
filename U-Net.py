import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import subprocess
from pydub import AudioSegment
import soundfile as sf
import torchaudio
from mir_eval.separation import bss_eval_sources
import torch._dynamo
import shutil

torch._dynamo.config.suppress_errors = True

# Hyperparameters
SAMPLE_RATE = 22050
DURATION = 4
N_FFT = 768
HOP_LENGTH = N_FFT // 8
EPOCHS = 100
BATCH_SIZE = 2
NUM_STEMS = 5
NUM_OUTPUT_STEMS = 4
DB_MIN = -80.0
DB_MAX = 0.0
ACCUM_STEPS = 4
STEREO = False
OVERLAP_RATIO = 0.25  # 25% overlap for chunks

# Utility Functions
def normalize_db(spectrogram_db):
    return (spectrogram_db - DB_MIN) / (DB_MAX - DB_MIN)

def denormalize_db(normalized_db):
    return normalized_db * (DB_MAX - DB_MIN) + DB_MIN

def get_stem_count(file_path):
    return 5

# Custom Loss Function
class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, sc_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight
        self.sc_weight = sc_weight
    
    def spectral_convergence(self, input, target):
        return torch.norm(target - input, p='fro') / torch.norm(target, p='fro').clamp(min=1e-4)
    
    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        sc_loss = self.spectral_convergence(input, target)
        return self.mse_weight * mse_loss + self.sc_weight * sc_loss

# Collate Function
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None, None
    min_time = min(b[0].shape[2] for b in batch)
    X = torch.stack([b[0][:, :, :min_time] for b in batch])
    Y = torch.stack([b[1][:, :, :min_time] for b in batch])
    P = torch.stack([b[2][:, :min_time] for b in batch])
    W = torch.stack([b[3][:, :min_time] for b in batch])
    return X, Y, P, W

# Data Loading Functions
def extract_stems(file_path, output_dir):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")
    
    os.makedirs(output_dir, exist_ok=True)
    stem_count = get_stem_count(file_path)
    if stem_count != NUM_STEMS:
        print(f"Skipping {file_path}: Expected {NUM_STEMS} stems, found {stem_count}")
        return None
    stems = []
    channels = 2 if STEREO else 1
    for i in range(NUM_STEMS):
        output_file = os.path.join(output_dir, f"stem_{i}.wav")
        try:
            subprocess.run([
                'ffmpeg', '-i', file_path,
                '-map', f'0:{i}',
                '-ac', str(channels),
                '-ar', str(SAMPLE_RATE),
                '-y', output_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stems.append(output_file)
        except subprocess.CalledProcessError as e:
            print(f"Error extracting stem {i} from {file_path}: {e}")
            return None
    return stems

def load_stems(file_path):
    try:
        stems = []
        lengths = []
        for i in range(NUM_STEMS):
            stem_file = os.path.join(file_path, f"stem_{i}.wav")
            waveform, sr = torchaudio.load(stem_file)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            if STEREO:
                waveform = waveform.mean(dim=0, keepdim=True)
            samples = waveform.numpy().squeeze()
            stems.append(samples)
            lengths.append(len(samples))
        
        common_length = min(lengths)
        processed_stems = []
        for s in stems:
            s = librosa.util.fix_length(s, size=common_length)
            processed_stems.append(s)
        
        return np.stack(processed_stems), SAMPLE_RATE
    except Exception as e:
        print(f"Error loading stems from {file_path}: {e}")
        return None, None

def augment_audio(audio):
    gain = np.random.uniform(0.9, 1.1)
    audio *= gain
    if random.random() > 0.5:
        shift = random.randint(-100, 100)
        audio = np.roll(audio, shift)
    return audio

def compute_spectrogram(audio, sr=SAMPLE_RATE):
    chunk_samples = int(DURATION * sr)
    overlap_samples = int(chunk_samples * OVERLAP_RATIO)
    step_samples = chunk_samples - overlap_samples
    total_samples = len(audio)
    
    mags = []
    phases = []
    
    for start in range(0, total_samples, step_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        chunk = librosa.util.fix_length(chunk, size=chunk_samples)
        
        stft = librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag, phase = np.abs(stft), np.angle(stft)
        mags.append(mag)
        phases.append(phase)
    
    return mags, phases

class MusicDataset(Dataset):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        self.dataset_path = dataset_path
        self.stem_files = []
        self.chunk_indices = []
        
        files = list(Path(dataset_path).rglob('*.stem.mp4'))
        for file in files:
            output_dir = os.path.join(dataset_path, 'extracted', file.stem)
            if not os.path.exists(output_dir):
                if extract_stems(str(file), output_dir) is None:
                    continue
            if self.validate_stem_dir(output_dir):
                self.stem_files.append(output_dir)
        
        if not self.stem_files:
            raise ValueError(f"No valid stem files found in {dataset_path}")
        
        # Calculate chunk indices for each file
        for idx, stem_dir in enumerate(self.stem_files):
            stems, sr = load_stems(stem_dir)
            if stems is None:
                continue
            total_samples = len(stems[0])
            chunk_samples = int(DURATION * sr)
            overlap_samples = int(chunk_samples * OVERLAP_RATIO)
            step_samples = chunk_samples - overlap_samples
            num_chunks = max(1, (total_samples - overlap_samples) // step_samples)
            for chunk_idx in range(num_chunks):
                self.chunk_indices.append((idx, chunk_idx))
    
    def validate_stem_dir(self, stem_dir):
        return all(os.path.exists(os.path.join(stem_dir, f"stem_{i}.wav")) for i in range(NUM_STEMS))

    def __len__(self):
        return len(self.chunk_indices)

    def __getitem__(self, idx):
        file_idx, chunk_idx = self.chunk_indices[idx]
        stem_dir = self.stem_files[file_idx]
        stems, sr = load_stems(stem_dir)
        
        if stems is None:
            return None
        
        stems = np.array([augment_audio(s) for s in stems])
        
        chunk_samples = int(DURATION * sr)
        overlap_samples = int(chunk_samples * OVERLAP_RATIO)
        step_samples = chunk_samples - overlap_samples
        start = chunk_idx * step_samples
        end = min(start + chunk_samples, len(stems[0]))
        chunk_stems = np.array([s[start:end] for s in stems])
        chunk_stems = np.array([librosa.util.fix_length(s, size=chunk_samples) for s in chunk_stems])
        
        mix_mags, mix_phases = compute_spectrogram(chunk_stems[0], sr)
        target_mags = []
        target_phases = []
        waveforms = []
        
        for i in range(1, NUM_STEMS):
            mags, phases = compute_spectrogram(chunk_stems[i], sr)
            target_mags.append(mags[0])  # Only one chunk per call
            target_phases.append(phases[0])
            waveforms.append(chunk_stems[i])
        
        target_mags = np.stack(target_mags)
        target_phases = np.stack(target_phases)
        waveforms = np.stack(waveforms)
        
        mix_mag = librosa.amplitude_to_db(np.maximum(mix_mags[0], 1e-8), ref=1.0)
        target_mags = librosa.amplitude_to_db(np.maximum(target_mags, 1e-8), ref=1.0)
        mix_mag = normalize_db(mix_mag)
        target_mags = normalize_db(target_mags)
        
        return (
            torch.tensor(mix_mag, dtype=torch.float32).unsqueeze(0),
            torch.tensor(target_mags, dtype=torch.float32),
            torch.tensor(mix_phases[0], dtype=torch.float32),
            torch.tensor(waveforms, dtype=torch.float32)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class EnhancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = ResidualBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2, ceil_mode=True)
        
        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)
        
        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, ceil_mode=True)
        
        self.bottleneck = ResidualBlock(256, 512)
        
        self.upconv3 = self._upconv(512, 256)
        self.dec3 = ResidualBlock(512, 256)
        
        self.upconv2 = self._upconv(256, 128)
        self.dec2 = ResidualBlock(256, 128)
        
        self.upconv1 = self._upconv(128, 64)
        self.dec1 = ResidualBlock(128, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, NUM_OUTPUT_STEMS, 1),
            nn.Sigmoid()
        )
        
        self.attn_gate = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.Sigmoid()
        )
    
    def _upconv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
    
    def forward(self, x):
        input_size = x.size()
        
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.upconv3(b)
        if e3.size()[2:] != d3.size()[2:]:
            e3 = F.interpolate(e3, size=d3.size()[2:], mode='bilinear', align_corners=True)
        attn = self.attn_gate(e3)
        d3 = torch.cat([d3, e3 * attn], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        if e2.size()[2:] != d2.size()[2:]:
            e2 = F.interpolate(e2, size=d2.size()[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        if e1.size()[2:] != d1.size()[2:]:
            e1 = F.interpolate(e1, size=d1.size()[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        output = self.final(d1)
        if output.size()[2:] != input_size[2:]:
            output = F.interpolate(output, size=input_size[2:], mode='bilinear', align_corners=True)
        
        return output

def reconstruct_audio_from_mag_and_phase(output_tensor, phase_tensor, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    output = output_tensor.cpu().numpy()
    phase = phase_tensor.cpu().numpy()
    stem_names = ['drums', 'bass', 'other', 'vocals']
    
    for i in range(NUM_OUTPUT_STEMS):
        db_spec = denormalize_db(output[i])
        mag = librosa.db_to_amplitude(db_spec)
        complex_spec = mag * np.exp(1j * phase)
        waveform = librosa.istft(complex_spec, hop_length=HOP_LENGTH, win_length=N_FFT)
        
        output_path = os.path.join(output_dir, f"{stem_names[i]}_reconstructed.wav")
        sf.write(output_path, waveform, SAMPLE_RATE)
        print(f"Saved: {output_path}")
    
    return stem_names

def train_model(model, train_loader, epochs=EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.compile(model)
    model.to(device)
    
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    scaler = torch.cuda.amp.GradScaler()
    
    best_loss = float('inf')
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, (X_batch, Y_batch, P_batch, _) in enumerate(train_loader):
            if X_batch is None:
                continue
            
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                output = model(X_batch)
                loss = criterion(output, Y_batch) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            if (batch_idx + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * ACCUM_STEPS
            valid_batches += 1
            torch.cuda.empty_cache()
        
        if valid_batches == 0:
            print(f"Epoch {epoch+1}: No valid batches")
            continue
        
        avg_loss = total_loss / valid_batches
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if epoch < 10:
            for g in optimizer.param_groups:
                g['lr'] = 0.001 * (epoch + 1) / 10

def eval_model(model, test_loader, dataset=None, save_outputs=False, output_root='outputs'):
     # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load saved state dict
    state_dict = torch.load("best_model.pth")

    # Remove '_orig_mod.' prefix from keys
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load into fresh model
    model = EnhancedUNet()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    criterion = CombinedLoss()
    total_loss = 0.0
    total_sdr = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for X_batch, Y_batch, P_batch, W_batch in (test_loader):
            if X_batch is None:
                continue
            
            X_batch, Y_batch, P_batch, W_batch = X_batch.to(device), Y_batch.to(device), P_batch.to(device), W_batch.to(device)
            
            with torch.cuda.amp.autocast():
                output = model(X_batch)
                loss = criterion(output, Y_batch)
            
            total_loss += loss.item()
            if save_outputs:
                sdr = reconstruct_audio_from_mag_and_phase(output[0], P_batch[0], output_root)
                total_sdr += sdr
            valid_batches += 1
            torch.cuda.empty_cache()
    
    if valid_batches == 0:
        print("No valid batches in test set.")
        return float('inf'), float('inf')
    
    avg_loss = total_loss / valid_batches
    avg_sdr = total_sdr / valid_batches if save_outputs else float('inf')
    print(f"Evaluation Loss: {avg_loss:.6f}, Average SDR: {avg_sdr:.6f}")
    return avg_loss, avg_sdr

def infer(model, input_audio_path, output_dir='inferred_outputs', model_path='best_model.pth'):
     # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load saved state dict
    state_dict = torch.load("best_model.pth")

    # Remove '_orig_mod.' prefix from keys
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load into fresh model
    model = EnhancedUNet()
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    try:
        waveform, sr = torchaudio.load(input_audio_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio = waveform.numpy().squeeze()
    except Exception as e:
        raise RuntimeError(f"Error loading audio {input_audio_path}: {e}")
    
    chunk_samples = int(DURATION * SAMPLE_RATE)
    overlap_samples = chunk_samples // 4
    step_samples = chunk_samples - overlap_samples
    total_samples = len(audio)
    
    separated_waveforms = [np.zeros(total_samples) for _ in range(NUM_OUTPUT_STEMS)]
    window = np.hanning(overlap_samples * 2)
    
    with torch.no_grad():
        for start in range(0, total_samples, step_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]
            chunk = librosa.util.fix_length(chunk, size=chunk_samples)
            
            mag, phase = compute_spectrogram(chunk)
            mag_db = librosa.amplitude_to_db(np.maximum(mag[0], 1e-8), ref=1.0)
            mag_norm = normalize_db(mag_db)
            
            input_tensor = torch.tensor(mag_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast():
                output = model(input_tensor)
            output = output.squeeze(0)
            
            output_np = output.cpu().numpy()
            phase_np = phase[0]
            
            for i in range(NUM_OUTPUT_STEMS):
                db_spec = denormalize_db(output_np[i])
                mag = librosa.db_to_amplitude(db_spec)
                complex_spec = mag * np.exp(1j * phase_np)
                waveform = librosa.istft(complex_spec, hop_length=HOP_LENGTH, win_length=N_FFT)
                
                chunk_len = len(waveform)
                target_len = min(chunk_len, total_samples - start)
                if overlap_samples > 0 and start > 0:
                    overlap_end = min(start + overlap_samples, total_samples)
                    overlap_len = overlap_end - start
                    waveform[:overlap_len] *= window[:overlap_len]
                    separated_waveforms[i][start:start+overlap_len] *= window[overlap_len:overlap_len*2]
                
                separated_waveforms[i][start:start+target_len] += waveform[:target_len]
            
            torch.cuda.empty_cache()
    
    os.makedirs(output_dir, exist_ok=True)
    stem_names = ['drums', 'bass', 'other', 'vocals']
    for i, waveform in enumerate(separated_waveforms):
        output_path = os.path.join(output_dir, f"{stem_names[i]}_reconstructed.wav")
        sf.write(output_path, waveform, SAMPLE_RATE)
        print(f"Saved: {output_path}")
    
    return stem_names

if __name__ == "__main__":
    model = EnhancedUNet()
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    dataset = MusicDataset("./musdb18/train/")
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    train_model(model, train_loader)
    print("Training complete! Best model saved to best_model.pth")
    
    test_dataset = MusicDataset("./musdb18/test/")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )
    
    model.load_state_dict(torch.load("./best_model.pth"))
    eval_model(model, test_loader, dataset=test_dataset, save_outputs=True)
    
    input_audio = "path/to/your/song.wav"
    if os.path.exists(input_audio):
        print(f"Running inference on {input_audio}")
        infer(model, input_audio, output_dir="inferred_outputs")
    else:
        print(f"Inference skipped: {input_audio} does not exist. Please provide a valid audio file path.")