import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import streamlit as st
import tempfile
import shutil
from pathlib import Path

# Hyperparameters
SAMPLE_RATE = 22050
DURATION = 4
N_FFT = 768
HOP_LENGTH = N_FFT // 8
NUM_OUTPUT_STEMS = 4
DB_MIN = -80.0
DB_MAX = 0.0
OVERLAP_RATIO = 0.25

# Utility Functions
def normalize_db(spectrogram_db):
    spectrogram_db = np.clip(spectrogram_db, DB_MIN, DB_MAX)
    return (spectrogram_db - DB_MIN) / (DB_MAX - DB_MIN)

def denormalize_db(normalized_db):
    return normalized_db * (DB_MAX - DB_MIN) + DB_MIN

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
        if np.any(np.isnan(mag)) or np.any(np.isinf(mag)):
            st.warning(f"Warning: NaN or Inf in spectrogram magnitude at chunk {start}-{end}")
            continue
        mags.append(mag)
        phases.append(phase)
    
    if not mags:
        st.error("Error: No valid spectrograms computed")
        return None, None
    
    return mags, phases

# Model Definition
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

# Inference Function (Unchanged from latest.py)
def infer(model, input_audio_path, output_dir='inferred_outputs', model_path='best_model.pth'):
    # Device setup
    device = torch.device('cpu')
    # Load saved state dict
    state_dict = torch.load("best_model.pth",map_location=device)

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
            if mag is None:
                st.warning(f"Warning: Invalid spectrogram for inference chunk {start}-{end}")
                continue
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
    output_paths = []
    for i, waveform in enumerate(separated_waveforms):
        output_path = os.path.join(output_dir, f"{stem_names[i]}_reconstructed.wav")
        sf.write(output_path, waveform, SAMPLE_RATE)
        output_paths.append(output_path)
    
    return stem_names, output_paths

# Streamlit App
def main():
    st.title("Audio Source Separation App")
    st.write("Upload an audio file (.wav or .mp4) to separate it into drums, bass, other, and vocals using a pre-trained model.")

    # Check for model weights
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        st.error("Model weights (best_model.pth) not found. Please place the file in the same directory as this app.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp4"])
    
    if uploaded_file is not None:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        output_dir = os.path.join(temp_dir, "outputs")
        
        # Save uploaded file
        input_audio_path = os.path.join(temp_dir, uploaded_file.name)
        with open(input_audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        # Initialize model
        # Device setup
        device = torch.device('cpu')
    # Load saved state dict
        state_dict = torch.load("best_model.pth",map_location=device)

    # Remove '_orig_mod.' prefix from keys
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Load into fresh model
        model = EnhancedUNet()
        model.load_state_dict(new_state_dict)
        #model.to(device)
        #model.eval()
        
        # Run inference when button is clicked
        if st.button("Separate Audio"):
            with st.spinner("Separating audio... This may take a few minutes."):
                try:
                    stem_names, output_paths = infer(model, input_audio_path, output_dir=output_dir, model_path=model_path)
                    st.success("Separation complete!")
                    
                    # Display download buttons for each stem
                    st.subheader("Download Separated Stems")
                    for stem_name, output_path in zip(stem_names, output_paths):
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label=f"Download {stem_name}.wav",
                                data=f,
                                file_name=f"{stem_name}_reconstructed.wav",
                                mime="audio/wav"
                            )
                
                except Exception as e:
                    st.error(f"Error during separation: {str(e)}")
                
                finally:
                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
    
    else:
        st.info("Please upload an audio file to begin.")

if __name__ == "__main__":
    main()
