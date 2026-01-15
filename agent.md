# MoDA Agent Context

## Project Overview

**MoDA (Multi-modal Diffusion Architecture)** is a talking head generation system that animates a static face image using audio input. It generates realistic facial motion sequences synchronized with speech, supporting emotion conditioning and classifier-free guidance.

This is a fork of the original [MoDA repository](https://github.com/lixinyyang/MoDA) with model weights hosted on [Hugging Face](https://huggingface.co/lixinyizju/moda).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MoDA Pipeline                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Audio      │───▶│   Motion     │───▶│   Motion     │───▶│   Video   │  │
│  │   Input      │    │   Diffusion  │    │   Processor  │    │   Output  │  │
│  │   (.wav)     │    │   (DiT)      │    │ (LivePortrait)│    │   (.mp4)  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│        │                    │                    │                           │
│        ▼                    ▼                    ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   HuBERT     │    │ Flow Matching│    │  SPADE       │                   │
│  │   Encoder    │    │  Scheduler   │    │  Generator   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
MoDA/
├── app.py                              # Gradio web interface
├── configs/
│   └── audio2motion/
│       ├── inference/
│       │   └── inference.yaml          # Main inference config
│       └── model/
│           ├── audio_processer_config.yaml
│           ├── config.yaml             # Motion generator config
│           ├── crop_config.yaml
│           ├── liveportrait_config.yaml
│           └── models.yaml
├── src/
│   ├── datasets/
│   │   ├── mean.pt                     # Motion normalization stats
│   │   └── preprocess/extract_features/
│   │       ├── audio_processer.py      # Audio feature extraction
│   │       ├── motion_processer.py     # Face motion extraction/rendering
│   │       └── face_segmentation/      # BiSeNet face parsing
│   ├── models/
│   │   ├── audio/                      # Audio encoder models
│   │   │   ├── hubert.py               # Modified HuBERT
│   │   │   └── wav2vec_modified.py     # Modified Wav2Vec
│   │   ├── dit/                        # Diffusion Transformer
│   │   │   ├── talking_head_dit.py     # Main DiT architecture
│   │   │   ├── talking_head_diffusion.py # Diffusion wrapper
│   │   │   ├── blocks.py               # DiT block implementations
│   │   │   ├── embedders.py            # Audio/Motion/Timestep embedders
│   │   │   └── modules.py              # Attention, MLP, RMSNorm
│   │   ├── inference/
│   │   │   └── moda_test.py            # Main inference script
│   │   └── schedulers/
│   │       ├── flow_matching.py        # Flow matching scheduler
│   │       └── scheduling_ddim.py      # DDIM scheduler
│   ├── thirdparty/
│   │   └── liveportrait/               # LivePortrait face animation
│   └── utils/
│       ├── filter.py                   # Kalman smoothing
│       └── util.py                     # Utility functions
└── pretrained_weights/                 # Model checkpoints (not in repo)
```

## Core Components

### 1. Audio Processing (`src/datasets/preprocess/extract_features/audio_processer.py`)

**Class: `AudioProcessor`**

Extracts audio features using HuBERT or Wav2Vec models:
- Loads audio at 16kHz sample rate
- Optionally separates vocals using audio-separator
- Extracts embeddings from all 12 transformer layers (shape: `[T, 12, 768]`)
- Supports Chinese HuBERT model

**Key Methods:**
- `preprocess(audio_path)` - Extract features from audio file
- `get_long_audio_emb(audio_path)` - Handle long audio by chunking
- `add_silent_audio(audio_path, silent_audio_path)` - Pad audio for smoother transitions

### 2. Motion Diffusion Model (`src/models/dit/`)

**Class: `MotionDiffusion`** (`talking_head_diffusion.py`)

Wraps the DiT model with diffusion sampling logic:
- **Input**: Audio embeddings, reference keypoints, previous motion
- **Output**: Motion sequence (70 dims: 63 expression + 7 pose parameters)
- Uses Flow Matching scheduler (10 inference steps)
- Supports classifier-free guidance (CFG)
- Handles long sequences via overlapping segments

**Key Parameters:**
- `n_pred_frames`: 80 frames per segment
- `overlap_len`: 16 frames for segment fusion
- `cfg_scale`: Guidance scale (default: 1.2)

**Class: `TalkingHeadDiT`** (`talking_head_dit.py`)

The core transformer architecture:
- **Blocks**: 3 four-stream + 6 double-stream + 12 single-stream blocks
- **Hidden size**: 768, 12 attention heads
- **Embedders**:
  - `MotionEmbedder`: Projects motion to hidden dim
  - `TimestepEmbedder`: Sinusoidal timestep encoding
  - `AudioEmbedder`: Projects audio features
  - `LabelEmbedder`: Emotion class embedding (8 emotions + null)
- Uses rotary position embeddings (RoPE)
- AdaLN-Zero conditioning on timestep and emotion

### 3. Motion Processing (`src/datasets/preprocess/extract_features/motion_processer.py`)

**Class: `MotionProcesser`**

Handles face detection, motion extraction, and video rendering using LivePortrait components:

**Loaded Models:**
1. `appearance_feature_extractor` - Extract 3D face features
2. `motion_extractor` - Extract keypoint info (exp, scale, t, pitch, yaw, roll)
3. `stitching_retargeting_module` - Blend generated motion with source
4. `warping_module` - Warp source features
5. `spade_generator` - Decode to final image
6. `face_parser` (BiSeNet) - Face segmentation for paste-back

**Key Methods:**
- `prepare_source(src_img)` - Extract source face features (f_s) and keypoints (x_s_info)
- `transform_keypoint(kp_info)` - Apply pose/expression transforms
- `driven(f_s, x_s_info, kp_infos)` - Generate animated frames
- `driven_by_audio(src_img, kp_infos, save_path)` - Full pipeline with paste-back

### 4. Schedulers (`src/models/schedulers/`)

**Class: `ModelSamplingDiscreteFlow`** (`flow_matching.py`)

Implements Flow Matching for fast generation:
- Linear interpolation between noise and data
- 10 inference steps (configurable)
- Euler ODE solver

**Key Methods:**
- `add_noise(sample, noise, timesteps)` - Forward process
- `step(model_output, timestep, sample)` - Reverse step

### 5. Inference Pipeline (`src/models/inference/moda_test.py`)

**Class: `LiveVASAPipeline`**

Main inference orchestrator:

```python
# Usage
pipeline = LiveVASAPipeline(cfg_path="configs/audio2motion/inference/inference.yaml")
video_path = pipeline.driven_sample(
    image_path="image.jpg",
    audio_path="audio.wav",
    cfg_scale=1.2,
    emo=8  # 0-7 for emotions, 8 for None
)
```

**Pipeline Steps:**
1. Load and process audio → HuBERT embeddings
2. Load source image → crop face, extract features
3. Get initial motion from source face
4. Run diffusion sampling with audio conditioning
5. Post-process: lip modulation, denormalization
6. Render frames using LivePortrait
7. Paste back onto original image with face mask
8. Combine with audio to create final video

## Motion Representation

Motion vectors are 70-dimensional:
- **dims 0-62 (63)**: Expression coefficients (21 keypoints × 3 coords)
- **dim 63**: Scale
- **dims 64-66 (3)**: Translation (tx, ty, tz)
- **dim 67**: Pitch
- **dim 68**: Yaw  
- **dim 69**: Roll

Motion is normalized using precomputed mean/std (`src/datasets/mean.pt`).

## Emotion Classes

The model supports 8 emotion classes (plus "None"):
```python
emo_map = {
    0: 'Anger',
    1: 'Contempt', 
    2: 'Disgust',
    3: 'Fear',
    4: 'Happiness',
    5: 'Neutral',
    6: 'Sadness',
    7: 'Surprise',
    8: 'None'  # No emotion conditioning
}
```

## Configuration Files

### `configs/audio2motion/inference/inference.yaml`
```yaml
# Key paths
motion_generator_path: pretrained_weights/moda/net-200.pth
audio_model_config: configs/audio2motion/model/audio_processer_config.yaml
motion_processer_config: configs/audio2motion/model/liveportrait_config.yaml

# Processing parameters
device_id: 0
batch_size: 50
source_max_dim: 1280
input_height: 256
input_width: 256
```

### `configs/audio2motion/model/config.yaml`
```yaml
model_name: TalkingHeadDiT-B  # Base model size

motion_generator:
  input_dim: 70
  output_dim: 70
  n_pred_frames: 80
  norm_type: rms_norm
  qk_norm: rms_norm

audio_projector:
  sequence_length: 1
  blocks: 12
  audio_feat_dim: 768
  audio_cond_dim: 63

noise_scheduler:
  type: flow_matching
  num_inference_steps: 10
  time_shifting: True
```

## Key Dependencies

- **PyTorch 2.6.0** with CUDA 12.4
- **transformers** - HuBERT/Wav2Vec models
- **diffusers** - Model utilities
- **xformers** - Efficient attention
- **insightface** - Face detection/analysis
- **mediapipe** - Landmark detection
- **decord** - Video reading
- **gradio** - Web interface

## Running Inference

### Command Line
```bash
python src/models/inference/moda_test.py \
    --image_path src/examples/reference_images/6.jpg \
    --audio_path src/examples/driving_audios/5.wav \
    --cfg_scale 1.2 \
    --save_dir output/
```

### Gradio App
```bash
python app.py
```

## Model Weights

Download from Hugging Face:
```bash
git clone https://huggingface.co/lixinyizju/moda pretrained_weights
```

Required weight structure:
```
pretrained_weights/
├── audio/
│   ├── chinese-hubert-base/
│   └── audio_separator/
├── decode/v1/first_stage/
│   ├── base_models/
│   │   ├── appearance_feature_extractor.pth
│   │   ├── motion_extractor.pth
│   │   ├── spade_generator.pth
│   │   └── warping_module.pth
│   └── retargeting_models/
│       └── stitching_retargeting_module.pth
└── moda/
    └── net-200.pth  # Main motion diffusion model
```

## Technical Notes

### Flow Matching
The model uses rectified flow / flow matching instead of traditional DDPM:
- Learns the velocity field v = x₁ - x₀
- Straight-line interpolation: xₜ = t·x₁ + (1-t)·x₀
- Only 10 sampling steps required (vs 50-1000 for DDPM)

### Classifier-Free Guidance
During sampling, conditions are dropped with probability:
- Audio: 20%
- Previous motion: 0%
- Emotion: 10%

At inference, CFG is applied: `output = uncond + cfg_scale * (cond - uncond)`

### Segment Fusion
Long audio is processed in 80-frame segments with 16-frame overlap:
- Linear blending in overlap region
- Kalman smoothing for temporal consistency

### Face Paste-Back
Generated faces are composited back:
1. Compute face mask using BiSeNet
2. Apply perspective transform to match original
3. Laplacian blending for seamless integration

## Common Modifications

### Change Inference Speed
Edit `configs/audio2motion/model/config.yaml`:
```yaml
noise_scheduler:
  num_inference_steps: 5  # Faster but lower quality
```

### Adjust Lip Sync
In `moda_test.py`, modify `modulate_lip()` parameters:
```python
alpha=5    # Lip motion amplification
beta=0.1   # Smoothing factor
```

### Change Audio Model
Edit `configs/audio2motion/model/audio_processer_config.yaml`:
```yaml
model_params:
  model_name: wav2vec  # Alternative to hubert
  is_chinese: False    # Use English model
```

## Debugging Tips

1. **Check audio format**: Must be 16kHz mono WAV
2. **Face detection fails**: Ensure clear frontal face in image
3. **OOM errors**: Reduce `batch_size` in inference.yaml
4. **Lip sync issues**: Try adjusting `cfg_scale` (1.0-2.0 range)
5. **Jerky motion**: Increase `overlap_len` or enable Kalman smoothing
