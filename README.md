# MoDA (Fork)

This is a fork of the original MoDA repository. For the official codebase,
installation, inference instructions, and updates, please refer to upstream:

- https://github.com/lixinyyang/MoDA

Model weights are hosted here:

- https://huggingface.co/lixinyizju/moda

## Local additions

Download model weights into `pretrained_weights`:


## ⚙️ Installation (UV)

```bash
# 1. Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies
uv sync

# 3. Install ffmpeg
sudo apt-get update  
sudo apt-get install ffmpeg -y

# 4. Activate the environment
source .venv/bin/activate
```
## 📥 Download Model Weights

Download the weights from Hugging Face into `pretrained_weights`:

```bash
# 1. Install Git LFS (if not already installed)
sudo apt-get update
sudo apt-get install git-lfs -y
git lfs install

# 2. Clone the weights repo into ./pretrained_weights
git clone https://huggingface.co/lixinyizju/moda pretrained_weights
```
## &#x1F680; Inference
```python
python src/models/inference/moda_test.py  --image_path src/examples/reference_images/6.jpg  --audio_path src/examples/driving_audios/5.wav 
```

## Acknowledgements

This project is based on the original MoDA repository:

- https://github.com/lixinyyang/MoDA
