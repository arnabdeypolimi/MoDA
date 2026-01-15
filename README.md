<h1 align='center'>MoDA: Multi-modal Diffusion Architecture for Talking Head Generation</h1>

<div align="center">

<strong>Authors</strong> <br><br>

Xinyang&nbsp;Li<sup>1,2</sup>,&nbsp;
Gen&nbsp;Li<sup>2</sup>,&nbsp;
Zhihui&nbsp;Lin<sup>1,3</sup>,&nbsp;
Yichen&nbsp;Qian<sup>1,3&nbsp;†</sup>,&nbsp;
Gongxin&nbsp;Yao<sup>2</sup>,&nbsp;
Weinan&nbsp;Jia<sup>1</sup>,&nbsp;
Aowen&nbsp;Wang<sup>1</sup>,&nbsp;
Weihua&nbsp;Chen<sup>1,3</sup>,&nbsp;
Fan&nbsp;Wang<sup>1,3</sup> <br><br>

<sup>1</sup>Xunguang&nbsp;Team,&nbsp;DAMO&nbsp;Academy,&nbsp;Alibaba&nbsp;Group&nbsp;&nbsp;&nbsp;
<sup>2</sup>Zhejiang&nbsp;University&nbsp;&nbsp;&nbsp;
<sup>3</sup>Hupan&nbsp;Lab <br><br>

<sup>†</sup>Corresponding authors: yichen.qyc@alibaba-inc.com,&nbsp;l_xyang@zju.edu.cn

</div>
<br>
<div align='center'>
    <a href='https://lixinyyang.github.io/MoDA.github.io/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
    <a href='https://arxiv.org/abs/2507.03256'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
</div>

##  📂 Updates

* [2025.08.08] 🔥 We release our inference [codes](https://github.com/lixinyyang/MoDA/) and [models](https://huggingface.co/lixinyizju/moda/).

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
## ⚖️ Disclaimer
This project is intended for academic research, and we explicitly disclaim any responsibility for user-generated content. Users are solely liable for their actions while using the generative model. The project contributors have no legal affiliation with, nor accountability for, users' behaviors. It is imperative to use the generative model responsibly, adhering to both ethical and legal standards.

## 🙏🏻 Acknowledgements

We would like to thank the contributors to the [LivePortrait](https://github.com/KwaiVGI/LivePortrait), and [echomimic](https://github.com/antgroup/echomimic),[JoyVasa](https://github.com/jdh-algo/JoyVASA/),[Ditto](https://github.com/antgroup/ditto-talkinghead/), [Open Facevid2vid](https://github.com/zhanglonghao1992/One-Shot_Free-View_Neural_Talking_Head_Synthesis), [InsightFace](https://github.com/deepinsight/insightface), [X-Pose](https://github.com/IDEA-Research/X-Pose), [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk), [Hallo](https://github.com/fudan-generative-vision/hallo), [wav2vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec), [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain), [Q-Align](https://github.com/Q-Future/Q-Align), [Syncnet](https://github.com/joonson/syncnet_python), and [VBench](https://github.com/Vchitect/VBench) repositories, for their open research and extraordinary work.
If we missed any open-source projects or related articles, we would like to complement the acknowledgement of this specific work immediately.
## 📑 Citation

If you use MoDA in your research, please cite:

```bibtex
@article{li2025moda,
  title={MoDA: Multi-modal Diffusion Architecture for Talking Head Generation},
  author={Li, Xinyang and Li, Gen and Lin, Zhihui and Qian, Yichen and Yao, GongXin and Jia, Weinan and Chen, Weihua and Wang, Fan},
  journal={arXiv preprint arXiv:2507.03256},
  year={2025}
}
```
