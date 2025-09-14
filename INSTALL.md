# SemanticBox Installation

## Conda Setup 

```bash
conda create -n bound python=3.12
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy randaugment ffmpeg wandb tqdm pyyaml python-csv pprintpp dotmap pathlib
python -m pip install opencv-python ftfy regex omegaconf scikit-learn
```

## Florence Install

```bash
python -m pip install tranforms==4.49.0 einops timm peft
```
