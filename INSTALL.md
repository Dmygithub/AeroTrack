# AeroTrack Environment Setup

## 1. Weight Assets

Place the following checkpoint files under the `checkpoints/` directory:

| Model | Path | Usage | Download |
| :--- | :--- | :--- | :--- |
| **SAM3** | `checkpoints/sam3.pt` | Mask propagation and relocalization for BL2-BL4 | [HuggingFace gated access](https://huggingface.co/facebook/sam3) |
| **SAM2.1 Hiera-L** | `checkpoints/sam2.1_hiera_large.pt` | Mask propagation for BL5/BL6 (857 MB) | [HuggingFace](https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt) |
| **YOLO-World** | `checkpoints/l_stage1-7d280586.pth` | Open-vocabulary detection for BL3/BL5 | [HuggingFace](https://huggingface.co/wondervictor/YOLO-World-V2.1/blob/main/l_stage1-7d280586.pth) |
| **GroundingDINO** | `checkpoints/groundingdino_swint_ogc.pth` | Open-vocabulary detection for BL4/BL6 | [GitHub Release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| **CLIP** | `checkpoints/Clip-vit-base-patch32/` | YOLO-World text encoder for offline use | [HuggingFace](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) |
| **BERT** | `checkpoints/bert-base-uncased/` | GroundingDINO text encoder for offline use | [HuggingFace](https://huggingface.co/google-bert/bert-base-uncased/tree/main) |

> **Offline setup:** CLIP and BERT can be downloaded manually.
> Place CLIP under `checkpoints/Clip-vit-base-patch32/` and BERT under `checkpoints/bert-base-uncased/`.

---

## 2. Prerequisites

- **OS**: Linux (recommended) / Windows
- **CUDA Toolkit**: 12.1+
- **Python**: 3.10

---

## 3. Environment Setup

### 1) Base Environment

```bash
# Create the environment.
conda create -n uav python=3.10 -c conda-forge -y
conda activate uav

# Install PyTorch.
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install Triton.
python -m pip install triton==2.3.0
```

### 2) OpenMMLab Dependencies (YOLO-World)
Version lock: `mmcv==2.1.0` / `mmengine==0.10.7` / `mmdet==3.3.0` / `mmyolo==0.6.0`
`mmcv` 2.0.x has no prebuilt wheel for `torch==2.1.0 + cu121`, so this setup uses **mmcv 2.1.0** and patches the `mmyolo` version assertion.

```bash
# Install MMCV.
python -m pip install -U pip openmim
mim install "mmcv==2.1.0"

# Install detection components.
mim install "mmengine==0.10.7" "mmdet==3.3.0" "mmyolo==0.6.0"

python -c "
import sys, glob
pat = 'lib/python*/site-packages/mmyolo/__init__.py'
files = glob.glob(sys.prefix + '/' + pat)
assert files, 'mmyolo not found under ' + sys.prefix
f = files[0]; s = open(f).read()
open(f,'w').write(s.replace('assert (mmcv_version >= digit_version(mmcv_minimum_version)','assert (True or mmcv_version >= digit_version(mmcv_minimum_version)'))
print('patched', f)
"

# Pin the Transformers version.
python -m pip install "transformers==4.36.2"
```

### 3) Install aerotrack_core

```bash
# Install aerotrack_core.
cd aerotrack_core
python -m pip install -e .
cd ..

```

### 4) GroundingDINO CUDA Operators


```bash
cd aerotrack_core/models/groundingdino_sam3
bash compile.sh
cd ../../..
```

> **BERT/CLIP text encoders:** If `bert-base-uncased` and
> `Clip-vit-base-patch32` have been downloaded manually into `checkpoints/`,
> the code will read the local files directly and will not trigger online downloads.
> If they are not present locally, the code will fetch and cache them from HuggingFace Hub when network access is available.

### 5) Common Dependencies

```bash
# Install common dependencies.
python -m pip install -r requirements.txt

python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install "opencv-python<=4.11.0.86"
```

---

## 4. Quick Start

Replace `<video_or_frame_dir>` with your own video file or frame-sequence directory.

```bash
# BL2: SAM3
python demo_video.py --baseline 2 --video <video_or_frame_dir> --text vehicle --output outputs/videos/bl2_vehicle

# BL3: YOLO-World + SAM3
python demo_video.py --baseline 3 --video <video_or_frame_dir> --text vehicle --output outputs/videos/bl3_vehicle

# BL4: Grounding DINO + SAM3
python demo_video.py --baseline 4 --video <video_or_frame_dir> --text vehicle --output outputs/videos/bl4_vehicle

# BL5: YOLO-World + SAM2
python demo_video.py --baseline 5 --video <video_or_frame_dir> --text vehicle --output outputs/videos/bl5_vehicle

# BL6: Grounding DINO + SAM2
python demo_video.py --baseline 6 --video <video_or_frame_dir> --text vehicle --output outputs/videos/bl6_vehicle

```


---

## Appendix: Offline File Checklist for Large Model Weights

HuggingFace repositories often include weights for multiple deep-learning frameworks. If storage space is limited, keep only the files listed below. Large files such as `.h5` and `.msgpack` that are not listed here can be safely removed because they are TensorFlow and JAX weights, while AeroTrack uses PyTorch.

### `checkpoints/bert-base-uncased/` (5 required files)
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`
- `model.safetensors` 
- `pytorch_model.bin`

### `checkpoints/Clip-vit-base-patch32/` (7 required files)
- `config.json`
- `preprocessor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `pytorch_model.bin` (or `model.safetensors`)
