# AeroTrack

<p align="center">
  <b>UAV-OVVIS: Unmanned Aerial Vehicles Also Need Open-Vocabulary Video Instance Segmentation</b><br>
  <a href="https://arxiv.org/abs/2607.08075"><img src="https://img.shields.io/badge/arXiv-2607.08075-b31b1b.svg" alt="arXiv"></a>
</p>

**AeroTrack** is a **training-free** framework for **Unmanned Aerial Vehicle Open-Vocabulary Video Instance Segmentation (UAV-OVVIS)**. It periodically detects text-specified targets with open-vocabulary recognizers (Grounding DINO / YOLO-World / SAM3), propagates instance masks via SAM2/SAM3 within short segments, and maintains globally consistent identities through Lifecycle-aware ID Association (LIA). We release five pipeline variants (**BL2–BL6**) and the **AeroVIS** benchmark.

<p align="center">
  <img src="Image/UAV-OVVIS.gif" width="96%">
</p>


<p align="center">
  <img src="Image/AeroTrack.png" width="96%">
</p>

We construct **[AeroVIS](https://drive.google.com/file/d/1DMLagGZMPntrvxk5W0PsaIoybsE7WX56/view?usp=drive_link)** for evaluating open-vocabulary video instance segmentation in Unmanned Aerial Vehicle (UAV) scenes. Evaluation results are shown below:

<p align="center">
  <img src="Image/table1.png" width="96%">
</p>

Qualitative results:

<p align="center">
  <img src="Image/fig2.png" width="96%">
</p>

**Setup:** [INSTALL.md](INSTALL.md)

## Demo

`--baseline` selects BL2–BL6; `--text` is a short open-vocabulary prompt. A **single word** needs no quotes (e.g. `vehicle`); **multi-word phrases** should be quoted (e.g. `"road median fence"`, `"person riding bicycle"`).

```bash
python demo_video.py --baseline 2 --video <video_or_frame_dir> --text vehicle --output outputs/
python demo_video.py --baseline 2 --video <video_or_frame_dir> --text "road median fence" --output outputs/
```

| BL | Training-free pipeline |
| :--- | :--- |
| BL2 | SAM3 + SAM3 |
| BL3 | YOLO-World + SAM3 |
| BL4 | GroundingDINO + SAM3 |
| BL5 | YOLO-World + SAM2 |
| BL6 | GroundingDINO + SAM2 |

## AeroVIS

```text
data/AeroVIS/
├── aero_vis.json
└── sequences/
```

Categories and format: [data/AeroVIS/data.md](data/AeroVIS/data.md)

## Evaluation

Evaluate on the **AeroVIS** benchmark. `--baseline` supports **BL2–BL6** (five training-free pipelines). `--text` selects one of **9 categories**:

`person` · `car` · `truck` · `bus` · `bicycle` · `motorcycle` · `tricycle` · `boat` · `vehicle`

One category:

```bash
python evaluate.py --baseline 2 --text car --output outputs/bl2/car
```

All categories:

```bash
python evaluate.py --baseline 4 --output outputs/bl4
```

Useful flags: `--nooutput` skips visualization; `--json <predictions.json>` recomputes metrics only.

During evaluation, per-category tuning is loaded from `configs/categories/<category>.yaml`.

## Project Structure

```text
.
├── aerotrack_core/           # Core model, pipeline, and evaluation modules
├── configs/
│   ├── baselines/            # BL2-BL6 baseline YAML configs
│   └── categories/           # Per-category detector/segmenter tuning for evaluation
├── data/AeroVIS/             # AeroVIS dataset description and expected data layout
├── demo_video.py             # Video/frame-sequence demo entry point
├── evaluate.py               # Benchmark evaluation entry point
├── INSTALL.md                # Environment and checkpoint setup
└── requirements.txt          # Common Python dependencies
```

## Notes

- This repository does not include large model checkpoints.
- AeroVIS data files are distributed separately via Google Drive.
- AeroVIS annotation terms are described in [data/AeroVIS/data.md](data/AeroVIS/data.md).

## Citation

If AeroTrack or AeroVIS is helpful to your research, please cite our paper:

```bibtex
@article{dou2026uavovvis,
  title={UAV-OVVIS: Unmanned Aerial Vehicles Also Need Open-Vocabulary Video Instance Segmentation},
  author={Dou, Mingyu and Qiu, Shi and Hu, Ming and Chen, Yifan and Sun, Zhe},
  journal={arXiv preprint arXiv:2607.08075},
  year={2026}
}
```

Paper: [arXiv:2607.08075](https://arxiv.org/abs/2607.08075)

## Acknowledgments

**Code.** This project builds upon the following open-source works:

- [SAM3](https://github.com/facebookresearch/sam3)
- [SAM2](https://github.com/facebookresearch/sam2)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [YOLO-World](https://github.com/AILab-CVC/YOLO-World)

**Data.** AeroVIS is constructed from the following public UAV datasets:

- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)
- [UAVDT](https://sites.google.com/view/grli-uavdt)
- [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee)
