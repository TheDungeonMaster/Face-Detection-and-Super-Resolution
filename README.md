# Face-Detection-and-Super-Resolution
Deep Learning pipeline for face extraction from CCTV footage and Super Resolution of cropped face images for improved face recognition accuracy.


# Project Structure:
face-sr-det/
├─ README.md
├─ LICENSE
├─ pyproject.toml                  # or requirements.txt + setup.cfg
├─ .pre-commit-config.yaml
├─ .gitignore
├─ Makefile                        # handy shortcuts (see below)
├─ docker/
│  └─ Dockerfile
├─ scripts/
│  ├─ download_data.py             # fetch datasets to data/raw
│  ├─ prepare_splits.py            # identities-based split, CSVs
│  ├─ generate_lr.py               # synthetic degradations for SR
│  ├─ export_onnx.py               # export models to ONNX
│  └─ benchmark_rt.py              # runtime/throughput benchmarks
├─ configs/                        # Hydra configs
│  ├─ config.yaml                  # defaults + job name, paths
│  ├─ data/
│  │  ├─ ffhq.yaml
│  │  └─ cctv_eval.yaml
│  ├─ model/
│  │  ├─ detector.yaml             # e.g., retinaface/mtcnn
│  │  ├─ sr_main.yaml              # main SR (e.g., SwinIR-light)
│  │  └─ baselines.yaml            # Real-ESRGAN/GFPGAN/CodeFormer
│  ├─ train/
│  │  ├─ sr.yaml                   # losses, optim, sched, steps
│  │  └─ detector.yaml
│  └─ eval/
│     ├─ sr.yaml                   # PSNR/SSIM/LPIPS
│     └─ id.yaml                   # TAR@FAR, ROC-AUC, EER
├─ data/
│  ├─ raw/                         # original datasets (read-only)
│  ├─ interim/                     # intermediate (crops, splits)
│  ├─ processed/                   # LR/HR pairs for SR training
│  └─ external/                    # anything imported (Kaggle, etc.)
├─ notebooks/
│  ├─ 00_eda.ipynb
│  ├─ 10_visualize_detections.ipynb
│  └─ 20_eval_curves.ipynb
├─ src/
│  ├─ facesr/
│  │  ├─ __init__.py
│  │  ├─ utils/
│  │  │  ├─ io.py                  # read/write, caching, hashing
│  │  │  ├─ img.py                 # augmentations, degradations
│  │  │  └─ metrics.py             # PSNR, SSIM, LPIPS wrappers
│  │  ├─ data/
│  │  │  ├─ datasets.py            # FFHQ/DIV2K/cctv loaders
│  │  │  └─ transforms.py          # flips, rotations, to-tensor
│  │  ├─ detection/
│  │  │  ├─ detector_base.py
│  │  │  ├─ retinaface.py          # or mtcnn.py
│  │  │  └─ postprocess.py         # NMS, bbox to aligned crops
│  │  ├─ sr/
│  │  │  ├─ model_base.py
│  │  │  ├─ swinir_light.py
│  │  │  ├─ realesrgan.py
│  │  │  ├─ gfpgan.py
│  │  │  └─ codeformer.py
│  │  ├─ id/
│  │  │  ├─ arcface.py             # embedding + identity loss
│  │  │  └─ verifier.py            # TAR@FAR, ROC-AUC, EER calc
│  │  ├─ train/
│  │  │  ├─ lit_sr_module.py       # LightningModule (SR)
│  │  │  ├─ lit_detector_module.py
│  │  │  └─ loops.py               # custom loops/callbacks
│  │  ├─ evaluate/
│  │  │  ├─ eval_sr.py             # PSNR/SSIM/LPIPS
│  │  │  └─ eval_id.py             # verification metrics
│  │  └─ infer/
│  │     ├─ pipeline.py            # detect → align → SR → verify (opt)
│  │     └─ cli.py                 # single entrypoint for inference
│  └─ cli/
│     ├─ train_sr.py               # hydra.main() → Lightning Trainer
│     ├─ train_detector.py
│     ├─ eval_sr.py
│     ├─ eval_id.py
│     └─ detect_and_upscale.py     # batch video/image inference
├─ experiments/
│  ├─ sr_main/                     # Hydra logs, configs, metrics
│  └─ detector/
├─ models/
│  ├─ detector/                     # checkpoints
│  └─ sr/
├─ tests/
│  ├─ test_transforms.py
│  ├─ test_detector_io.py
│  ├─ test_sr_forward.py
│  ├─ test_metrics.py
│  └─ test_infer_pipeline.py
└─ docs/
   ├─ system_design.md
   ├─ data_card.md
   └─ model_card.md
