<div align="center">

# Rethinking Multi-Scale Representations in Deep Deraining Transformer (AAAI2024)

</div>

<!-- > Rethinking Multi-Scale Representations in Deep Deraining Transformer -->
Welcome to visit our website (专注底层视觉领域的信息服务平台) for low-level vision: https://lowlevelcv.com/


## 🛠️ Training and Testing
1. Please put datasets in the folder `Datasets/`.
2. Follow the instructions below to begin training our model.
```
bash train.sh
```
Run the script then you can find the generated experimental logs in the folder `checkpoints`.

3. Follow the instructions below to begin testing our model.
```
python test.py
```
Run the script then you can find the output visual results in the folder `results/`.


## 🤖 Pre-trained Models
| Models | MSDT |
|:-----: |:-----: |
| Rain200L | [Google Drive](https://drive.google.com/file/d/1qk8pUq7oM4Z4v2X-qmWJpE2LmUuweL4_/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1jikJhCuv51bvkl9vF2AkKw?pwd=8ajd) (8ajd) 
| Rain200H | [Google Drive](https://drive.google.com/file/d/1y8gjAvnt0kkf1dSEyauVFu2weLi53LmF/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1jr01T_hzl8K_h2VksrmlFQ?pwd=97lm) (97lm) 
| DID-Data | [Google Drive](https://drive.google.com/file/d/1RDvMFZn57UFrkeeojRHXwR7YbvXSGR5i/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1PJrRTDsG4vL4XwhNd8kfHg?pwd=5g4p) (5g4p) 
| DDN-Data | [Google Drive](https://drive.google.com/file/d/1p7FVQuZSw4n0nXEvLrsJPtYxzlMyOCK0/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1Y3YRkNO40m6bII-R3-Hi4g?pwd=b0b5) (b0b5) 
| SPA-Data | [Google Drive](https://drive.google.com/file/d/1hEpYFrFG0qhKassfYAZmXwUnNUYmGMLs/view?usp=drive_link) / [Baidu Netdisk](https://pan.baidu.com/s/1CO7wlaZyhu2egjfdaavFeQ?pwd=x0i5) (x0i5) 


## 🚨 Performance Evaluation
See folder "evaluations" 

1) *for Rain200L/H and SPA-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/sauchm/MSDT/tree/main/evaluations/Evalution_Rain200L_Rain200H_SPA-Data).

2) *for DID-Data and DDN-Data datasets*: 
PSNR and SSIM results are computed by using this [Matlab Code](https://github.com/sauchm/MSDT/tree/main/evaluations/Evaluation_DID-Data_DDN-Data).



## 🚀 Visual Deraining Results

| Methods | MSDT |
|:-----: |:-----: |
| Rain200L | [Baidu Netdisk](https://pan.baidu.com/s/1us3smvwhAe3azJPnunWs8w?pwd=1xkc) (1xkc) 
| Rain200H | [Baidu Netdisk](https://pan.baidu.com/s/1S__NNB0jV2ING2ngR0PjiA?pwd=yr3n) (yr3n) 
| DID-Data | [Baidu Netdisk](https://pan.baidu.com/s/1Rif4QC1AuDF4ccHteg_A4A?pwd=242e) (242e) 
| DDN-Data | [Baidu Netdisk](https://pan.baidu.com/s/1JFHyrTMSdsFotOJ6pKokow?pwd=2pwk) (2pwk) 
| SPA-Data | [Baidu Netdisk](https://pan.baidu.com/s/14fSFf_T7AOD44ktso56Rxw?pwd=cag0) (cag0) 


## 👍 Acknowledgement
Thanks for their awesome works ([DeepRFT](https://github.com/INVOKERer/DeepRFT) and [DRSformer](https://github.com/cschenxiang/DRSformer)).

## RaindropClarity single-model baselines

This repository also provides two fair, independent single-MSDT experiments:

- `MSDT baseline`: the original MSDT parameter path; scene labels are neither loaded nor used.
- `MSDT + Scene`: the same MSDT with a zero-initialized four-class FiLM module at the shared 1/4-resolution bottleneck.

Both supplied configs share the split manifest, seed, augmentations, loss, optimizer,
scheduler, batch size, and epoch count. No pseudo-GT, ensemble, TTA, scene classifier,
test-time tuning, or unrelated pretrained deraining weights are used.

Install the additional metric/config dependencies:

```bash
pip install -r requirements-raindrop.txt
```

### Data layouts and split

Flat paired data use exact filename matching (not sorted positional matching):

```text
DATA_ROOT/
  Drop/Day_00001_xxx.png
  Clear/Day_00001_xxx.png
  Drop_scen_pred.json
```

The original release is also detected, including a combined day/night parent:

```text
DATA_ROOT/
  DayRainDrop/{Drop,Blur,Clear}/00001/frame.png
  NightRainDrop/{Drop,Blur,Clear}/00001/frame.png
```

For the official raw release, each `Drop/<scene>/<frame>` is one raindrop-removal
input and its same-relative-path `Clear` image is the GT. `Blur` is the auxiliary
raindrop-free blurry background, not a second rainy input. Following the dataset
authors' code, `Blur == Clear` identifies a background-focused pair; otherwise the
item is a raindrop-focused triplet. Every frame inside each scene/triplet directory
is expanded into an MSDT sample. Treating `Blur -> Clear` as another deraining sample
would change the task into extra deblurring training, so it is deliberately excluded.

Flat names are grouped by the default `Day_00001`/`Night_00001` prefix. Change
`data.group_regex` if local names use another convention. Raw data are grouped by
their scene directory. The first run creates `splits/raindrop_split.json`; later runs
reuse it and fail if its groups no longer match, preventing scene leakage.

### Train, validate, and resume

```bash
# A: no scene labels
python train_raindrop.py --config configs/raindrop_no_scene.yaml --data-root /path/to/DATA_ROOT

# B: four-class scene conditioning
python train_raindrop.py --config configs/raindrop_scene.yaml --data-root /path/to/DATA_ROOT \
  --scene-json /path/to/DATA_ROOT/Drop_scen_pred.json

# Resume all model/optimizer/scheduler/scaler state
python train_raindrop.py --config configs/raindrop_scene.yaml --data-root /path/to/DATA_ROOT \
  --scene-json /path/to/DATA_ROOT/Drop_scen_pred.json \
  --resume checkpoints/raindrop_scene/model_latest.pth

# Standalone fixed-split validation
python eval_raindrop.py --config configs/raindrop_no_scene.yaml \
  --weights checkpoints/raindrop_no_scene/model_best.pth --data-root /path/to/DATA_ROOT
```

Validation reports `PSNR_Y`, `SSIM_Y`, AlexNet `LPIPS`, and
`Score = PSNR_Y + 10*SSIM_Y - 5*LPIPS`. Only `model_best.pth` and
`model_latest.pth` are maintained.

### Inference

```bash
# One image, no-scene model
python infer_raindrop.py --config configs/raindrop_no_scene.yaml \
  --weights checkpoints/raindrop_no_scene/model_best.pth \
  --input image.png --output-dir results/no_scene

# One image with a manual scene ID
python infer_raindrop.py --config configs/raindrop_scene.yaml \
  --weights checkpoints/raindrop_scene/model_best.pth \
  --input image.png --scene-id 3 --output-dir results/scene

# Folder labels by exact filename, with overlap-tile inference
python infer_raindrop.py --config configs/raindrop_scene.yaml \
  --weights checkpoints/raindrop_scene/model_best.pth \
  --input /path/to/images --scene-json labels.json --output-dir results/scene \
  --tile-size 512 --tile-overlap 64

# Fixed validation split inference
python infer_raindrop.py --config configs/raindrop_scene.yaml \
  --weights checkpoints/raindrop_scene/model_best.pth --validation \
  --data-root /path/to/DATA_ROOT --scene-json /path/to/Drop_scen_pred.json \
  --output-dir results/scene_val
```

Images are reflect-padded to the network multiple and cropped back to their original
size. Inference saves only `output[0]` as lossless PNG without TTA or ensembling.

### Fair ablation and smoke test

```bash
bash scripts/train_scene_ablation.sh /path/to/DATA_ROOT /path/to/Drop_scen_pred.json
bash scripts/eval_scene_ablation.sh /path/to/DATA_ROOT /path/to/Drop_scen_pred.json
bash scripts/smoke_test_raindrop.sh
```

`train_scene_ablation.sh` is the unified single-RTX-5090 launcher. It exposes
one GPU, uses `cuda:0`, AMP from both configs, batch size 1, and defaults to four
DataLoader workers. Override the physical card or worker count without editing it:

```bash
GPU_ID=0 NUM_WORKERS=8 bash scripts/train_scene_ablation.sh \
  /path/to/DATA_ROOT /path/to/Drop_scen_pred.json
```

`smoke_test_raindrop.sh` likewise exposes one GPU and runs on `cuda:0` by default.
For a CPU-only diagnostic fallback, use `DEVICE=cpu bash scripts/smoke_test_raindrop.sh`.

Evaluation writes `scene_ablation.csv` and `scene_ablation.md`, including the raw
`MSDT + Scene - MSDT baseline` metric deltas. These experiments are baselines and a
scene-conditioning ablation; they are not claims of reproducing a challenge Top-1 result.

## 📘 Citation
Please consider citing our work as follows if it is helpful.
```
@inproceedings{chen2024rethinking,
  title={Rethinking Multi-Scale Representations in Deep Deraining Transformer},
  author={Chen, Hongming and Chen, Xiang and Lu, Jiyang and Li, Yufeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={1046--1053},
  year={2024}
}
```

