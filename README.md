# EviJEPA — Evidence-Aware Vision-Language Representation Learning for 3D Chest CT

EviJEPA is an evidence-aware CT–report representation learning framework. Instead of compressing a
whole CT volume and its radiology report into a single global embedding, EviJEPA keeps **multiple
evidence representations** so that several distinct pathological findings in one scan can each be
aligned with the part of the report that describes them.

The pretrained EviJEPA visual encoder was adapted to two tracks of the **MICCAI 2026 VLM3D Challenge**:

| Track | Rank | Primary metric |
|---|---|---|
| CT Abnormality Classification | **3rd** | CRG 0.5703 · AUROC 0.8646 |
| CT Report Generation | **3rd** | CRG 0.5555 · F1 0.6109 |

> ### ⚠️ Model weights are not available
> **The trained checkpoints (EviJEPA pretrained encoder, the abnormality classifier, and the report
> generation model) have been deleted and are no longer distributed with this repository.**
> This repository documents the methodology, configurations, and results. Reproducing the submitted
> numbers requires re-running pretraining and downstream training from the configurations below.
> No `.pt` / `.pth` / `weights.zip` artifacts are included.

---

## Table of Contents

1. [Team](#team)
2. [Methodology](#methodology)
3. [Pretrained Models](#pretrained-models)
4. [Data](#data)
5. [Preprocessing](#preprocessing)
6. [Training Settings](#training-settings)
7. [Installation](#installation)
8. [Usage — Input and Output Formats](#usage--input-and-output-formats)
9. [Leaderboard Submission Configurations](#leaderboard-submission-configurations)
10. [Results](#results)
11. [Limits and Limitations](#limits-and-limitations)
12. [References](#references)
13. [Usage Terms](#usage-terms)

---

## Team

| Name | Affiliation |
|---|---|
| Hyunjun Park | Korea Advanced Institute of Science and Technology (KAIST) |
| Woojin Na | Korea Advanced Institute of Science and Technology (KAIST) |
| Jinah Park | Korea Advanced Institute of Science and Technology (KAIST) |

---

## Methodology

We developed **EviJEPA**, an evidence-aware vision-language representation learning framework designed
to capture fine-grained clinical evidence from volumetric CT scans and radiology reports. Unlike
conventional approaches that compress CT volumes and reports into single global representations,
EviJEPA preserves multiple evidence representations to capture diverse pathological findings.

Each CT volume was resampled to a spacing of 1 × 1 × 2.5 mm and resized to 384 × 384 × 128. Three
Hounsfield-unit windows (lung, soft tissue, and wide) were used to capture complementary anatomical
information. The volumes were tokenized using z-axis pixel unshuffle and 3D convolutions, followed by
a LoRA-adapted visual encoder that extracts organ-wise visual tokens. Following the VL-JEPA
architecture, a trainable predictor transforms visual representations toward the semantic embedding
space of the text encoder.

EviJEPA employs two complementary alignment objectives. The **global branch** aggregates visual tokens
through learnable attention pooling and aligns them with report representations using contrastive
learning. The **local branch** employs expert-choice routing to extract multiple evidence
representations from visual and textual tokens. These representations are matched using **Sinkhorn
optimal transport** to establish fine-grained cross-modal correspondence. Additionally, **route
diversity** and **evidence diversity** regularization are introduced to prevent redundant token
selection and encourage experts to capture complementary clinical evidence. The framework is trained
with global and local InfoNCE objectives.

For the challenge submissions, the pretrained EviJEPA visual encoder was adapted to two downstream
tasks. Abnormality classification employed **label-wise queries** for 18 findings, while report
generation used a **two-stage projector and language-model fine-tuning** strategy based on DCP-PD with
MedGemma.

### Architecture details

**Visual encoder**

| Component | Setting |
|---|---|
| Backbone | V-JEPA 2.1 ViT-L (`vjepa2_1_vitl_dist_vitG_384`), embed dim 1024, 24 layers |
| Adaptation | LoRA (r = 16, α = 32, dropout 0.05) on attention projections |
| Patch embedding | 3D conv, patch 16 × 16, tubelet 2, **fully trained** (warm-started from the widened pretrained kernel) |
| Z-axis pixel unshuffle | factor 2 — packs 2 adjacent slices into the channel dimension (3 windows → 6 input channels) |
| Input channels | 3 HU windows → 6 after z-unshuffle |
| Token grid | 32 × 24 × 24 = **18,432 tokens** for a 384 × 384 × 128 volume |
| Positional encoding | 3D factorized RoPE |

**Organ-wise tokenization.** Organ masks (lung, heart, esophagus, aorta) are max-pooled onto the token
grid, so each organ selects the subset of visual tokens that overlaps it. Each organ branch is trained
against the organ-specific paragraph of the report.

**Predictor and text encoder**

| Component | Setting |
|---|---|
| Predictor | First 8 layers of Llama-3.2-1B, operating on organ tokens |
| Pooling (global branch) | **Per-organ learnable query** attention pooling (one query per organ) |
| Text encoder | EmbeddingGemma-300m, masked-mean pooling |
| Shared embedding dim | 512 |
| Max organ tokens | 2,560 per organ (random subsample above this) |

**Evidence branch (local alignment)**

| Component | Setting |
|---|---|
| Routing | Expert-Choice MoE, **E = 8 experts**, soft routing (τ = 0.5) |
| Capacity | 32 tokens per vision expert, 16 per text expert |
| Evidence dim | 512 |
| Matching | Sinkhorn optimal transport (5 iterations, ε = 0.05) |
| Regularizers | route diversity λ = 0.02, evidence diversity λ = 0.02 |
| Evidence loss weight | λ_ev = 1.0 |

**Contrastive objective.** Bidirectional InfoNCE (image→text and text→image) per organ, temperature
0.07, with cross-GPU negative gathering. Following fVLM, a **false-negative correction** treats
off-diagonal pairs as soft positives when both samples are normal for that organ or carry identical
report text, preventing two normal organs from being pushed apart.

### Downstream task heads

**(a) CT Abnormality Classification.** A mask-free label-attention head: **18 learnable label queries**
(one per finding) cross-attend *all* 18,432 visual tokens (8 heads, dropout 0.1) and produce one logit
per label. The EviJEPA visual encoder (LoRA + patch embedding) is used as a warm start and fine-tuned
jointly with the head.

**(b) CT Report Generation.** A two-stage adaptation on top of the shared EviJEPA encoder, following
DCP-PD with **MedGemma 1.5 4B**:
1. *Stage 1 — vision adaptation / projector*: **256 visual queries** compress the volumetric tokens
   into a fixed visual prefix aligned to the language model embedding space.
2. *Stage 2 — language model fine-tuning*: the generator is fine-tuned to produce the radiology report
   from the visual prefix. Greedy decoding, `max_new_tokens = 512`.
3. *Classifier cue integration*: the 18-label classifier (with 4-view TTA and calibration) produces
   positive findings; a bundled RadBERT check verifies whether each positive finding is already
   mentioned in the generated report, and any missed finding is prepended as a short cue sentence.

---

## Pretrained Models

| Model | Role | Availability |
|---|---|---|
| V-JEPA 2.1 ViT-L (`vjepa2_1_vitl_dist_vitG_384`) | Visual backbone initialization | Public (Meta V-JEPA 2) |
| Llama-3.2-1B | Predictor (first 8 layers) | Public (Meta, gated) |
| EmbeddingGemma-300m | Text encoder | Public (Google) |
| MedGemma 1.5 4B (`medgemma-1.5-4b-it`) | Report generation language model | Public (Google) |
| TotalSegmentator | Organ mask generation (preprocessing only) | Public |
| RadBERT-CT | Report finding verification (cue integration) | Public |
| **EviJEPA pretrained encoder** | Our pretrained CT–report model | ❌ **deleted — not available** |
| **Abnormality classifier checkpoint** | Challenge submission | ❌ **deleted — not available** |
| **Report generation checkpoint** | Challenge submission | ❌ **deleted — not available** |

All *base* models above are third-party and must be obtained from their original sources under their
own licenses. Only our fine-tuned checkpoints were removed.

---

## Data

**Dataset: CT-RATE (only).** No other dataset was used for training or evaluation.

| Split | Use |
|---|---|
| CT-RATE **training split** | All training (EviJEPA pretraining, classifier, report generator) |
| CT-RATE **validation split** | Model selection, ablations, calibration, zero-shot evaluation |
| CT-RATE **test split** | Official challenge leaderboard evaluation (held out, run by organizers) |

**Access.** CT-RATE is distributed through its official release on Hugging Face and requires accepting
the dataset license and access conditions. See [References](#references).

**Components used**
- Volumetric chest CT (`*.nii.gz`)
- Free-text radiology reports (`train_reports.csv`, `validation_reports.csv`) — the `Findings` and
  `Impressions` sections, split into organ-level paragraphs
- Multi-abnormality labels (`train_predicted_labels.csv`, `valid_predicted_labels.csv`) — 18 binary
  findings per volume

**The 18 abnormality labels**

```
Medical material · Arterial wall calcification · Cardiomegaly · Pericardial effusion
Coronary artery wall calcification · Hiatal hernia · Lymphadenopathy · Emphysema
Atelectasis · Lung nodule · Lung opacity · Pulmonary fibrotic sequela · Pleural effusion
Mosaic attenuation pattern · Peribronchial thickening · Consolidation · Bronchiectasis
Interlobular septal thickening
```

---

## Preprocessing

Identical preprocessing is used for pretraining, both downstream tasks, and inference.

| Step | Setting |
|---|---|
| 1. Resample | Target spacing **1 × 1 × 2.5 mm** (trilinear) |
| 2. Resize / crop-pad | Target volume **384 × 384 × 128** (X × Y × Z), center crop or zero pad |
| 3. HU windowing | Stored as the **wide** window `[-1150, 350] → [0, 1]` |
| 4. Three-window expansion (in-model) | **Lung** `[-1000, 400]`, **Soft tissue** `[-150, 250]`, **Wide** `[-1150, 350]` → 3 channels |
| 5. Normalization | ImageNet mean/std per window channel |
| 6. Z-axis pixel unshuffle | factor 2 → 6 channels, halves the temporal token count |
| 7. Organ masks | TotalSegmentator → lung / heart / esophagus / aorta, max-pooled onto the token grid |

Reports are split into organ-level paragraphs (conclusion + description per organ). When an organ has
no abnormal conclusion, a normal template sentence (`"{organ} shows no significant abnormalities."`)
is used, and such samples are handled by the false-negative correction in the contrastive loss.

---

## Training Settings

### Stage 1 — EviJEPA pretraining (CT ↔ report contrastive)

Config: [`configs/pretrain_evijepa.yaml`](configs/pretrain_evijepa.yaml)

| Setting | Value |
|---|---|
| Epochs | 10 |
| Batch size | 16 per GPU, 2 GPUs (DDP) with cross-GPU negative gathering |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rate | 1e-4 (predictor / pooling / projections) |
| LR multipliers | ×0.1 for the visual encoder (LoRA + patch embed) and the text encoder |
| Warmup | 500 steps |
| Temperature | 0.07 |
| Grad clipping | 1.0 |
| Precision | bf16, gradient checkpointing enabled |
| Organs | lung, heart, esophagus, aorta |
| Max text length | 384 tokens |

**Organ gating (fVLM-style).** An organ contributes to the loss in a step only when at least one
sample in the (gathered) batch has an abnormal finding for that organ.

### Stage 2a — Abnormality classification

Config: [`configs/abnormality_classification.yaml`](configs/abnormality_classification.yaml)

| Setting | Value |
|---|---|
| Initialization | EviJEPA pretrained visual encoder (LoRA + patch embedding) |
| Head | Mask-free label attention, 18 queries, 8 heads, dropout 0.1 |
| Loss | BCE with per-label `pos_weight` (capped at 10) |
| Epochs | 25 |
| Batch | 2 × gradient accumulation 4 (effective 16) |
| LR | head 1e-3, visual encoder 1e-5 |
| Weight decay | 0.05, warmup 200 steps, grad clip 1.0, bf16 |
| Augmentation | flip p=0.5; affine p=0.2 (±15° rotation, 0.9–1.1 zoom, ±8 px translation); Gaussian noise p=0.15 (σ=0.02); brightness / contrast / gamma p=0.15 each |

> **Note on intensity augmentation.** Brightness / contrast / gamma are applied **inside the encoder,
> after** the three HU windows are derived — applying them to the stored `[0,1]` volume beforehand
> corrupts the HU semantics that the windowing depends on.

**Test-time augmentation (TTA).** Four views — original, in-plane X flip, in-plane Y flip, and XY flip
— are averaged in probability space.

**Calibration.** A per-label threshold `t` is chosen on the validation split and converted to a logit
bias `b = ln((1 - t) / t)`, so that the operating point moves to 0.5 while leaving the ranking (and
therefore AUROC) unchanged. Three variants were studied:
- **F1-optimal** — maximizes macro F1
- **CRG-optimal** — maximizes `2r·TP − FP`
- **Balanced** — maximizes `0.5·F1_norm + 0.5·CRG_norm` (used for the final submission)

### Stage 2b — Report generation

| Setting | Value |
|---|---|
| Visual encoder | Shared EviJEPA encoder (frozen / adapted via the stage-1 vision adaptation module) |
| Visual queries | 256 |
| Language model | MedGemma 1.5 4B (`medgemma-1.5-4b-it`) |
| Training | Two-stage (projector → language-model fine-tuning), DCP-PD style |
| Submitted checkpoint | **Stage-2, epoch 9** |
| Decoding | Greedy, `max_new_tokens = 512`, batch 4 |
| Cue integration | 18-label classifier (epoch 19, 4-view TTA, calibrated) + RadBERT verification; missed positive findings are prepended as cue sentences |

### Evaluation settings

- **Zero-shot classification**: prompt-based — for each finding, a positive and a negative prompt are
  encoded by the text encoder, and the volume is scored by the softmax over the two similarities
  (temperature 0.07). Global and evidence branches are combined (`combine` mode, α = 0.5).
- **Supervised classification**: macro AUROC over 18 labels; F1 and CRG at threshold 0.5 after
  calibration.
- **Report generation**: CRG, macro F1 (findings extracted from the generated report), BLEU, ROUGE-L,
  METEOR, accuracy — computed by the official challenge evaluation.

---

## Installation

### Requirements

- Linux, NVIDIA GPU with >= 40 GB memory (pretraining and classifier fine-tuning used 2 x 80 GB)
- CUDA 12.x driver
- Python 3.10

### 1. Clone this repository

```bash
git clone <this-repository> EviJEPA
cd EviJEPA
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch and dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install TotalSegmentator
```

### 3. Clone the V-JEPA 2 source repository

The visual backbone definition is loaded from Meta's V-JEPA 2 repository, which is **not** vendored
here. Clone it and expose it through the `VJEPA_REPO` environment variable:

```bash
git clone https://github.com/facebookresearch/vjepa2.git ../vjepa2
export VJEPA_REPO=$(realpath ../vjepa2)
```

`evijepa/vision_encoder.py` inserts `$VJEPA_REPO` into `sys.path` and imports `src.hub.backbones`.

### 4. Download the base models

All base models are third-party and must be obtained under their own licenses. Llama-3.2 and MedGemma
are gated, so `huggingface-cli login` is required first.

```bash
huggingface-cli login

huggingface-cli download meta-llama/Llama-3.2-1B      --local-dir models/Llama-3.2-1B
huggingface-cli download google/embeddinggemma-300m   --local-dir models/embeddinggemma-300m
huggingface-cli download google/medgemma-1.5-4b-it    --local-dir models/medgemma-1.5-4b-it
```

The V-JEPA 2.1 ViT-L checkpoint (`vjepa2_1_vitl_dist_vitG_384.pt`) is obtained from the V-JEPA 2
release described in that repository; place it under `models/`.

### 5. Download CT-RATE

CT-RATE requires accepting its license and access conditions. After approval:

```bash
huggingface-cli download ibrahimhamamci/CT-RATE --repo-type dataset --local-dir data/CT-RATE
```

The following parts are used:

```
data/CT-RATE/dataset/
├── train/ , valid/                              volumetric CT (*.nii.gz)
├── radiology_text_reports/
│   ├── train_reports.csv
│   └── validation_reports.csv
└── multi_abnormality_labels/
    ├── train_predicted_labels.csv
    └── valid_predicted_labels.csv
```

### 6. Prepare the training data

Training does **not** read raw NIfTI directly. `prepare_data.py` converts each volume into the
preprocessed `.npz` consumed by the dataloaders, and builds the organ-level report JSON files.

Each produced `.npz` contains:

| Key | Shape | Content |
|---|---|---|
| `ct` | `(384, 384, 128)` float16 | HU-windowed `[-1150, 350] -> [0, 1]`, resampled to 1 x 1 x 2.5 mm |
| `mask` | `(384, 384, 128)` int8 | organ labels: 1 lung, 2 heart, 3 esophagus, 4 aorta, 0 background |

```bash
python prepare_data.py \
  --ct_dir      data/CT-RATE/dataset/train \
  --out_dir     data/prepared/train \
  --reports_csv data/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
  --desc_info   data/desc_info.json \
  --conc_info   data/conc_info.json \
  --seg_cache   data/seg_cache

python prepare_data.py \
  --ct_dir    data/CT-RATE/dataset/valid \
  --out_dir   data/prepared/valid \
  --seg_cache data/seg_cache
```

Organ masks come from TotalSegmentator and are the slowest step. The run is resumable (existing
`.npz` files are skipped) and can be sharded across GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python prepare_data.py --ct_dir ... --out_dir ... --nshard 4 --shard 0 &
CUDA_VISIBLE_DEVICES=1 python prepare_data.py --ct_dir ... --out_dir ... --nshard 4 --shard 1 &
```

`--skip_masks` writes zero masks and is only useful for the mask-free classifier, which never reads
them.

> **Note on organ-level reports.** `prepare_data.py` splits each report into sentences and assigns
> them to the four organs with a keyword rule, producing `desc_info.json` (findings) and
> `conc_info.json` (impressions). Organs with no matched sentence fall back to a normal template
> during training. This is a simple reimplementation of the organ-splitting step and may differ in
> detail from the splitter used for the submitted models.

### 7. Point the configs at your paths

Edit `configs/*.yaml` so that `train_dir`, `valid_dir`, `desc_info`, `conc_info`, `labels`,
`vjepa21_ckpt`, `llama_name` and `gemma_name` match the locations created above.

---

## Usage — Input and Output Formats

### Input

Both challenge tracks consume **3D chest CT volumes in NIfTI format**:

```
/input/
├── <case_id_1>.nii.gz
├── <case_id_2>.nii.gz
└── ...
```

- Single-channel volumetric CT in **Hounsfield units**, with voxel spacing in the NIfTI header.
- Arbitrary input size and spacing — resampling to 1 × 1 × 2.5 mm and 384 × 384 × 128 is performed
  internally.

### Output — CT Abnormality Classification

One probability per finding per case, in the Grand-Challenge wrapped format written to
`/output/predictions.json`:

```json
[
  {
    "outputs": [
      {
        "value": {
          "name": "Generated probabilities",
          "type": "Abnormality Classification",
          "version": { "major": 1, "minor": 0 },
          "predictions": [
            {
              "input_image_name": "case_id_1",
              "probabilities": {
                "Medical material": 0.03,
                "Arterial wall calcification": 0.81,
                "Cardiomegaly": 0.12,
                "...": 0.0,
                "Interlobular septal thickening": 0.07
              }
            }
          ]
        }
      }
    ]
  }
]
```

Probabilities are **calibrated**, i.e. the 0.5 cut corresponds to the chosen operating point.
AUROC is unaffected by this calibration because the transform is monotonic.

### Output — CT Report Generation

```json
[
  {
    "outputs": [
      {
        "value": {
          "name": "Generated reports",
          "type": "Report generation",
          "version": "1.0",
          "generated_reports": [
            {
              "input_image_name": "case_id_1",
              "report": "FINDINGS:\nTrachea and both main bronchi are open. ...\n\nIMPRESSION:\n..."
            }
          ]
        }
      }
    ]
  }
]
```

### Running

> Because the trained checkpoints are **not distributed**, the commands below are the interfaces used
> for the submissions; they require checkpoints produced by re-running the training stages.

**Stage 1 — EviJEPA pretraining**

```bash
accelerate launch --num_processes 2 --gpu_ids 0,1 \
  train_pretrain.py --config configs/pretrain_evijepa.yaml
```

**Stage 2 — abnormality classifier**

```bash
accelerate launch --num_processes 2 --gpu_ids 0,1 \
  train_classifier.py --config configs/abnormality_classification.yaml
```

**Inference — abnormality classification**

```bash
python infer_classifier.py \
  --config configs/abnormality_classification.yaml \
  --ckpt  /path/to/classifier_epoch19.pt \
  --input  /path/to/input \
  --output /path/to/output \
  --calibration /path/to/calib_bias.json
```

Add `--no-tta` to disable the four-view test-time augmentation.

**Containerized inference** (the submission format). Weights are mounted rather than baked into the
image:

```bash
docker build -t evijepa-abnclass .
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  -v /path/to/weights:/weights:ro \
  evijepa-abnclass
```

Environment variables recognized by the inference entrypoints:

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_DIR` / `FORITHMUS_INPUT` | `/input` | Directory of `*.nii.gz` volumes |
| `OUTPUT_DIR` / `FORITHMUS_OUTPUT` | `/output` | Where `predictions.json` is written |
| `WEIGHTS_DIR` | `/weights` | Mounted checkpoint directory |
| `GEN_BATCH` | `4` | Report-generation batch size |
| `MAX_NEW_TOKENS` | `512` | Report-generation decoding length |

---

## Leaderboard Submission Configurations

### Submission 1 — CT Abnormality Classification (3rd place)

| Item | Value |
|---|---|
| Config file | [`configs/abnormality_classification.yaml`](configs/abnormality_classification.yaml) |
| Backbone init | EviJEPA pretrained encoder |
| Head | Mask-free label attention (`head_type: attn`), 18 queries, 8 heads |
| Loss | BCE + per-label `pos_weight` (cap 10) |
| Training | 25 epochs, effective batch 16, head LR 1e-3 / vision LR 1e-5 |
| Augmentation | flip + affine + noise + brightness/contrast/gamma (see config) |
| Submitted epoch | **19** |
| Inference | **4-view TTA** (original, X flip, Y flip, XY flip), probability-averaged |
| Calibration | **Balanced** (F1 + CRG) per-label thresholds → logit bias |
| Container | CUDA 12.1 runtime base, PyTorch cu128, thin image with weights mounted at `/weights` |

### Submission 2 — CT Report Generation (3rd place)

| Item | Value |
|---|---|
| Visual encoder | Shared EviJEPA encoder + stage-1 vision adaptation |
| Visual queries | 256 |
| Language model | MedGemma 1.5 4B (`medgemma-1.5-4b-it`) |
| Submitted checkpoint | Stage-2 **epoch 9** |
| Decoding | Greedy, `max_new_tokens = 512`, batch 4 |
| Auxiliary classifier | Abnormality classifier epoch 19, 4-view TTA, calibrated |
| Cue integration | RadBERT verifies which positive findings are missing from the raw report; missing ones are prepended as cue sentences |
| Container | Offline image; base models and checkpoints mounted at `/weights` |

---

## Results

All validation numbers below are computed on the **official CT-RATE validation set**; the challenge
results are computed by the organizers on the held-out **test set**.

### 1. Zero-shot multi-label classification

To evaluate the semantic alignment capability of EviJEPA independently of task-specific classifier
training, we performed zero-shot multi-label classification on the CT-RATE validation set.

| Method | AUC | ACC (Macro) | F1 (Macro) | Precision |
|---|---:|---:|---:|---:|
| CT-CLIP | 0.731 | 0.668 | 0.707 | 0.323 |
| BIUD | 0.713 | 0.681 | 0.716 | 0.338 |
| Merlin | 0.728 | 0.672 | 0.709 | 0.337 |
| RadZero3D | 0.762 | 0.701 | 0.742 | 0.369 |
| COLIPRI | 0.778 | 0.702 | 0.752 | — |
| fVLM | 0.778 | 0.718 | 0.751 | 0.379 |
| ViSD-Boost | 0.790 | 0.731 | 0.759 | 0.387 |
| **EviJEPA (Ours)** | **0.820** | **0.753** | **0.781** | **0.417** |

EviJEPA achieved the best performance across all four reported metrics, demonstrating the
effectiveness of evidence-aware CT–report alignment.

### 2. Downstream task — CT abnormality classification

For abnormality classification, we evaluated the effects of binary cross-entropy training, data
augmentation, and test-time augmentation on the CT-RATE validation set.

| Configuration | AUC | CRG (Macro) | F1 (Macro) |
|---|---:|---:|---:|
| EviJEPA (zero-shot) | 0.820 | — | 0.781 |
| BCE loss | 0.8705 | 0.5595 | 0.6235 |
| + Augmentation | 0.8837 | 0.5900 | 0.6333 |
| + TTA | **0.8862** | **0.5935** | **0.6415** |

The configuration with data augmentation and test-time augmentation achieved the highest validation
performance among the supervised configurations.

### 3. Official challenge results

Our submissions achieved **third place in both** CT abnormality classification and CT report
generation.

**CT Abnormality Classification**

| Rank | CRG Score | AUROC (Macro) | F1 (Macro) | Accuracy |
|---|---:|---:|---:|---:|
| **3rd** | 0.5703 | 0.8646 | 0.6270 | 0.8071 |

**CT Report Generation**

| Rank | CRG Score | F1 (Macro) | BLEU | ROUGE-L | METEOR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **3rd** | 0.5555 | 0.6109 | 0.2576 | 0.2615 | 0.2333 | 0.8004 |

Both challenge submissions leveraged the shared EviJEPA representation, with task-specific adaptation
applied to their respective downstream objectives.

---

## Limits and Limitations

**Reproducibility**
- **Trained weights are deleted and not distributed.** The reported numbers cannot be reproduced by
  inference alone; all stages must be retrained from the configurations in this repository.
- Training used 2 × 80 GB GPUs; the 18,432-token sequence makes the pipeline memory-intensive and
  effectively inaccessible on small GPUs without gradient checkpointing and micro-batching.

**Supervision and evaluation**
- The 18 abnormality labels in CT-RATE are **automatically extracted from reports** (RadBERT-derived),
  not radiologist-verified ground truth. Both training supervision and validation metrics inherit this
  label noise.
- Report-generation metrics (BLEU / ROUGE-L / METEOR) reward surface-level lexical overlap and
  correlate only loosely with clinical correctness; CRG and macro F1 are the more clinically meaningful
  numbers in our tables.

**Method**
- **Organ-level granularity.** Contrastive alignment is performed per organ, so a single organ
  embedding must encode *all* findings in that organ simultaneously. This dilutes rare or focal
  findings; lung nodule and pulmonary fibrotic sequela were consistently the weakest labels.
- **Only four organs** (lung, heart, esophagus, aorta) are routed. Findings that do not belong to these
  organs — notably *Medical material* and *Lymphadenopathy* — receive no organ-specific supervision
  during pretraining.
- **Hard mask-based token selection.** Tokens are assigned to an organ by a binary mask overlap, so
  boundary findings (e.g. pleural effusion at the lung margin) can fall outside the selected tokens.
- **Rare findings are under-sampled.** With a batch of 16 volumes, findings with ~7–10 % prevalence
  appear roughly once per batch, providing almost no positive pairs for contrastive learning.
- **False-negative correction is string-based.** Two patients with the same disease but differently
  worded reports are still treated as negatives.

**Generalization**
- Training and evaluation used **CT-RATE only**. Robustness to other scanners, protocols, institutions
  or populations is unverified. We observed that whole-volume CT representations can cluster strongly
  by acquisition parameters (e.g. slice thickness), so distribution shift is a realistic risk.
- A validation-to-test gap was observed (validation AUROC 0.8862 vs. test 0.8646), consistent with
  mild overfitting from checkpoint selection and threshold calibration on the validation split.

**Report generation**
- The generator can **over-call findings** on normal studies (false positives) and occasionally
  produces template-like text patterns inherited from the training corpus.
- Generated reports are **not clinically validated** and must not be used for diagnosis.

---

## References

<!-- Add citations here. Suggested entries: CT-RATE / CT-CLIP, V-JEPA 2, fVLM, Query2Label,
     Sinkhorn / optimal transport matching, Expert-Choice routing, MedGemma, Llama 3.2,
     EmbeddingGemma, TotalSegmentator, RadBERT, DCP-PD, and the VLM3D Challenge. -->

1.
2.
3.
4.
5.

---

## Usage Terms

- **Research use only.** This repository is released for academic and research purposes. It is **not a
  medical device**, has not been clinically validated, and must not be used for diagnosis, treatment,
  or any clinical decision-making.
- **Code license.** <!-- TODO: add a LICENSE file and name it here (e.g. MIT, Apache-2.0). -->
  To be specified. Unless stated otherwise, the code is provided "as is", without warranty of any kind.
- **Data.** CT-RATE is governed by its own license and data use agreement. This repository distributes
  **no** CT data, reports, or labels; users must obtain the dataset directly from the official source
  and comply with its terms.
- **Third-party models.** V-JEPA 2, Llama-3.2, EmbeddingGemma, MedGemma, TotalSegmentator and RadBERT
  are the property of their respective owners and are subject to their own licenses and acceptable-use
  policies. Access to some of these models is gated.
- **Model weights.** Our trained checkpoints have been deleted and are **not** redistributed here.
- **Attribution.** If you use this work, please cite the entries listed in
  [References](#references) and this repository.

---

## Repository Structure

```
EviJEPA/
├── README.md
├── requirements.txt
├── prepare_data.py
├── train_pretrain.py
├── train_classifier.py
├── infer_classifier.py
├── configs/
│   ├── pretrain_evijepa.yaml
│   └── abnormality_classification.yaml
└── evijepa/
    ├── preprocess.py
    ├── data.py
    ├── vision_encoder.py
    ├── predictor.py
    ├── text_encoder.py
    ├── moe.py
    ├── losses.py
    ├── model.py
    └── classifier.py
```

| File | Contents |
|---|---|
| `prepare_data.py` | CT-RATE -> preprocessed `.npz` (CT + organ mask) and organ-level report JSON |
| `evijepa/preprocess.py` | Resampling to 1 × 1 × 2.5 mm, crop/pad to 384 × 384 × 128, HU windowing |
| `evijepa/data.py` | CT-RATE organ-text dataset, label dataset, spatial augmentation |
| `evijepa/vision_encoder.py` | V-JEPA ViT-L + LoRA, z-unshuffle, three HU windows, organ token flags |
| `evijepa/predictor.py` | Llama predictor with per-organ learnable query pooling |
| `evijepa/text_encoder.py` | EmbeddingGemma text encoder with mean pooling |
| `evijepa/moe.py` | Expert-Choice routing, route and evidence diversity losses |
| `evijepa/losses.py` | Organ InfoNCE with false-negative correction, Sinkhorn evidence InfoNCE |
| `evijepa/model.py` | EviJEPA global and local alignment |
| `evijepa/classifier.py` | Label-attention head for the 18 findings |

> Paths inside the configuration files reflect the original training environment and must be adapted
> to your own dataset and checkpoint locations.
