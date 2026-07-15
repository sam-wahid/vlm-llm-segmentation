# 🧠 VLM-LLM-SAM Distillation Framework

> A unified knowledge distillation framework for compressing **Vision-Language Models (VLM)**, **Large Language Models (LLM)**, and **Segment Anything Models (SAM)** into lightweight, deployable models for specialized domains (e.g. medical imaging, resource-constrained edge devices).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebooks](#running-the-notebooks)
- [Requirements](#requirements)
- [Experimental Results](#experimental-results)
- [Roadmap](#roadmap)
- [Conclusion](#conclusion)
- [References](#references)

---

## Overview

Foundation models — VLMs, LLMs, and SAM — deliver strong general-purpose performance but are often too large, slow, or expensive to deploy in specialized, latency- or resource-constrained settings (e.g. clinical decision support, embedded/edge inference, on-device assistants). This project builds a **reproducible knowledge distillation (KD) framework** that transfers capability from large "teacher" foundation models into smaller, efficient "student" models while preserving task performance within a target domain.

The framework currently spans three tracks, each following a consistent pipeline — dataset loading, teacher/student model setup, distillation training, and benchmark evaluation:

| Track | Focus | Status |
|---|---|---|
| **VLM** | Distilling vision-language backbones (CLIP, DINOv1/v2, SigLIP) and task-specific VLMs (e.g. BLIP for medical VQA) into compact student networks | Active |
| **LLM** | Distilling large language models for specialized-domain text tasks |  Planned |
| **SAM** | Distilling Segment Anything Model variants for efficient, domain-specific segmentation |  Planned |

---

## Repository Structure

```
vlm-llm-segmentation/
│
├── README.md
├── requirements.txt
│
└── Notebooks/
    ├── VLM/
    │   ├── CLIP_benchmark.ipynb
    |   ├── Openended-VQA-RAD.ipynb
    │   ├── siglipbenchmark.ipynb
    │   ├── dinov1benchmark.ipynb
    │   ├── dinov2benchmark.ipynb
    │   ├── KD-Beansdataset.ipynb
    │   ├── VQA-RAD-kd.ipynb
    │   ├── VQA-RAD-kd(BLIP-Mobilenetv2).ipynb
    |   ├── VQAv2-Open ended(Qwen).ipynb
    │   ├── VQA_RAD_Report.docx
    │   └── README.md
    │
    └── LLM/
        └── README.md          
```

> A `Notebooks/SAM/` track for Segment Anything Model distillation is planned — see [Roadmap](#roadmap).

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sam-wahid/vlm-llm-segmentation.git
cd vlm-llm-segmentation
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```

---

## Running the Notebooks

| Notebook | Description |
|---|---|
| `VLM/CLIP_benchmark.ipynb` | Teacher/student setup and benchmarking for CLIP-based vision-language distillation |
| `VLM/siglipbenchmark.ipynb` | Benchmarking distillation from SigLIP as the teacher backbone |
| `VLM/dinov1benchmark.ipynb` | Distillation benchmarking using DINOv1 self-supervised features as the teacher |
| `VLM/dinov2benchmark.ipynb` | Distillation benchmarking using DINOv2 self-supervised features as the teacher |
| `VLM/KD-Beansdataset.ipynb` | Knowledge distillation pipeline evaluated on the Beans (plant disease) image classification dataset |
| `VLM/VQA-RAD-kd.ipynb` | Knowledge distillation for medical Visual Question Answering on the VQA-RAD dataset |
| `VLM/VQA-RAD-kd(BLIP-Mobilenetv2).ipynb` | BLIP → MobileNetV2 distillation for lightweight medical VQA inference |
| `VLM/VQAv2-Open ended(Qwen).ipynb` | Qwen2.5-VL-7B → Qwen3-VL-2B distillation (QLoRA teacher fine-tuning + LoRA KD) for open-ended general-domain VQA on VQAv2 |
>  Each notebook is self-contained (dataset download, teacher/student setup, training, evaluation) — run them independently based on the model/dataset you're interested in.

Datasets and pretrained teacher checkpoints are loaded directly from [HuggingFace Datasets](https://huggingface.co/datasets) and [HuggingFace Hub](https://huggingface.co/models) — no manual download required.

---

## Requirements

- **Python** 3.9+
- **GPU** recommended for teacher inference and distillation training (CUDA-enabled)

### Key dependencies

```
torch
torchvision
torchaudio
transformers
timm
huggingface_hub
datasets
onnxruntime==1.20.1
onnx==1.20.1
Pillow
opencv-python
numpy
scipy
pandas
matplotlib
seaborn
tqdm
jiwer
jsonlines
jupyter
ipykernel
```

See [`requirements.txt`](requirements.txt) for the full list.

---

## Experimental Results

Distillation results compare **teacher** (large foundation model) vs. **student** (compact model) performance on domain-specific benchmarks, tracking accuracy retention alongside compute/parameter reduction.

> Results will be populated here as each notebook's benchmarking run is finalized.

### Vision-Language Distillation (VLM)

| Task / Dataset | Teacher | Student | Teacher Baseline | Student baseline | Student after KD |
|---|---|---|---|---|---|
| Medical VQA — VQA-RAD | CLIP vit L/14 | MobileNetV2 | 69.44% | 56.02%| 58.33% |
| Medical VQA — VQA-RAD | BLIP | MobileNetV2 | 57.87% | 46.76%| 54.17% |
| Image Classification — Beans | merve/beans-vit-224| Mobilenetv2 | 93.75% |33.59% | 59.38% |
| Medical VQA — VQA-RAD(OE) | BLIP-2 | BLIP VQA BASE | 78.84% | 66.31%| 73.06% |
| General VQA(Open ended)-VQAv2 | Qwen2.5-VL-7B-Instruct | Qwen3-VL-2B-Instruct | 95.04% | 81.80% | 88.92% |

---

## Roadmap

- [x] VLM distillation: CLIP, SigLIP, DINOv1/v2 backbone benchmarking
- [x] VLM distillation: BLIP → MobileNetV2 for medical VQA (VQA-RAD)
- [x] VLM distillation: general image classification (Beans dataset)
- [ ] LLM distillation track for specialized-domain text tasks
- [ ] SAM distillation track for efficient, domain-specific segmentation
- [ ] Unified evaluation harness across VLM / LLM / SAM tracks
- [ ] Cross-modal distillation experiments (shared student backbones)

---

## Conclusion

This repository lays the groundwork for a cross-modal distillation framework spanning vision-language understanding, language modeling, and segmentation. The VLM track is the most mature so far, with multiple teacher backbones (CLIP, SigLIP, DINOv1/v2) benchmarked and a working BLIP-to-MobileNetV2 distillation pipeline validated on a real specialized domain — medical VQA (VQA-RAD). The LLM and SAM tracks are scaffolded and next in line, with the long-term goal of a single, consistent framework for producing small, deployable models across all three modalities without sacrificing domain-specific accuracy.

---

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.*
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).*
- Zhai, X. et al. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).*
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).*
- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- Li, J. et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training.*
- Kirillov, A. et al. (2023). *Segment Anything (SAM).*
- Lau, J. et al. (2018). *VQA-RAD: A Dataset of Clinically Generated Visual Questions and Answers about Radiology Images.*

---

<p align="center">
  Built for efficient, specialized multimodal AI research
</p>
