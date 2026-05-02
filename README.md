# COMP8221 Assignment 1: DDIM Diffusion Model

**Unit:** COMP8221 Advanced Machine Learning  
**Assignment:** Assignment 1  
**Option Selected:** Option 3 – Diffusion Models  
**Model:** Denoising Diffusion Implicit Model (DDIM)  
**Framework:** PyTorch  

---

## Project Overview

This project implements a DDIM-based diffusion model for image generation as part of COMP8221 Assignment 1.

The objective is to build a complete diffusion pipeline from scratch, including:
- Forward diffusion (noise addition)
- Reverse sampling (DDIM)
- Noise prediction network (U-Net with time embeddings)
- Training and evaluation pipeline

All core components are implemented using standard `torch.nn` layers.  
No high-level generative model libraries (e.g., HuggingFace `diffusers`, `timm`) are used.

External libraries are used only for:
- Dataset handling (CIFAR-10)
- Visualization and plotting
- Quantitative evaluation (FID)

---

## Key Features

- DDIM implementation (non-Markovian sampling)
- U-Net architecture with sinusoidal time embeddings
- Noise prediction using MSE loss
- CIFAR-10 dataset with preprocessing
- Loss curve visualization
- Generated image samples from noise
- Reverse diffusion visualization
- FID score evaluation

---

## Folder Structure

```text
COMP8221_Assignment1/
|-- notebook/
|   `-- 2026S1_COMP8221_Assignment1_MD_IFTHEKHER_UDDIN_CHY.ipynb
|-- outputs/
|   |-- real_samples_grid.png
|   |-- forward_noising_grid.png
|   |-- loss_curve.png
|   |-- generated_samples_grid.png
|   |-- reverse_diffusion_grid.png
|   |-- fid_score.txt
|   `-- experiment_summary.csv
|-- requirements.txt
|-- README.md
`-- references.md