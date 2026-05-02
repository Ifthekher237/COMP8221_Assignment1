# COMP8221 Assignment 1: DDIM Diffusion Model

Unit: COMP8221 Advanced Machine Learning

Assignment: Assignment 1

Selected option: Option 3 Diffusion Models

Selected model: DDIM

Framework: PyTorch

## Project Overview

This project is prepared for implementing a Denoising Diffusion Implicit Model (DDIM) for COMP8221 Assignment 1.

The core model will be implemented from scratch using `torch.nn` layers. No high-level generative model libraries such as HuggingFace `diffusers`, `timm`, or ready-made diffusion/model packages will be used.

External libraries are reserved for appropriate support tasks such as dataset loading, plotting, augmentation, and evaluation metrics.

## Folder Structure

```text
COMP8221_Assignment1/
|-- notebook/
|   `-- 2026S1_COMP8221_Assignment1_STUDENTID_NAME.ipynb
|-- report/
|   `-- .gitkeep
|-- src/
|   |-- __init__.py
|   |-- dataset.py
|   |-- model.py
|   |-- diffusion.py
|   |-- train.py
|   |-- evaluate.py
|   `-- visualize.py
|-- outputs/
|   `-- .gitkeep
|-- checkpoints/
|   `-- .gitkeep
|-- data/
|   `-- README.md
|-- requirements.txt
|-- README.md
`-- references.md
```

## Setup Instructions

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Open the notebook:

```bash
jupyter notebook notebook/2026S1_COMP8221_Assignment1_STUDENTID_NAME.ipynb
```

3. Run cells from top to bottom.

## Expected Future Outputs

The following files are expected to be generated later during implementation and experimentation:

- `real_samples_grid.png`
- `forward_noising_grid.png`
- `loss_curve.png`
- `generated_samples_grid.png`
- `reverse_diffusion_grid.png`
- `fid_score.txt`
- `experiment_summary.csv`
