# Checkpoint Storage Guide

Trained model checkpoints for DroughtCast are versioned and stored using Zenodo open-access data repositories and HuggingFace Hub rather than tracked directly in the git tree to keep the repository lightweight.

## Getting Checkpoints

### Option 1: Train from scratch (Recommended for full reproducibility)
```bash
python train.py --config configs/config.yaml --seed 42
```
This will train the model on the NASA POWER + NOAA observational panel and save the best checkpoint to:
`checkpoints/best_research_model.pt`

### Option 2: Pre-trained Model Checkpoint
To obtain pre-trained weights for the multi-seed benchmark:
- DOI: `10.5281/zenodo.droughtcast`
- Release tag: `v1.0-research-weights`
Place the downloaded `.pt` files directly into the `checkpoints/` directory:
```bash
# Example structure:
checkpoints/
└── best_research_model.pt
```
