# CycleGAN for Image-to-Image Translation

This project implements a Cycle-Consistent Generative Adversarial Network (CycleGAN) for unpaired image-to-image translation tasks. Based on the original [CycleGAN paper](https://arxiv.org/abs/1703.10593), this implementation learns to translate images between two domains without paired training examples.

## Project Overview

The implementation includes:
- Custom dataset loader for unpaired image datasets
- Generator with residual blocks using instance normalization
- PatchGAN discriminator architecture
- Cycle-consistency loss for domain translation
- Adversarial loss with both MSE and BCE implementations
- Training pipeline with visualization of results

## Key Features

- **Flexible Architecture**: Supports different normalization layers and activation functions
- **Comprehensive Loss Functions**: Implements cycle-consistency loss and adversarial loss
- **Visualization Tools**: Includes functions to visualize training progress and generated images
- **Efficient Training**: Batch processing and GPU acceleration support
- **Model Checkpointing**: Automatic saving of model states during training

## Installation ⚙️

1. Clone repository:
```bash
git clone https://github.com/Ansarill/cycle_gans.git
cd cycle_gans
```
2. Create virtual env (**Python 3.10.12**):
```bash
python3.10 -m venv my_env
source my_env/bin/activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the following structure (or follow notebook to download one of datasets) and change variables in the notebook accordingly:
```
datasets/
└── img2img/
    └── your_dataset/
        ├── trainA/
        ├── trainB/
        ├── testA/
        └── testB/
```

2. Configure the training parameters in the Jupyter notebook

3. Run the training

## License 📄
This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

---
**Inspired by** The Great Impressionist Painter Vincent van Gogh  
**Powered by** Pure PyTorch and Enthusiasm