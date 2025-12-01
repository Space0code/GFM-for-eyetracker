#!/bin/bash
# setup_env.sh - Minimal environment setup for GFM-for-eyetracker
# Optimized for NVIDIA RTX 4070 with CUDA 13.0 driver

# Create conda environment with Python 3.10
conda create -n gfm python=3.10 -y

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gfm

# Install PyTorch with CUDA 12.6 support (compatible with CUDA 13.0 driver)
# Using pip for better CUDA version control
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install PyTorch Geometric
pip install torch_geometric

# Install PyG optional dependencies for better performance (optional but recommended)
# Note: pyg_lib may not have prebuilt wheels for all configurations - it's optional
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.9.0+cu126.html

# Install scientific computing packages
conda install numpy pandas scipy scikit-learn -y

# Install visualization and utilities
conda install matplotlib pyyaml -y

# Install Jupyter for notebook support
conda install jupyter ipykernel -y

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch_geometric; print(f'PyTorch Geometric: {torch_geometric.__version__}')"

echo ""
echo "Environment 'gfm' setup complete!"
echo "Assumed Hardware Configuration:"
echo "  GPU: NVIDIA GeForce RTX 4070"
echo "  CUDA Driver: 13.0 | PyTorch CUDA: 12.6"