# CS521 Ithemal Setup Guide

## Ithemal Data Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│              Data Collection Stage (Optional)                │
│  Real Hardware → DynamoRIO → Measure Timing → MySQL DB      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Preparation Stage                     │
│        MySQL DB → save_data.py → .data files                │
│     (Contains: code_id, timing, code_intel, code_xml)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Stage                      │
│   .data files → run_ithemal.py → Trained Model              │
│    (input)      (training script)   (predictor.dump + .mdl) │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Model Prediction Stage                     │
│        New Code → predict.py → Predicted Timing             │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages Explained

1. **Data Collection (Optional)**: Use DynamoRIO to instrument real binaries running on target hardware, collect execution timing data, store in MySQL
2. **Data Preparation**: Export MySQL data to `.data` format using `save_data.py` for training
3. **Model Training**: Train the graph neural network model using `run_ithemal.py` on `.data` files
4. **Model Prediction**: Use trained model to predict timing for new assembly code sequences

**For BHive Testing**: You can skip Stage 1 and use pre-collected `.data` files directly (e.g., `haswell_sample1000.data`)

## Project Overview

This is a modernized version of [Ithemal](https://github.com/ithemal/Ithemal) for CS521 Fall 2025 final project, focusing on BHive training with updated dependencies:

- **Python**: 3.11 (upgraded from Python 2.7)
- **PyTorch**: 2.8.0 (upgraded from 1.x)
- **Docker**: Containerized environment with MySQL 5.7

## Completed Work Summary

### 1. Python 3.11 Migration ✅

**Files Modified**:
- `common/common_libs`
  - Fixed: Python2 printing error
  - Type annotation
- `learning/pytorch/ithemal/run_ithemal.py`
  - Fixed: `import Queue` → `import queue`
  
- `learning/pytorch/experiments/experiment.py`
  - Fixed: `import urlparse` → `from urllib import parse as urlparse`
  
- `learning/pytorch/models/model_utils.py`
  - Fixed: PyTorch 2.x storage API
  - Changed: `.storage()._share_filename_()` → `.untyped_storage()._share_filename_cpu_()`
  

### 2. Docker Environment Setup ✅

**Files Modified**:
- `docker/Dockerfile`
  - Base image: `debian:stretch` → `debian:bullseye`
  - Python: `python-dev` → `python3-dev`
  - Qt: `qt5-default` → `qtbase5-dev`
  - MySQL: Server removed (using docker-compose service)
  - Conda: Miniconda2 → Miniforge3 (avoid Anaconda ToS)
  - User: Fixed UID to 1000 (removed system user flag)
  - DynamoRIO: Added execute permissions
  - Jupyter: Direct configuration (removed patch file)

- `docker/docker-compose.yml`
  - MySQL 5.7 service configured
  - Environment variables set (ITHEMAL_HOME, MYSQL_HOST)

### 3. Test Infrastructure ✅

**New Test Files Created**:
- `testing/test_environment.py` - Verify Python, PyTorch, MySQL setup
- `testing/test_data_loading.py` - Test Ithemal data format loading
- `testing/test_model_components.py` - Test model module imports
- `testing/test_training_pipeline.py` - End-to-end training test

**Files Modified**:
- `testing/conftest.py`
  - Added automatic ITHEMAL_HOME detection
  - Fixed paths to use absolute references
  - Added DYNAMORIO_HOME default for skipping tests
  - Fixed regex patterns with raw strings

### 4. Data Pipeline ✅

**Training Data Format Verified**:
- Downloaded and tested `haswell_sample1000.data` (1000 samples)
- Format: `[(code_id, timing, code_intel, code_xml), ...]`
- Successfully loaded with PyTorch 2.8.0

---

## Docker Environment Setup

### Prerequisites

- Docker Engine (20.10+)
- Docker Compose (v2.x or standalone)

### Initial Setup (One-time)

#### 1. Clone Repository

```bash
git clone https://github.com/Aldo-Zhang/CS521_Ithemal.git
cd CS521_Ithemal
```

#### 2. Build Docker Image

```bash
cd docker
sudo ./docker_build.sh
```

**Build time**: ~10-15 minutes (downloads and installs all dependencies)

**Expected output**: 
```
✓ Debian packages installed
✓ DynamoRIO downloaded and configured
✓ Miniforge3 installed
✓ Conda environment created (Python 3.11)
✓ PyTorch 2.8.0 installed
✓ User 'ithemal' created
✓ Image built: ithemal:latest
```

#### 3. Start Container and Connect

```bash
sudo ./docker_connect.sh
```

This will:
1. Start MySQL and Ithemal containers
2. Run `build_all.sh` (builds DynamoRIO components, installs common_libs)
3. Drop you into a tmux session inside the container

**Container Info**:
- Working directory: `/home/ithemal/ithemal`
- User: `ithemal` (UID 1000)
- Python environment: `conda activate ithemal` (auto-activated)
- MySQL: accessible at `db:3306`

### Stopping and Restarting

#### Stop Container

```bash
# From host machine
cd docker
sudo docker-compose down
```

#### Restart Container

```bash
sudo ./docker_connect.sh
```

**Note**: Docker-compose will restart existing containers if they exist.

#### Detach from tmux without stopping

```
Control-b d
```

To reconnect:
```bash
sudo ./docker_connect.sh
```

---

## Running Tests

### Test Hierarchy

```
Level 1: Environment Tests (Quick, ~5s)
  ├── test_environment.py - Python, PyTorch, MySQL connectivity
  └── test_data_loading.py - Data format verification

Level 2: Component Tests (Medium, ~10s)
  └── test_model_components.py - Module import checks

Level 3: Integration Tests (Slow, ~15s+)
  └── test_training_pipeline.py - End-to-end training
  └── test_predict_pipeline.py - End-to-end predict
```

### Inside Container

#### Run All Tests

```bash
cd /home/ithemal/ithemal
pytest testing/ -v
```

#### Run Specific Test Suites

```bash
# Quick environment check
pytest testing/test_environment.py -v

# Data loading test
pytest testing/test_data_loading.py -v

# Training pipeline (slow)
pytest testing/test_training_pipeline.py::test_mini_training -v -s
```
---


## Next Steps for BHive Training

### 1. Obtain BHive Dataset

Download or prepare BHive training data in Ithemal format:
```python
# Expected format:
[(code_id, timing, assembly, xml_tokens), ...]
```

### 2. Convert to .data Format

```python
import torch

# Load your BHive data
bhive_data = [...]  # Your data in the format above

# Save as .data file
torch.save(bhive_data, 'bhive_training.data')
```

### 3. Run Full Training

```bash
python learning/pytorch/ithemal/run_ithemal.py \
  --data bhive_training.data \
  --use-rnn \
  train \
  --experiment-name bhive_final \
  --experiment-time $(date +%s) \
  --sgd \
  --threads 4 \
  --trainers 6 \
  --weird-lr \
  --decay-lr \
  --epochs 100
```

### 4. Evaluate Results

Results will be saved in:
```
learning/pytorch/saved/bhive_final/[timestamp]/
├── trained.mdl              # Model weights
├── predictor.dump           # Predictor for inference
├── loss_report.log          # Training loss history
└── validation_results.txt   # Test set performance
```

---

## Additional Resources

- **Original Ithemal Paper**: [ICML 2019](https://arxiv.org/abs/1808.07412)
- **Original Repository**: [github.com/ithemal/Ithemal](https://github.com/ithemal/Ithemal)
- **Ithemal Models**: [github.com/ithemal/Ithemal-models](https://github.com/ithemal/Ithemal-models)
- **PyTorch Documentation**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)

---
