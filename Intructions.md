# Oracle V4 Installation Guide

## Why GPU Acceleration Matters

Oracle V4 achieves **~6x performance improvement** with GPU acceleration compared to CPU-only execution. What would take **days** on CPU completes in **hours** on GPU. It's fundamental to Oracle's design philosophy.

**Performance Comparison:**
- CPU-only: ~48+ hours for full hurricane simulation
- GPU-accelerated: ~8-10 hours for same simulation
- Real-time forecasting: **Only possible with GPU**

## System Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with compute capability 6.0 or higher
- **VRAM:** 8GB+ recommended (16GB+ for extended simulations)
- **RAM:** 16GB system memory
- **Storage:** 50GB+ for code, data, and outputs
- **OS:** Windows 10/11, Linux, or macOS (with compatible GPU)

### Recommended Setup
- **GPU:** NVIDIA RTX 3060+ or better
- **VRAM:** 16GB+
- **RAM:** 32GB+
- **Storage:** SSD with 100GB+ free space

### Check GPU Compatibility
To verify your GPU supports CUDA:
```bash
nvidia-smi
```

Look for compute capability 6.0+ in the output.

---

## Installation

###  Recommended: Conda (GPU-Accelerated) 

**This is the ONLY recommended installation method.** Conda automatically handles all CUDA dependencies, ensuring proper GPU integration.

#### Step 1: Install Miniconda/Anaconda
If you don't have conda installed:
- Download Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Or Anaconda: https://www.anaconda.com/download

#### Step 2: Create Oracle Environment
```bash
# Clone the repository
git clone https://github.com/justingwatford-dev/Woe-Solver-Suite/new/main
cd oracle-v4

# Create conda environment (this may take 10-15 minutes)
conda env create -f environment.yml

# Activate the environment
conda activate oracle_env
```

#### Step 3: Verify GPU Installation
```python
import cupy as cp
print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"GPU device: {cp.cuda.Device(0).compute_capability}")
```

If you see CUDA available and your GPU info, you're ready to go! 

---

###  Alternative: pip (NOT Recommended)

**Warning:** This approach requires manual CUDA toolkit installation and is error-prone. Performance may be degraded without proper GPU setup.

**Only use pip if:**
- You're an experienced CUDA developer
- You have CUDA toolkit 12.x/13.x already installed
- You understand CUDA environment configuration

**Manual steps required:**
1. Install NVIDIA CUDA Toolkit (12.x or 13.x)
2. Install cuDNN libraries
3. Configure environment variables (PATH, LD_LIBRARY_PATH)
4. Install from requirements.txt: `pip install -r requirements.txt`

**We cannot provide support for pip installations.** Use conda.

---

## Configuration

### ECMWF API Access (Required for Data)
Oracle V4 requires ECMWF API access for ERA5 and GFS data:

1. Create free account: https://cds.climate.copernicus.eu/
2. Get API credentials
3. Create `~/.cdsapirc` file:
```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### First Run
```bash
# Activate environment
conda activate oracle_env

# Run test simulation
python oracle_test.py
```

---

## Troubleshooting

### "CUDA out of memory" errors
- Reduce domain size in configuration
- Close other GPU-intensive applications
- Consider upgrading GPU VRAM

### Slow performance despite GPU
- Verify CuPy is using GPU: Check with `nvidia-smi` during run
- Ensure conda environment is activated
- Check CUDA version matches GPU drivers

### Installation errors
- Update conda: `conda update conda`
- Clear conda cache: `conda clean --all`
- Try creating environment with: `conda env create -f environment.yml --force`

---

## Performance Tips

1. **Use GPU for everything:** Oracle is designed GPU-first
2. **Batch simulations:** Run multiple forecast members in parallel
3. **SSD storage:** Faster I/O for data loading
4. **Monitor VRAM:** Keep VRAM usage below 90% for stability

---

## Updating Oracle

```bash
# Pull latest changes
git pull origin main

# Update conda environment
conda env update -f environment.yml --prune

# Reactivate environment
conda activate oracle_env
```

---

## Questions?

- **Documentation:** See `/docs` folder
- **Issues:** Open GitHub issue
- **Performance:** Check GPU usage with `nvidia-smi -l 1`

**Remember:** Oracle V4 is built for GPU acceleration. The conda installation ensures you get the full performance benefit that makes real-time hurricane forecasting possible.
