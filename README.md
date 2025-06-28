# PartPacker Cog Wrapper

This repository contains a Cog wrapper for the PartPacker 3D object generation model.

## 🚀 Try it on Replicate

**Model API**: [`ayushunleashed/partpacker`](https://replicate.com/ayushunleashed/partpacker)

## Overview

PartPacker performs efficient part-level 3D object generation from single-view images using dual volume packing. This Cog wrapper provides a convenient API for running the model on Replicate.

Original Paper: [PartPacker: Efficient Part-level 3D Object Generation via Dual Volume Packing](https://research.nvidia.com/labs/dir/partpacker/)

## Repository Structure

```
partpacker-cog/
├── cog.yaml                    # Cog configuration
├── predict.py                  # Cog prediction interface
├── download_weights.py         # Weight downloader script
├── PartPacker/                 # Git submodule (original repo)
│   ├── README.md
│   ├── requirements.txt
│   ├── app.py
│   ├── flow/
│   ├── vae/
│   └── ...
└── README.md                   # This file
```

## Original Repository

This wrapper is built for the [PartPacker](https://github.com/NVlabs/PartPacker) project. The original repository contains the core implementation which is included as a submodule here.

## Local Testing with Cog

### 1. Clone with Submodule

```bash
# Clone the repository
git clone https://github.com/your-username/partpacker-cog.git
cd partpacker-cog

# Initialize and update the submodule
git submodule update --init --recursive
```

### 2. Install Cog

Follow the [official Cog installation guide](https://cog.run/getting-started/):

```bash
# On macOS
brew install replicate/tap/cog

# On Linux/Windows WSL
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

### 3. Test Locally

```bash
# Download weights manually (optional)
python download_weights.py

# Build the Docker image
cog build

# Test with an image
cog predict -i image=@path/to/your/image.jpg
```

## Usage

### Input Requirements

- **Image**: JPEG, PNG formats supported
- Single-view image with clear object visibility
- Automatic background removal if no alpha channel present

### API Parameters

- `image` (required): Input image file
- `num_steps`: Number of inference steps (1-100, default: 50)
- `cfg_scale`: Classifier-free guidance scale (1-20, default: 7.0)
- `grid_resolution`: Grid resolution for mesh extraction (256-512, default: 384)
- `seed`: Random seed for reproducible results (optional)
- `simplify_mesh`: Whether to simplify the output mesh (default: False)
- `target_num_faces`: Target number of faces for simplification (10k-1M, default: 100k)

### Example Usage

#### Local Testing with Cog

```bash
# Basic usage
cog predict -i image=@input.jpg

# With custom parameters
cog predict \
    -i image=@input.jpg \
    -i num_steps=80 \
    -i cfg_scale=9.0 \
    -i grid_resolution=512 \
    -i seed=42 \
    -i simplify_mesh=true \
    -i target_num_faces=50000
```

#### Python API (Using Deployed Model)

```python
import replicate

output = replicate.run(
    "your-username/partpacker",
    input={
        "image": open("input.jpg", "rb"),
        "num_steps": 50,
        "cfg_scale": 7.0,
        "grid_resolution": 384,
        "seed": 42
    }
)

print(f"Output GLB file: {output}")
```

## Model Details

- **Architecture**: Diffusion Transformer (DiT) with Flow Matching
- **Input**: Single RGB image (518x518 processed)
- **Output**: GLB file with part-separated 3D mesh
- **Part Generation**: Dual volume packing for efficient part-level generation
- **Memory Requirements**: ~8-12GB GPU memory for typical usage

## Performance Tips

1. **Quality vs Speed**:
   - Lower `num_steps` (30-40) = faster generation
   - Higher `num_steps` (70-100) = better quality

2. **Memory Management**:
   - Lower `grid_resolution` (256-320) = less memory usage
   - Higher `grid_resolution` (448-512) = more detail

3. **Mesh Optimization**:
   - Enable `simplify_mesh` for smaller file sizes
   - Adjust `target_num_faces` based on your needs

## Output Format

The model outputs a GLB file containing:
- Multiple mesh parts with different colors
- Each part can be separated and manipulated individually
- Optimized for 3D printing and game engine import

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `grid_resolution` or use smaller input images
2. **Poor Quality**: Increase `num_steps` or `cfg_scale`
3. **Large File Size**: Enable `simplify_mesh` with lower `target_num_faces`

### Input Image Tips

- Use high-contrast objects with clear boundaries
- Avoid cluttered backgrounds (auto-removal works best with simple backgrounds)
- Center the object in the image
- Use good lighting conditions

## License

This Cog wrapper follows the same license as the original PartPacker project. See the [original repository](https://github.com/NVlabs/PartPacker) for license details.

## Citation

If you use this model, please cite the original PartPacker paper:

```bibtex
@article{tang2024partpacker,
  title={Efficient Part-level 3D Object Generation via Dual Volume Packing},
  author={Tang, Jiaxiang and Lu, Ruijie and Li, Zhaoshuo and Hao, Zekun and Li, Xuan and Wei, Fangyin and Song, Shuran and Zeng, Gang and Liu, Ming-Yu and Lin, Tsung-Yi},
  journal={arXiv preprint arXiv:2506.09980},
  year={2025}
}
```