# MODEL_GENERATOR_V2

**Ultra-quality single-image-to-3D-mesh generation based on Tencent Hunyuan3D-2.1**

Geometry-only pipeline — generates smooth, clean, optimized 3D meshes from a single input image. All texture-related components have been removed to focus entirely on mesh quality.

---

## Architecture

```
Image → Background Removal → DINOv2 Conditioning → DiT Diffusion
→ ShapeVAE Decode → Marching Cubes → Post-Processing → Export
```

### Core Components
| Component | Description | Source |
|-----------|-------------|--------|
| **ImageConditioner** | DINOv2-Giant feature extraction at 518×518 | Adapted from Hunyuan3D |
| **Hunyuan3DDiT** | Dual-stream + single-stream flow-matching DiT | Adapted from Hunyuan3D |
| **ShapeVAE** | Latent → SDF → Marching Cubes mesh extraction | Adapted from Hunyuan3D |
| **PostProcessingPipeline** | 8-step mesh repair, smoothing, decimation | New |
| **Exporters** | GLB, OBJ, STL, PLY format export | New |

---

## Quick Start

### Installation
```bash
git clone <repo-url> MODEL_GENERATOR_V2
cd MODEL_GENERATOR_V2
pip install -r requirements.txt
pip install -e .
```

### CLI Usage
```bash
# Basic generation
python generate.py --image input.png --output output.glb

# With quality preset
python generate.py --image input.png --preset ultra --output output.glb

# Custom parameters
python generate.py --image input.png --steps 75 --resolution 448 \
    --format obj --output output.obj

# Multi-format export
python generate.py --image input.png --preset ultra \
    --format glb obj stl ply --output-dir ./outputs

# Full options
python generate.py \
    --image input.png \
    --preset ultra \
    --steps 100 \
    --resolution 512 \
    --format glb obj stl ply \
    --output-dir ./outputs \
    --seed 42 \
    --device cuda:0 \
    --fp16 \
    --target-faces 150000
```

### Python API
```python
from MODEL_GENERATOR_V2.generation import GeometryPipeline
from MODEL_GENERATOR_V2.postprocessing import PostProcessingPipeline
from MODEL_GENERATOR_V2.exporters import get_exporter
from MODEL_GENERATOR_V2.configs.presets import get_preset_config

# Load pipeline
config = get_preset_config('ultra')
pipeline = GeometryPipeline.from_pretrained(
    model_path='tencent/Hunyuan3D-2',
    preset='ultra',
)

# Generate raw mesh
mesh = pipeline('input.png')

# Post-process
postprocessor = PostProcessingPipeline(config.postprocessing)
mesh = postprocessor(mesh)

# Export to multiple formats
for fmt in ['glb', 'obj', 'stl', 'ply']:
    exporter = get_exporter(fmt)
    exporter.export(mesh, f'output.{fmt}')
```

---

## Quality Presets

| Preset | Steps | Resolution | Target Faces | Smoothing | Approx. Time | VRAM |
|--------|-------|------------|--------------|-----------|---------------|------|
| **FAST** | 25 | 256 | 50K | 3 iterations | ~15s | ~6GB |
| **BALANCED** | 50 | 384 | 100K | 5 iterations | ~45s | ~10GB |
| **ULTRA** | 100 | 512 | 200K | 10 iterations | ~120s | ~16GB |

---

## Post-Processing Pipeline

The 8-step post-processing chain significantly improves raw generation output:

1. **Mesh Repair** — Removes degenerate/duplicate faces, merges close vertices
2. **Floater Removal** — Removes small disconnected components
3. **Normal Fixing** — Reorients faces for consistent normals
4. **Hole Filling** — Fills small holes in the surface
5. **Taubin/HC Smoothing** — Volume-preserving surface smoothing
6. **Loop Subdivision** — Adaptive detail enhancement (BALANCED/ULTRA)
7. **Quadric Decimation** — Reduces to target face count preserving quality
8. **Validation** — Checks watertightness, manifoldness, quality metrics

---

## Project Structure

```
MODEL_GENERATOR_V2/
├── configs/           # Configuration dataclasses and quality presets
├── core/              # ML models (DiT, VAE, conditioner, scheduler)
│   └── vae/           # ShapeVAE, attention blocks, surface extractor
├── preprocessing/     # Background removal, image processing
├── generation/        # Pipeline orchestration, diffusion runner, model loading
├── postprocessing/    # Mesh repair, smoothing, subdivision, decimation, validation
├── exporters/         # GLB, OBJ, STL, PLY exporters
├── utils/             # Logging, timing, device management, memory
├── tests/             # Unit, integration, and inference tests
├── outputs/           # Generated mesh output directory
├── generate.py        # CLI entry point
├── requirements.txt   # Python dependencies
└── setup.py           # Package installation
```

---

## Dependencies

**Core ML**: torch, transformers, diffusers, accelerate, einops, safetensors  
**3D Processing**: trimesh, pymeshlab, scikit-image  
**Image Processing**: Pillow, opencv-python, rembg  
**Config**: omegaconf, pyyaml  

Removed from Hunyuan3D: xatlas, gradio, fastapi, Real-ESRGAN, custom rasterizer, differentiable renderer, Blender

---

## Testing

```bash
# Unit tests (no GPU required)
pytest tests/unit/ -v

# Integration tests (no GPU required)
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=MODEL_GENERATOR_V2

# Inference tests (requires GPU + model weights)
pytest tests/inference/ -v
```

---

## Model Weights

The system loads pretrained weights from HuggingFace:
- **v2.0**: `tencent/Hunyuan3D-2`
- **v2.1**: `tencent/Hunyuan3D-2.1`

Key weight subfolders:
- `hunyuan3d-dit-v2-0` — DiT model weights
- `hunyuan3d-vae-v2-0-withencoder` — ShapeVAE weights

---

## License

Based on Tencent Hunyuan3D-2, licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT.
