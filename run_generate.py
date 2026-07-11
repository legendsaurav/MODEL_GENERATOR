#!/usr/bin/env python3
"""
MODEL_GENERATOR_V2 — Production 3D Mesh Generator with Hidden State Capture
============================================================================
Uses the OFFICIAL Hunyuan3D-2.1 pipeline for correct weight loading and
high-quality mesh generation. Optionally captures DiT hidden states for
the Geometry Engine pipeline.

Usage:
    # Basic mesh generation
    python run_generate.py --image input.png

    # Ultra quality with hidden state capture (for AI CAD OS pipeline)
    python run_generate.py --image input.png --preset ultra --capture-states

    # Specify model path explicitly
    python run_generate.py --image input.png --model-path tencent/Hunyuan3D-2.1

Requirements:
    pip install hy3dgen trimesh torch

Author: AI CAD OS Project
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
#  Module aliasing for Hunyuan3D v2.0 ↔ v2.1 compatibility
# ---------------------------------------------------------------------------
# The downloaded v2.1 checkpoint configs reference 'hy3dshape' module paths
# but the installed hy3dgen package uses 'hy3dgen.shapegen'. We register
# aliases BEFORE any hy3dgen imports to ensure the model loads correctly.

import importlib

def _setup_module_aliases():
    """Register hy3dshape → hy3dgen.shapegen module aliases.

    The HuggingFace model configs reference the old ``hy3dshape`` package name.
    This function maps every referenced sub-module to its ``hy3dgen.shapegen``
    counterpart so ``instantiate_from_config`` can resolve class targets.

    IMPORTANT: The v2.1 weights use ``HunYuanDiTPlain`` from ``hunyuandit.py``
    (U-Net + MoE architecture), NOT ``Hunyuan3DDiT`` from ``hunyuan3ddit.py``
    (Flux-style architecture).  These are two completely different models.
    """
    try:
        import hy3dgen.shapegen
        import hy3dgen.shapegen.models
        import hy3dgen.shapegen.models.denoisers
        import hy3dgen.shapegen.models.denoisers.hunyuandit
        import hy3dgen.shapegen.models.autoencoders
        import hy3dgen.shapegen.schedulers
        import hy3dgen.shapegen.preprocessors

        sys.modules['hy3dshape'] = hy3dgen.shapegen
        sys.modules['hy3dshape.models'] = hy3dgen.shapegen.models
        sys.modules['hy3dshape.models.denoisers'] = hy3dgen.shapegen.models.denoisers
        sys.modules['hy3dshape.models.denoisers.hunyuandit'] = (
            hy3dgen.shapegen.models.denoisers.hunyuandit
        )
        sys.modules['hy3dshape.models.autoencoders'] = hy3dgen.shapegen.models.autoencoders
        sys.modules['hy3dshape.schedulers'] = hy3dgen.shapegen.schedulers
        sys.modules['hy3dshape.preprocessors'] = hy3dgen.shapegen.preprocessors
        sys.modules['hy3dshape.pipelines'] = hy3dgen.shapegen.pipelines
    except ImportError:
        pass  # hy3dgen not installed; dependency check will catch this later

_setup_module_aliases()


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_generator_v2")


# ---------------------------------------------------------------------------
#  Preset configurations
# ---------------------------------------------------------------------------

PRESETS = {
    "fast": {
        "steps": 25,
        "resolution": 256,
        "guidance_scale": 7.5,
        "target_faces": 50_000,
    },
    "balanced": {
        "steps": 50,
        "resolution": 384,
        "guidance_scale": 7.5,
        "target_faces": 100_000,
    },
    "ultra": {
        "steps": 100,
        "resolution": 512,
        "guidance_scale": 7.5,
        "target_faces": 200_000,
    },
}


# ---------------------------------------------------------------------------
#  Post-processing (trimesh only)
# ---------------------------------------------------------------------------

def postprocess_mesh(mesh, target_faces=100_000, smooth_iterations=3):
    """Clean and optimize the raw mesh using trimesh.

    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target face count for decimation
        smooth_iterations: Number of Laplacian smoothing passes

    Returns:
        Cleaned trimesh.Trimesh
    """
    import trimesh
    import numpy as np

    logger.info("Post-processing mesh (%d verts, %d faces)...",
                len(mesh.vertices), len(mesh.faces))

    # Remove degenerate faces
    try:
        mask = mesh.nondegenerate_faces()
        if mask is not None and not mask.all():
            removed = (~mask).sum()
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
            logger.info("  [1/5] Removed %d degenerate faces", removed)
        else:
            logger.info("  [1/5] No degenerate faces")
    except Exception as e:
        logger.warning("  [1/5] Skip degenerate check: %s", e)

    # Merge duplicate vertices
    try:
        pre_v = len(mesh.vertices)
        mesh.merge_vertices()
        merged = pre_v - len(mesh.vertices)
        if merged > 0:
            logger.info("  [2/5] Merged %d duplicate vertices", merged)
        else:
            logger.info("  [2/5] No duplicate vertices")
    except Exception as e:
        logger.warning("  [2/5] Skip vertex merge: %s", e)

    # Remove small disconnected components
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = len(mesh.faces)
            min_faces = max(100, int(total_faces * 0.005))
            large = [c for c in components if len(c.faces) >= min_faces]
            if not large:
                large = [max(components, key=lambda c: len(c.faces))]
            if len(large) < len(components):
                mesh = trimesh.util.concatenate(large)
                logger.info("  [3/5] Removed %d small components",
                            len(components) - len(large))
            else:
                logger.info("  [3/5] All %d components significant", len(components))
        else:
            logger.info("  [3/5] Single connected component")
    except Exception as e:
        logger.warning("  [3/5] Skip component removal: %s", e)

    # Smooth
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iterations)
            logger.info("  [4/5] Laplacian smoothing: %d iterations", smooth_iterations)
        except Exception as e:
            logger.warning("  [4/5] Skip smoothing: %s", e)
    else:
        logger.info("  [4/5] Smoothing disabled")

    # Fix normals
    try:
        mesh.fix_normals()
        logger.info("  [5/5] Normals fixed")
    except Exception as e:
        logger.warning("  [5/5] Skip normal fix: %s", e)

    logger.info("Post-processing complete: %d vertices, %d faces",
                len(mesh.vertices), len(mesh.faces))
    return mesh


# ---------------------------------------------------------------------------
#  Hidden State Bridge integration
# ---------------------------------------------------------------------------

def setup_bridge(pipeline, capture_timesteps):
    """Attach HiddenStateBridge to the pipeline's transformer model.

    Args:
        pipeline: The Hunyuan3DDiTFlowMatchingPipeline instance.
        capture_timesteps: List of timestep values to capture.

    Returns:
        HiddenStateBridge instance, or None if setup fails.
    """
    try:
        # Add MODEL_GENERATOR_V2 to path for bridge import
        project_root = Path(__file__).parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from core.hidden_state_bridge import HiddenStateBridge

        bridge = HiddenStateBridge()

        # Find the transformer model inside the pipeline
        transformer = None
        for attr_name in ['model', 'transformer', 'dit', 'denoiser']:
            candidate = getattr(pipeline, attr_name, None)
            if candidate is not None and hasattr(candidate, 'parameters'):
                transformer = candidate
                logger.info("Found transformer model at pipeline.%s", attr_name)
                break

        if transformer is None:
            logger.warning("Could not find transformer model in pipeline. "
                           "Hidden state capture disabled.")
            return None

        bridge.register_hooks(transformer)
        bridge.set_capture_timesteps(capture_timesteps)
        logger.info("HiddenStateBridge attached with %d capture timesteps",
                     len(capture_timesteps))
        return bridge

    except Exception as e:
        logger.warning("Failed to set up HiddenStateBridge: %s", e)
        return None


def save_captured_states(bridge, output_dir: str, image_stem: str):
    """Save captured hidden states to disk for offline analysis.

    Saves:
        - {image_stem}_states.pt — PyTorch tensor dict
        - {image_stem}_states_meta.json — metadata (shapes, timesteps)
    """
    import torch

    states = bridge.get_captured_states()
    if not states:
        logger.warning("No hidden states captured — nothing to save")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save as .pt file
    pt_path = os.path.join(output_dir, f"{image_stem}_states.pt")
    torch.save(states, pt_path)
    logger.info("Saved hidden states: %s", pt_path)

    # Save metadata
    meta = {
        "timesteps": list(states.keys()),
        "layers_per_timestep": {},
        "total_tensors": 0,
    }
    for ts_key, layer_dict in states.items():
        layer_info = {}
        for layer_name, tensor in layer_dict.items():
            layer_info[layer_name] = list(tensor.shape)
            meta["total_tensors"] += 1
        meta["layers_per_timestep"][ts_key] = layer_info

    meta_path = os.path.join(output_dir, f"{image_stem}_states_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved state metadata: %s (%d timesteps, %d total tensors)",
                meta_path, len(states), meta["total_tensors"])


# ---------------------------------------------------------------------------
#  Main generation
# ---------------------------------------------------------------------------

def generate(args):
    """Run the full generation pipeline."""
    import torch
    import trimesh
    from PIL import Image

    total_start = time.perf_counter()

    # Resolve preset
    preset = PRESETS.get(args.preset, PRESETS["balanced"])
    steps = args.steps or preset["steps"]
    resolution = args.resolution or preset["resolution"]
    guidance = args.guidance_scale or preset["guidance_scale"]
    target_faces = args.target_faces or preset["target_faces"]

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print(f"  MODEL_GENERATOR_V2 — 3D Mesh Generation")
    print(f"{'='*60}")
    print(f"  Image:      {args.image}")
    print(f"  Preset:     {args.preset.upper()}")
    print(f"  Steps:      {steps}")
    print(f"  Resolution: {resolution}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_path}")
    print(f"  Capture:    {'YES' if args.capture_states else 'NO'}")
    print(f"{'='*60}\n")

    # ── Load pipeline ─────────────────────────────────────────────
    logger.info("Loading Hunyuan3D-2.1 pipeline...")
    t0 = time.perf_counter()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
    )

    # Move to GPU — v2.1 .to() may return None, so don't reassign
    if device != "cpu":
        pipeline.to(device)

    load_time = time.perf_counter() - t0
    logger.info("Pipeline loaded in %.1fs", load_time)

    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB)", gpu_name, mem)

    # ── Set up Hidden State Bridge (optional) ─────────────────────
    bridge = None
    if args.capture_states:
        capture_ts = [0.0, 0.25, 0.5, 0.75, 1.0]
        bridge = setup_bridge(pipeline, capture_ts)

    # ── Load image ────────────────────────────────────────────────
    image = Image.open(args.image)
    logger.info("Input image: %d×%d", image.size[0], image.size[1])

    # ── Generate ──────────────────────────────────────────────────
    logger.info("Generating 3D mesh (%d steps, %d resolution)...", steps, resolution)
    t0 = time.perf_counter()

    # Set seed
    generator = None
    if args.seed is not None:
        gen_device = device if device.startswith("cuda") else "cpu"
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(args.seed)

    # Call official pipeline
    result = pipeline(
        image=image,
        num_inference_steps=steps,
        octree_resolution=resolution,
        guidance_scale=guidance,
        generator=generator,
        output_type="trimesh",
    )

    # Extract mesh from result
    if isinstance(result, list):
        mesh = result[0]
    elif isinstance(result, trimesh.Trimesh):
        mesh = result
    elif hasattr(result, 'meshes'):
        mesh = result.meshes[0]
    else:
        mesh = result

    gen_time = time.perf_counter() - t0
    logger.info("Raw mesh: %d vertices, %d faces (%.1fs)",
                len(mesh.vertices), len(mesh.faces), gen_time)

    # ── Save hidden states ────────────────────────────────────────
    if bridge is not None:
        image_stem = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.dirname(args.output) or "./outputs"
        save_captured_states(bridge, output_dir, image_stem)
        bridge.clear()

    # ── Post-process ──────────────────────────────────────────────
    if not args.no_postprocess:
        mesh = postprocess_mesh(
            mesh,
            target_faces=target_faces,
            smooth_iterations=0 if args.no_smooth else 3,
        )

    # ── Export ─────────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output) or "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_stem = os.path.splitext(os.path.basename(args.image))[0]

    formats = args.format or ["glb", "obj"]
    for fmt in formats:
        if len(formats) == 1 and args.output:
            out_path = args.output
        else:
            out_path = os.path.join(output_dir, f"{image_stem}.{fmt}")

        mesh.export(out_path)
        logger.info("Exported: %s", out_path)

    total_time = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  Generation complete!")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Vertices:    {len(mesh.vertices):,}")
    print(f"  Faces:       {len(mesh.faces):,}")
    print(f"  Output:      {output_dir}/")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="run_generate",
        description="3D mesh generation from a single image using Hunyuan3D-2.1",
    )

    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default="./outputs/output.glb",
                        help="Output file path")
    parser.add_argument("--preset", "-p", choices=["fast", "balanced", "ultra"],
                        default="balanced")
    parser.add_argument("--steps", "-s", type=int, default=None,
                        help="Diffusion steps")
    parser.add_argument("--resolution", "-r", type=int, default=None,
                        help="Octree resolution")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cuda:0, cpu")
    parser.add_argument("--format", "-f", nargs="+", default=None,
                        help="Export formats: glb obj stl ply")
    parser.add_argument("--target-faces", type=int, default=None,
                        help="Target face count")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable smoothing")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Skip all post-processing")
    parser.add_argument("--capture-states", action="store_true",
                        help="Capture DiT hidden states for Geometry Engine")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    generate(args)


if __name__ == "__main__":
    main()
