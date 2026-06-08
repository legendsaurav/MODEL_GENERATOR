#!/usr/bin/env python3
"""
MODEL_GENERATOR_V2 — Standalone 3D Mesh Generator
==================================================
Uses the OFFICIAL Hunyuan3D-2.1 pipeline (hy3dgen) for correct
weight loading and high-quality mesh generation.

This script is STANDALONE — it does NOT depend on any of the custom
MODEL_GENERATOR_V2 modules. Just run this file directly.

Usage:
    python run_generate.py --image input.png
    python run_generate.py --image input.png --steps 100 --resolution 512
    python run_generate.py --image input.png --preset ultra --output output.glb

Requirements:
    pip install hy3dgen trimesh torch

Author: MODEL_GENERATOR_V2 Project
"""

import argparse
import os
import sys
import time
import subprocess


# ═══════════════════════════════════════════════════════════════════════════
#  Dependency check
# ═══════════════════════════════════════════════════════════════════════════

def check_and_install_deps():
    """Check for required packages and offer to install them."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import hy3dgen
    except ImportError:
        missing.append("hy3dgen")

    try:
        import trimesh
    except ImportError:
        missing.append("trimesh")

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"Installing with pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            *missing, "-q"
        ])
        print("✓ Dependencies installed\n")


# ═══════════════════════════════════════════════════════════════════════════
#  Preset configurations
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "fast": {
        "steps": 25,
        "resolution": 256,
        "guidance_scale": 7.5,
        "target_faces": 50000,
    },
    "balanced": {
        "steps": 50,
        "resolution": 384,
        "guidance_scale": 7.5,
        "target_faces": 100000,
    },
    "ultra": {
        "steps": 100,
        "resolution": 512,
        "guidance_scale": 7.5,
        "target_faces": 200000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  Post-processing (trimesh only — no pymeshlab needed)
# ═══════════════════════════════════════════════════════════════════════════

def postprocess_mesh(mesh, target_faces=100000, smooth_iterations=3):
    """Clean and optimize the raw mesh using trimesh only.

    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target face count for decimation
        smooth_iterations: Number of Laplacian smoothing passes

    Returns:
        Cleaned trimesh.Trimesh
    """
    import trimesh
    import numpy as np

    print(f"\n  Post-processing...")
    original_v = len(mesh.vertices)
    original_f = len(mesh.faces)

    # Step 1: Remove degenerate faces
    try:
        mask = mesh.nondegenerate_faces()
        if mask is not None and not mask.all():
            removed = (~mask).sum()
            mesh.update_faces(mask)
            mesh.remove_unreferenced_vertices()
            print(f"    [1/6] Removed {removed} degenerate faces")
        else:
            print(f"    [1/6] No degenerate faces")
    except Exception as e:
        print(f"    [1/6] Skip degenerate check: {e}")

    # Step 2: Merge duplicate vertices
    try:
        pre_v = len(mesh.vertices)
        mesh.merge_vertices()
        merged = pre_v - len(mesh.vertices)
        if merged > 0:
            print(f"    [2/6] Merged {merged} duplicate vertices")
        else:
            print(f"    [2/6] No duplicate vertices")
    except Exception as e:
        print(f"    [2/6] Skip vertex merge: {e}")

    # Step 3: Remove small disconnected components
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
                removed_comp = len(components) - len(large)
                print(f"    [3/6] Removed {removed_comp} small components")
            else:
                print(f"    [3/6] All {len(components)} components are significant")
        else:
            print(f"    [3/6] Single connected component")
    except Exception as e:
        print(f"    [3/6] Skip component removal: {e}")

    # Step 4: Smooth
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(
                mesh, iterations=smooth_iterations
            )
            print(f"    [4/6] Laplacian smoothing: {smooth_iterations} iterations")
        except Exception as e:
            print(f"    [4/6] Skip smoothing: {e}")
    else:
        print(f"    [4/6] Smoothing disabled")

    # Step 5: Decimate if needed
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
            print(f"    [5/6] Decimated: {original_f} → {len(mesh.faces)} faces")
        except Exception as e:
            print(f"    [5/6] Skip decimation: {e}")
    else:
        print(f"    [5/6] Already below target ({len(mesh.faces)} ≤ {target_faces})")

    # Step 6: Fix normals
    try:
        mesh.fix_normals()
        print(f"    [6/6] Normals fixed")
    except Exception as e:
        print(f"    [6/6] Skip normal fix: {e}")

    print(f"  Post-processing complete: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
#  Main generation
# ═══════════════════════════════════════════════════════════════════════════

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
    print(f"  MODEL_GENERATOR_V2 — 3D Mesh Generation (Official Pipeline)")
    print(f"{'='*60}")
    print(f"  Image:      {args.image}")
    print(f"  Preset:     {args.preset.upper()}")
    print(f"  Steps:      {steps}")
    print(f"  Resolution: {resolution}")
    print(f"  Device:     {device}")
    print(f"  Model:      {args.model_path}")
    print(f"{'='*60}\n")

    # ── Load pipeline ──────────────────────────────────────────────────
    print("  Loading official Hunyuan3D pipeline...")
    t0 = time.perf_counter()

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
    )

    if device.startswith("cuda"):
        pipeline = pipeline.to(device)

    load_time = time.perf_counter() - t0
    print(f"  ✓ Pipeline loaded in {load_time:.1f}s")

    if device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  GPU: {gpu_name} ({mem:.1f} GB)")

    # ── Load image ─────────────────────────────────────────────────────
    image = Image.open(args.image)
    print(f"  Input image: {image.size[0]}×{image.size[1]}")

    # ── Generate ───────────────────────────────────────────────────────
    print(f"\n  Generating 3D mesh...")
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
    print(f"  ✓ Raw mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces ({gen_time:.1f}s)")

    # ── Post-process ───────────────────────────────────────────────────
    if not args.no_postprocess:
        mesh = postprocess_mesh(
            mesh,
            target_faces=target_faces,
            smooth_iterations=0 if args.no_smooth else 3,
        )

    # ── Export ─────────────────────────────────────────────────────────
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
        print(f"  ✓ Exported: {out_path}")

    total_time = time.perf_counter() - total_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="run_generate",
        description="Ultra-quality single-image-to-3D-mesh generation using official Hunyuan3D-2.1",
    )

    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", default="./outputs/output.glb", help="Output file path")
    parser.add_argument("--preset", "-p", choices=["fast", "balanced", "ultra"], default="balanced")
    parser.add_argument("--steps", "-s", type=int, default=None, help="Diffusion steps")
    parser.add_argument("--resolution", "-r", type=int, default=None, help="Octree resolution")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1", help="HuggingFace model ID")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, cuda:0, cpu")
    parser.add_argument("--format", "-f", nargs="+", default=None, help="Export formats: glb obj stl ply")
    parser.add_argument("--target-faces", type=int, default=None, help="Target face count")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing")
    parser.add_argument("--no-postprocess", action="store_true", help="Skip all post-processing")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    check_and_install_deps()
    generate(args)


if __name__ == "__main__":
    main()
