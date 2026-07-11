#!/usr/bin/env python3
"""
MODEL_GENERATOR_V2 — CLI Entry Point

Generates ultra-quality 3D meshes from a single input image.

Usage:
    python generate.py --image input.png --output output.glb
    python generate.py --image input.png --preset ultra --output output.glb
    python generate.py --image input.png --steps 75 --resolution 448 \
        --format obj --output output.obj
    python generate.py --image input.png --preset ultra \
        --format glb obj stl ply --output-dir ./outputs

Example with all options:
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
        --target-faces 150000 \
        --no-smooth \
        --no-bg-removal
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="MODEL_GENERATOR_V2",
        description=(
            "Ultra-quality single-image-to-3D-mesh generation "
            "based on Hunyuan3D-2.1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the input image file.",
    )

    # Output
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (e.g., output.glb).",
    )
    output_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (filename auto-generated).",
    )

    # Quality preset
    parser.add_argument(
        "--preset", "-p",
        type=str,
        choices=["fast", "balanced", "ultra"],
        default="balanced",
        help="Quality preset (default: balanced).",
    )

    # Generation overrides
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=None,
        help="Number of diffusion steps (overrides preset).",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int,
        default=None,
        help="Octree resolution (overrides preset).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="tencent/Hunyuan3D-2",
        help="HuggingFace model ID or local path.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto, cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (default: True).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Use FP32 precision.",
    )

    # Export formats
    parser.add_argument(
        "--format", "-f",
        type=str,
        nargs="+",
        default=None,
        help="Export formats: glb, obj, stl, ply (default: from preset).",
    )

    # Post-processing overrides
    parser.add_argument(
        "--target-faces",
        type=int,
        default=None,
        help="Target face count after decimation.",
    )
    parser.add_argument(
        "--smoothing-iterations",
        type=int,
        default=None,
        help="Number of smoothing passes.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable mesh smoothing.",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable mesh repair.",
    )
    parser.add_argument(
        "--no-decimate",
        action="store_true",
        help="Disable mesh decimation.",
    )
    parser.add_argument(
        "--no-bg-removal",
        action="store_true",
        help="Skip background removal.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Skip all post-processing.",
    )

    # Misc
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offloading (reduces VRAM).",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for CLI mesh generation."""
    args = parse_args()

    # Validate input
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Import after arg parsing for faster --help
    from MODEL_GENERATOR_V2.configs.presets import get_preset_config
    from MODEL_GENERATOR_V2.generation.pipeline import GeometryPipeline
    from MODEL_GENERATOR_V2.postprocessing.pipeline import PostProcessingPipeline
    from MODEL_GENERATOR_V2.exporters import get_exporter
    from MODEL_GENERATOR_V2.utils.logging import setup_logging, get_logger

    import logging
    setup_logging(logging.WARNING if args.quiet else logging.INFO)
    logger = get_logger("model_generator_v2.cli")

    total_start = time.perf_counter()

    # ── Build config ──────────────────────────────────────────────
    config = get_preset_config(args.preset)

    # Apply overrides
    if args.steps is not None:
        config.generation.num_inference_steps = args.steps
    if args.resolution is not None:
        config.generation.octree_resolution = args.resolution
    if args.guidance_scale is not None:
        config.generation.guidance_scale = args.guidance_scale
    if args.seed is not None:
        config.generation.seed = args.seed
    if args.format is not None:
        config.export.formats = args.format
    if args.target_faces is not None:
        config.postprocessing.target_faces = args.target_faces
    if args.smoothing_iterations is not None:
        config.postprocessing.smoothing_iterations = args.smoothing_iterations
    if args.no_smooth:
        config.postprocessing.enable_smoothing = False
    if args.no_repair:
        config.postprocessing.enable_repair = False
    if args.no_decimate:
        config.postprocessing.enable_decimation = False

    dtype_str = "float32" if args.fp32 else "float16"

    config.validate()

    print(f"\n{'='*50}")
    print("  MODEL_GENERATOR_V2 — 3D Mesh Generation")
    print(f"{'='*50}")
    print(f"  Image:      {args.image}")
    print(f"  Preset:     {args.preset.upper()}")
    print(f"  Steps:      {config.generation.num_inference_steps}")
    print(f"  Resolution: {config.generation.octree_resolution}")
    print(f"  Device:     {args.device}")
    print(f"  Precision:  {dtype_str.upper()}")
    print(f"  Formats:    {', '.join(config.export.formats)}")
    print(f"{'='*50}\n")

    # ── Initialize pipeline ───────────────────────────────────────
    logger.info("Initializing pipeline...")
    pipeline = GeometryPipeline.from_pretrained(
        model_path=args.model_path,
        preset=args.preset,
        config=config,
        device=args.device,
        dtype_str=dtype_str,
        enable_cpu_offload=args.cpu_offload,
    )

    # ── Generate mesh ─────────────────────────────────────────────
    logger.info("Generating mesh...")
    raw_mesh = pipeline(
        image=args.image,
        num_inference_steps=config.generation.num_inference_steps,
        octree_resolution=config.generation.octree_resolution,
        seed=config.generation.seed,
        remove_background=not args.no_bg_removal,
        show_progress=not args.quiet,
    )

    if raw_mesh is None:
        print("Error: Mesh generation failed.", file=sys.stderr)
        sys.exit(1)

    print(
        f"\n  Raw mesh: {len(raw_mesh.vertices):,} vertices, "
        f"{len(raw_mesh.faces):,} faces"
    )

    # ── Post-process ──────────────────────────────────────────────
    if not args.no_postprocess:
        logger.info("Post-processing mesh...")
        postprocessor = PostProcessingPipeline(config.postprocessing)
        mesh = postprocessor(raw_mesh, verbose=not args.quiet)
    else:
        mesh = raw_mesh

    print(
        f"  Final mesh: {len(mesh.vertices):,} vertices, "
        f"{len(mesh.faces):,} faces"
    )

    # ── Export ─────────────────────────────────────────────────────
    output_dir = args.output_dir or os.path.dirname(args.output or "./outputs/mesh")
    if not output_dir:
        output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    image_stem = Path(args.image).stem

    for fmt in config.export.formats:
        if args.output and len(config.export.formats) == 1:
            out_path = args.output
        else:
            out_path = os.path.join(output_dir, f"{image_stem}.{fmt}")

        exporter = get_exporter(fmt)
        exported_path = exporter.export(mesh, out_path)
        print(f"  Exported: {exported_path}")

    total_time = time.perf_counter() - total_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
