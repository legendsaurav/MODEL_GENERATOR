"""
MODEL_GENERATOR_V2 — Ultra-quality single-image-to-3D-mesh generation.

Based on Tencent Hunyuan3D-2.1, stripped to geometry-only pipeline.
Generates smooth, clean, optimized 3D meshes from a single input image.

Usage:
    from MODEL_GENERATOR_V2.generation import GeometryPipeline
    from MODEL_GENERATOR_V2.postprocessing import PostProcessingPipeline
    from MODEL_GENERATOR_V2.exporters import get_exporter

    pipeline = GeometryPipeline.from_pretrained(preset='ultra')
    mesh = pipeline('input.png')
"""

__version__ = "2.0.0"
__author__ = "MODEL_GENERATOR_V2"
