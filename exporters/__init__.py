# MODEL_GENERATOR_V2 - Exporters Package
from typing import Callable, Dict

from .base_exporter import MeshExporter
from .glb_exporter import GLBExporter
from .obj_exporter import OBJExporter
from .stl_exporter import STLExporter
from .ply_exporter import PLYExporter


def get_exporter(fmt: str) -> MeshExporter:
    """Factory function to get an exporter by format name.

    Args:
        fmt: Format string ('glb', 'obj', 'stl', 'ply').

    Returns:
        Appropriate MeshExporter instance.

    Raises:
        ValueError: If format is not supported.
    """
    exporters: Dict[str, Callable[[], MeshExporter]] = {
        "glb": GLBExporter,
        "gltf": GLBExporter,
        "obj": OBJExporter,
        "stl": STLExporter,
        "ply": PLYExporter,
    }
    fmt_lower = fmt.lower().strip(".")
    if fmt_lower not in exporters:
        raise ValueError(
            f"Unsupported format '{fmt}'. "
            f"Supported: {list(exporters.keys())}"
        )
    return exporters[fmt_lower]()


__all__ = [
    "MeshExporter",
    "GLBExporter",
    "OBJExporter",
    "STLExporter",
    "PLYExporter",
    "get_exporter",
]
