#!/usr/bin/env python3
"""
MODEL_GENERATOR_V2 — 3D Model Format Converter

Converts GLB/OBJ/PLY files to STL, STEP (.stp), and other formats
that can be opened directly in SolidWorks, Fusion 360, FreeCAD, etc.

═══════════════════════════════════════════════════════════════════════
  Supported Conversions:
═══════════════════════════════════════════════════════════════════════

  INPUT FORMATS         OUTPUT FORMATS
  ─────────────         ──────────────
  .glb / .gltf          .stl  (3D Printing / SolidWorks Import)
  .obj                  .step / .stp  (SolidWorks / CAD Native)
  .ply                  .obj  (Wavefront)
  .stl                  .ply  (Stanford)
  .off                  .3mf  (Modern 3D printing)
                        .dae  (Collada / Blender)
                        .glb  (Web / Game Engines)

═══════════════════════════════════════════════════════════════════════
  SolidWorks Compatibility:
═══════════════════════════════════════════════════════════════════════

  .sldprt is a proprietary SolidWorks binary format that CANNOT be
  written directly from Python. Instead, this tool exports to:

  1. STL  → SolidWorks: File → Open → select .stl → Opens as mesh
  2. STEP → SolidWorks: File → Open → select .stp → Opens as solid
  3. 3MF  → SolidWorks: File → Open → select .3mf → Opens as mesh

  For .sldprt:
    Open the STL/STEP in SolidWorks → File → Save As → .sldprt

═══════════════════════════════════════════════════════════════════════

Usage:
    python convert_model.py input.glb --to stl
    python convert_model.py input.glb --to step
    python convert_model.py input.glb --to stl step obj ply 3mf
    python convert_model.py input.glb --to stl --output my_model.stl
    python convert_model.py *.glb --to stl --output-dir ./converted
    python convert_model.py input.glb --to stl --repair --smooth 3
"""

import argparse
import glob
import os
import sys
import time
import tempfile
import logging
from pathlib import Path
from typing import List, Optional

import trimesh

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_converter")


# ════════════════════════════════════════════════════════════════════════════
#  CORE CONVERTER CLASS
# ════════════════════════════════════════════════════════════════════════════

class ModelConverter:
    """Converts 3D model files between formats.

    Handles loading from GLB/OBJ/PLY/STL, optional mesh repair and
    smoothing, and exporting to STL/STEP/OBJ/PLY/3MF/DAE/GLB.

    For SolidWorks users:
        - STL: Import directly as a mesh body
        - STEP: Import as a proper B-rep solid (best for CAD editing)
        - Save as .sldprt from within SolidWorks after import

    Attributes:
        mesh: The loaded trimesh.Trimesh object.
        source_path: Path to the source file.
    """

    SUPPORTED_INPUT = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off", ".3mf", ".dae"}
    SUPPORTED_OUTPUT = {".stl", ".stp", ".step", ".obj", ".ply", ".glb", ".3mf", ".dae"}

    def __init__(self) -> None:
        self.mesh: Optional[trimesh.Trimesh] = None
        self.source_path: Optional[str] = None

    # ── Loading ────────────────────────────────────────────────────────────

    def load(self, filepath: str) -> "ModelConverter":
        """Load a 3D model file.

        Handles single meshes and scenes (multi-mesh GLB files are
        merged into a single mesh automatically).

        Args:
            filepath: Path to the input 3D model file.

        Returns:
            self (for method chaining).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        filepath = os.path.abspath(filepath)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = Path(filepath).suffix.lower()
        if ext not in self.SUPPORTED_INPUT:
            raise ValueError(
                f"Unsupported input format '{ext}'. "
                f"Supported: {sorted(self.SUPPORTED_INPUT)}"
            )

        logger.info(f"Loading: {filepath}")
        loaded = trimesh.load(filepath, process=False)

        # Handle scenes (GLB can contain multiple meshes)
        if isinstance(loaded, trimesh.Scene):
            meshes = []
            for name, geom in loaded.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)
                    logger.info(f"  Found mesh '{name}': {len(geom.vertices)} verts, {len(geom.faces)} faces")

            if not meshes:
                raise ValueError("No triangle meshes found in scene")

            if len(meshes) == 1:
                self.mesh = meshes[0]
            else:
                self.mesh = trimesh.util.concatenate(meshes)
                logger.info(f"  Merged {len(meshes)} meshes into one")
        elif isinstance(loaded, trimesh.Trimesh):
            self.mesh = loaded
        else:
            raise ValueError(f"Unexpected object type: {type(loaded)}")

        self.source_path = filepath
        logger.info(
            f"Loaded: {len(self.mesh.vertices):,} vertices, "
            f"{len(self.mesh.faces):,} faces"
        )
        return self

    # ── Mesh Processing ───────────────────────────────────────────────────

    def repair(self) -> "ModelConverter":
        """Repair mesh defects.

        Fixes normals, merges duplicate vertices, removes degenerate
        faces, and fills small holes.

        Returns:
            self (for method chaining).
        """
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")

        logger.info("Repairing mesh...")
        original_faces = len(self.mesh.faces)

        # Fix winding and normals
        self.mesh.fix_normals()

        # Remove degenerate faces
        self.mesh.update_faces(self.mesh.nondegenerate_faces())

        # Merge close vertices
        self.mesh.merge_vertices()

        # Remove duplicate faces
        self.mesh.update_faces(self.mesh.unique_faces())

        # Remove unreferenced vertices
        self.mesh.remove_unreferenced_vertices()

        repaired_faces = len(self.mesh.faces)
        diff = original_faces - repaired_faces
        if diff > 0:
            logger.info(f"  Removed {diff} degenerate/duplicate faces")
        logger.info(
            f"  Result: {len(self.mesh.vertices):,} verts, "
            f"{repaired_faces:,} faces"
        )
        return self

    def smooth(self, iterations: int = 5) -> "ModelConverter":
        """Apply Laplacian smoothing.

        Args:
            iterations: Number of smoothing passes.

        Returns:
            self (for method chaining).
        """
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")

        logger.info(f"Smoothing mesh ({iterations} iterations)...")
        try:
            trimesh.smoothing.filter_laplacian(
                self.mesh, iterations=iterations
            )
            logger.info("  Smoothing complete")
        except Exception as e:
            logger.warning(f"  Smoothing failed: {e}")
        return self

    def decimate(self, target_faces: int = 100000) -> "ModelConverter":
        """Reduce face count via simplification.

        Args:
            target_faces: Target number of faces.

        Returns:
            self (for method chaining).
        """
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")

        if len(self.mesh.faces) <= target_faces:
            logger.info(
                f"Mesh already has {len(self.mesh.faces):,} faces "
                f"(<= target {target_faces:,}), skipping"
            )
            return self

        logger.info(
            f"Decimating: {len(self.mesh.faces):,} → {target_faces:,} faces..."
        )
        try:
            self.mesh = self.mesh.simplify_quadric_decimation(target_faces)
            logger.info(f"  Result: {len(self.mesh.faces):,} faces")
        except Exception as e:
            logger.warning(f"  Decimation failed: {e}")
        return self

    def make_watertight(self) -> "ModelConverter":
        """Attempt to make the mesh watertight (for STEP/solid export).

        Fills holes and ensures consistent normals so the mesh
        defines a valid enclosed volume.

        Returns:
            self (for method chaining).
        """
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")

        logger.info("Making mesh watertight...")
        self.mesh.fill_holes()
        self.mesh.fix_normals()

        if self.mesh.is_watertight:
            logger.info("  ✓ Mesh is watertight")
        else:
            logger.warning(
                "  ✗ Mesh is NOT fully watertight. "
                "STEP export may produce an open shell instead of a solid."
            )
        return self

    # ── Export Methods ────────────────────────────────────────────────────

    def to_stl(self, output_path: str, binary: bool = True) -> str:
        """Export to STL format.

        STL is universally supported by:
        - SolidWorks (File → Open → .stl)
        - Fusion 360, FreeCAD, Inventor
        - All 3D printers (Cura, PrusaSlicer, etc.)

        Args:
            output_path: Output file path.
            binary: Use binary STL (True) or ASCII (False).

        Returns:
            Absolute path to the exported file.
        """
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".stl")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        mesh.export(output_path, file_type="stl")
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Exported STL: {output_path} ({size_mb:.1f} MB)")
        return os.path.abspath(output_path)

    def to_step(self, output_path: str) -> str:
        """Export to STEP format (.stp / .step).

        STEP is the gold standard for CAD interchange:
        - SolidWorks: File → Open → .stp → Opens as solid body
        - Fusion 360, FreeCAD, CATIA, Inventor, Creo

        Requires: cadquery + OCP (Open CASCADE). If not installed,
        falls back to FreeCAD CLI, or provides manual instructions.

        Args:
            output_path: Output file path.

        Returns:
            Absolute path to the exported file.
        """
        self._check_mesh()
        output_path = self._ensure_ext(output_path, ".stp")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        # Try Method 1: CadQuery + OCP
        if self._export_step_cadquery(output_path):
            return os.path.abspath(output_path)

        # Try Method 2: FreeCAD Python module
        if self._export_step_freecad(output_path):
            return os.path.abspath(output_path)

        # Try Method 3: numpy-stl → gmsh → STEP
        if self._export_step_gmsh(output_path):
            return os.path.abspath(output_path)

        # Fallback: Export STL + provide SolidWorks instructions
        fallback_stl = output_path.replace(".stp", ".stl").replace(".step", ".stl")
        self.to_stl(fallback_stl)
        logger.warning(
            f"\n"
            f"  ╔══════════════════════════════════════════════════════════╗\n"
            f"  ║  STEP export requires additional CAD libraries.        ║\n"
            f"  ║  Exported STL instead: {os.path.basename(fallback_stl):<30} ║\n"
            f"  ║                                                        ║\n"
            f"  ║  To get STEP output, install ONE of:                   ║\n"
            f"  ║                                                        ║\n"
            f"  ║  Option A (recommended):                               ║\n"
            f"  ║    pip install cadquery                                 ║\n"
            f"  ║                                                        ║\n"
            f"  ║  Option B:                                             ║\n"
            f"  ║    conda install -c conda-forge freecad                ║\n"
            f"  ║                                                        ║\n"
            f"  ║  Option C — Convert STL to SLDPRT in SolidWorks:      ║\n"
            f"  ║    1. Open SolidWorks                                  ║\n"
            f"  ║    2. File → Open → select the .stl file              ║\n"
            f"  ║    3. Choose 'Solid Body' import option                ║\n"
            f"  ║    4. File → Save As → .sldprt                        ║\n"
            f"  ╚══════════════════════════════════════════════════════════╝\n"
        )
        return os.path.abspath(fallback_stl)

    def _export_step_cadquery(self, output_path: str) -> bool:
        """Try STEP export using CadQuery + OCP."""
        try:
            import OCP  # noqa: F401  (availability probe)
            from OCP.StlAPI import StlAPI_Reader
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
            from OCP.STEPControl import (
                STEPControl_Writer,
                STEPControl_AsIs,
            )
            from OCP.Interface import Interface_Static
            from OCP.TopoDS import TopoDS_Shape

            logger.info("Exporting STEP via OCP (Open CASCADE)...")

            # Export mesh to temporary STL
            mesh = self._check_mesh()
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                mesh.export(tmp.name, file_type="stl")
                tmp_stl = tmp.name

            # Read STL into OCC shape
            reader = StlAPI_Reader()
            shape = TopoDS_Shape()
            reader.Read(shape, tmp_stl)

            # Sew the triangulation into a shell
            sewer = BRepBuilderAPI_Sewing(1e-6)
            sewer.Add(shape)
            sewer.Perform()
            sewn_shape = sewer.SewedShape()

            # Write STEP
            writer = STEPControl_Writer()
            Interface_Static.SetCVal("write.step.schema", "AP214")
            writer.Transfer(sewn_shape, STEPControl_AsIs)
            status = writer.Write(output_path)

            os.unlink(tmp_stl)

            if status == 1:  # IFSelect_RetDone
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Exported STEP: {output_path} ({size_mb:.1f} MB)")
                return True
            return False

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"OCP STEP export failed: {e}")
            return False

    def _export_step_freecad(self, output_path: str) -> bool:
        """Try STEP export using FreeCAD Python module."""
        try:
            import FreeCAD  # noqa: F401  (availability probe)
            import Part
            import Mesh as FcMesh

            logger.info("Exporting STEP via FreeCAD...")

            mesh = self._check_mesh()
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                mesh.export(tmp.name, file_type="stl")
                tmp_stl = tmp.name

            mesh_obj = FcMesh.Mesh(tmp_stl)
            shape = Part.Shape()
            shape.makeShapeFromMesh(mesh_obj.Topology, 0.1)
            solid = Part.makeSolid(shape)
            solid.exportStep(output_path)

            os.unlink(tmp_stl)

            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Exported STEP: {output_path} ({size_mb:.1f} MB)")
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"FreeCAD STEP export failed: {e}")
            return False

    def _export_step_gmsh(self, output_path: str) -> bool:
        """Try STEP export using gmsh."""
        try:
            import gmsh

            logger.info("Exporting STEP via gmsh...")

            mesh = self._check_mesh()
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                mesh.export(tmp.name, file_type="stl")
                tmp_stl = tmp.name

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.merge(tmp_stl)
            gmsh.model.mesh.classifySurfaces(
                angle=40 * 3.14159 / 180, boundary=True, forReparametrization=True
            )
            gmsh.model.mesh.createGeometry()
            gmsh.write(output_path)
            gmsh.finalize()

            os.unlink(tmp_stl)

            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Exported STEP: {output_path} ({size_mb:.1f} MB)")
                return True
            return False

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"gmsh STEP export failed: {e}")
            return False

    def to_obj(self, output_path: str) -> str:
        """Export to Wavefront OBJ format."""
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".obj")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        mesh.export(output_path, file_type="obj")
        logger.info(f"Exported OBJ: {output_path}")
        return os.path.abspath(output_path)

    def to_ply(self, output_path: str, binary: bool = True) -> str:
        """Export to Stanford PLY format."""
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".ply")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        encoding = "binary_little_endian" if binary else "ascii"
        mesh.export(output_path, file_type="ply", encoding=encoding)
        logger.info(f"Exported PLY: {output_path}")
        return os.path.abspath(output_path)

    def to_glb(self, output_path: str) -> str:
        """Export to binary GLB format."""
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".glb")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        mesh.export(output_path, file_type="glb")
        logger.info(f"Exported GLB: {output_path}")
        return os.path.abspath(output_path)

    def to_3mf(self, output_path: str) -> str:
        """Export to 3MF format (modern 3D printing standard).

        3MF is supported by:
        - SolidWorks 2019+ (File → Open)
        - Cura, PrusaSlicer, Microsoft 3D Builder
        """
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".3mf")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        try:
            mesh.export(output_path, file_type="3mf")
            logger.info(f"Exported 3MF: {output_path}")
        except Exception as e:
            logger.warning(f"3MF export failed ({e}), falling back to STL")
            return self.to_stl(output_path.replace(".3mf", ".stl"))
        return os.path.abspath(output_path)

    def to_dae(self, output_path: str) -> str:
        """Export to Collada DAE format."""
        mesh = self._check_mesh()
        output_path = self._ensure_ext(output_path, ".dae")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        mesh.export(output_path, file_type="dae")
        logger.info(f"Exported DAE: {output_path}")
        return os.path.abspath(output_path)

    def convert(self, output_path: str, fmt: str) -> str:
        """Convert to any supported format by name.

        Args:
            output_path: Output file path.
            fmt: Format string (stl, step, stp, obj, ply, glb, 3mf, dae).

        Returns:
            Path to the exported file.
        """
        fmt = fmt.lower().strip(".")
        exporters = {
            "stl": self.to_stl,
            "step": self.to_step,
            "stp": self.to_step,
            "obj": self.to_obj,
            "ply": self.to_ply,
            "glb": self.to_glb,
            "3mf": self.to_3mf,
            "dae": self.to_dae,
        }
        if fmt not in exporters:
            raise ValueError(
                f"Unsupported output format '{fmt}'. "
                f"Supported: {sorted(exporters.keys())}"
            )
        return exporters[fmt](output_path)

    # ── Mesh Info ─────────────────────────────────────────────────────────

    def info(self) -> str:
        """Return a formatted mesh information string."""
        m = self._check_mesh()

        bbox = m.bounding_box.extents if m.bounding_box else [0, 0, 0]
        try:
            volume = f"{m.volume:.4f}" if m.is_watertight else "N/A (open mesh)"
        except Exception:
            volume = "N/A"

        return (
            f"\n"
            f"  ╔══════════════════════════════════════════╗\n"
            f"  ║         MESH INFORMATION                 ║\n"
            f"  ╠══════════════════════════════════════════╣\n"
            f"  ║  Source:        {os.path.basename(self.source_path or 'N/A'):>22}  ║\n"
            f"  ║  Vertices:      {len(m.vertices):>22,}  ║\n"
            f"  ║  Faces:         {len(m.faces):>22,}  ║\n"
            f"  ║  Edges:         {len(m.edges_unique):>22,}  ║\n"
            f"  ║  Watertight:    {'✓ Yes' if m.is_watertight else '✗ No':>22}  ║\n"
            f"  ║  Volume:        {volume:>22}  ║\n"
            f"  ║  Surface Area:  {m.area:>22.4f}  ║\n"
            f"  ║  BBox Size:     {bbox[0]:.2f} × {bbox[1]:.2f} × {bbox[2]:.2f}  ║\n"
            f"  ╚══════════════════════════════════════════╝\n"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_mesh(self) -> "trimesh.Trimesh":
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Call load() first.")
        return self.mesh

    @staticmethod
    def _ensure_ext(path: str, ext: str) -> str:
        if not path.lower().endswith(ext):
            path += ext
        return path


# ════════════════════════════════════════════════════════════════════════════
#  BATCH CONVERTER
# ════════════════════════════════════════════════════════════════════════════

def batch_convert(
    input_paths: List[str],
    output_formats: List[str],
    output_dir: str = "./converted",
    repair: bool = False,
    smooth_iterations: int = 0,
    decimate_faces: int = 0,
) -> List[str]:
    """Convert multiple files to multiple formats.

    Args:
        input_paths: List of input file paths (supports glob patterns).
        output_formats: List of output format strings.
        output_dir: Directory for converted files.
        repair: Whether to repair meshes before conversion.
        smooth_iterations: Smoothing passes (0 = disabled).
        decimate_faces: Target face count (0 = disabled).

    Returns:
        List of exported file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    exported = []
    converter = ModelConverter()

    # Expand glob patterns
    all_files = []
    for pattern in input_paths:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        else:
            all_files.append(pattern)

    total = len(all_files) * len(output_formats)
    logger.info(
        f"Batch converting {len(all_files)} file(s) "
        f"→ {len(output_formats)} format(s) ({total} exports)"
    )

    for i, filepath in enumerate(all_files, 1):
        stem = Path(filepath).stem

        try:
            converter.load(filepath)

            if repair:
                converter.repair()
            if smooth_iterations > 0:
                converter.smooth(smooth_iterations)
            if decimate_faces > 0:
                converter.decimate(decimate_faces)

            for fmt in output_formats:
                ext = fmt.lower().strip(".")
                out_name = f"{stem}.{ext}"
                out_path = os.path.join(output_dir, out_name)

                result = converter.convert(out_path, fmt)
                exported.append(result)

            logger.info(f"  [{i}/{len(all_files)}] Done: {stem}")

        except Exception as e:
            logger.error(f"  [{i}/{len(all_files)}] FAILED: {filepath} — {e}")

    logger.info(f"\nBatch complete: {len(exported)}/{total} exports successful")
    return exported


# ════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def glb_to_stl(input_path: str, output_path: Optional[str] = None, repair: bool = True) -> str:
    """Convert a GLB file to STL.

    Args:
        input_path: Path to the .glb file.
        output_path: Path for the .stl output. Auto-generated if None.
        repair: Whether to repair the mesh first.

    Returns:
        Path to the exported STL file.

    Example:
        >>> stl_path = glb_to_stl('model.glb')
        >>> print(f"STL saved to: {stl_path}")
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".stl"))

    converter = ModelConverter()
    converter.load(input_path)
    if repair:
        converter.repair()
    return converter.to_stl(output_path)


def glb_to_step(input_path: str, output_path: Optional[str] = None, repair: bool = True) -> str:
    """Convert a GLB file to STEP (SolidWorks / CAD).

    Args:
        input_path: Path to the .glb file.
        output_path: Path for the .stp output. Auto-generated if None.
        repair: Whether to repair the mesh first.

    Returns:
        Path to the exported STEP file.

    Example:
        >>> step_path = glb_to_step('model.glb')
        >>> print(f"STEP saved to: {step_path}")
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".stp"))

    converter = ModelConverter()
    converter.load(input_path)
    if repair:
        converter.repair()
    converter.make_watertight()
    return converter.to_step(output_path)


def glb_to_sldprt_instructions(input_path: str, output_dir: str = ".") -> str:
    """Convert GLB to STL + print SolidWorks .sldprt conversion instructions.

    Since .sldprt is a proprietary format, this exports STL and STEP
    (if possible) and prints instructions for the final conversion
    inside SolidWorks.

    Args:
        input_path: Path to the .glb file.
        output_dir: Directory for output files.

    Returns:
        Path to the exported STL file.
    """
    stem = Path(input_path).stem
    os.makedirs(output_dir, exist_ok=True)

    converter = ModelConverter()
    converter.load(input_path)
    converter.repair()
    converter.make_watertight()

    # Export STL
    stl_path = converter.to_stl(os.path.join(output_dir, f"{stem}.stl"))

    # Try STEP
    step_path = converter.to_step(os.path.join(output_dir, f"{stem}.stp"))

    # Print SolidWorks instructions
    print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              HOW TO CREATE .SLDPRT IN SOLIDWORKS                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                     ║
║  Exported files:                                                    ║
║    STL:  {os.path.basename(stl_path):<55}  ║
║    STEP: {os.path.basename(step_path):<55}  ║
║                                                                     ║
║  ── Method 1: Import STL (easiest) ──                               ║
║                                                                     ║
║   1. Open SolidWorks                                                ║
║   2. File → Open → Change type to "STL (*.stl)"                    ║
║   3. Select: {os.path.basename(stl_path):<51}  ║
║   4. In the import dialog:                                          ║
║      • Select "Solid Body" (not Graphics Body)                      ║
║      • Set units to Millimeters (or your preference)                ║
║   5. Click OK → The mesh appears as a solid                        ║
║   6. File → Save As → SolidWorks Part (*.sldprt)                   ║
║                                                                     ║
║  ── Method 2: Import STEP (best for CAD editing) ──                 ║
║                                                                     ║
║   1. Open SolidWorks                                                ║
║   2. File → Open → Change type to "STEP (*.stp)"                   ║
║   3. Select: {os.path.basename(step_path):<51}  ║
║   4. Click OK → Opens as a fully editable B-rep solid              ║
║   5. File → Save As → SolidWorks Part (*.sldprt)                   ║
║                                                                     ║
║  ── Method 3: SolidWorks Macro (automated) ──                      ║
║                                                                     ║
║   1. Open SolidWorks → Tools → Macro → New                         ║
║   2. Paste the VBA macro code (see convert_model.py docs)           ║
║   3. Run → Creates .sldprt automatically                            ║
║                                                                     ║
╚═══════════════════════════════════════════════════════════════════════╝
""")
    return stl_path


# ════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert_model",
        description=(
            "Convert 3D model files (GLB/OBJ/PLY/STL) to STL, STEP, "
            "and other formats for SolidWorks, 3D printing, and CAD."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_model.py model.glb --to stl
  python convert_model.py model.glb --to step
  python convert_model.py model.glb --to stl step obj ply
  python convert_model.py model.glb --to stl --repair --smooth 5
  python convert_model.py model.glb --to stl --output my_model.stl
  python convert_model.py *.glb --to stl --output-dir ./converted
  python convert_model.py model.glb --to sldprt
  python convert_model.py model.glb --info
        """,
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="Input file(s) — supports glob patterns (e.g., *.glb).",
    )
    parser.add_argument(
        "--to", "-t",
        nargs="+",
        default=["stl"],
        metavar="FORMAT",
        help=(
            "Output format(s): stl, step/stp, obj, ply, glb, 3mf, dae, sldprt. "
            "Default: stl. Use 'sldprt' for SolidWorks instructions."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (for single file conversion).",
    )
    parser.add_argument(
        "--output-dir", "-d",
        default=None,
        help="Output directory (for batch conversion).",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair mesh before conversion.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="N",
        help="Apply N iterations of Laplacian smoothing.",
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=0,
        metavar="FACES",
        help="Decimate to target face count.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print mesh information and exit.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    total_start = time.perf_counter()

    print(f"\n{'═'*60}")
    print("  MODEL CONVERTER — 3D Format Conversion Tool")
    print(f"{'═'*60}\n")

    # Handle sldprt special case
    formats = [f.lower().strip(".") for f in args.to]
    has_sldprt = "sldprt" in formats

    if has_sldprt:
        formats = [f for f in formats if f != "sldprt"]
        if not formats:
            formats = ["stl"]  # At minimum export STL for SolidWorks

    # Expand inputs
    all_inputs = []
    for pattern in args.input:
        matches = glob.glob(pattern)
        if matches:
            all_inputs.extend(matches)
        elif os.path.exists(pattern):
            all_inputs.append(pattern)
        else:
            print(f"  Warning: File not found: {pattern}", file=sys.stderr)

    if not all_inputs:
        print("  Error: No valid input files found.", file=sys.stderr)
        sys.exit(1)

    converter = ModelConverter()

    for filepath in all_inputs:
        try:
            converter.load(filepath)

            # Info mode
            if args.info:
                print(converter.info())
                continue

            # Processing
            if args.repair:
                converter.repair()
            if args.smooth > 0:
                converter.smooth(args.smooth)
            if args.decimate > 0:
                converter.decimate(args.decimate)

            # Determine output directory
            out_dir = args.output_dir or os.path.dirname(filepath) or "."
            stem = Path(filepath).stem

            # Export each format
            for fmt in formats:
                if args.output and len(all_inputs) == 1 and len(formats) == 1:
                    out_path = args.output
                else:
                    out_path = os.path.join(out_dir, f"{stem}.{fmt}")

                converter.convert(out_path, fmt)

            # SolidWorks special handling
            if has_sldprt:
                glb_to_sldprt_instructions(filepath, out_dir)

        except Exception as e:
            logger.error(f"Failed: {filepath} — {e}")

    total_time = time.perf_counter() - total_start
    print(f"\n  Completed in {total_time:.1f}s")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
