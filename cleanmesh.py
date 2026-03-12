"""
Clean surface .msh files from Ath4 for BEMPP workflows.

1) Merges coincident (or near-coincident) vertices within a tolerance.
2) Rebuilds triangle connectivity.
3) Removes collapsed and duplicate triangles.
4) Removes unused vertices.
5) Reports topology stats before/after (boundary/open edges, non-manifold edges, etc.).

Notes:
- This script targets triangle surface meshes (common for BEM boundary meshes).
- It preserves triangle physical tags (gmsh:physical) when present.
- If true geometric holes exist, they will remain open; this script only stitches seams
  caused by duplicated/near-coincident vertices.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import meshio
import numpy as np


# =========================
# User settings (no-args run)
# =========================
# Edit these values to run this module directly without command-line arguments.
INPUT_MSH = "samplemesh.msh"
OUTPUT_MSH = "samplemesh_clean.msh"
MERGE_TOL = 1e-9
AREA_TOL = 0.0
WRITE_BINARY = False


@dataclass
class MeshStats:
    vertices: int
    triangles: int
    boundary_edges: int
    nonmanifold_edges: int
    duplicate_faces: int
    degenerate_faces: int
    components: int


def _find_triangle_block(mesh: meshio.Mesh) -> Tuple[str, np.ndarray]:
    cells_dict = mesh.cells_dict
    if "triangle" in cells_dict:
        return "triangle", np.asarray(cells_dict["triangle"], dtype=np.int64)
    if "triangle3" in cells_dict:
        return "triangle3", np.asarray(cells_dict["triangle3"], dtype=np.int64)
    raise ValueError("No triangle/triangle3 cell block found in mesh.")


def _extract_triangle_cell_data(mesh: meshio.Mesh, tri_key: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for data_name, by_cell_type in mesh.cell_data_dict.items():
        if tri_key in by_cell_type:
            out[data_name] = np.asarray(by_cell_type[tri_key])
    return out


def _edge_counts(triangles: np.ndarray) -> Dict[Tuple[int, int], int]:
    counts: Dict[Tuple[int, int], int] = {}
    for a, b, c in triangles:
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            key = (int(u), int(v))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _connected_components(triangles: np.ndarray) -> int:
    if len(triangles) == 0:
        return 0

    edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
    for face_index, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            if u > v:
                u, v = v, u
            edge_to_faces.setdefault((int(u), int(v)), []).append(face_index)

    adjacency: List[set] = [set() for _ in range(len(triangles))]
    for face_ids in edge_to_faces.values():
        if len(face_ids) < 2:
            continue
        for i in range(len(face_ids)):
            for j in range(i + 1, len(face_ids)):
                f0 = face_ids[i]
                f1 = face_ids[j]
                adjacency[f0].add(f1)
                adjacency[f1].add(f0)

    seen = np.zeros(len(triangles), dtype=bool)
    components = 0
    for start in range(len(triangles)):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)

    return components


def _degenerate_mask(points: np.ndarray, triangles: np.ndarray, area_tol: float) -> np.ndarray:
    v0 = points[triangles[:, 0]]
    v1 = points[triangles[:, 1]]
    v2 = points[triangles[:, 2]]

    repeated_vertex = (triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2]) | (triangles[:, 0] == triangles[:, 2])
    area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # 2 * area
    tiny_area = area2 <= (2.0 * area_tol)
    return repeated_vertex | tiny_area


def _mesh_stats(points: np.ndarray, triangles: np.ndarray, area_tol: float) -> MeshStats:
    deg_mask = _degenerate_mask(points, triangles, area_tol)

    sorted_faces = np.sort(triangles, axis=1)
    unique_faces = {tuple(row) for row in sorted_faces}
    duplicate_faces = len(sorted_faces) - len(unique_faces)

    edge_count = _edge_counts(triangles)
    boundary_edges = sum(1 for c in edge_count.values() if c == 1)
    nonmanifold_edges = sum(1 for c in edge_count.values() if c > 2)

    components = _connected_components(triangles)

    return MeshStats(
        vertices=len(points),
        triangles=len(triangles),
        boundary_edges=boundary_edges,
        nonmanifold_edges=nonmanifold_edges,
        duplicate_faces=duplicate_faces,
        degenerate_faces=int(np.sum(deg_mask)),
        components=components,
    )


def _spatial_hash_merge(points: np.ndarray, tol: float) -> np.ndarray:
    """
    Returns representative index for each original point.
    Points within tol are merged (transitively) via local grid-neighborhood checks.
    """
    if tol <= 0:
        return np.arange(len(points), dtype=np.int64)

    cell_size = tol
    inv = 1.0 / cell_size
    cell_coords = np.floor(points * inv).astype(np.int64)

    # Build cell -> point list
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, c in enumerate(cell_coords):
        key = (int(c[0]), int(c[1]), int(c[2]))
        grid.setdefault(key, []).append(idx)

    parent = np.arange(len(points), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Neighbor offsets in 3D grid
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for key, idxs in grid.items():
        # same cell comparisons
        for i in range(len(idxs)):
            ii = idxs[i]
            pi = points[ii]
            for j in range(i + 1, len(idxs)):
                jj = idxs[j]
                if np.linalg.norm(pi - points[jj]) <= tol:
                    union(ii, jj)

        # neighbor cells comparisons (only forward keys to avoid duplicate work)
        kx, ky, kz = key
        for dx, dy, dz in offsets:
            nk = (kx + dx, ky + dy, kz + dz)
            if nk <= key:
                continue
            if nk not in grid:
                continue
            neigh = grid[nk]
            for ii in idxs:
                pi = points[ii]
                for jj in neigh:
                    if np.linalg.norm(pi - points[jj]) <= tol:
                        union(ii, jj)

    rep = np.array([find(i) for i in range(len(points))], dtype=np.int64)
    return rep


def _remove_duplicate_faces(triangles: np.ndarray, cell_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    seen: Dict[Tuple[int, int, int], int] = {}
    keep_indices: List[int] = []
    removed = 0

    for idx, tri in enumerate(triangles):
        key = tuple(sorted((int(tri[0]), int(tri[1]), int(tri[2]))))
        if key in seen:
            removed += 1
            continue
        seen[key] = idx
        keep_indices.append(idx)

    keep = np.asarray(keep_indices, dtype=np.int64)
    triangles_out = triangles[keep]
    cell_data_out = {name: arr[keep] for name, arr in cell_data.items()}
    return triangles_out, cell_data_out, removed


def _compact_vertices(points: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles.ravel())
    new_index = -np.ones(len(points), dtype=np.int64)
    new_index[used] = np.arange(len(used), dtype=np.int64)

    points_compact = points[used]
    triangles_compact = new_index[triangles]
    return points_compact, triangles_compact


def clean_mesh(mesh: meshio.Mesh, merge_tol: float, area_tol: float) -> Tuple[meshio.Mesh, Dict[str, int], MeshStats, MeshStats]:
    tri_key, triangles = _find_triangle_block(mesh)
    points = np.asarray(mesh.points, dtype=float)
    cell_data = _extract_triangle_cell_data(mesh, tri_key)

    stats_before = _mesh_stats(points, triangles, area_tol)

    # 1) Merge near-coincident points
    rep = _spatial_hash_merge(points, merge_tol)
    unique_reps, inverse = np.unique(rep, return_inverse=True)
    points_merged = points[unique_reps]
    triangles_merged = inverse[triangles]

    merged_vertices = len(points) - len(points_merged)

    # 2) Remove degenerate faces
    deg_mask = _degenerate_mask(points_merged, triangles_merged, area_tol)
    keep = ~deg_mask
    triangles_clean = triangles_merged[keep]
    cell_data_clean = {name: arr[keep] for name, arr in cell_data.items()}
    removed_degenerate = int(np.sum(deg_mask))

    # 3) Remove duplicate faces
    triangles_clean, cell_data_clean, removed_duplicate = _remove_duplicate_faces(triangles_clean, cell_data_clean)

    # 4) Compact vertex list to used vertices only
    points_clean, triangles_clean = _compact_vertices(points_merged, triangles_clean)

    # Build output mesh preserving field_data and point_data where possible
    out_mesh = meshio.Mesh(
        points=points_clean,
        cells=[("triangle", triangles_clean)],
        point_data={},
        cell_data={name: [arr] for name, arr in cell_data_clean.items()},
        field_data=mesh.field_data,
    )

    stats_after = _mesh_stats(points_clean, triangles_clean, area_tol)

    changes = {
        "merged_vertices": int(merged_vertices),
        "removed_degenerate_faces": int(removed_degenerate),
        "removed_duplicate_faces": int(removed_duplicate),
        "removed_unused_vertices": int(len(points_merged) - len(points_clean)),
    }

    return out_mesh, changes, stats_before, stats_after


def _print_stats(label: str, s: MeshStats) -> None:
    print(f"{label}")
    print(f"  vertices          : {s.vertices}")
    print(f"  triangles         : {s.triangles}")
    print(f"  boundary edges    : {s.boundary_edges}")
    print(f"  nonmanifold edges : {s.nonmanifold_edges}")
    print(f"  duplicate faces   : {s.duplicate_faces}")
    print(f"  degenerate faces  : {s.degenerate_faces}")
    print(f"  components        : {s.components}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean/stitch a triangle .msh surface mesh for BEM.")
    parser.add_argument("input_msh", nargs="?", default=INPUT_MSH, help="Input .msh file")
    parser.add_argument("output_msh", nargs="?", default=OUTPUT_MSH, help="Output cleaned .msh file")
    parser.add_argument(
        "--merge-tol",
        type=float,
        default=MERGE_TOL,
        help="Vertex merge tolerance in mesh units (default: 1e-9)",
    )
    parser.add_argument(
        "--area-tol",
        type=float,
        default=AREA_TOL,
        help="Area tolerance for removing tiny triangles in mesh units^2 (default: 0.0)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=WRITE_BINARY,
        help="Write binary .msh (default is ASCII gmsh22 for compatibility)",
    )

    args = parser.parse_args()

    mesh = meshio.read(args.input_msh)
    cleaned, changes, before, after = clean_mesh(mesh, merge_tol=args.merge_tol, area_tol=args.area_tol)

    _print_stats("Before:", before)
    print("Changes:")
    for k, v in changes.items():
        print(f"  {k:24s}: {v}")
    _print_stats("After:", after)

    file_format = "gmsh22"
    meshio.write(args.output_msh, cleaned, file_format=file_format, binary=args.binary)
    print(f"\nWrote cleaned mesh: {args.output_msh}")

    if after.boundary_edges > 0:
        print("Warning: mesh still has open edges. This usually means real holes (not just unstitched seams).")


if __name__ == "__main__":
    main()
