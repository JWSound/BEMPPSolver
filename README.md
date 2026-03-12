# BEMPPSolver

Basic utilities for running Boundary Element Method (BEM) simulations on loudspeaker surface meshes and generating directivity and impedance plots.

The project is organized as a simple four-step workflow:

1. `cleanmesh.py`
2. `bemppsolver.py`
3. `prepare_visualization_data.py`
4. `visualizer.py`

## What It Does

Given a loudspeaker mesh in Gmsh `.msh` format with two surface groups comprising a rigid enclosure and a driven diaphragm, this project can:

- clean and stitch a triangle surface mesh for BEM use
- run a frequency sweep with `bempp-cl`
- compute normalized horizontal and vertical polar response
- compute acoustic impedance
- generate directivity and impedance plot images

## Requirements

This project is written in Python and depends on:

- `numpy`
- `meshio`
- `scipy`
- `matplotlib`
- `bempp-cl`
- `pyopencl`

Example install:

```bash
pip install numpy meshio scipy matplotlib bempp-cl pyopencl
```

To utilize OpenCL solving, you will need to download the Intel OpenCL runtime from this page: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html, which is compatible with both AMD/Intel CPUs.

## Mesh Expectations

The solver expects a triangular surface mesh with physical groups in the `.msh` file.

- One surface group should represent the rigid enclosure.
- One surface group should represent the driven diaphragm or throat surface.
- In `bemppsolver.py`, the driven surface is selected with `tag_throat`, which defaults to `2`.

If your mesh uses different physical tag values, update the configuration in `bemppsolver.py` before running the solver.

## Workflow

### 1. Clean the mesh

Use `cleanmesh.py` to merge coincident vertices, remove degenerate or duplicate triangles, and write a cleaned mesh.

Default input/output:

- input: `samplemesh.msh`
- output: `samplemesh_clean.msh`

Example:

```bash
python cleanmesh.py samplemesh.msh samplemesh_clean.msh --merge-tol 1e-9
```

If you run it without arguments, it uses the editable constants near the top of the file.

### 2. Run the BEM simulation

Use `bemppsolver.py` to run the acoustic simulation over a frequency sweep.

Default mesh:

- `samplemesh_clean.msh`

Default output:

- `pressure_data.npz`

Important settings are still defined in `SimulationConfig` in `bemppsolver.py`, including:

- frequency range and number of frequency points
- worker count for process-based parallel execution
- sound speed and density
- observation distance
- axial offset for polar origin
- `tag_throat`
- mesh scale factor

Run with defaults:

```bash
python bemppsolver.py
```

Run with CLI overrides:

```bash
python bemppsolver.py samplemesh_clean.msh --output-npz-base-path pressure_data --freq-min 200 --freq-max 20000 --freq-count 72 --polar-angle-step-deg 2.5 --polar-angle-min-deg -180 --polar-angle-max-deg 180 --observation-axial-offset-m 0.116 --workers 4
```

The solver currently supports these CLI inputs:

- positional `mesh_file`
- `--output-npz-base-path`
- `--freq-min`
- `--freq-max`
- `--freq-count`
- `--polar-angle-step-deg`
- `--polar-angle-min-deg`
- `--polar-angle-max-deg`
- `--observation-axial-offset-m`
- `--workers`

`--workers` uses a spawn-based process pool and splits the frequency sweep into equally sized chunks. This is process-based rather than thread-based so it works consistently on Windows, macOS, and Linux.

### 3. Prepare visualization data

Use `prepare_visualization_data.py` to convert raw solver output into plot-ready arrays.

Default input:

- `pressure_data.npz`

Default output:

- `pressure_data_formatted.npz`

Run with defaults:

```bash
python prepare_visualization_data.py
```

This step also applies clipping, interpolation, and fractional-octave smoothing for the isobar plots.

Run with CLI overrides:

```bash
python prepare_visualization_data.py pressure_data.npz pressure_data_formatted.npz --min-db -30 --max-db 0 --isobar-angle-samples-smooth 250 --isobar-freq-samples-smooth 500 --isobar-octave-smooth-fraction 24 --horizontal-reference-angle-deg 10 --vertical-reference-angle-deg 10
```

The prep stage currently supports these CLI inputs:

- `input_polar_npz`
- `output_npz`
- `--min-db`
- `--max-db`
- `--isobar-angle-samples-smooth`
- `--isobar-freq-samples-smooth`
- `--isobar-octave-smooth-fraction`
- `--horizontal-reference-angle-deg`
- `--vertical-reference-angle-deg`

### 4. Generate plots

Use `visualizer.py` to generate PNG outputs.

Default input:

- `pressure_data_formatted.npz`

Default outputs:

- `horizontal_isobar.png`
- `vertical_isobar.png`
- `acoustic_impedance.png`

Sample output previews:

<img src="sample_horizontal_isobar.png" alt="Sample horizontal isobar" width="520" />

<img src="sample_vertical_isobar.png" alt="Sample vertical isobar" width="520" />

<img src="sample_acoustic_impedance.png" alt="Sample acoustic impedance" width="520" />

Run with defaults:

```bash
python visualizer.py
```

The visualizer supports these CLI inputs:

- positional `input_npz`
- `--output-horizontal-png`
- `--output-vertical-png`
- `--output-impedance-png`
- `--isobar-interp-freq-factor`

## Typical Run Sequence

```bash
python cleanmesh.py
python bemppsolver.py
python prepare_visualization_data.py
python visualizer.py
```

## Notes

- `cleanmesh.py` writes Gmsh 2.2 format for compatibility.
- `bemppsolver.py` is configured to use CPU devices by default via OpenCL