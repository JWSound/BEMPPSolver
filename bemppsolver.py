"""
Simulates radiation of a freestanding loudspeaker using the Boundary Element Method (BEM)
via the Bempp-cl library.

- Requires a mesh file with two surface groups representing a rigid enclosure and a driven disc
- Outputs normalized horizontal and vertical polar SPL around the device
- Outputs the real + imaginary acoustic impedance as well.

"""

import argparse
import time
from dataclasses import dataclass
from typing import Tuple
import bempp_cl.api
import meshio
import numpy as np
import warnings
from pyopencl import CompilerWarning
warnings.filterwarnings("ignore", category=CompilerWarning)

# ==========================================
# Configuration
# ==========================================
@dataclass
class SimulationConfig:
    mesh_file: str
    sound_speed: float = 343.0      # m/s
    rho: float = 1.21               # kg/m^3
    distance: float = 2.0           # meters
    observation_axial_offset_m: float = 0.116  # meters; shifts polar origin along +Z axis
    polar_angle_step_deg: float = 2.5  # angular precision for polar sampling
    polar_angle_min_deg: float = -180
    polar_angle_max_deg: float = 180
    freq_min: float = 200.0
    freq_max: float = 20000.0
    freq_count: int = 72
    tag_throat: int = 2             # Mesh physical tag index for the disc representing the compression driver
    scale_factor: float = 0.001     # Mesh should be scaled to mm
    use_burton_miller: bool = True  # Use Burton-Miller formulation to mitigate fictitious resonances

    # Output controls
    output_npz_base_path: str = "pressure_data"

    # BEMPP Device Configuration
    bempp_cl.api.BOUNDARY_OPERATOR_DEVICE_TYPE = "cpu"
    bempp_cl.api.POTENTIAL_OPERATOR_DEVICE_TYPE = "cpu"
    bempp_cl.api.DEFAULT_PRECISION = 'single'

# Global instance for easy configuration editing
CONFIG = SimulationConfig(
    mesh_file="samplemesh_clean.msh",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BEM frequency sweep on a loudspeaker mesh.")
    parser.add_argument(
        "mesh_file",
        nargs="?",
        default=CONFIG.mesh_file,
        help="Path to the input mesh file",
    )
    parser.add_argument(
        "--output-npz-base-path",
        default=CONFIG.output_npz_base_path,
        help="Base path for the solver output NPZ file, without the .npz suffix",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=CONFIG.freq_min,
        help="Minimum frequency in Hz",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=CONFIG.freq_max,
        help="Maximum frequency in Hz",
    )
    parser.add_argument(
        "--freq-count",
        type=int,
        default=CONFIG.freq_count,
        help="Number of frequency points in the sweep",
    )
    parser.add_argument(
        "--polar-angle-step-deg",
        type=float,
        default=CONFIG.polar_angle_step_deg,
        help="Angular step for polar evaluation in degrees",
    )
    parser.add_argument(
        "--polar-angle-min-deg",
        type=float,
        default=CONFIG.polar_angle_min_deg,
        help="Minimum polar angle in degrees",
    )
    parser.add_argument(
        "--polar-angle-max-deg",
        type=float,
        default=CONFIG.polar_angle_max_deg,
        help="Maximum polar angle in degrees",
    )
    parser.add_argument(
        "--observation-axial-offset-m",
        type=float,
        default=CONFIG.observation_axial_offset_m,
        help="Shift the polar evaluation origin along +Z in meters",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        mesh_file=args.mesh_file,
        sound_speed=CONFIG.sound_speed,
        rho=CONFIG.rho,
        distance=CONFIG.distance,
        observation_axial_offset_m=args.observation_axial_offset_m,
        polar_angle_step_deg=args.polar_angle_step_deg,
        polar_angle_min_deg=args.polar_angle_min_deg,
        polar_angle_max_deg=args.polar_angle_max_deg,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        freq_count=args.freq_count,
        tag_throat=CONFIG.tag_throat,
        scale_factor=CONFIG.scale_factor,
        use_burton_miller=CONFIG.use_burton_miller,
        output_npz_base_path=args.output_npz_base_path,
    )

# ==========================================
# Solver Class
# ==========================================
class HornBEMSolver:
    def __init__(self, config: SimulationConfig):
        self.cfg = config

        print(f"Loading mesh: {self.cfg.mesh_file}...")
        self.grid, self.physical_tags = self._load_mesh()
        
        # Setup Spaces
        # P1: Continuous linear elements (for Pressure)
        # DP0: Discontinuous constant elements (for Velocity/Flux)
        self.p1_space = bempp_cl.api.function_space(self.grid, "P", 1)
        self.dp0_space = bempp_cl.api.function_space(self.grid, "DP", 0)
        
        # Pre-compute Geometry info
        self._setup_driver_geometry()
        self._setup_polar_evaluation_points()
        
        # Pre-compute Identity Operator (Frequency Independent)
        self.lhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.p1_space, self.p1_space, self.p1_space
        )
        self.rhs_identity = bempp_cl.api.operators.boundary.sparse.identity(
            self.dp0_space, self.p1_space, self.p1_space
        )

        # Create Unit Velocity Excitation (to scale later)
        self.unit_velocity_fun = self._create_unit_velocity()

    def _load_mesh(self) -> Tuple[bempp_cl.api.Grid, np.ndarray]:
        #Load mesh and extract physical tags.
        mesh_data = meshio.read(self.cfg.mesh_file)
        vertices = mesh_data.points * self.cfg.scale_factor
        
        # Handle meshio cell key variations
        if 'triangle' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle']
            tri_key = 'triangle'
        elif 'triangle3' in mesh_data.cells_dict:
            elements = mesh_data.cells_dict['triangle3']
            tri_key = 'triangle3'
        else:
            raise ValueError("No triangular elements found in mesh.")

        physical_tags = None
        for key in mesh_data.cell_data_dict:
            if 'gmsh:physical' in key and tri_key in mesh_data.cell_data_dict[key]:
                physical_tags = mesh_data.cell_data_dict[key][tri_key]
                break
        
        if physical_tags is None:
            raise ValueError("No physical tags found in mesh.")

        grid = bempp_cl.api.Grid(vertices.T, elements.T)
        return grid, physical_tags

    def _setup_driver_geometry(self):
        #Identify throat elements for impedance calculation
        # In DP0, DOFs map 1:1 to elements
        self.driver_dofs = [
            i for i in range(self.dp0_space.global_dof_count) 
            if self.physical_tags[i] == self.cfg.tag_throat
        ]

        if len(self.driver_dofs) == 0:
            raise ValueError(
                f"No throat elements found for tag_throat={self.cfg.tag_throat}. "
                "Check mesh physical tags."
            )

        self.enclosure_dofs = [
            i for i in range(self.dp0_space.global_dof_count)
            if self.physical_tags[i] != self.cfg.tag_throat
        ]
        
        # Geometry for impedance integration
        self.throat_element_areas = self.grid.volumes[self.driver_dofs]
        self.throat_p1_dofs = self.p1_space.local2global[self.driver_dofs]
        print(f"Driven surface identified with {len(self.driver_dofs)} elements. "
              f"Enclosure identified with {len(self.enclosure_dofs)} elements.")

    def _create_unit_velocity(self):
        #Create a normal velocity boundary condition with magnitude 1.0 on the throat.
        coeffs = np.zeros(self.dp0_space.global_dof_count, dtype=np.complex128)
        coeffs[self.driver_dofs] = 1.0
        return bempp_cl.api.GridFunction(self.dp0_space, coefficients=coeffs)

    def _setup_polar_evaluation_points(self):
        #Generate horizontal and vertical polar evaluation points.
        step = float(self.cfg.polar_angle_step_deg)
        if step <= 0:
            raise ValueError("polar_angle_step_deg must be positive.")

        angle_min = float(self.cfg.polar_angle_min_deg)
        angle_max = float(self.cfg.polar_angle_max_deg)
        if angle_min < -180.0 or angle_max > 180.0:
            raise ValueError("polar angle range must stay within [-180, 180] degrees.")
        if angle_max < angle_min:
            raise ValueError("polar_angle_max_deg must be >= polar_angle_min_deg.")
        if not (angle_min <= 0.0 <= angle_max):
            raise ValueError("polar angle range must include 0 degrees for on-axis normalization.")

        self.polar_angles_deg = np.arange(angle_min, angle_max + 0.5 * step, step, dtype=np.float32)
        self.polar_angles_deg = np.clip(self.polar_angles_deg, angle_min, angle_max)
        angles_rad = np.deg2rad(self.polar_angles_deg.astype(float))

        x_h = np.sin(angles_rad)
        y_h = np.zeros_like(x_h)
        z_h = np.cos(angles_rad)

        x_v = np.zeros_like(angles_rad)
        y_v = np.sin(angles_rad)
        z_v = np.cos(angles_rad)

        r_dist = float(self.cfg.distance)
        axial_offset_m = float(self.cfg.observation_axial_offset_m)
        axial_shift = np.array([[0.0], [0.0], [axial_offset_m]], dtype=float)

        self.horizontal_eval_points = r_dist * np.vstack([x_h, y_h, z_h]) + axial_shift
        self.vertical_eval_points = r_dist * np.vstack([x_v, y_v, z_v]) + axial_shift
        self.on_axis_idx = int(np.argmin(np.abs(self.polar_angles_deg)))

    def solve_sweep(self) -> Tuple[list, np.ndarray]:
        frequencies = np.logspace(
            np.log10(self.cfg.freq_min),
            np.log10(self.cfg.freq_max),
            self.cfg.freq_count
        )

        print(f"Starting solver: {len(frequencies)} frequencies.")

        results_polar = []
        results_imp = []
        for i, freq in enumerate(frequencies):
            res_h, res_v, res_z = self._solve_single_frequency(freq)
            results_polar.append((freq, res_h, res_v))
            results_imp.append(res_z)
            print(f"[{i+1}/{len(frequencies)}] {freq:.1f} Hz")

        imp_matrix = np.asarray(results_imp, dtype=np.float32)
        return results_polar, imp_matrix

    def _solve_single_frequency(self, freq):
        omega = 2 * np.pi * freq
        k = omega / self.cfg.sound_speed
        
        # 1. Update Boundary Conditions
        # v = 1 m/s. Use normal velocity on the throat only.
        velocity_fun = self.unit_velocity_fun
        neumann_fun = 1j * self.cfg.rho * omega * velocity_fun

        # 2. Assemble Operators
        dlp = bempp_cl.api.operators.boundary.helmholtz.double_layer(
            self.p1_space, self.p1_space, self.p1_space, k
        )
        slp = bempp_cl.api.operators.boundary.helmholtz.single_layer(
            self.dp0_space, self.p1_space, self.p1_space, k
        )

        # 3. Formulate LHS and RHS
        if self.cfg.use_burton_miller:
            hyp = bempp_cl.api.operators.boundary.helmholtz.hypersingular(
                self.p1_space, self.p1_space, self.p1_space, k
            )
            adlp = bempp_cl.api.operators.boundary.helmholtz.adjoint_double_layer(
                self.dp0_space, self.p1_space, self.p1_space, k
            )
            # Exterior Neumann, Burton-Miller (BEMPP sign conventions)
            # Note that BEMPP negates the hypersingular operator
            coupling = 1j / k
            lhs = 0.5 * self.lhs_identity - dlp - coupling * -hyp
            rhs = (-slp - coupling * (adlp + 0.5 * self.rhs_identity)) * neumann_fun
        else:
            # Exterior Neumann (classical)
            lhs = dlp - 0.5 * self.lhs_identity
            rhs = slp * neumann_fun

        # 4. Solve System
        dirichlet_fun, info = bempp_cl.api.linalg.gmres(lhs, rhs, tol=1E-3)
        if info != 0:
            print(f"  Warning: Solver did not converge at {freq:.1f}Hz")

        # 5. Post-Processing
        z_data = self._calculate_impedance(freq, dirichlet_fun)
        horizontal_spl = self._evaluate_field(self.horizontal_eval_points, k, dirichlet_fun, neumann_fun, omega)
        vertical_spl = self._evaluate_field(self.vertical_eval_points, k, dirichlet_fun, neumann_fun, omega)
        horizontal_spl_norm, vertical_spl_norm = self._normalize_polar_to_on_axis(horizontal_spl, vertical_spl)
        
        return horizontal_spl_norm, vertical_spl_norm, z_data

    def _calculate_impedance(self, freq, dirichlet_fun):
        # Pressure at local P1 dofs of throat elements.
        # Do not index with raw mesh vertex ids: P1 global dof numbering may differ.
        p_at_vertices = dirichlet_fun.coefficients[self.throat_p1_dofs]
        p_avg = np.mean(p_at_vertices, axis=1)
        
        # Force = Integral(p dS) ~ sum(p_avg * area)
        total_force = np.sum(p_avg * self.throat_element_areas) * 10
        
        # Z = Force / Velocity (v=1)
        return [freq, np.real(total_force)/2, -np.imag(total_force)/2]

    def _evaluate_field(self, points, k, dirichlet_fun, neumann_fun, omega):
        slp_pot = bempp_cl.api.operators.potential.helmholtz.single_layer(
            self.dp0_space, points, k, device_interface="opencl"
        )
        dlp_pot = bempp_cl.api.operators.potential.helmholtz.double_layer(
            self.p1_space, points, k, device_interface="opencl"
        )

        p_field = (dlp_pot * dirichlet_fun - slp_pot * neumann_fun).ravel()
        
        # Convert to SPL
        # Ref pressure = 20e-6 Pa
        return 20 * np.log10(np.abs(p_field) / 20e-6)

    def _normalize_polar_to_on_axis(self, horizontal_spl, vertical_spl):
        on_axis_ref = horizontal_spl[self.on_axis_idx]
        return horizontal_spl - on_axis_ref, vertical_spl - on_axis_ref

    def save_outputs(self, polar_results, imp_matrix):
        base = self.cfg.output_npz_base_path

        freqs = np.array([freq for freq, _, _ in polar_results], dtype=np.float32)
        horizontal_spl = np.vstack([h_spl for _, h_spl, _ in polar_results]).astype(np.float32, copy=False)
        vertical_spl = np.vstack([v_spl for _, _, v_spl in polar_results]).astype(np.float32, copy=False)
        z_freq_hz = imp_matrix[:, 0].astype(np.float32, copy=False)
        z_real = imp_matrix[:, 1].astype(np.float32, copy=False)
        z_imag = imp_matrix[:, 2].astype(np.float32, copy=False)

        np.savez_compressed(
            f"{base}.npz",
            freq_hz=freqs,
            polar_angle_deg=self.polar_angles_deg.astype(np.float32, copy=False),
            horizontal_spl_norm_db=horizontal_spl,
            vertical_spl_norm_db=vertical_spl,
            impedance_freq_hz=z_freq_hz,
            impedance_real=z_real,
            impedance_imag=z_imag,
            observation_axial_offset_m=np.float32(self.cfg.observation_axial_offset_m),
        )
        print(f"Saved {base}.npz")


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    config = _config_from_args(args)
    t_start = time.time()
    solver = HornBEMSolver(config)
    polar_results, imp_matrix = solver.solve_sweep()

    # Save Results
    solver.save_outputs(polar_results, imp_matrix)
    
    print(f"Total Analysis Time: {time.time() - t_start:.2f}s")
    print("Analysis Complete.")