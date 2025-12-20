import pyvista as pv
import numpy as np
import os
import matplotlib.pyplot as plt

# Import unit conversion utilities
from utils import ms_to_kts

class Visualizer:
    """
    3D and 2D visualization system for hurricane simulation.
    
    Generates:
    - 3D VTK files for ParaView visualization
    - 2D wind maps with pressure contours
    - Diagnostic overlays comparing simulated vs. historical tracks
    
    UNIT HANDLING:
        - Input velocities: dimensionless (from simulation)
        - Display velocities: converted to m/s or knots as appropriate
        - All conversions documented and explicit
    """
    
    def __init__(self, sim_instance):
        self.sim = sim_instance
        self.output_dir_3d = os.path.join(self.sim.plot_dir, "vtk_frames_final")
        if not os.path.exists(self.output_dir_3d):
            os.makedirs(self.output_dir_3d)
        print("Visualizer initialized with Diagnostic Overlay capabilities.")
        print("  -> All velocity fields will be converted to physical units for display")

    def generate_3d_scene(self, frame, pressure, q, T, u, v, w):
        """
        Generate 3D VTK scene for ParaView visualization.
        
        Creates a structured grid with all thermodynamic fields.
        Wind speeds are kept in dimensionless units in the VTK file
        since ParaView users can apply their own scaling.
        
        Args:
            frame: Current simulation frame
            pressure: 3D pressure field (dimensionless)
            q: 3D specific humidity field (kg/kg)
            T: 3D temperature field (°C)
            u, v, w: 3D velocity components (dimensionless)
        """
        grid = pv.StructuredGrid()
        
        # Create coordinate arrays (dimensionless domain)
        x_coords = np.arange(self.sim.nx) * self.sim.dx
        y_coords = np.arange(self.sim.ny) * self.sim.dy
        z_coords = np.arange(self.sim.nz) * self.sim.dz
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        grid.points = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T
        grid.dimensions = (self.sim.nx, self.sim.ny, self.sim.nz)

        # Add thermodynamic fields
        grid.point_data["pressure"] = pressure.ravel(order='F')
        grid.point_data["temperature_c"] = T.ravel(order='F')
        grid.point_data["moisture_kg_kg"] = q.ravel(order='F')
        
        # Wind speed in dimensionless units
        # Note: For physical units, multiply by (domain_size_m / lx) / dt_solver
        wind_speed = np.sqrt(u**2 + v**2 + w**2)
        grid.point_data["wind_speed_dimensionless"] = wind_speed.ravel(order='F')

        # Save as multiblock for future extensibility
        scene = pv.MultiBlock()
        scene.append(grid, name="PrimaryDataGrid")

        output_filename = os.path.join(self.output_dir_3d, f"scene_{frame:06d}.vtm")
        scene.save(output_filename, binary=True)

    def generate_2d_slice(self, frame, p, u, v):
        """
        Generate 2D wind map at mid-level with pressure contours.
        
        Shows:
        - Pressure field as filled contours
        - Wind vectors colored by intensity (white/yellow/red)
        - Proper wind speed in knots
        
        Args:
            frame: Current simulation frame
            p: 3D pressure field (dimensionless)
            u, v: 3D velocity components (dimensionless)
        """
        # Extract mid-level slice
        z_index = self.sim.nz // 3
        p_slice = p[:, :, z_index]
        u_slice = u[:, :, z_index]
        v_slice = v[:, :, z_index]
        
        # === CONVERT WIND SPEED TO KNOTS FOR DISPLAY ===
        # Step 1: Calculate dimensionless wind speed
        wind_speed_dim = np.sqrt(u_slice**2 + v_slice**2)
        
        # Step 2: Convert to physical velocity (m/s) using U_CHAR
        wind_speed_ms = self.sim.U_CHAR * wind_speed_dim
        
        # Step 3: Convert to knots (meteorological standard)
        wind_speed_slice_kts = ms_to_kts(wind_speed_ms)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Pressure contours
        contour = ax.contourf(p_slice.T, cmap='viridis', levels=20)
        fig.colorbar(contour, ax=ax, label='Pressure (dimensionless)')

        # Wind vectors (subsampled for clarity)
        skip = 8
        x = np.arange(0, self.sim.nx, skip)
        y = np.arange(0, self.sim.ny, skip)

        # Color code by wind intensity (Beaufort-inspired scale)
        wind_colors = np.full(wind_speed_slice_kts[::skip, ::skip].T.shape, 'white', dtype=object)
        wind_colors[wind_speed_slice_kts[::skip, ::skip].T > 34.0] = 'yellow'  # Gale force
        wind_colors[wind_speed_slice_kts[::skip, ::skip].T > 96.0] = 'red'     # Hurricane Cat 2+

        # Plot wind vectors (in dimensionless units for visual consistency)
        ax.quiver(
            x, y,
            u_slice[::skip, ::skip].T,
            v_slice[::skip, ::skip].T,
            color=wind_colors.flatten()
        )

        ax.set_title(f"Wind Map at z-index {z_index} | Frame {frame}")
        ax.set_xlabel("X-index")
        ax.set_ylabel("Y-index")
        ax.set_aspect('equal', adjustable='box')

        output_filename = os.path.join(self.output_dir_3d, f"wind_map_{frame:06d}.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"    VISUALIZER: Saved 2D Wind Map to {output_filename}")
        
    def generate_diagnostic_overlay(self, frame, storm_tracker, data_interface):
        """
        Generate diagnostic overlay map showing:
        - ERA5 steering flow vectors
        - HURDAT2 historical track (green)
        - Simulated track (red)
        - Current storm position (red star)
        
        All tracks are in geographic coordinates (lat/lon).
        
        Args:
            frame: Current simulation frame
            storm_tracker: StormTracker instance with simulated path
            data_interface: DataInterface instance with ERA5 and HURDAT2 data
        """
        print(f"    VISUALIZER: Generating Diagnostic Overlay for frame {frame}...")
        fig, ax = plt.subplots(figsize=(12, 12))

        # Get domain boundaries in geographic coordinates
        lon_min, lon_max = data_interface.lon_bounds
        lat_min, lat_max = data_interface.lat_bounds
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

        # === 1. PLOT ERA5 STEERING VECTORS ===
        # Note: These are dimensionless velocities from the simulation
        # For a geographic map, we plot them as directional indicators
        # (The actual magnitude scaling is handled by the 'scale' parameter)
        u_steer, v_steer = data_interface.u_target, data_interface.v_target
        skip = 8
        lon_coords = np.linspace(lon_min, lon_max, self.sim.nx)
        lat_coords = np.linspace(lat_min, lat_max, self.sim.ny)
        
        ax.quiver(
            lon_coords[::skip], lat_coords[::skip],
            u_steer[::skip, ::skip].T, v_steer[::skip, ::skip].T,
            color='blue', alpha=0.4, scale=6.0,
            label='ERA5 Steering Flow'
        )
        # Note: 'scale' controls visual size, not physical units
        # Larger scale = smaller arrows

        # === 2. PLOT HURDAT2 HISTORICAL TRACK ===
        hist_track = data_interface.historical_track
        ax.plot(
            hist_track['longitude'], 
            hist_track['latitude'], 
            'g-', lw=2, label='HURDAT2 Track'
        )
        ax.plot(
            hist_track['longitude'], 
            hist_track['latitude'], 
            'go', markersize=4
        )

        # === 3. PLOT SIMULATED TRACK ===
        # === CRITICAL FIX V21: Use GEOGRAPHIC coordinates directly ===
        if storm_tracker.storm_path_geo_smoothed:
            path_geo = np.array(storm_tracker.storm_path_geo_smoothed)
            # === END FIX V21 ===
            ax.plot(
                path_geo[:, 0], path_geo[:, 1], 
                'r-', lw=2.5, label='Simulated Track'
            )
            # Mark current position
            ax.plot(
                path_geo[-1, 0], path_geo[-1, 1], 
                'r*', markersize=15
            )

        # Formatting
        ax.set_title(f"Diagnostic Overlay | Frame {frame}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')

        output_filename = os.path.join(self.output_dir_3d, f"diagnostic_overlay_{frame:06d}.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"    VISUALIZER: Saved Diagnostic Overlay to {output_filename}")