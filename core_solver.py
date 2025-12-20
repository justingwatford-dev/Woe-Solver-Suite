import numpy as np
import sys

# === GPU ACCELERATION TOGGLE ===
USE_GPU = True

try:
    if USE_GPU:
        import cupy as xp
        import cupyx.scipy.ndimage as ndimage
        import cupyx.scipy.fft as fft
        print(f"[{__name__}] 🚀 GPU Acceleration ENABLED (CuPy)")
    else:
        raise ImportError
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndimage
    import scipy.fft as fft
    print(f"[{__name__}] 🐢 GPU Acceleration DISABLED (NumPy)")


class CoreSolver:
    def __init__(self, sim_instance):
        self.sim = sim_instance
        self.kx = xp.fft.fftfreq(self.sim.nx, d=self.sim.dx)
        self.ky = xp.fft.fftfreq(self.sim.ny, d=self.sim.dy)
        self.kz = xp.fft.fftfreq(self.sim.nz, d=self.sim.dz)
        self.k_squared = self.kx[:, xp.newaxis, xp.newaxis]**2 + \
                         self.ky[xp.newaxis, :, xp.newaxis]**2 + \
                         self.kz[xp.newaxis, xp.newaxis, :]**2
        # PATCH V38: k=0 mode is now handled properly in project() method
        
        self.grid_points = xp.mgrid[0:self.sim.nx, 0:self.sim.ny, 0:self.sim.nz]

    def gradient_x(self, f):
        return fft.ifftn(1j * self.kx[:, xp.newaxis, xp.newaxis] * fft.fftn(f)).real

    def gradient_y(self, f):
        return fft.ifftn(1j * self.ky[xp.newaxis, :, xp.newaxis] * fft.fftn(f)).real

    def gradient_z(self, f):
        return fft.ifftn(1j * self.kz[xp.newaxis, xp.newaxis, :] * fft.fftn(f)).real

    def laplacian(self, f):
        return fft.ifftn(-self.k_squared * fft.fftn(f)).real

    def advect(self, f, u, v, w):
        departure_x = self.grid_points[0] - u * self.sim.dt_solver / self.sim.dx
        departure_y = self.grid_points[1] - v * self.sim.dt_solver / self.sim.dy
        departure_z = self.grid_points[2] - w * self.sim.dt_solver / self.sim.dz
        
        departure_points = xp.array([departure_x, departure_y, departure_z])
        
        f_advected = ndimage.map_coordinates(f, departure_points, order=1, mode='wrap')
        
        return f_advected

    def project(self, u, v, w, damping_factor_h, damping_factor_w):
        """
        PATCH V38: SPECTRAL POISSON SOLVER (The Iron Foundation).
        Solves ∇²p = ∇·u to enforce incompressibility.
        
        V50.5 UPDATE: Surgical Governor application.
        """
        
        # 1. SEPARATE MEAN FLOW (The "DC Component")
        # The solver only works on the fluctuating component.
        # We must preserve the mean flow (translation) separately.
        u_mean = xp.mean(u)
        v_mean = xp.mean(v)
        w_mean = xp.mean(w)
        
        u -= u_mean
        v -= v_mean
        w -= w_mean

        # 2. COMPUTE DIVERGENCE
        divergence = self.gradient_x(u) + self.gradient_y(v) + self.gradient_z(w)
        
        # 3. TRANSFORM TO FREQUENCY SPACE
        div_hat = fft.fftn(divergence)
        
        # 4. ENFORCE GLOBAL MASS CONSERVATION
        # The total divergence of the domain must be zero for incompressible flow.
        div_hat[0, 0, 0] = 0.0
        
        # 5. SOLVE POISSON EQUATION: k² p_hat = -div_hat
        k_squared_safe = self.k_squared.copy()
        k_squared_safe[0, 0, 0] = 1.0  # Dummy value to avoid Inf/NaN
        
        p_hat = -div_hat / k_squared_safe
        
        # 6. SET MEAN PRESSURE GAUGE
        p_hat[0, 0, 0] = 0.0
        
        # 7. TRANSFORM BACK TO PHYSICAL SPACE
        p = fft.ifftn(p_hat).real
        
        # 8. APPLY PRESSURE CORRECTION (Project onto divergence-free space)
        inv_rho = 1.0 / self.sim.rho
        
        u -= damping_factor_h * inv_rho * self.gradient_x(p)
        v -= damping_factor_h * inv_rho * self.gradient_y(p)
        w -= damping_factor_w * inv_rho * self.gradient_z(p)

        # === V50.5 PATCH: SURGICAL INTENSITY GOVERNOR ===
        # === V53.1 UPDATE: Proper 3D Velocity Clamping ===
        # === V54 UPDATE: Stronger Damping (Gemini's Fix) ===
        # MOVED UP: We apply the governor to the PERTURBATION velocity (spin)
        # BEFORE adding back the mean flow (steering). 
        # This prevents "The Parking Brake Effect" where high winds killed forward motion.
        
        # V53.1 FIX: The original V50.5 clamped each component separately,
        # but this allowed the TOTAL 3D magnitude to exceed the cap.
        # Solution: Clamp the velocity VECTOR magnitude, not components.
        
        # V54 FIX: Strengthen progressive damping to prevent oscillations
        # (Gemini's diagnosis: ±40kt oscillations from too-soft damping)
        
        MAX_REALISTIC_WIND_MS = 95.0  # ~185 kts (Strongest observed Atlantic hurricane)
        EMERGENCY_DAMPING_THRESHOLD = 85.0  # ~165 kts (Start damping at strong H5)
        
        # Calculate current max 3D spin magnitude (perturbation)
        velocity_magnitude_3d = xp.sqrt(u**2 + v**2 + w**2)
        max_spin_ms = float(xp.max(velocity_magnitude_3d) * self.sim.U_CHAR)
        
        if max_spin_ms > EMERGENCY_DAMPING_THRESHOLD:
            # === V54 DAMPING ENHANCEMENT (Gemini's Fix) ===
            # Diagnosis: ±40kt oscillations at H5 intensity
            # Cause: Damping ramp (0.15) too gentle, acting like a soft spring
            # Fix: Stiffen the spring to 0.35 to prevent bounce-back
            
            overshoot = max_spin_ms - EMERGENCY_DAMPING_THRESHOLD
            excess_ratio = overshoot / EMERGENCY_DAMPING_THRESHOLD
            
            # V54: Stronger progressive damping (0.35 max vs 0.15)
            # Steepened slope (0.7 vs 0.3) - "Stiffer spring"
            emergency_damping = 1.0 - min(0.35, excess_ratio * 0.7)
            
            u *= emergency_damping
            v *= emergency_damping
            w *= emergency_damping
            # === END V54 DAMPING ENHANCEMENT ===
            
            # Recalculate after damping
            velocity_magnitude_3d = xp.sqrt(u**2 + v**2 + w**2)
            max_spin_ms = float(xp.max(velocity_magnitude_3d) * self.sim.U_CHAR)
            
            if max_spin_ms > MAX_REALISTIC_WIND_MS:
                # === V53.1 VECTOR MAGNITUDE CLAMPING (PRESERVED!) ===
                # Scale the velocity VECTOR to cap at MAX_REALISTIC_WIND_MS
                velocity_magnitude_3d_physical = velocity_magnitude_3d * self.sim.U_CHAR
                
                # Create scale factor FIELD: only where velocity exceeds cap
                scale_factor = xp.where(
                    velocity_magnitude_3d_physical > MAX_REALISTIC_WIND_MS,
                    MAX_REALISTIC_WIND_MS / (velocity_magnitude_3d_physical + 1e-12),
                    1.0
                )
                
                u *= scale_factor
                v *= scale_factor
                w *= scale_factor
                
                # Log when actively clamping (not just damping)
                if max_spin_ms > MAX_REALISTIC_WIND_MS:
                     print(f"    ⚠️ V54 SURGICAL GOVERNOR: Clamped 3D Spin {max_spin_ms:.1f} -> {MAX_REALISTIC_WIND_MS:.1f} m/s")
                # === END V53.1 VECTOR CLAMPING ===
        # === END V50.5/V53.1/V54 PATCH ===
        
        # 9. RESTORE MEAN FLOW (Steering is preserved 100%)
        u += u_mean
        v += v_mean
        w += w_mean
        
        return u, v, w, p


    def get_max_velocity(self, u, v, w):
        # Add a check for NaN/Inf values
        if not xp.all(xp.isfinite(u)):
            return xp.inf
        return float(xp.max(xp.sqrt(u**2 + v**2 + w**2)))

    def curl(self, u, v, w):
        dw_dy = self.gradient_y(w); dv_dz = self.gradient_z(v)
        du_dz = self.gradient_z(u); dw_dx = self.gradient_x(w)
        dv_dx = self.gradient_x(v); du_dy = self.gradient_y(u)
        return (dw_dy - dv_dz, du_dz - dw_dx, dv_dx - du_dy)

    def generate_3d_divergence_free_noise(self, amplitude):
        potential_shape = (self.sim.nx // 4, self.sim.ny // 4, self.sim.nz // 4)
        coords = xp.mgrid[0:self.sim.nx, 0:self.sim.ny, 0:self.sim.nz] / 4.0
        
        if USE_GPU:
             noise_x = xp.random.randn(*potential_shape)
             noise_y = xp.random.randn(*potential_shape)
             noise_z = xp.random.randn(*potential_shape)
        else:
             noise_x = xp.asarray(np.random.randn(*potential_shape))
             noise_y = xp.asarray(np.random.randn(*potential_shape))
             noise_z = xp.asarray(np.random.randn(*potential_shape))

        Ax = ndimage.map_coordinates(noise_x, coords, order=3, mode='wrap')
        Ay = ndimage.map_coordinates(noise_y, coords, order=3, mode='wrap')
        Az = ndimage.map_coordinates(noise_z, coords, order=3, mode='wrap')
        
        u_turb, v_turb, w_turb = self.curl(Ax, Ay, Az)
        norm_factor = amplitude / (xp.mean(xp.sqrt(u_turb**2 + v_turb**2 + w_turb**2)) + 1e-12)
        return u_turb * norm_factor, v_turb * norm_factor, w_turb * norm_factor