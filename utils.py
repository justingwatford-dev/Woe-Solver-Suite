"""
===============================================================================
ORACLE V4 - UNIT CONVERSION UTILITIES
===============================================================================

This module provides ALL unit conversions for the hurricane simulation system.
The goal: Keep internal physics in pure SI units, convert only at boundaries.

INTERNAL PHYSICS UNITS (SI):
    - Velocity: m/s
    - Distance: m (meters)
    - Pressure: Pa
    - Temperature: K
    - Time: s
    - Mass: kg

EXTERNAL INTERFACE UNITS:
    - Wind speed: knots (for meteorological convention)
    - Distance: km (for reporting)
    - Track positions: degrees lat/lon

DOMAIN GEOMETRY:
    - Simulation domain: dimensionless [0, lx] × [0, ly] × [0, lz]
    - Physical domain: 2000 km × 2000 km × 20 km (typical)
    - Grid spacing scales with domain size

===============================================================================
"""

import numpy as np

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

# Earth
EARTH_RADIUS_M = 6.371e6  # meters
EARTH_RADIUS_KM = 6371.0  # kilometers

# Atmospheric
OMEGA_EARTH = 7.2921e-5   # Earth's angular velocity (rad/s)
GRAVITY = 9.81            # m/s²
SPECIFIC_HEAT_AIR = 1004.0  # J/(kg·K) at constant pressure
LATENT_HEAT_VAPORIZATION = 2.4e6  # J/kg

# ============================================================================
# VELOCITY CONVERSIONS
# ============================================================================

def kts_to_ms(speed_kts):
    """
    Convert wind speed from knots to meters per second.
    
    Args:
        speed_kts: Wind speed in knots
        
    Returns:
        Wind speed in m/s
        
    Note:
        1 knot = 1.852 km/h = 0.51444... m/s
    """
    return speed_kts * 0.51444

def ms_to_kts(speed_ms):
    """
    Convert wind speed from meters per second to knots.
    
    Args:
        speed_ms: Wind speed in m/s
        
    Returns:
        Wind speed in knots
    """
    return speed_ms / 0.51444

# ============================================================================
# DISTANCE CONVERSIONS
# ============================================================================

def km_to_m(distance_km):
    """Convert kilometers to meters."""
    return distance_km * 1000.0

def m_to_km(distance_m):
    """Convert meters to kilometers."""
    return distance_m / 1000.0

def deg_to_km(distance_deg, latitude=25.0):
    """
    Convert distance in degrees to kilometers.
    
    Args:
        distance_deg: Distance in degrees
        latitude: Reference latitude (default: 25°N, typical for Atlantic hurricanes)
        
    Returns:
        Distance in kilometers
        
    Note:
        1 degree latitude ≈ 111 km everywhere
        1 degree longitude varies with latitude: 111 km * cos(latitude)
    """
    return distance_deg * 111.0

def km_to_deg(distance_km):
    """
    Convert distance in kilometers to degrees (approximate).
    
    Args:
        distance_km: Distance in kilometers
        
    Returns:
        Distance in degrees
        
    Note:
        Uses 111 km/degree as standard approximation
    """
    return distance_km / 111.0

# ============================================================================
# HAVERSINE DISTANCE (Great Circle)
# ============================================================================

def haversine_distance_km(lon1, lat1, lon2, lat2):
    """
    Calculate great circle distance between two points on Earth.
    
    Args:
        lon1, lat1: Longitude and latitude of point 1 (degrees)
        lon2, lat2: Longitude and latitude of point 2 (degrees)
        
    Returns:
        Distance in kilometers
        
    Note:
        Uses the haversine formula for accuracy over long distances.
        More accurate than simple degree-to-km conversion.
    """
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return EARTH_RADIUS_KM * c

# ============================================================================
# DOMAIN SCALING (Dimensionless ↔ Physical)
# ============================================================================

class DomainScaler:
    """
    Handles conversions between dimensionless simulation domain and physical space.
    
    The simulation operates on a dimensionless domain [0, lx] × [0, ly] × [0, lz].
    This class converts between dimensionless coordinates and physical distances.
    
    Example:
        >>> scaler = DomainScaler(lx=2.0, ly=2.0, lz=1.0, 
        ...                       physical_lx_km=2000.0, physical_ly_km=2000.0, physical_lz_km=20.0)
        >>> # Convert 1.0 dimensionless units to physical meters
        >>> physical_distance = scaler.dimensionless_to_physical(1.0, axis='x')
    """
    
    def __init__(self, lx, ly, lz, physical_lx_km, physical_ly_km, physical_lz_km):
        """
        Initialize domain scaler.
        
        Args:
            lx, ly, lz: Dimensionless domain extents
            physical_lx_km, physical_ly_km, physical_lz_km: Physical domain sizes in km
        """
        self.lx = lx
        self.ly = ly
        self.lz = lz
        
        # Physical domain sizes in meters
        self.physical_lx_m = km_to_m(physical_lx_km)
        self.physical_ly_m = km_to_m(physical_ly_km)
        self.physical_lz_m = km_to_m(physical_lz_km)
        
        # Scaling factors (meters per dimensionless unit)
        self.scale_x = self.physical_lx_m / lx
        self.scale_y = self.physical_ly_m / ly
        self.scale_z = self.physical_lz_m / lz
        
    def dimensionless_to_physical_x(self, x_dim):
        """Convert dimensionless x-coordinate to physical distance (meters)."""
        return x_dim * self.scale_x
    
    def dimensionless_to_physical_y(self, y_dim):
        """Convert dimensionless y-coordinate to physical distance (meters)."""
        return y_dim * self.scale_y
    
    def dimensionless_to_physical_z(self, z_dim):
        """Convert dimensionless z-coordinate to physical distance (meters)."""
        return z_dim * self.scale_z
    
    def physical_to_dimensionless_x(self, x_phys_m):
        """Convert physical distance (meters) to dimensionless x-coordinate."""
        return x_phys_m / self.scale_x
    
    def physical_to_dimensionless_y(self, y_phys_m):
        """Convert physical distance (meters) to dimensionless y-coordinate."""
        return y_phys_m / self.scale_y
    
    def physical_to_dimensionless_z(self, z_phys_m):
        """Convert physical distance (meters) to dimensionless z-coordinate."""
        return z_phys_m / self.scale_z

# ============================================================================
# VALIDATION & TESTING
# ============================================================================

def validate_conversions():
    """
    Run self-tests to validate conversion functions.
    Prints results and raises AssertionError if any test fails.
    """
    print("\n" + "="*80)
    print("VALIDATING UNIT CONVERSIONS")
    print("="*80)
    
    # Test velocity conversions
    print("\n1. Velocity Conversions:")
    v_kts = 50.0
    v_ms = kts_to_ms(v_kts)
    v_kts_back = ms_to_kts(v_ms)
    print(f"   50 knots = {v_ms:.4f} m/s")
    print(f"   {v_ms:.4f} m/s = {v_kts_back:.4f} knots")
    assert abs(v_kts - v_kts_back) < 1e-6, "Velocity round-trip failed!"
    print("   ✓ Velocity conversions validated")
    
    # Test distance conversions
    print("\n2. Distance Conversions:")
    d_km = 100.0
    d_m = km_to_m(d_km)
    d_km_back = m_to_km(d_m)
    print(f"   100 km = {d_m:.1f} m")
    assert abs(d_km - d_km_back) < 1e-6, "Distance round-trip failed!"
    print("   ✓ Distance conversions validated")
    
    # Test haversine
    print("\n3. Haversine Distance:")
    lat1, lon1 = 25.0, -80.0  # Miami
    lat2, lon2 = 30.0, -85.0  # Near Tallahassee
    dist = haversine_distance_km(lon1, lat1, lon2, lat2)
    print(f"   Miami to Tallahassee: {dist:.1f} km")
    assert 700 < dist < 800, "Haversine distance seems wrong!"
    print("   ✓ Haversine formula validated")
    
    # Test domain scaling
    print("\n4. Domain Scaling:")
    domain_scaler = DomainScaler(
        lx=2.0, ly=2.0, lz=1.0,
        physical_lx_km=2000.0, physical_ly_km=2000.0, physical_lz_km=20.0
    )
    
    x_dim = 1.0
    x_phys_m = domain_scaler.dimensionless_to_physical_x(x_dim)
    x_phys_km = m_to_km(x_phys_m)
    print(f"   1.0 dimensionless unit = {x_phys_km:.1f} km")
    assert abs(x_phys_km - 1000.0) < 1, "Domain scaling failed!"
    print("   ✓ Domain scaling validated")
    
    print("\n" + "="*80)
    print("ALL VALIDATIONS PASSED ✓")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Run validation when executed directly
    validate_conversions()