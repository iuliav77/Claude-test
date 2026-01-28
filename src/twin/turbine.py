"""
Turbine model with specifications and operational status.

Defines the physical turbine and its constraints.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .config import TurbineConfig


class OperationalStatus(Enum):
    """Turbine operational status."""
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    RATED_POWER = 3
    STOPPING = 4
    EMERGENCY_STOP = 5
    FAULT = 6
    MAINTENANCE = 7


@dataclass(frozen=True)
class PowerCurvePoint:
    """Single point on the power curve."""
    wind_speed_ms: float
    power_kw: float


class Turbine:
    """
    Wind turbine model with specifications and operational constraints.
    
    This is a simplified physics model for Phase 2, focusing on:
    - Correct power curve behavior
    - Reasonable operational constraints
    - Stable state transitions
    """
    
    def __init__(self, config: TurbineConfig):
        """
        Initialize turbine with configuration.
        
        Args:
            config: TurbineConfig with specifications
        """
        self.config = config
        
        # Derived parameters
        self.rotor_area_m2 = np.pi * (config.rotor_diameter_m / 2) ** 2
        self.tip_speed_ratio = 7.0  # Typical for modern turbines
        
    def calculate_power(
        self,
        wind_speed_ms: float,
        air_density_kgm3: float = 1.225
    ) -> float:
        """
        Calculate power output based on wind speed.
        
        Uses simplified power curve with three regions:
        - Region I (v < cut-in): P = 0
        - Region II (cut-in ≤ v < rated): P increases with v³
        - Region III (rated ≤ v < cut-out): P = rated
        - Region IV (v ≥ cut-out): P = 0
        
        Args:
            wind_speed_ms: Wind speed in m/s
            air_density_kgm3: Air density in kg/m³
            
        Returns:
            Power output in kW
        """
        # Region I: Below cut-in
        if wind_speed_ms < self.config.cut_in_speed_ms:
            return 0.0
        
        # Region IV: Above cut-out
        if wind_speed_ms >= self.config.cut_out_speed_ms:
            return 0.0
        
        # Region III: Rated power
        if wind_speed_ms >= self.config.rated_speed_ms:
            return self.config.rated_power_kw
        
        # Region II: Partial load (cubic relationship)
        # Normalize to [0, 1] range
        v_range = self.config.rated_speed_ms - self.config.cut_in_speed_ms
        v_norm = (wind_speed_ms - self.config.cut_in_speed_ms) / v_range
        
        # Cubic power curve (simplified)
        power_fraction = v_norm ** 3
        
        # Apply density correction
        density_correction = air_density_kgm3 / 1.225  # Standard air density
        
        # Apply efficiency
        total_efficiency = (
            self.config.mechanical_efficiency * 
            self.config.electrical_efficiency
        )
        
        power_kw = (
            self.config.rated_power_kw * 
            power_fraction * 
            density_correction * 
            total_efficiency
        )
        
        return max(0.0, power_kw)
    
    def calculate_rotor_speed(self, wind_speed_ms: float) -> float:
        """
        Calculate optimal rotor speed based on wind speed.
        
        Uses tip-speed ratio to determine optimal RPM:
        ω = (TSR × v × 60) / (π × D)
        
        Args:
            wind_speed_ms: Wind speed in m/s
            
        Returns:
            Rotor speed in RPM
        """
        if wind_speed_ms < self.config.cut_in_speed_ms:
            return 0.0
        
        if wind_speed_ms >= self.config.cut_out_speed_ms:
            return 0.0
        
        # Calculate optimal RPM
        rpm = (
            (self.tip_speed_ratio * wind_speed_ms * 60) / 
            (np.pi * self.config.rotor_diameter_m)
        )
        
        # Clamp to operational limits
        rpm = np.clip(rpm, self.config.min_rotor_rpm, self.config.max_rotor_rpm)
        
        return rpm
    
    def determine_status(
        self,
        wind_speed_ms: float,
        power_kw: float,
        is_faulted: bool = False,
        is_maintenance: bool = False
    ) -> OperationalStatus:
        """
        Determine operational status based on conditions.
        
        Args:
            wind_speed_ms: Current wind speed
            power_kw: Current power output
            is_faulted: Whether turbine has a fault
            is_maintenance: Whether in maintenance mode
            
        Returns:
            OperationalStatus
        """
        # Check special states first
        if is_maintenance:
            return OperationalStatus.MAINTENANCE
        
        if is_faulted:
            return OperationalStatus.FAULT
        
        # Check wind conditions
        if wind_speed_ms >= self.config.cut_out_speed_ms:
            return OperationalStatus.EMERGENCY_STOP
        
        if wind_speed_ms < self.config.cut_in_speed_ms:
            return OperationalStatus.STOPPED
        
        # Check power output
        if power_kw < 10:  # Minimal power threshold
            return OperationalStatus.STARTING
        
        if power_kw >= self.config.rated_power_kw * 0.95:
            return OperationalStatus.RATED_POWER
        
        return OperationalStatus.RUNNING
    
    def check_limits(
        self,
        generator_temp_c: float,
        gearbox_temp_c: float,
        vibration_mms: float,
        rotor_rpm: float
    ) -> tuple[bool, Optional[str]]:
        """
        Check if operational limits are violated.
        
        Args:
            generator_temp_c: Generator temperature
            gearbox_temp_c: Gearbox temperature
            vibration_mms: Vibration level
            rotor_rpm: Rotor speed
            
        Returns:
            Tuple of (is_ok, fault_message)
        """
        # Temperature checks
        if generator_temp_c > self.config.max_generator_temp_c:
            return False, f"Generator temperature too high: {generator_temp_c:.1f}°C"
        
        if gearbox_temp_c > self.config.max_gearbox_temp_c:
            return False, f"Gearbox temperature too high: {gearbox_temp_c:.1f}°C"
        
        # Vibration check
        if vibration_mms > self.config.max_vibration_mms:
            return False, f"Excessive vibration: {vibration_mms:.2f} mm/s"
        
        # Speed check
        if rotor_rpm > self.config.max_rotor_rpm:
            return False, f"Rotor overspeed: {rotor_rpm:.1f} RPM"
        
        return True, None
    
    def get_power_curve(
        self,
        wind_speeds: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate complete power curve for plotting.
        
        Args:
            wind_speeds: Array of wind speeds. If None, uses default range.
            
        Returns:
            Tuple of (wind_speeds, power_outputs)
        """
        if wind_speeds is None:
            wind_speeds = np.linspace(0, 30, 100)
        
        power_outputs = np.array([
            self.calculate_power(v) for v in wind_speeds
        ])
        
        return wind_speeds, power_outputs


if __name__ == "__main__":
    # Test turbine model
    from .config import get_config
    
    config = get_config()
    turbine = Turbine(config.turbine)
    
    print("Turbine Model Test")
    print("=" * 50)
    print(f"Turbine: {config.turbine.name}")
    print(f"Rated Power: {config.turbine.rated_power_kw} kW")
    print(f"Rotor Diameter: {config.turbine.rotor_diameter_m} m")
    print()
    
    # Test power calculations
    test_speeds = [0, 2, 3, 5, 8, 12, 15, 20, 25, 30]
    print("Power Curve:")
    print(f"{'Wind Speed (m/s)':<20} {'Power (kW)':<15} {'RPM':<10} {'Status'}")
    print("-" * 70)
    
    for v in test_speeds:
        power = turbine.calculate_power(v)
        rpm = turbine.calculate_rotor_speed(v)
        status = turbine.determine_status(v, power)
        print(f"{v:<20.1f} {power:<15.0f} {rpm:<10.1f} {status.name}")
