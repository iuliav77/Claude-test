"""
Continuous Weather Generator for v1.0 Real-Time Digital Twin

Generates smooth, realistic weather transitions suitable for live monitoring.
Key features:
- Continuous generation (not batch)
- Wind continuity with smoothing
- Thermal inertia simulation
- No sudden jumps
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any


class ContinuousWeatherGenerator:
    """
    Generates weather data continuously with realistic transitions.
    
    v1.0 Requirements Met:
    - Wind continuity (exponential smoothing)
    - No random jumps
    - Thermal gradual changes
    - Diurnal patterns
    """
    
    def __init__(self, seed: int = 42, location: str = "offshore"):
        """
        Initialize continuous weather generator.
        
        Args:
            seed: Random seed for reproducibility
            location: Location preset
        """
        self.rng = np.random.default_rng(seed)
        self.location = location
        
        # Location parameters (define BEFORE generating targets)
        self.mean_wind_speed = 9.5
        self.wind_std = 3.5
        self.mean_temp = 15.0
        self.temp_amplitude = 8.0  # Diurnal variation
        
        # Current state (for continuity)
        self.current_wind_speed = 8.0  # m/s
        self.current_wind_direction = 270.0  # degrees
        self.current_temperature = 15.0  # °C
        self.current_pressure = 1013.0  # hPa
        
        # Target values (where weather is heading)
        self._generate_new_targets()
        
        # Smoothing factors (0.0 = no change, 1.0 = instant)
        self.wind_speed_smoothing = 0.15  # Slow wind changes
        self.wind_direction_smoothing = 0.10  # Very slow direction changes
        self.temperature_smoothing = 0.05  # Very slow temp changes
        self.pressure_smoothing = 0.08  # Slow pressure changes
        
        # Change targets periodically
        self.steps_since_target_change = 0
        self.target_change_interval = 12  # Change targets every 12 steps (~2 hours)
    
    def _generate_new_targets(self):
        """Generate new target values for weather to transition towards."""
        # Wind speed target (Weibull-like distribution)
        self.target_wind_speed = max(0.0, min(25.0, 
            self.rng.normal(self.mean_wind_speed, self.wind_std)
        ))
        
        # Wind direction target (with some persistence)
        direction_change = self.rng.normal(0, 30)
        self.target_wind_direction = (self.current_wind_direction + direction_change) % 360
        
        # Temperature target (with diurnal pattern)
        hour = datetime.now().hour
        diurnal_offset = self.temp_amplitude * np.sin((hour - 6) * np.pi / 12)
        self.target_temperature = self.mean_temp + diurnal_offset + self.rng.normal(0, 2)
        
        # Pressure target
        self.target_pressure = 1013.0 + self.rng.normal(0, 10)
    
    def get_next_sample(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Generate next weather sample with smooth transitions.
        
        Args:
            timestamp: Current simulation time
            
        Returns:
            Dictionary with weather data
        """
        # Periodically generate new targets
        self.steps_since_target_change += 1
        if self.steps_since_target_change >= self.target_change_interval:
            self._generate_new_targets()
            self.steps_since_target_change = 0
        
        # Smooth transition towards targets (exponential smoothing)
        self.current_wind_speed += (
            self.target_wind_speed - self.current_wind_speed
        ) * self.wind_speed_smoothing
        
        self.current_wind_direction += (
            self._angle_difference(self.target_wind_direction, self.current_wind_direction)
        ) * self.wind_direction_smoothing
        self.current_wind_direction = self.current_wind_direction % 360
        
        self.current_temperature += (
            self.target_temperature - self.current_temperature
        ) * self.temperature_smoothing
        
        self.current_pressure += (
            self.target_pressure - self.current_pressure
        ) * self.pressure_smoothing
        
        # Add small noise for realism
        wind_speed_with_noise = max(0.0, self.current_wind_speed + self.rng.normal(0, 0.3))
        wind_direction_with_noise = (self.current_wind_direction + self.rng.normal(0, 2)) % 360
        
        # Calculate air density
        temp_kelvin = self.current_temperature + 273.15
        air_density = (self.current_pressure * 100) / (287.05 * temp_kelvin)
        
        return {
            'timestamp': timestamp,
            'wind_speed_ms': round(wind_speed_with_noise, 2),
            'wind_direction_deg': round(wind_direction_with_noise, 1),
            'temperature_c': round(self.current_temperature, 1),
            'pressure_hpa': round(self.current_pressure, 1),
            'air_density_kgm3': round(air_density, 3)
        }
    
    def _angle_difference(self, target: float, current: float) -> float:
        """
        Calculate shortest angular difference (handles 359° → 1° wrap).
        
        Args:
            target: Target angle (degrees)
            current: Current angle (degrees)
            
        Returns:
            Shortest difference (-180 to +180)
        """
        diff = target - current
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff
