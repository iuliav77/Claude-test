"""
SCADA data generator for wind turbine simulation.

Generates realistic synthetic SCADA data including:
- Power output based on wind speed
- Rotor speed and generator parameters
- Temperature profiles
- Vibration levels
- Operational status
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class SCADAGenerator:
    """
    Generate synthetic SCADA data based on weather conditions.
    
    Simulates realistic turbine behavior including:
    - Power curve following physics
    - Temperature dynamics
    - Vibration patterns
    - Occasional anomalies
    """
    
    def __init__(
        self,
        rated_power_kw: float = 2500,
        rotor_diameter_m: float = 90,
        hub_height_m: float = 80,
        cut_in_speed: float = 3.0,
        rated_speed: float = 12.0,
        cut_out_speed: float = 25.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the SCADA generator with turbine specifications.
        
        Args:
            rated_power_kw: Maximum power output in kW
            rotor_diameter_m: Rotor diameter in meters
            hub_height_m: Hub height in meters
            cut_in_speed: Minimum wind speed for operation (m/s)
            rated_speed: Wind speed at rated power (m/s)
            cut_out_speed: Maximum wind speed for operation (m/s)
            seed: Random seed for reproducibility
        """
        self.rated_power_kw = rated_power_kw
        self.rotor_diameter_m = rotor_diameter_m
        self.hub_height_m = hub_height_m
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Calculate derived parameters
        self.rotor_area = np.pi * (rotor_diameter_m / 2) ** 2
        self.tip_speed_ratio = 7.0  # Typical for modern turbines
        self.power_coefficient = 0.45  # Betz limit consideration
        
    def generate(
        self,
        weather_data: pd.DataFrame,
        anomaly_probability: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate SCADA data based on weather conditions.
        
        Args:
            weather_data: DataFrame with weather parameters
            anomaly_probability: Probability of anomaly at each timestep
            
        Returns:
            DataFrame with SCADA measurements
        """
        periods = len(weather_data)
        
        # Extract weather parameters
        wind_speed = weather_data['wind_speed_ms'].values
        temperature = weather_data['temperature_c'].values
        air_density = weather_data['air_density_kgm3'].values
        
        # Calculate power output
        power_output = self._calculate_power(wind_speed, air_density)
        
        # Calculate rotor speed
        rotor_speed_rpm = self._calculate_rotor_speed(wind_speed)
        
        # Calculate temperatures
        generator_temp, gearbox_temp = self._calculate_temperatures(
            power_output, temperature, periods
        )
        
        # Calculate vibration
        vibration = self._calculate_vibration(rotor_speed_rpm, power_output)
        
        # Determine operational status
        status = self._determine_status(wind_speed, power_output)
        
        # Inject anomalies
        if anomaly_probability > 0:
            power_output, generator_temp, vibration, status = self._inject_anomalies(
                power_output, generator_temp, vibration, status, anomaly_probability
            )
        
        # Calculate nacelle position (yaw angle follows wind direction)
        nacelle_position = self._calculate_nacelle_position(
            weather_data['wind_direction_deg'].values
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': weather_data['timestamp'],
            'power_kw': power_output,
            'rotor_speed_rpm': rotor_speed_rpm,
            'generator_temp_c': generator_temp,
            'gearbox_temp_c': gearbox_temp,
            'nacelle_position_deg': nacelle_position,
            'vibration_mms': vibration,
            'status': status
        })
        
        return df
    
    def _calculate_power(
        self,
        wind_speed: np.ndarray,
        air_density: np.ndarray
    ) -> np.ndarray:
        """
        Calculate power output using simplified power curve.
        
        Power curve regions:
        1. Below cut-in: 0 kW
        2. Cut-in to rated: Cubic relationship
        3. Rated to cut-out: Rated power
        4. Above cut-out: 0 kW (shutdown)
        """
        power = np.zeros_like(wind_speed)
        
        # Region 2: Cut-in to rated (cubic curve)
        region2 = (wind_speed >= self.cut_in_speed) & (wind_speed < self.rated_speed)
        if region2.any():
            # Simplified power calculation
            # P = 0.5 * ρ * A * Cp * v³
            v_cubed = wind_speed[region2] ** 3 - self.cut_in_speed ** 3
            max_v_cubed = self.rated_speed ** 3 - self.cut_in_speed ** 3
            power[region2] = self.rated_power_kw * (v_cubed / max_v_cubed)
        
        # Region 3: Rated to cut-out
        region3 = (wind_speed >= self.rated_speed) & (wind_speed < self.cut_out_speed)
        power[region3] = self.rated_power_kw
        
        # Add realistic noise and efficiency variations
        efficiency = self.rng.normal(1.0, 0.02, len(power))
        power *= efficiency
        
        # Add measurement noise
        noise = self.rng.normal(0, 5, len(power))
        power += noise
        
        # Ensure non-negative
        power = np.maximum(power, 0)
        
        return power
    
    def _calculate_rotor_speed(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        Calculate rotor speed based on wind speed and tip-speed ratio.
        """
        # Rotor speed in RPM
        # ω = (TSR * v * 60) / (π * D)
        rpm = np.zeros_like(wind_speed)
        
        operating = wind_speed >= self.cut_in_speed
        rpm[operating] = (
            (self.tip_speed_ratio * wind_speed[operating] * 60) / 
            (np.pi * self.rotor_diameter_m)
        )
        
        # Typical range: 6-18 RPM for large turbines
        rpm = np.clip(rpm, 0, 20)
        
        # Add noise
        rpm += self.rng.normal(0, 0.1, len(rpm))
        
        return rpm
    
    def _calculate_temperatures(
        self,
        power_output: np.ndarray,
        ambient_temp: np.ndarray,
        periods: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate generator and gearbox temperatures.
        
        Temperature depends on:
        - Ambient temperature
        - Power output (heat generation)
        - Thermal inertia (slow changes)
        """
        # Initialize temperatures at ambient + offset
        generator_temp = np.zeros(periods)
        gearbox_temp = np.zeros(periods)
        
        generator_temp[0] = ambient_temp[0] + 40
        gearbox_temp[0] = ambient_temp[0] + 35
        
        # Thermal time constant (in timesteps)
        tau = 20
        
        for i in range(1, periods):
            # Heat generation proportional to power
            heat_gen = power_output[i] / self.rated_power_kw * 30
            
            # Temperature update with thermal inertia
            generator_temp[i] = (
                generator_temp[i-1] * (1 - 1/tau) + 
                (ambient_temp[i] + heat_gen) * (1/tau)
            )
            
            gearbox_temp[i] = (
                gearbox_temp[i-1] * (1 - 1/tau) + 
                (ambient_temp[i] + heat_gen * 0.8) * (1/tau)
            )
        
        # Add noise
        generator_temp += self.rng.normal(0, 1.0, periods)
        gearbox_temp += self.rng.normal(0, 0.8, periods)
        
        # Realistic bounds
        generator_temp = np.clip(generator_temp, ambient_temp - 5, 110)
        gearbox_temp = np.clip(gearbox_temp, ambient_temp - 5, 100)
        
        return generator_temp, gearbox_temp
    
    def _calculate_vibration(
        self,
        rotor_speed: np.ndarray,
        power_output: np.ndarray
    ) -> np.ndarray:
        """
        Calculate vibration levels based on operation.
        
        Vibration increases with speed and load.
        """
        # Base vibration when operating
        base_vibration = 0.5
        
        # Additional vibration from speed and power
        speed_factor = rotor_speed / 15.0  # Normalized
        power_factor = power_output / self.rated_power_kw
        
        vibration = base_vibration + 2.0 * speed_factor + 1.5 * power_factor
        
        # Random variation
        vibration += self.rng.normal(0, 0.3, len(vibration))
        
        # Ensure non-negative and reasonable
        vibration = np.clip(vibration, 0, 10)
        
        return vibration
    
    def _determine_status(
        self,
        wind_speed: np.ndarray,
        power_output: np.ndarray
    ) -> np.ndarray:
        """
        Determine operational status at each timestep.
        
        Status codes:
        0: Stopped (no wind or maintenance)
        1: Starting
        2: Running
        3: Stopping
        4: Fault
        """
        status = np.zeros(len(wind_speed), dtype=int)
        
        # Running when producing power
        running = power_output > 10
        status[running] = 2
        
        # Stopped when wind too low or too high
        stopped = (wind_speed < self.cut_in_speed) | (wind_speed >= self.cut_out_speed)
        status[stopped] = 0
        
        return status
    
    def _inject_anomalies(
        self,
        power: np.ndarray,
        temperature: np.ndarray,
        vibration: np.ndarray,
        status: np.ndarray,
        probability: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject realistic anomalies into the data.
        
        Anomaly types:
        1. Power degradation
        2. Temperature spike
        3. Excessive vibration
        4. Unexpected shutdown
        """
        anomalies = self.rng.random(len(power)) < probability
        
        for i in np.where(anomalies)[0]:
            anomaly_type = self.rng.integers(0, 4)
            
            if anomaly_type == 0:  # Power degradation
                duration = self.rng.integers(5, 20)
                end = min(i + duration, len(power))
                power[i:end] *= self.rng.uniform(0.6, 0.85)
                
            elif anomaly_type == 1:  # Temperature spike
                duration = self.rng.integers(10, 30)
                end = min(i + duration, len(temperature))
                temperature[i:end] += self.rng.uniform(15, 30)
                
            elif anomaly_type == 2:  # Excessive vibration
                duration = self.rng.integers(3, 15)
                end = min(i + duration, len(vibration))
                vibration[i:end] *= self.rng.uniform(2.0, 4.0)
                
            elif anomaly_type == 3:  # Unexpected shutdown
                duration = self.rng.integers(5, 20)
                end = min(i + duration, len(status))
                status[i:end] = 4  # Fault
                power[i:end] = 0
        
        return power, temperature, vibration, status
    
    def _calculate_nacelle_position(self, wind_direction: np.ndarray) -> np.ndarray:
        """
        Calculate nacelle yaw position following wind direction.
        
        Nacelle tracks wind direction with some lag.
        """
        position = np.zeros_like(wind_direction)
        position[0] = wind_direction[0]
        
        # Yaw tracking with lag
        alpha = 0.95
        for i in range(1, len(wind_direction)):
            position[i] = alpha * position[i-1] + (1-alpha) * wind_direction[i]
        
        # Wrap to 0-360
        position = position % 360
        
        return position
    
    def save(self, df: pd.DataFrame, filepath: str):
        """Save SCADA data to CSV."""
        df.to_csv(filepath, index=False)
        print(f"SCADA data saved to: {filepath}")
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics for the generated data."""
        stats = {
            'mean_power_kw': df['power_kw'].mean(),
            'max_power_kw': df['power_kw'].max(),
            'capacity_factor': df['power_kw'].mean() / self.rated_power_kw,
            'total_energy_mwh': df['power_kw'].sum() * (10/60) / 1000,  # 10-min intervals
            'operating_time_pct': (df['status'] == 2).sum() / len(df) * 100,
            'fault_time_pct': (df['status'] == 4).sum() / len(df) * 100,
            'samples': len(df)
        }
        return stats


if __name__ == "__main__":
    # Example usage
    from weather_generator import WeatherGenerator
    
    # Generate weather data first
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate(
        start_date="2024-01-01",
        days=7,
        interval_minutes=10
    )
    
    # Generate SCADA data
    scada_gen = SCADAGenerator(seed=42)
    scada_data = scada_gen.generate(weather_data, anomaly_probability=0.05)
    
    print("Generated SCADA data:")
    print(scada_data.head())
    print("\nStatistics:")
    print(scada_gen.get_statistics(scada_data))
