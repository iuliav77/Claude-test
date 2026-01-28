"""
Weather data generator for wind turbine simulation.

Generates realistic synthetic weather data including:
- Wind speed and direction
- Temperature and pressure
- Diurnal and seasonal patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple


class WeatherGenerator:
    """
    Generate synthetic weather data with realistic patterns.
    
    The generator simulates:
    - Wind speed with Weibull distribution
    - Wind direction with directional bias
    - Temperature with diurnal and seasonal cycles
    - Atmospheric pressure variations
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        location: str = "offshore_north_sea"
    ):
        """
        Initialize the weather generator.
        
        Args:
            seed: Random seed for reproducibility
            location: Location preset (affects wind patterns)
        """
        self.seed = seed
        self.location = location
        self.rng = np.random.default_rng(seed)
        
        # Location-specific parameters
        self.params = self._get_location_params(location)
        
    def _get_location_params(self, location: str) -> dict:
        """Get weather parameters based on location."""
        locations = {
            "offshore_north_sea": {
                "mean_wind_speed": 9.5,  # m/s
                "wind_speed_std": 3.5,
                "prevailing_direction": 250,  # degrees (WSW)
                "direction_std": 45,
                "mean_temp": 11.0,  # °C
                "temp_seasonal_amplitude": 8.0,
                "temp_diurnal_amplitude": 2.0,
                "mean_pressure": 1013.25,  # hPa
            },
            "onshore_plains": {
                "mean_wind_speed": 7.0,
                "wind_speed_std": 3.0,
                "prevailing_direction": 270,
                "direction_std": 60,
                "mean_temp": 12.0,
                "temp_seasonal_amplitude": 15.0,
                "temp_diurnal_amplitude": 8.0,
                "mean_pressure": 1013.25,
            }
        }
        return locations.get(location, locations["offshore_north_sea"])
    
    def generate(
        self,
        start_date: str,
        days: int,
        interval_minutes: int = 10
    ) -> pd.DataFrame:
        """
        Generate weather time series data.
        
        Args:
            start_date: Start date as 'YYYY-MM-DD'
            days: Number of days to generate
            interval_minutes: Time interval between samples
            
        Returns:
            DataFrame with weather parameters
        """
        # Generate timestamps
        start = pd.to_datetime(start_date)
        periods = int((days * 24 * 60) / interval_minutes)
        timestamps = pd.date_range(
            start=start,
            periods=periods,
            freq=f'{interval_minutes}min'  # Use 'min' instead of deprecated 'T'
        )
        
        # Generate wind speed using Weibull distribution
        wind_speed = self._generate_wind_speed(periods)
        
        # Generate wind direction
        wind_direction = self._generate_wind_direction(periods)
        
        # Generate temperature with cycles
        temperature = self._generate_temperature(timestamps)
        
        # Generate atmospheric pressure
        pressure = self._generate_pressure(periods)
        
        # Calculate air density (kg/m³) from temp and pressure
        air_density = self._calculate_air_density(temperature, pressure)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed_ms': wind_speed,
            'wind_direction_deg': wind_direction,
            'temperature_c': temperature,
            'pressure_hpa': pressure,
            'air_density_kgm3': air_density
        })
        
        return df
    
    def _generate_wind_speed(self, periods: int) -> np.ndarray:
        """
        Generate wind speed using Weibull distribution with persistence.
        
        Wind speed has autocorrelation (persistence) - it changes gradually.
        """
        # Weibull shape parameter (k) - affects distribution shape
        k = 2.0  # Rayleigh distribution (common for wind)
        
        # Scale parameter based on mean wind speed
        scale = self.params["mean_wind_speed"] / 0.886  # For k=2
        
        # Generate base wind speed from Weibull
        base_wind = self.rng.weibull(k, periods) * scale
        
        # Add temporal persistence using AR(1) process
        alpha = 0.95  # Persistence coefficient
        noise = self.rng.normal(0, 0.5, periods)
        
        wind_speed = np.zeros(periods)
        wind_speed[0] = base_wind[0]
        
        for i in range(1, periods):
            wind_speed[i] = (
                alpha * wind_speed[i-1] + 
                (1-alpha) * base_wind[i] + 
                noise[i]
            )
        
        # Add occasional gusts
        gust_probability = 0.05
        gusts = self.rng.random(periods) < gust_probability
        wind_speed[gusts] += self.rng.uniform(3, 8, gusts.sum())
        
        # Ensure non-negative and reasonable bounds
        wind_speed = np.clip(wind_speed, 0, 35)
        
        return wind_speed
    
    def _generate_wind_direction(self, periods: int) -> np.ndarray:
        """
        Generate wind direction with prevailing direction and variability.
        """
        # Base direction with normal distribution
        base_direction = self.rng.normal(
            self.params["prevailing_direction"],
            self.params["direction_std"],
            periods
        )
        
        # Add persistence
        alpha = 0.98
        direction = np.zeros(periods)
        direction[0] = base_direction[0]
        
        for i in range(1, periods):
            direction[i] = (
                alpha * direction[i-1] + 
                (1-alpha) * base_direction[i]
            )
        
        # Wrap to 0-360 degrees
        direction = direction % 360
        
        return direction
    
    def _generate_temperature(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """
        Generate temperature with seasonal and diurnal cycles.
        """
        periods = len(timestamps)
        
        # Seasonal cycle (annual)
        day_of_year = timestamps.dayofyear
        seasonal_component = self.params["temp_seasonal_amplitude"] * np.sin(
            2 * np.pi * (day_of_year - 80) / 365  # Peak around day 170 (summer)
        )
        
        # Diurnal cycle (daily)
        hour_of_day = timestamps.hour + timestamps.minute / 60
        diurnal_component = self.params["temp_diurnal_amplitude"] * np.sin(
            2 * np.pi * (hour_of_day - 6) / 24  # Peak around 14:00
        )
        
        # Random variation
        noise = self.rng.normal(0, 1.0, periods)
        
        # Combine all components
        temperature = (
            self.params["mean_temp"] + 
            seasonal_component + 
            diurnal_component + 
            noise
        )
        
        return temperature
    
    def _generate_pressure(self, periods: int) -> np.ndarray:
        """
        Generate atmospheric pressure with weather fronts.
        """
        # Base pressure with slow-moving fronts
        trend = np.cumsum(self.rng.normal(0, 0.5, periods))
        pressure = self.params["mean_pressure"] + trend
        
        # Add high-frequency noise
        pressure += self.rng.normal(0, 2.0, periods)
        
        # Keep in realistic range
        pressure = np.clip(pressure, 980, 1045)
        
        return pressure
    
    def _calculate_air_density(
        self,
        temperature: np.ndarray,
        pressure: np.ndarray
    ) -> np.ndarray:
        """
        Calculate air density using ideal gas law.
        
        ρ = P / (R * T)
        where:
        - P is pressure in Pa
        - R is specific gas constant for air (287.05 J/(kg·K))
        - T is temperature in Kelvin
        """
        R = 287.05  # J/(kg·K)
        
        # Convert to SI units
        T_kelvin = temperature + 273.15
        P_pascal = pressure * 100  # hPa to Pa
        
        density = P_pascal / (R * T_kelvin)
        
        return density
    
    def save(self, df: pd.DataFrame, filepath: str):
        """Save weather data to CSV."""
        df.to_csv(filepath, index=False)
        print(f"Weather data saved to: {filepath}")
    
    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Calculate summary statistics for the generated data."""
        stats = {
            'mean_wind_speed': df['wind_speed_ms'].mean(),
            'max_wind_speed': df['wind_speed_ms'].max(),
            'min_wind_speed': df['wind_speed_ms'].min(),
            'mean_temperature': df['temperature_c'].mean(),
            'mean_pressure': df['pressure_hpa'].mean(),
            'samples': len(df)
        }
        return stats


if __name__ == "__main__":
    # Example usage
    generator = WeatherGenerator(seed=42)
    weather_data = generator.generate(
        start_date="2024-01-01",
        days=7,
        interval_minutes=10
    )
    
    print("Generated weather data:")
    print(weather_data.head())
    print("\nStatistics:")
    print(generator.get_statistics(weather_data))
