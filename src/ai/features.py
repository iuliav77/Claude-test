"""
Feature engineering for wind turbine forecasting.

Converts turbine state history into ML-ready features including:
- Lag features (past values)
- Rolling statistics (mean, std, min, max)
- Time-based features (hour, day, season)
- Derived features (power coefficient, efficiency)
- Weather trend features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from twin.state import TurbineState


class FeatureEngineer:
    """
    Transform turbine state history into features for ML forecasting.
    
    Creates features suitable for predicting:
    - Power output (next 1-6 hours)
    - Component failures
    - Performance degradation
    """
    
    def __init__(
        self,
        lag_hours: List[int] = [1, 2, 3, 6, 12, 24],
        rolling_windows: List[int] = [6, 12, 24],  # In timesteps (10-min intervals)
        include_time_features: bool = True,
        include_derived_features: bool = True
    ):
        """
        Initialize feature engineer.
        
        Args:
            lag_hours: Hours to create lag features for
            rolling_windows: Window sizes for rolling statistics (in timesteps)
            include_time_features: Whether to include time-based features
            include_derived_features: Whether to include derived features
        """
        self.lag_hours = lag_hours
        self.rolling_windows = rolling_windows
        self.include_time_features = include_time_features
        self.include_derived_features = include_derived_features
    
    def states_to_dataframe(self, states: List[TurbineState]) -> pd.DataFrame:
        """
        Convert list of TurbineState objects to pandas DataFrame.
        
        Args:
            states: List of TurbineState objects
            
        Returns:
            DataFrame with all state parameters
        """
        if not states:
            return pd.DataFrame()
        
        # Extract all fields from states
        records = []
        for state in states:
            record = {
                'timestamp': state.timestamp,
                'simulation_time_s': state.simulation_time_s,
                
                # Power system
                'power_kw': state.power_output_kw,
                'rotor_speed_rpm': state.rotor_speed_rpm,
                'pitch_angle_deg': state.pitch_angle_deg,
                
                # Temperatures
                'generator_temp_c': state.generator_temp_c,
                'gearbox_temp_c': state.gearbox_temp_c,
                'bearing_temp_c': state.bearing_temp_c,
                
                # Mechanical
                'vibration_mms': state.vibration_mms,
                'torque_nm': state.torque_nm,
                
                # Position
                'yaw_angle_deg': state.yaw_angle_deg,
                'nacelle_position_deg': state.nacelle_position_deg,
                
                # Operational
                'status': state.status.value,
                'is_faulted': int(state.is_faulted),
                
                # Cumulative
                'total_energy_mwh': state.total_energy_mwh,
                'operating_hours': state.operating_hours,
                
                # Weather
                'wind_speed_ms': state.wind_speed_ms,
                'wind_direction_deg': state.wind_direction_deg,
                'ambient_temp_c': state.ambient_temp_c,
                'air_density_kgm3': state.air_density_kgm3,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lag_hours: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features (past values).
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lag_hours: Hours to lag (uses instance default if None)
            
        Returns:
            DataFrame with lag features added
        """
        lag_hours = lag_hours or self.lag_hours
        df_out = df.copy()
        
        # Assuming 10-minute intervals (6 per hour)
        timesteps_per_hour = 6
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for hours in lag_hours:
                lag_steps = hours * timesteps_per_hour
                lag_col_name = f'{col}_lag_{hours}h'
                df_out[lag_col_name] = df[col].shift(lag_steps)
        
        return df_out
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistics (mean, std, min, max).
        
        Args:
            df: Input DataFrame
            columns: Columns to compute rolling stats for
            windows: Window sizes in timesteps (uses instance default if None)
            
        Returns:
            DataFrame with rolling features added
        """
        windows = windows or self.rolling_windows
        df_out = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                window_hours = window / 6  # Convert timesteps to hours
                
                # Rolling mean
                df_out[f'{col}_roll_mean_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )
                
                # Rolling std
                df_out[f'{col}_roll_std_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )
                
                # Rolling min
                df_out[f'{col}_roll_min_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                
                # Rolling max
                df_out[f'{col}_roll_max_{window}'] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )
        
        return df_out
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with time features added
        """
        df_out = df.copy()
        
        # Extract time components
        df_out['hour'] = df_out['timestamp'].dt.hour
        df_out['day_of_week'] = df_out['timestamp'].dt.dayofweek
        df_out['day_of_year'] = df_out['timestamp'].dt.dayofyear
        df_out['month'] = df_out['timestamp'].dt.month
        df_out['quarter'] = df_out['timestamp'].dt.quarter
        
        # Cyclical encoding (important for ML models)
        df_out['hour_sin'] = np.sin(2 * np.pi * df_out['hour'] / 24)
        df_out['hour_cos'] = np.cos(2 * np.pi * df_out['hour'] / 24)
        
        df_out['day_sin'] = np.sin(2 * np.pi * df_out['day_of_week'] / 7)
        df_out['day_cos'] = np.cos(2 * np.pi * df_out['day_of_week'] / 7)
        
        df_out['month_sin'] = np.sin(2 * np.pi * df_out['month'] / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * df_out['month'] / 12)
        
        # Season (meteorological)
        df_out['season'] = df_out['month'].apply(self._get_season)
        
        # Is weekend
        df_out['is_weekend'] = (df_out['day_of_week'] >= 5).astype(int)
        
        return df_out
    
    @staticmethod
    def _get_season(month: int) -> int:
        """Get season from month (0=winter, 1=spring, 2=summer, 3=fall)."""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived/engineered features from existing columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with derived features added
        """
        df_out = df.copy()
        
        # Power coefficient (Cp) approximation
        # Cp = P / (0.5 * ρ * A * v³)
        # For a 2.5MW turbine with 90m rotor diameter
        rotor_area = np.pi * (90 / 2) ** 2  # m²
        
        df_out['wind_power_available'] = (
            0.5 * df_out['air_density_kgm3'] * 
            rotor_area * 
            (df_out['wind_speed_ms'] ** 3) / 1000  # Convert to kW
        )
        
        # Avoid division by zero
        df_out['power_coefficient'] = np.where(
            df_out['wind_power_available'] > 1,
            df_out['power_kw'] / df_out['wind_power_available'],
            0
        )
        
        # Clip to valid Betz limit range [0, 0.593]
        df_out['power_coefficient'] = np.clip(df_out['power_coefficient'], 0, 0.593)
        
        # Capacity factor (current)
        rated_power = 2500  # kW
        df_out['capacity_factor'] = df_out['power_kw'] / rated_power
        
        # Temperature differences (stress indicators)
        df_out['temp_diff_gen_ambient'] = (
            df_out['generator_temp_c'] - df_out['ambient_temp_c']
        )
        df_out['temp_diff_gearbox_ambient'] = (
            df_out['gearbox_temp_c'] - df_out['ambient_temp_c']
        )
        
        # Wind direction change (turbulence indicator)
        df_out['wind_dir_change'] = df_out['wind_direction_deg'].diff().abs()
        # Handle wrap-around (359° -> 1° should be 2°, not 358°)
        df_out['wind_dir_change'] = np.where(
            df_out['wind_dir_change'] > 180,
            360 - df_out['wind_dir_change'],
            df_out['wind_dir_change']
        )
        
        # Wind speed change rate (m/s per timestep)
        df_out['wind_speed_change'] = df_out['wind_speed_ms'].diff()
        
        # Tip speed ratio (TSR = ω * R / v)
        rotor_radius = 90 / 2  # meters
        # Convert RPM to rad/s: ω = RPM * 2π / 60
        omega_rad_s = df_out['rotor_speed_rpm'] * 2 * np.pi / 60
        df_out['tip_speed_ratio'] = np.where(
            df_out['wind_speed_ms'] > 1,
            (omega_rad_s * rotor_radius) / df_out['wind_speed_ms'],
            0
        )
        
        # Power density (kW per m² of rotor area)
        df_out['power_density'] = df_out['power_kw'] / rotor_area
        
        # Energy rate (MWh per hour)
        df_out['energy_rate'] = df_out['power_kw'] / 1000  # kW to MWh/h
        
        return df_out
    
    def create_target_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'power_kw',
        forecast_hours: List[int] = [1, 3, 6]
    ) -> pd.DataFrame:
        """
        Create future target values for supervised learning.
        
        Args:
            df: Input DataFrame
            target_col: Column to forecast
            forecast_hours: Hours ahead to forecast
            
        Returns:
            DataFrame with target features added
        """
        df_out = df.copy()
        
        # Assuming 10-minute intervals (6 per hour)
        timesteps_per_hour = 6
        
        for hours in forecast_hours:
            steps = hours * timesteps_per_hour
            target_name = f'{target_col}_future_{hours}h'
            df_out[target_name] = df[target_col].shift(-steps)
        
        return df_out
    
    def engineer_features(
        self,
        states: List[TurbineState],
        for_forecasting: bool = True,
        forecast_hours: List[int] = [1, 3, 6]
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            states: List of TurbineState objects
            for_forecasting: Whether to create future target columns
            forecast_hours: Hours ahead to create targets for
            
        Returns:
            DataFrame with all engineered features
        """
        # Convert states to dataframe
        df = self.states_to_dataframe(states)
        
        if df.empty:
            return df
        
        print(f"Starting feature engineering with {len(df)} samples...")
        
        # Define key columns for lag/rolling features
        power_cols = ['power_kw', 'rotor_speed_rpm']
        weather_cols = ['wind_speed_ms', 'wind_direction_deg', 'ambient_temp_c']
        temp_cols = ['generator_temp_c', 'gearbox_temp_c', 'bearing_temp_c']
        mechanical_cols = ['vibration_mms']
        
        # Create lag features
        print("  Creating lag features...")
        df = self.create_lag_features(df, power_cols + weather_cols)
        
        # Create rolling features
        print("  Creating rolling statistics...")
        df = self.create_rolling_features(df, power_cols + weather_cols + temp_cols)
        
        # Create time features
        if self.include_time_features:
            print("  Creating time features...")
            df = self.create_time_features(df)
        
        # Create derived features
        if self.include_derived_features:
            print("  Creating derived features...")
            df = self.create_derived_features(df)
        
        # Create target features (for training)
        if for_forecasting:
            print("  Creating target features...")
            df = self.create_target_features(df, 'power_kw', forecast_hours)
        
        print(f"Feature engineering complete: {len(df.columns)} features created")
        
        return df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get logical groupings of features for analysis/selection.
        
        Returns:
            Dictionary mapping group names to feature name patterns
        """
        return {
            'raw_power': ['power_kw', 'rotor_speed_rpm', 'pitch_angle_deg'],
            'raw_weather': ['wind_speed_ms', 'wind_direction_deg', 'ambient_temp_c', 'air_density_kgm3'],
            'raw_temps': ['generator_temp_c', 'gearbox_temp_c', 'bearing_temp_c'],
            'raw_mechanical': ['vibration_mms', 'torque_nm'],
            'lag_features': ['_lag_'],
            'rolling_features': ['_roll_'],
            'time_features': ['hour', 'day_', 'month', 'season', 'quarter', '_sin', '_cos', 'is_weekend'],
            'derived_features': ['power_coefficient', 'capacity_factor', 'temp_diff_', 
                                'wind_dir_change', 'wind_speed_change', 'tip_speed_ratio',
                                'power_density', 'energy_rate'],
            'targets': ['_future_']
        }
    
    def get_feature_names_by_group(self, df: pd.DataFrame, group: str) -> List[str]:
        """
        Get feature names belonging to a specific group.
        
        Args:
            df: DataFrame with features
            group: Group name from get_feature_groups()
            
        Returns:
            List of column names in that group
        """
        groups = self.get_feature_groups()
        if group not in groups:
            return []
        
        patterns = groups[group]
        matching_cols = []
        
        for col in df.columns:
            for pattern in patterns:
                if pattern in col:
                    matching_cols.append(col)
                    break
        
        return matching_cols


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    drop_na: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train and test sets (temporal split).
    
    Args:
        df: Feature dataframe
        test_size: Fraction for test set
        drop_na: Whether to drop rows with NaN values
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if drop_na:
        df = df.dropna()
    
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


if __name__ == "__main__":
    # Test feature engineering
    import sys
    sys.path.insert(0, '/home/claude/windtwin-ai/src')
    
    from data.weather_generator import WeatherGenerator
    from twin.simulator import Simulator
    
    print("Feature Engineering Test")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating simulation data...")
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate(
        start_date="2024-01-01",
        days=7,
        interval_minutes=10
    )
    
    # Run simulation
    sim = Simulator(weather_data)
    print(f"   Running {len(weather_data)} simulation steps...")
    
    for i in range(len(weather_data)):
        sim.step()
        if i % 144 == 0:  # Every day
            print(f"   Day {i // 144 + 1} complete")
    
    # Get state history
    states = sim.state_manager.get_history()
    print(f"\n2. Retrieved {len(states)} states from history")
    
    # Engineer features
    print("\n3. Engineering features...")
    engineer = FeatureEngineer(
        lag_hours=[1, 3, 6],
        rolling_windows=[6, 12, 24],
        include_time_features=True,
        include_derived_features=True
    )
    
    features_df = engineer.engineer_features(
        states,
        for_forecasting=True,
        forecast_hours=[1, 3, 6]
    )
    
    print(f"\n4. Feature summary:")
    print(f"   Total samples: {len(features_df)}")
    print(f"   Total features: {len(features_df.columns)}")
    print(f"   Date range: {features_df['timestamp'].min()} to {features_df['timestamp'].max()}")
    
    # Show feature groups
    print(f"\n5. Feature groups:")
    for group, features in engineer.get_feature_groups().items():
        matching = engineer.get_feature_names_by_group(features_df, group)
        if matching:
            print(f"   {group}: {len(matching)} features")
    
    # Show sample
    print(f"\n6. Sample features (first 5 rows, key columns):")
    key_cols = ['timestamp', 'power_kw', 'wind_speed_ms', 'power_kw_lag_1h', 
                'wind_speed_ms_roll_mean_6', 'hour', 'power_coefficient']
    available_cols = [c for c in key_cols if c in features_df.columns]
    print(features_df[available_cols].head())
    
    # Train/test split
    print(f"\n7. Train/test split:")
    train_df, test_df = prepare_train_test_split(features_df, test_size=0.2)
    print(f"   Train: {len(train_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    print("\n✅ Feature engineering working correctly!")
