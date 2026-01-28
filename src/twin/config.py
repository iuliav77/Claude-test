"""
Configuration management for WINDTWIN-AI.

Loads and validates configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class TurbineConfig:
    """Turbine specification configuration."""
    name: str
    rated_power_kw: float
    rotor_diameter_m: float
    hub_height_m: float
    cut_in_speed_ms: float
    rated_speed_ms: float
    cut_out_speed_ms: float
    max_rotor_rpm: float
    min_rotor_rpm: float
    rotor_inertia_kgm2: float
    max_generator_temp_c: float
    max_gearbox_temp_c: float
    max_bearing_temp_c: float
    max_vibration_mms: float
    power_coefficient: float
    mechanical_efficiency: float
    electrical_efficiency: float


@dataclass
class PhysicsConfig:
    """Physics model configuration."""
    thermal: Dict[str, float] = field(default_factory=dict)
    mechanical: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Simulation settings configuration."""
    time_step_seconds: int
    default_mode: str
    fast_mode_multipliers: list
    history_buffer_size: int
    auto_checkpoint_enabled: bool
    auto_checkpoint_interval_minutes: int
    max_checkpoints_to_keep: int


@dataclass
class DashboardConfig:
    """Dashboard display configuration."""
    refresh_rate_hz: float
    power_gauge_width: int
    show_events_count: int
    status_colors: Dict[str, str] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    log_to_file: bool
    log_file: str
    max_log_size_mb: int
    backup_count: int


@dataclass
class StorageConfig:
    """Data storage configuration."""
    simulation_data_dir: str
    checkpoint_dir: str
    export_format: str


@dataclass
class Config:
    """Main configuration object."""
    turbine: TurbineConfig
    physics: PhysicsConfig
    simulation: SimulationConfig
    dashboard: DashboardConfig
    logging: LoggingConfig
    storage: StorageConfig


class ConfigLoader:
    """Load and manage configuration."""
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> Config:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. If None, uses default.
            
        Returns:
            Config object with all settings.
        """
        if config_path is None:
            # Use default config - go up to project root
            # __file__ is in src/twin/, so go up 3 levels to project root
            default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
            config_path = str(default_path)
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigLoader._dict_to_config(config_dict)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config dataclass."""
        
        # Parse turbine config
        turbine_data = config_dict['turbine']
        turbine = TurbineConfig(**turbine_data)
        
        # Parse physics config
        physics = PhysicsConfig(**config_dict['physics'])
        
        # Parse simulation config
        simulation = SimulationConfig(**config_dict['simulation'])
        
        # Parse dashboard config
        dashboard = DashboardConfig(**config_dict['dashboard'])
        
        # Parse logging config
        logging = LoggingConfig(**config_dict['logging'])
        
        # Parse storage config
        storage = StorageConfig(**config_dict['storage'])
        
        return Config(
            turbine=turbine,
            physics=physics,
            simulation=simulation,
            dashboard=dashboard,
            logging=logging,
            storage=storage
        )
    
    @staticmethod
    def save(config: Config, output_path: str):
        """Save configuration to YAML file."""
        # Convert dataclasses back to dict
        config_dict = {
            'turbine': config.turbine.__dict__,
            'physics': config.physics.__dict__,
            'simulation': config.simulation.__dict__,
            'dashboard': config.dashboard.__dict__,
            'logging': config.logging.__dict__,
            'storage': config.storage.__dict__
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Global config instance (loaded once)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration singleton.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = ConfigLoader.load(config_path)
    return _config


def reload_config(config_path: Optional[str] = None):
    """Reload configuration from file."""
    global _config
    _config = ConfigLoader.load(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Turbine: {config.turbine.name}")
    print(f"Rated Power: {config.turbine.rated_power_kw} kW")
    print(f"Time Step: {config.simulation.time_step_seconds}s")
    print(f"Dashboard Refresh: {config.dashboard.refresh_rate_hz} Hz")
