"""
Simulation engine for wind turbine digital twin.

This is the core time-stepping engine that:
- Advances simulation time
- Coordinates all subsystems
- Processes events
- Manages state transitions
- Supports multiple simulation modes
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from enum import Enum

from .config import Config, get_config
from .turbine import Turbine, OperationalStatus
from .physics import PhysicsEngine
from .state import TurbineState, StateManager
from .events import EventProcessor, Event, EventType, EventSeverity


class SimulationMode(Enum):
    """Simulation execution modes."""
    REALTIME = "realtime"  # 1 second real = 1 second simulated
    FAST = "fast"  # Run as fast as possible
    STEP = "step"  # Manual step-by-step control


class Simulator:
    """
    Main simulation engine for wind turbine digital twin.
    
    Coordinates all components and advances simulation through time.
    """
    
    def __init__(
        self,
        weather_data: pd.DataFrame,
        config: Optional[Config] = None,
        start_date: Optional[datetime] = None,
        forecaster_path: Optional[str] = None
    ):
        """
        Initialize simulator with weather data.
        
        Args:
            weather_data: DataFrame with weather time series
            config: Configuration object (uses default if None)
            start_date: Simulation start date (uses weather data start if None)
            forecaster_path: Path to trained forecaster model (optional)
        """
        # Configuration
        self.config = config or get_config()
        
        # Weather data
        self.weather_data = weather_data.copy()
        self.weather_index = 0
        
        # Time management
        self.start_date = start_date or pd.to_datetime(weather_data['timestamp'].iloc[0])
        self.current_time = self.start_date
        self.simulation_time_s = 0.0
        self.time_step_s = self.config.simulation.time_step_seconds
        
        # Components
        self.turbine = Turbine(self.config.turbine)
        self.physics = PhysicsEngine(self.config.physics)
        self.state_manager = StateManager(
            history_size=self.config.simulation.history_buffer_size
        )
        self.event_processor = EventProcessor()
        
        # Forecasting (Phase 4)
        self.forecaster = None
        self.feature_engineer = None
        self.forecast_enabled = False
        
        if forecaster_path:
            self._load_forecaster(forecaster_path)
        
        # Simulation control
        self.mode = SimulationMode.REALTIME
        self.speed_multiplier = 1.0
        self.is_running = False
        self.is_paused = False
        
        # Observers (for dashboard updates)
        self._observers: list[Callable] = []
        
        # Initialize state
        self._initialize_state()
        
        # Create simulation start event
        self._create_event(
            EventType.SIMULATION_START,
            EventSeverity.INFO,
            "Simulation initialized"
        )
    
    def _initialize_state(self):
        """Initialize the turbine state (v1.0: realistic initial conditions)."""
        # Get first weather conditions
        weather_row = self.weather_data.iloc[0]
        initial_wind = weather_row['wind_speed_ms']
        
        # v1.0 REALISM FIX: Calculate realistic initial conditions
        # If wind is sufficient, turbine should already be running
        initial_power = 0.0
        initial_rpm = 0.0
        initial_status = OperationalStatus.STOPPED
        
        if initial_wind >= self.config.turbine.cut_in_speed_ms:
            # Wind is strong enough - turbine should be operating
            initial_power = self.turbine.calculate_power(
                initial_wind, 
                weather_row['air_density_kgm3']
            )
            initial_rpm = self.turbine.calculate_rotor_speed(initial_wind)
            
            # Determine realistic status
            if initial_power >= self.config.turbine.rated_power_kw * 0.95:
                initial_status = OperationalStatus.RATED_POWER
            elif initial_power > 10:
                initial_status = OperationalStatus.RUNNING
            else:
                initial_status = OperationalStatus.STARTING
        
        # Create initial state with realistic values
        initial_state = TurbineState(
            timestamp=self.start_date,
            simulation_time_s=0.0,
            power_output_kw=initial_power,
            rotor_speed_rpm=initial_rpm,
            pitch_angle_deg=0.0,
            generator_temp_c=weather_row['temperature_c'] + 5,
            gearbox_temp_c=weather_row['temperature_c'] + 3,
            bearing_temp_c=weather_row['temperature_c'] + 2,
            vibration_mms=0.5,
            status=initial_status,
            wind_speed_ms=weather_row['wind_speed_ms'],
            wind_direction_deg=weather_row['wind_direction_deg'],
            ambient_temp_c=weather_row['temperature_c'],
            air_density_kgm3=weather_row['air_density_kgm3']
        )
        
        self.state_manager.update_state(initial_state)
    
    def add_observer(self, callback: Callable[[TurbineState], None]):
        """
        Add observer for state changes.
        
        Observers are notified when state updates (useful for dashboard).
        
        Args:
            callback: Function called with new state
        """
        self._observers.append(callback)
    
    def remove_observer(self, callback: Callable):
        """Remove an observer."""
        if callback in self._observers:
            self._observers.remove(callback)
    
    def _notify_observers(self):
        """Notify all observers of state change."""
        current_state = self.state_manager.current_state
        if current_state:
            for observer in self._observers:
                try:
                    observer(current_state)
                except Exception as e:
                    print(f"Error notifying observer: {e}")
    
    def _load_forecaster(self, forecaster_path: str):
        """
        Load trained forecaster model for real-time predictions.
        
        Args:
            forecaster_path: Path to saved forecaster model
        """
        try:
            # Import here to avoid circular dependencies
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            
            from ai.forecaster import PowerForecaster
            from ai.features import FeatureEngineer
            
            # Load forecaster
            self.forecaster = PowerForecaster()
            self.forecaster.load(forecaster_path)
            
            # Initialize feature engineer with same config
            self.feature_engineer = FeatureEngineer(
                lag_hours=[1, 3, 6],
                rolling_windows=[6, 12, 24],
                include_time_features=True,
                include_derived_features=True
            )
            
            self.forecast_enabled = True
            
            print(f"✓ Forecaster loaded: {forecaster_path}")
            print(f"  Model: {self.forecaster.model_name}")
            print(f"  Horizon: {self.forecaster.forecast_horizon_hours}h")
            print(f"  Features: {len(self.forecaster.feature_names)}")
            
        except FileNotFoundError:
            print(f"⚠ Forecaster not found: {forecaster_path}")
            print("  Continuing without forecasts...")
            self.forecast_enabled = False
        except Exception as e:
            print(f"⚠ Error loading forecaster: {e}")
            print("  Continuing without forecasts...")
            self.forecast_enabled = False
    
    def _generate_forecast(self) -> Optional[float]:
        """
        Generate 1-hour ahead power forecast using current state history.
        
        Returns:
            Forecasted power in kW, or None if forecast cannot be generated
        """
        if not self.forecast_enabled:
            return None
        
        try:
            # Get recent history (need enough for lag features)
            # For 1h, 3h, 6h lags we need at least 36 timesteps (6 hours)
            min_history = 36
            history = self.state_manager.get_history(last_n=min_history)
            
            if len(history) < min_history:
                # Not enough history yet
                return None
            
            # Convert states to features
            features_df = self.feature_engineer.states_to_dataframe(history)
            
            if features_df.empty:
                return None
            
            # Engineer features (without targets for prediction)
            features_df = self.feature_engineer.create_lag_features(
                features_df,
                columns=['power_kw', 'rotor_speed_rpm', 'wind_speed_ms', 
                        'wind_direction_deg', 'ambient_temp_c']
            )
            features_df = self.feature_engineer.create_rolling_features(
                features_df,
                columns=['power_kw', 'rotor_speed_rpm', 'wind_speed_ms',
                        'generator_temp_c', 'gearbox_temp_c']
            )
            features_df = self.feature_engineer.create_time_features(features_df)
            features_df = self.feature_engineer.create_derived_features(features_df)
            
            # Get the latest row for prediction
            latest_features = features_df.iloc[[-1]]  # Keep as DataFrame
            
            # Make prediction
            forecast = self.forecaster.predict(latest_features, return_dataframe=False)
            
            return float(forecast[0]) if len(forecast) > 0 else None
            
        except Exception as e:
            # Silently fail and return None
            return None
    
    def _create_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        message: str,
        data: Optional[Dict] = None
    ):
        """Create and process an event."""
        event = self.event_processor.create_event(
            event_type=event_type,
            severity=severity,
            message=message,
            timestamp=self.current_time,
            simulation_time_s=self.simulation_time_s,
            data=data
        )
        
        self.event_processor.process_event(
            event,
            context={'state': self.state_manager.current_state}
        )
    
    def step(self) -> TurbineState:
        """
        Execute one simulation time step.
        
        This is the core simulation loop:
        1. Get current weather
        2. Calculate power from turbine model
        3. Update physics (temperatures, RPM, vibration)
        4. Check operational limits
        5. Update cumulative metrics
        6. Create new state
        7. Notify observers
        
        Returns:
            New TurbineState after step
        """
        current_state = self.state_manager.current_state
        
        # 1. Get current weather conditions
        if self.weather_index >= len(self.weather_data):
            # Reached end of weather data
            self.stop()
            return current_state
        
        weather_row = self.weather_data.iloc[self.weather_index]
        wind_speed = weather_row['wind_speed_ms']
        wind_direction = weather_row['wind_direction_deg']
        ambient_temp = weather_row['temperature_c']
        air_density = weather_row['air_density_kgm3']
        
        # 2. Calculate power output (if not faulted)
        if not current_state.is_faulted:
            power_output = self.turbine.calculate_power(wind_speed, air_density)
            target_rpm = self.turbine.calculate_rotor_speed(wind_speed)
        else:
            power_output = 0.0
            target_rpm = 0.0
        
        # 3. Update physics
        physics_state = {
            'rotor_speed_rpm': current_state.rotor_speed_rpm,
            'generator_temp_c': current_state.generator_temp_c,
            'gearbox_temp_c': current_state.gearbox_temp_c,
            'bearing_temp_c': current_state.bearing_temp_c,
            'vibration_mms': current_state.vibration_mms
        }
        
        updated_physics = self.physics.step(
            current_state=physics_state,
            target_rpm=target_rpm,
            power_output_kw=power_output,
            rated_power_kw=self.config.turbine.rated_power_kw,
            ambient_temp_c=ambient_temp,
            time_step_s=self.time_step_s
        )
        
        # 4. Check operational limits
        is_ok, fault_msg = self.turbine.check_limits(
            generator_temp_c=updated_physics['generator_temp_c'],
            gearbox_temp_c=updated_physics['gearbox_temp_c'],
            vibration_mms=updated_physics['vibration_mms'],
            rotor_rpm=updated_physics['rotor_speed_rpm']
        )
        
        is_faulted = current_state.is_faulted
        if not is_ok and not is_faulted:
            # New fault detected
            is_faulted = True
            self._create_event(
                EventType.COMPONENT_FAILURE,
                EventSeverity.ERROR,
                fault_msg,
                {'fault_type': 'limit_violation'}
            )
        
        # 5. Determine operational status
        status = self.turbine.determine_status(
            wind_speed,
            power_output,
            is_faulted=is_faulted
        )
        
        # 6. Update cumulative metrics
        time_step_h = self.time_step_s / 3600.0
        
        total_energy = current_state.total_energy_mwh
        if power_output > 0:
            total_energy += (power_output * time_step_h) / 1000.0  # kWh to MWh
        
        operating_hours = current_state.operating_hours
        if status in [OperationalStatus.RUNNING, OperationalStatus.RATED_POWER]:
            operating_hours += time_step_h
        
        start_count = current_state.start_count
        if (current_state.status == OperationalStatus.STOPPED and 
            status == OperationalStatus.STARTING):
            start_count += 1
        
        fault_count = current_state.fault_count
        if not current_state.is_faulted and is_faulted:
            fault_count += 1
        
        # 7. Generate forecast (Phase 4)
        forecast_power = None
        forecast_available = False
        
        if self.forecast_enabled:
            forecast_power = self._generate_forecast()
            if forecast_power is not None:
                forecast_available = True
        
        # 8. Create new state
        new_state = TurbineState(
            timestamp=self.current_time,
            simulation_time_s=self.simulation_time_s,
            power_output_kw=power_output,
            rotor_speed_rpm=updated_physics['rotor_speed_rpm'],
            pitch_angle_deg=0.0,
            generator_temp_c=updated_physics['generator_temp_c'],
            gearbox_temp_c=updated_physics['gearbox_temp_c'],
            bearing_temp_c=updated_physics['bearing_temp_c'],
            vibration_mms=updated_physics['vibration_mms'],
            status=status,
            is_faulted=is_faulted,
            fault_message=fault_msg if is_faulted else None,
            total_energy_mwh=total_energy,
            operating_hours=operating_hours,
            start_count=start_count,
            fault_count=fault_count,
            wind_speed_ms=wind_speed,
            wind_direction_deg=wind_direction,
            ambient_temp_c=ambient_temp,
            air_density_kgm3=air_density,
            nacelle_position_deg=wind_direction,  # Simplified yaw tracking
            forecast_power_1h=forecast_power,  # Phase 4
            forecast_available=forecast_available  # Phase 4
        )
        
        # Update state manager
        self.state_manager.update_state(new_state)
        
        # Advance time
        self.current_time += timedelta(seconds=self.time_step_s)
        self.simulation_time_s += self.time_step_s
        self.weather_index += 1
        
        # Notify observers
        self._notify_observers()
        
        return new_state
    
    def run(
        self,
        duration_s: Optional[float] = None,
        mode: SimulationMode = SimulationMode.REALTIME,
        speed_multiplier: float = 1.0
    ):
        """
        Run simulation for specified duration.
        
        Args:
            duration_s: Duration in seconds (None = until weather data ends)
            mode: Simulation mode
            speed_multiplier: Speed multiplier for fast mode
        """
        self.mode = mode
        self.speed_multiplier = speed_multiplier
        self.is_running = True
        self.is_paused = False
        
        start_sim_time = self.simulation_time_s
        target_sim_time = (
            start_sim_time + duration_s if duration_s else float('inf')
        )
        
        while (self.is_running and 
               self.simulation_time_s < target_sim_time and
               self.weather_index < len(self.weather_data)):
            
            # Handle pause
            while self.is_paused:
                time.sleep(0.1)
                if not self.is_running:
                    return
            
            # Execute step
            step_start_time = time.time()
            self.step()
            step_duration = time.time() - step_start_time
            
            # Handle real-time mode timing
            if mode == SimulationMode.REALTIME:
                target_duration = self.time_step_s / speed_multiplier
                sleep_time = target_duration - step_duration
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Fast mode: no delay
            # Step mode: handled externally
    
    def start(
        self,
        mode: SimulationMode = SimulationMode.REALTIME,
        speed_multiplier: float = 1.0
    ):
        """Start simulation in specified mode."""
        self.run(mode=mode, speed_multiplier=speed_multiplier)
    
    def pause(self):
        """Pause simulation."""
        self.is_paused = True
        self._create_event(
            EventType.CUSTOM,
            EventSeverity.INFO,
            "Simulation paused"
        )
    
    def resume(self):
        """Resume paused simulation."""
        self.is_paused = False
        self._create_event(
            EventType.CUSTOM,
            EventSeverity.INFO,
            "Simulation resumed"
        )
    
    def stop(self):
        """Stop simulation."""
        self.is_running = False
        self._create_event(
            EventType.SIMULATION_STOP,
            EventSeverity.INFO,
            f"Simulation stopped at {self.simulation_time_s:.0f}s"
        )
    
    def set_speed(self, multiplier: float):
        """
        Set simulation speed multiplier.
        
        Args:
            multiplier: Speed multiplier (1.0 = real-time, 10.0 = 10x, etc.)
        """
        self.speed_multiplier = multiplier
        self._create_event(
            EventType.CUSTOM,
            EventSeverity.INFO,
            f"Simulation speed set to {multiplier}x"
        )
    
    def inject_fault(self, fault_type: str, duration_s: float = 300):
        """
        Inject a fault for testing.
        
        Args:
            fault_type: Type of fault to inject
            duration_s: Duration of fault in seconds
        """
        self._create_event(
            EventType.COMPONENT_FAILURE,
            EventSeverity.ERROR,
            f"Injected fault: {fault_type}",
            {'fault_type': fault_type, 'duration': duration_s}
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get simulation summary statistics.
        
        Returns:
            Dict with summary metrics
        """
        current_state = self.state_manager.current_state
        if not current_state:
            return {}
        
        capacity_factor = self.state_manager.calculate_capacity_factor()
        availability = self.state_manager.calculate_availability()
        
        return {
            'simulation_time_s': self.simulation_time_s,
            'simulation_time_h': self.simulation_time_s / 3600,
            'current_time': self.current_time.isoformat(),
            'current_power_kw': current_state.power_output_kw,
            'current_status': current_state.status.name,
            'total_energy_mwh': current_state.total_energy_mwh,
            'operating_hours': current_state.operating_hours,
            'capacity_factor': capacity_factor,
            'availability': availability,
            'start_count': current_state.start_count,
            'fault_count': current_state.fault_count,
            'weather_data_remaining': len(self.weather_data) - self.weather_index
        }


if __name__ == "__main__":
    # Test simulator with sample weather data
    import sys
    sys.path.insert(0, '/home/claude/windtwin-ai/src')
    
    from data.weather_generator import WeatherGenerator
    
    print("Simulator Test")
    print("=" * 50)
    
    # Generate weather data
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate(
        start_date="2024-01-01",
        days=1,  # 1 day test
        interval_minutes=10
    )
    
    print(f"Generated {len(weather_data)} weather samples")
    
    # Create simulator
    sim = Simulator(weather_data)
    
    print(f"Simulation initialized")
    print(f"Time step: {sim.time_step_s}s")
    print()
    
    # Run a few steps manually
    print("Running 10 simulation steps...")
    for i in range(10):
        state = sim.step()
        print(f"Step {i+1}: Power={state.power_output_kw:.0f}kW, "
              f"RPM={state.rotor_speed_rpm:.1f}, "
              f"Status={state.status.name}")
    
    # Get summary
    print("\nSimulation Summary:")
    summary = sim.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Simulator working correctly!")
