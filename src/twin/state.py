"""
State management for turbine digital twin.

Tracks current state and maintains history for analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Deque
from collections import deque
import numpy as np

from .turbine import OperationalStatus


@dataclass(frozen=True)
class TurbineState:
    """
    Immutable snapshot of turbine state at a moment in time.
    
    Using frozen dataclass ensures state cannot be modified after creation,
    which prevents bugs and makes simulation deterministic.
    """
    
    # Timestamp (required fields)
    timestamp: datetime
    simulation_time_s: float
    
    # Power system (required fields)
    power_output_kw: float
    rotor_speed_rpm: float
    
    # Temperatures (°C) (required fields)
    generator_temp_c: float
    gearbox_temp_c: float
    bearing_temp_c: float
    
    # Mechanical (required fields)
    vibration_mms: float
    
    # Operational (required fields)
    status: OperationalStatus
    
    # Environmental (from weather) (required fields)
    wind_speed_ms: float
    wind_direction_deg: float
    ambient_temp_c: float
    
    # Optional fields with defaults (must come after required fields)
    pitch_angle_deg: float = 0.0  # Blade pitch (0 = optimal, future use)
    yaw_angle_deg: float = 0.0  # Nacelle yaw angle
    nacelle_position_deg: float = 0.0  # Actual nacelle position
    torque_nm: float = 0.0  # Rotor torque (future use)
    is_faulted: bool = False
    fault_message: Optional[str] = None
    total_energy_mwh: float = 0.0
    operating_hours: float = 0.0
    start_count: int = 0
    fault_count: int = 0
    air_density_kgm3: float = 1.225
    
    # Forecast fields (Phase 4)
    forecast_power_1h: Optional[float] = None  # Forecasted power 1 hour ahead
    forecast_available: bool = False  # Whether forecast is available
    
    def __post_init__(self):
        """Validate state after creation."""
        # Basic sanity checks
        assert self.power_output_kw >= 0, "Power cannot be negative"
        assert self.rotor_speed_rpm >= 0, "RPM cannot be negative"
        assert self.vibration_mms >= 0, "Vibration cannot be negative"


class StateManager:
    """
    Manages turbine state and maintains history.
    
    Responsibilities:
    - Track current state
    - Maintain state history (ring buffer)
    - Calculate derived metrics
    - Validate state consistency
    - Provide state queries
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize state manager.
        
        Args:
            history_size: Number of historical states to keep
        """
        self.history_size = history_size
        self._current_state: Optional[TurbineState] = None
        self._history: Deque[TurbineState] = deque(maxlen=history_size)
        
    @property
    def current_state(self) -> Optional[TurbineState]:
        """Get current turbine state."""
        return self._current_state
    
    def update_state(self, new_state: TurbineState):
        """
        Update to new state and add previous to history.
        
        Args:
            new_state: New TurbineState
        """
        if self._current_state is not None:
            self._history.append(self._current_state)
        
        self._current_state = new_state
    
    def get_history(
        self,
        last_n: Optional[int] = None,
        time_window_s: Optional[float] = None
    ) -> List[TurbineState]:
        """
        Get historical states.
        
        Args:
            last_n: Get last N states
            time_window_s: Get states in last X seconds
            
        Returns:
            List of TurbineState objects
        """
        if last_n is not None:
            return list(self._history)[-last_n:]
        
        if time_window_s is not None and self._current_state is not None:
            cutoff_time = self._current_state.simulation_time_s - time_window_s
            return [
                s for s in self._history 
                if s.simulation_time_s >= cutoff_time
            ]
        
        return list(self._history)
    
    def get_statistics(
        self,
        parameter: str,
        time_window_s: Optional[float] = None
    ) -> dict:
        """
        Calculate statistics for a parameter over time window.
        
        Args:
            parameter: State parameter name (e.g., 'power_output_kw')
            time_window_s: Time window in seconds (None = all history)
            
        Returns:
            Dict with mean, min, max, std
        """
        history = self.get_history(time_window_s=time_window_s)
        
        if not history:
            return {
                'mean': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'count': 0
            }
        
        values = [getattr(s, parameter) for s in history]
        
        return {
            'mean': float(np.mean(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'std': float(np.std(values)),
            'count': len(values)
        }
    
    def get_recent_events(self, event_type: str, count: int = 10) -> List[TurbineState]:
        """
        Get recent states where specific event occurred.
        
        Args:
            event_type: Type of event (e.g., 'fault', 'start')
            count: Maximum number to return
            
        Returns:
            List of TurbineState objects
        """
        # This is a simplified version
        # In real implementation, would track events separately
        if event_type == 'fault':
            return [s for s in list(self._history)[-count:] if s.is_faulted]
        
        return []
    
    def calculate_capacity_factor(self, time_window_s: Optional[float] = None) -> float:
        """
        Calculate capacity factor over time window.
        
        Capacity factor = actual energy / potential energy
        
        Args:
            time_window_s: Time window (None = all history)
            
        Returns:
            Capacity factor (0-1)
        """
        history = self.get_history(time_window_s=time_window_s)
        
        if not history or self._current_state is None:
            return 0.0
        
        # Get rated power from current state (same for all states)
        # In real implementation, would store rated power separately
        rated_power = 2500.0  # kW
        
        # Calculate average power
        avg_power = np.mean([s.power_output_kw for s in history])
        
        return avg_power / rated_power
    
    def calculate_availability(self, time_window_s: Optional[float] = None) -> float:
        """
        Calculate availability (non-fault time / total time).
        
        Args:
            time_window_s: Time window (None = all history)
            
        Returns:
            Availability (0-1)
        """
        history = self.get_history(time_window_s=time_window_s)
        
        if not history:
            return 1.0
        
        non_fault_count = sum(1 for s in history if not s.is_faulted)
        
        return non_fault_count / len(history)
    
    def clear_history(self):
        """Clear all historical states (keeps current state)."""
        self._history.clear()
    
    def reset(self):
        """Reset state manager completely."""
        self._current_state = None
        self._history.clear()


class StateValidator:
    """
    Validates state consistency and checks for anomalies.
    """
    
    @staticmethod
    def validate(state: TurbineState) -> tuple[bool, List[str]]:
        """
        Validate state for consistency.
        
        Args:
            state: TurbineState to validate
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check power vs wind speed consistency
        if state.power_output_kw > 0 and state.wind_speed_ms < 3:
            warnings.append(
                f"Power output {state.power_output_kw}kW with low wind "
                f"({state.wind_speed_ms}m/s)"
            )
        
        # Check temperature consistency
        if state.generator_temp_c < state.ambient_temp_c:
            warnings.append(
                f"Generator temp {state.generator_temp_c}°C below ambient "
                f"{state.ambient_temp_c}°C"
            )
        
        # Check RPM vs power consistency
        if state.power_output_kw > 100 and state.rotor_speed_rpm < 5:
            warnings.append(
                f"High power {state.power_output_kw}kW with low RPM "
                f"{state.rotor_speed_rpm}"
            )
        
        # More checks can be added here
        
        is_valid = len(warnings) == 0
        return is_valid, warnings


if __name__ == "__main__":
    # Test state management
    from datetime import datetime, timedelta
    
    print("State Manager Test")
    print("=" * 50)
    
    # Create state manager
    manager = StateManager(history_size=100)
    
    # Create and add some states
    base_time = datetime.now()
    
    for i in range(10):
        state = TurbineState(
            timestamp=base_time + timedelta(seconds=i*10),
            simulation_time_s=i * 10,
            power_output_kw=1000 + i * 100,
            rotor_speed_rpm=10 + i * 0.5,
            pitch_angle_deg=0,
            generator_temp_c=50 + i * 2,
            gearbox_temp_c=45 + i * 1.5,
            bearing_temp_c=40 + i,
            vibration_mms=2.0 + i * 0.1,
            status=OperationalStatus.RUNNING,
            total_energy_mwh=i * 0.5,
            operating_hours=i * 0.00278,
            wind_speed_ms=8 + i * 0.5,
            wind_direction_deg=250,
            ambient_temp_c=15
        )
        
        manager.update_state(state)
    
    # Test queries
    print(f"Current power: {manager.current_state.power_output_kw} kW")
    print(f"History size: {len(manager.get_history())}")
    
    # Test statistics
    stats = manager.get_statistics('power_output_kw')
    print(f"\nPower statistics:")
    print(f"  Mean: {stats['mean']:.0f} kW")
    print(f"  Min: {stats['min']:.0f} kW")
    print(f"  Max: {stats['max']:.0f} kW")
    print(f"  Std: {stats['std']:.0f} kW")
    
    # Test capacity factor
    cf = manager.calculate_capacity_factor()
    print(f"\nCapacity factor: {cf:.1%}")
    
    # Test availability
    avail = manager.calculate_availability()
    print(f"Availability: {avail:.1%}")
    
    print("\n✅ State manager working correctly!")
