"""
Simplified physics engine for turbine simulation.

Phase 2 implementation focuses on:
- Correct behavior over high-fidelity
- Thermal dynamics with first-order model
- Rotor dynamics with acceleration/deceleration
- Stability and predictability
"""

import numpy as np
from typing import Tuple
from .config import PhysicsConfig


class ThermalModel:
    """
    Simplified thermal model for turbine components.
    
    Uses first-order thermal dynamics:
    T(t+Δt) = T(t) + (Q_gen - Q_loss) × Δt / C
    
    Where:
    - Q_gen: Heat generation from power output
    - Q_loss: Heat dissipation to ambient
    - C: Thermal capacity
    """
    
    def __init__(self, config: PhysicsConfig):
        """
        Initialize thermal model.
        
        Args:
            config: Physics configuration
        """
        self.config = config
        self.thermal_params = config.thermal
    
    def update_temperature(
        self,
        current_temp_c: float,
        ambient_temp_c: float,
        power_output_kw: float,
        rated_power_kw: float,
        component: str,
        time_step_s: float
    ) -> float:
        """
        Update component temperature based on power output.
        
        Args:
            current_temp_c: Current temperature
            ambient_temp_c: Ambient temperature
            power_output_kw: Current power output
            rated_power_kw: Rated power
            component: Component name (generator, gearbox, bearing)
            time_step_s: Time step in seconds
            
        Returns:
            New temperature in °C
        """
        # Get component-specific thermal capacity
        capacity_key = f"{component}_thermal_capacity"
        thermal_capacity = self.thermal_params.get(capacity_key, 50000)
        
        # Heat generation proportional to power output
        # More power = more losses = more heat
        power_fraction = power_output_kw / rated_power_kw
        heat_generation_w = power_output_kw * 1000 * 0.03 * power_fraction  # ~3% losses
        
        # Heat dissipation (Newton's law of cooling)
        temp_diff = current_temp_c - ambient_temp_c
        heat_dissipation_coeff = self.thermal_params.get(
            'heat_dissipation_coefficient', 100
        )
        heat_dissipation_w = heat_dissipation_coeff * temp_diff
        
        # Net heat change
        net_heat_w = heat_generation_w - heat_dissipation_w
        
        # Temperature change (first-order)
        temp_change = (net_heat_w * time_step_s) / thermal_capacity
        
        # Apply change
        new_temp = current_temp_c + temp_change
        
        # Ensure temperature doesn't go below ambient
        new_temp = max(new_temp, ambient_temp_c)
        
        return new_temp
    
    def update_all_temperatures(
        self,
        current_temps: dict,
        ambient_temp_c: float,
        power_output_kw: float,
        rated_power_kw: float,
        time_step_s: float
    ) -> dict:
        """
        Update all component temperatures.
        
        Args:
            current_temps: Dict of current temperatures
            ambient_temp_c: Ambient temperature
            power_output_kw: Current power output
            rated_power_kw: Rated power
            time_step_s: Time step in seconds
            
        Returns:
            Dict of updated temperatures
        """
        new_temps = {}
        
        for component in ['generator', 'gearbox', 'bearing']:
            current = current_temps.get(f'{component}_temp_c', ambient_temp_c + 20)
            new_temps[f'{component}_temp_c'] = self.update_temperature(
                current,
                ambient_temp_c,
                power_output_kw,
                rated_power_kw,
                component,
                time_step_s
            )
        
        return new_temps


class MechanicalModel:
    """
    Simplified mechanical model for rotor dynamics.
    
    Models rotor acceleration and deceleration with:
    - Inertia effects
    - Simple damping
    - Rate limits for realistic behavior
    """
    
    def __init__(self, config: PhysicsConfig):
        """
        Initialize mechanical model.
        
        Args:
            config: Physics configuration
        """
        self.config = config
        self.mechanical_params = config.mechanical
    
    def update_rotor_speed(
        self,
        current_rpm: float,
        target_rpm: float,
        time_step_s: float
    ) -> float:
        """
        Update rotor speed with acceleration/deceleration limits.
        
        Rotor doesn't instantly reach target speed due to inertia.
        
        Args:
            current_rpm: Current rotor speed
            target_rpm: Target rotor speed (from wind)
            time_step_s: Time step in seconds
            
        Returns:
            New rotor speed in RPM
        """
        # Get acceleration rates (RPM per second)
        accel_rate = self.mechanical_params.get('rotor_acceleration_rate', 0.5)
        decel_rate = self.mechanical_params.get('rotor_deceleration_rate', 0.3)
        damping = self.mechanical_params.get('damping_coefficient', 0.1)
        
        # Calculate speed difference
        speed_diff = target_rpm - current_rpm
        
        # Determine rate limit based on direction
        if speed_diff > 0:  # Accelerating
            max_change = accel_rate * time_step_s
        else:  # Decelerating
            max_change = -decel_rate * time_step_s
        
        # Apply damping (makes it smoother)
        damped_diff = speed_diff * (1 - damping)
        
        # Limit change rate
        actual_change = np.clip(damped_diff, max_change, -max_change)
        
        # Apply change
        new_rpm = current_rpm + actual_change
        
        # Ensure non-negative
        new_rpm = max(0.0, new_rpm)
        
        return new_rpm
    
    def calculate_vibration(
        self,
        rotor_rpm: float,
        power_kw: float,
        rated_power_kw: float,
        base_vibration: float = 0.5
    ) -> float:
        """
        Calculate vibration level based on operation.
        
        Vibration increases with:
        - Rotor speed
        - Power output (mechanical load)
        
        Args:
            rotor_rpm: Current rotor speed
            power_kw: Current power output
            rated_power_kw: Rated power
            base_vibration: Base vibration when idle
            
        Returns:
            Vibration level in mm/s
        """
        # Speed component (normalized to max 20 RPM)
        speed_factor = rotor_rpm / 20.0
        speed_vibration = 2.0 * speed_factor
        
        # Load component
        load_factor = power_kw / rated_power_kw
        load_vibration = 1.5 * load_factor
        
        # Total vibration
        vibration = base_vibration + speed_vibration + load_vibration
        
        # Add small random variation
        vibration += np.random.normal(0, 0.1)
        
        # Clamp to reasonable range
        vibration = np.clip(vibration, 0, 10)
        
        return vibration


class PhysicsEngine:
    """
    Main physics engine coordinating all physics models.
    """
    
    def __init__(self, config: PhysicsConfig):
        """
        Initialize physics engine.
        
        Args:
            config: Physics configuration
        """
        self.thermal_model = ThermalModel(config)
        self.mechanical_model = MechanicalModel(config)
    
    def step(
        self,
        current_state: dict,
        target_rpm: float,
        power_output_kw: float,
        rated_power_kw: float,
        ambient_temp_c: float,
        time_step_s: float
    ) -> dict:
        """
        Execute one physics time step.
        
        Args:
            current_state: Current turbine state dict
            target_rpm: Target rotor speed from wind
            power_output_kw: Current power output
            rated_power_kw: Rated power
            ambient_temp_c: Ambient temperature
            time_step_s: Time step in seconds
            
        Returns:
            Updated state dict with new values
        """
        updated_state = current_state.copy()
        
        # Update rotor speed (with inertia)
        updated_state['rotor_speed_rpm'] = self.mechanical_model.update_rotor_speed(
            current_state.get('rotor_speed_rpm', 0),
            target_rpm,
            time_step_s
        )
        
        # Update temperatures
        current_temps = {
            'generator_temp_c': current_state.get('generator_temp_c', ambient_temp_c + 20),
            'gearbox_temp_c': current_state.get('gearbox_temp_c', ambient_temp_c + 15),
            'bearing_temp_c': current_state.get('bearing_temp_c', ambient_temp_c + 10),
        }
        
        new_temps = self.thermal_model.update_all_temperatures(
            current_temps,
            ambient_temp_c,
            power_output_kw,
            rated_power_kw,
            time_step_s
        )
        
        updated_state.update(new_temps)
        
        # Update vibration
        updated_state['vibration_mms'] = self.mechanical_model.calculate_vibration(
            updated_state['rotor_speed_rpm'],
            power_output_kw,
            rated_power_kw
        )
        
        return updated_state


if __name__ == "__main__":
    # Test physics engine
    from .config import get_config
    
    config = get_config()
    physics = PhysicsEngine(config.physics)
    
    print("Physics Engine Test")
    print("=" * 50)
    
    # Simulate a few time steps
    state = {
        'rotor_speed_rpm': 0,
        'generator_temp_c': 20,
        'gearbox_temp_c': 20,
        'bearing_temp_c': 20,
        'vibration_mms': 0.5
    }
    
    print(f"{'Step':<6} {'RPM':<8} {'Gen °C':<10} {'Gear °C':<10} {'Vib mm/s':<10}")
    print("-" * 50)
    
    # Simulate startup and operation
    for i in range(20):
        target_rpm = 15.0 if i > 5 else 0.0
        power_kw = 2000 if i > 5 else 0
        
        state = physics.step(
            state,
            target_rpm=target_rpm,
            power_output_kw=power_kw,
            rated_power_kw=2500,
            ambient_temp_c=15,
            time_step_s=10
        )
        
        print(f"{i:<6} {state['rotor_speed_rpm']:<8.2f} "
              f"{state['generator_temp_c']:<10.1f} "
              f"{state['gearbox_temp_c']:<10.1f} "
              f"{state['vibration_mms']:<10.2f}")
