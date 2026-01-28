#!/usr/bin/env python3
"""
Quick test of Phase 5.1 real-time simulation loop.

This script verifies:
1. Simulator initializes correctly
2. Background thread starts
3. step() is called every 5 minutes (simulated as 5 seconds for testing)
4. Shutdown works cleanly
"""

import sys
sys.path.insert(0, 'src')

import time
import threading
from data.weather_generator import WeatherGenerator
from twin.simulator import Simulator

print("=" * 70)
print("Phase 5.1: Real-Time Simulation Loop Test")
print("=" * 70)
print()

# Initialize simulator
print("1. Initializing simulator...")
weather_gen = WeatherGenerator(seed=42)
weather_data = weather_gen.generate("2024-01-01", days=1, interval_minutes=10)
simulator = Simulator(weather_data)
print(f"   ✓ Simulator initialized with {len(weather_data)} weather samples")
print()

# Get initial state
initial_state = simulator.state_manager.current_state
print(f"2. Initial state:")
print(f"   Time: {initial_state.timestamp}")
print(f"   Power: {initial_state.power_output_kw:.0f} kW")
print()

# Simulate the background loop (with 5 second intervals for testing)
print("3. Running simulation loop (5 second intervals for testing)...")
print("   Press Ctrl+C to stop")
print()

shutdown_event = threading.Event()

def test_loop():
    """Test version of simulation_loop with 5 second intervals."""
    step_count = 0
    while not shutdown_event.is_set():
        # Wait 5 seconds (simulating 5 minutes in real deployment)
        for _ in range(5):
            if shutdown_event.is_set():
                break
            shutdown_event.wait(1)
        
        if shutdown_event.is_set():
            break
        
        # Advance simulation
        try:
            simulator.step()
            step_count += 1
            state = simulator.state_manager.current_state
            print(f"   Step {step_count}: {state.timestamp} | Power: {state.power_output_kw:.0f} kW | Wind: {state.wind_speed_ms:.1f} m/s")
        except Exception as e:
            print(f"   ⚠ Error: {e}")
            break

# Start test loop
loop_thread = threading.Thread(target=test_loop, daemon=False)
loop_thread.start()

try:
    # Let it run for ~15 seconds (3 steps)
    time.sleep(16)
except KeyboardInterrupt:
    print("\n   Interrupted by user")

# Shutdown
print()
print("4. Shutting down...")
shutdown_event.set()
loop_thread.join(timeout=5)

if loop_thread.is_alive():
    print("   ⚠ Thread did not stop")
else:
    print("   ✓ Thread stopped cleanly")

# Final state
final_state = simulator.state_manager.current_state
print()
print("5. Final state:")
print(f"   Time: {final_state.timestamp}")
print(f"   Power: {final_state.power_output_kw:.0f} kW")
print(f"   History size: {len(simulator.state_manager.get_history())}")
print()
print("=" * 70)
print("✓ Test Complete")
print("=" * 70)
