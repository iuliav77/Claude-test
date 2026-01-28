#!/usr/bin/env python3
"""
Phase 4 Demo: Real-Time Power Forecasting

This script demonstrates the complete Phase 4 workflow:
1. Generate synthetic data
2. Train a power forecasting model
3. Run simulation with live forecasting
4. Display forecasts in dashboard
"""

import sys
sys.path.insert(0, 'src')

import subprocess
from pathlib import Path

print("=" * 70)
print("WINDTWIN-AI: Phase 4 - Real-Time Power Forecasting Demo")
print("=" * 70)
print()

# Step 1: Train a forecasting model (if not exists)
model_path = Path("models/power_forecast_1h.pkl")

if model_path.exists():
    print("✓ Forecasting model already exists")
    print(f"  Location: {model_path}")
    print()
else:
    print("Step 1: Training power forecasting model...")
    print("-" * 70)
    
    # Import and run training
    from data.weather_generator import WeatherGenerator
    from twin.simulator import Simulator
    from ai.features import FeatureEngineer, prepare_train_test_split
    from ai.forecaster import PowerForecaster
    
    # Generate training data (30 days for better model)
    print("  Generating 30 days of training data...")
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate("2024-01-01", days=30, interval_minutes=10)
    
    # Run simulation to collect states
    print("  Running simulation to collect states...")
    sim = Simulator(weather_data)
    for i in range(len(weather_data)):
        sim.step()
        if i % 1440 == 0 and i > 0:
            print(f"    Day {i // 144}...")
    
    states = sim.state_manager.get_history()
    print(f"  Collected {len(states)} states")
    
    # Engineer features
    print("  Engineering features...")
    engineer = FeatureEngineer(
        lag_hours=[1, 3, 6],
        rolling_windows=[6, 12, 24],
        include_time_features=True,
        include_derived_features=True
    )
    features_df = engineer.engineer_features(states, for_forecasting=True)
    
    # Train/test split
    train_df, test_df = prepare_train_test_split(features_df, test_size=0.2, drop_na=True)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Train forecaster
    print("  Training model...")
    forecaster = PowerForecaster(
        model_name="power_forecast_1h",
        forecast_horizon_hours=1,
        scale_features=True
    )
    
    metrics = forecaster.train(
        train_df,
        val_df=test_df,
        filter_operational=True,
        min_power_kw=100.0
    )
    
    # Save model
    forecaster.save()
    print()
    print(f"✓ Model trained and saved to {model_path}")
    print()

# Step 2: Run simulation with forecasting
print("Step 2: Running simulation with live forecasting")
print("-" * 70)
print()
print("Starting simulation with:")
print("  - Duration: 1 day")
print("  - Mode: Realtime")
print("  - Dashboard: Enabled")
print("  - Forecasting: Enabled")
print()
print("The dashboard will show:")
print("  • Current power output")
print("  • 1-hour ahead forecast")
print("  • Power delta (forecast - current)")
print("  • Trend indicator (↑ ↓ →)")
print()
print("Press Ctrl+C to stop the simulation")
print()
input("Press Enter to start...")
print()

# Run simulation with forecasting
subprocess.run([
    sys.executable, "main.py", "simulate",
    "--days", "1",
    "--mode", "realtime",
    "--dashboard",
    "--forecast"
])

print()
print("=" * 70)
print("Demo Complete!")
print("=" * 70)
print()
print("What happened:")
print("  1. Generated/loaded training data")
print("  2. Trained LinearRegression forecaster")
print("  3. Saved model to models/power_forecast_1h.pkl")
print("  4. Ran simulation with real-time forecasting")
print("  5. Dashboard displayed live 1h power forecasts")
print()
print("Try it yourself:")
print("  python main.py simulate --days 1 --dashboard --forecast")
print()
