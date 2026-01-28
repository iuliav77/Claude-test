#!/usr/bin/env python3
"""
WINDTWIN-AI: Main entry point

Command-line interface for the wind turbine digital twin system.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.weather_generator import WeatherGenerator
from data.scada_generator import SCADAGenerator
from twin.simulator import Simulator, SimulationMode
from twin.persistence import CheckpointManager
from interface.cli_dashboard import CLIDashboard
import pandas as pd


def generate_data(args):
    """Generate synthetic weather and SCADA data."""
    print(f"🌀 WINDTWIN-AI: Generating {args.days} days of data...")
    print(f"   Start date: {args.start_date}")
    print(f"   Interval: {args.interval} minutes")
    print()
    
    # Create output directories
    weather_dir = Path("data/weather")
    scada_dir = Path("data/scada")
    weather_dir.mkdir(parents=True, exist_ok=True)
    scada_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate weather data
    print("📡 Generating weather data...")
    weather_gen = WeatherGenerator(seed=args.seed)
    weather_data = weather_gen.generate(
        start_date=args.start_date,
        days=args.days,
        interval_minutes=args.interval
    )
    
    # Save weather data
    date_str = datetime.now().strftime("%Y%m%d")
    weather_file = weather_dir / f"weather_{date_str}.csv"
    weather_gen.save(weather_data, str(weather_file))
    
    # Print weather statistics
    weather_stats = weather_gen.get_statistics(weather_data)
    print(f"   ✓ Samples: {weather_stats['samples']:,}")
    print(f"   ✓ Mean wind speed: {weather_stats['mean_wind_speed']:.2f} m/s")
    print(f"   ✓ Max wind speed: {weather_stats['max_wind_speed']:.2f} m/s")
    print(f"   ✓ Mean temperature: {weather_stats['mean_temperature']:.1f} °C")
    print()
    
    # Generate SCADA data
    print("⚡ Generating SCADA data...")
    scada_gen = SCADAGenerator(seed=args.seed)
    scada_data = scada_gen.generate(
        weather_data,
        anomaly_probability=args.anomaly_rate
    )
    
    # Save SCADA data
    scada_file = scada_dir / f"scada_{date_str}.csv"
    scada_gen.save(scada_data, str(scada_file))
    
    # Print SCADA statistics
    scada_stats = scada_gen.get_statistics(scada_data)
    print(f"   ✓ Samples: {scada_stats['samples']:,}")
    print(f"   ✓ Mean power: {scada_stats['mean_power_kw']:.0f} kW")
    print(f"   ✓ Max power: {scada_stats['max_power_kw']:.0f} kW")
    print(f"   ✓ Capacity factor: {scada_stats['capacity_factor']:.1%}")
    print(f"   ✓ Total energy: {scada_stats['total_energy_mwh']:.1f} MWh")
    print(f"   ✓ Operating time: {scada_stats['operating_time_pct']:.1f}%")
    print(f"   ✓ Fault time: {scada_stats['fault_time_pct']:.1f}%")
    print()
    
    print(f"✅ Data generation complete!")
    print(f"   Weather: {weather_file}")
    print(f"   SCADA: {scada_file}")


def show_stats(args):
    """Display statistics for existing data."""
    print("📊 WINDTWIN-AI: Data Statistics")
    print()
    
    # Check for data files
    weather_dir = Path("data/weather")
    scada_dir = Path("data/scada")
    
    weather_files = list(weather_dir.glob("*.csv")) if weather_dir.exists() else []
    scada_files = list(scada_dir.glob("*.csv")) if scada_dir.exists() else []
    
    if not weather_files and not scada_files:
        print("❌ No data found. Generate data first with:")
        print("   python main.py generate --days 7")
        return
    
    print(f"Weather files: {len(weather_files)}")
    for f in weather_files:
        print(f"   - {f.name}")
    
    print(f"\nSCADA files: {len(scada_files)}")
    for f in scada_files:
        print(f"   - {f.name}")
    
    print()
    print("💡 Use Jupyter notebook to explore data:")
    print("   jupyter notebook notebooks/exploration.ipynb")


def simulate(args):
    """Run digital twin simulation."""
    print(f"🌀 WINDTWIN-AI: Starting Simulation")
    print(f"   Duration: {args.days} days")
    print(f"   Mode: {args.mode}")
    if args.mode == 'fast':
        print(f"   Speed: {args.speed}x")
    print()
    
    # Generate or load weather data
    print("📡 Generating weather data...")
    weather_gen = WeatherGenerator(seed=args.seed)
    weather_data = weather_gen.generate(
        start_date=args.start_date,
        days=args.days,
        interval_minutes=10
    )
    print(f"   ✓ Generated {len(weather_data)} weather samples")
    print()
    
    # Create simulator
    print("⚙️  Initializing simulator...")
    
    # Check if forecast model exists and load if requested
    forecaster_path = None
    if args.forecast:
        from pathlib import Path
        default_model = Path("models/power_forecast_1h.pkl")
        if default_model.exists():
            forecaster_path = str(default_model)
        else:
            print("⚠️  Warning: Forecast model not found at models/power_forecast_1h.pkl")
            print("   Run demo to train a model first, or disable --forecast flag")
            print()
    
    simulator = Simulator(weather_data, forecaster_path=forecaster_path)
    print(f"   ✓ Turbine: {simulator.config.turbine.name}")
    print(f"   ✓ Rated Power: {simulator.config.turbine.rated_power_kw} kW")
    if simulator.forecast_enabled:
        print(f"   ✓ Forecast: Enabled (1h ahead)")
    print()
    
    # Setup checkpoint manager if enabled
    checkpoint_manager = None
    if args.checkpoint:
        checkpoint_manager = CheckpointManager()
        print("💾 Auto-checkpoint enabled")
        print()
    
    # Determine simulation mode
    if args.mode == 'realtime':
        mode = SimulationMode.REALTIME
        speed = args.speed
    elif args.mode == 'fast':
        mode = SimulationMode.FAST
        speed = args.speed
    else:
        mode = SimulationMode.STEP
        speed = 1.0
    
    # Run with or without dashboard
    if args.dashboard:
        print("🖥️  Launching dashboard...")
        print("   (Press Ctrl+C to stop)")
        print()
        
        # Create dashboard
        dashboard = CLIDashboard(simulator, refresh_rate=1.0)
        
        # Start simulation in background thread
        def run_simulation():
            simulator.run(mode=mode, speed_multiplier=speed)
        
        sim_thread = threading.Thread(target=run_simulation, daemon=True)
        sim_thread.start()
        
        # Run dashboard (blocks until stopped)
        try:
            dashboard.run()
        except KeyboardInterrupt:
            print("\n\nStopping simulation...")
            simulator.stop()
        
        # Show summary
        dashboard.display_summary()
        
    else:
        # Run without dashboard
        print("▶️  Running simulation...")
        simulator.run(mode=mode, speed_multiplier=speed)
        
        # Show summary
        summary = simulator.get_summary()
        print("\n✅ Simulation Complete!")
        print(f"   Time simulated: {summary['simulation_time_h']:.2f} hours")
        print(f"   Energy produced: {summary['total_energy_mwh']:.1f} MWh")
        print(f"   Capacity factor: {summary['capacity_factor']:.1%}")
        print(f"   Availability: {summary['availability']:.1%}")
    
    # Save checkpoint if requested
    if checkpoint_manager and args.checkpoint:
        checkpoint_id = checkpoint_manager.save_checkpoint(
            simulator,
            metadata={'cli_run': True, 'days': args.days}
        )
        print(f"\n💾 Checkpoint saved: {checkpoint_id}")


def show_stats(args):
    """Display statistics for existing data."""
    print("📊 WINDTWIN-AI: Data Statistics")
    print()
    
    # Check for data files
    weather_dir = Path("data/weather")
    scada_dir = Path("data/scada")
    
    weather_files = list(weather_dir.glob("*.csv")) if weather_dir.exists() else []
    scada_files = list(scada_dir.glob("*.csv")) if scada_dir.exists() else []
    
    if not weather_files and not scada_files:
        print("❌ No data found. Generate data first with:")
        print("   python main.py generate --days 7")
        return
    
    print(f"Weather files: {len(weather_files)}")
    for f in weather_files:
        print(f"   - {f.name}")
    
    print(f"\nSCADA files: {len(scada_files)}")
    for f in scada_files:
        print(f"   - {f.name}")
    
    print()
    print("💡 Use Jupyter notebook to explore data:")
    print("   jupyter notebook notebooks/exploration.ipynb")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WINDTWIN-AI: Wind Turbine Digital Twin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 7 days of data
  python main.py generate --days 7
  
  # Generate 30 days starting from specific date
  python main.py generate --days 30 --start-date 2024-01-01
  
  # Generate with custom seed and anomaly rate
  python main.py generate --days 14 --seed 123 --anomaly-rate 0.1
  
  # Show statistics for existing data
  python main.py stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to generate (default: 7)'
    )
    gen_parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date (YYYY-MM-DD, default: 2024-01-01)'
    )
    gen_parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Time interval in minutes (default: 10)'
    )
    gen_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    gen_parser.add_argument(
        '--anomaly-rate',
        type=float,
        default=0.05,
        help='Probability of anomalies (default: 0.05)'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show data statistics')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run digital twin simulation')
    sim_parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to simulate (default: 1)'
    )
    sim_parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date (YYYY-MM-DD, default: 2024-01-01)'
    )
    sim_parser.add_argument(
        '--mode',
        type=str,
        choices=['realtime', 'fast', 'step'],
        default='realtime',
        help='Simulation mode (default: realtime)'
    )
    sim_parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speed multiplier for fast mode (default: 1.0)'
    )
    sim_parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Show live dashboard'
    )
    sim_parser.add_argument(
        '--checkpoint',
        action='store_true',
        help='Enable auto-checkpoint'
    )
    sim_parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    sim_parser.add_argument(
        '--forecast',
        action='store_true',
        help='Enable real-time power forecasting (requires trained model)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'generate':
        generate_data(args)
    elif args.command == 'stats':
        show_stats(args)
    elif args.command == 'simulate':
        simulate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
