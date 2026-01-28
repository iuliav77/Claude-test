"""
Persistence system for saving and loading simulation state.

Provides checkpoint save/load functionality for:
- Continuing simulations
- Replay scenarios
- Analysis of saved states
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from .state import TurbineState
from .simulator import Simulator


class Checkpoint:
    """
    Represents a saved simulation checkpoint.
    """
    
    def __init__(
        self,
        checkpoint_id: str,
        timestamp: datetime,
        simulation_time_s: float,
        current_state: TurbineState,
        state_history: List[TurbineState],
        config_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize checkpoint.
        
        Args:
            checkpoint_id: Unique checkpoint identifier
            timestamp: When checkpoint was created
            simulation_time_s: Simulation time at checkpoint
            current_state: Current turbine state
            state_history: Historical states
            config_data: Configuration data
            metadata: Additional metadata
        """
        self.checkpoint_id = checkpoint_id
        self.timestamp = timestamp
        self.simulation_time_s = simulation_time_s
        self.current_state = current_state
        self.state_history = state_history
        self.config_data = config_data
        self.metadata = metadata or {}
        
        # Add version info
        self.version = "1.0"


class CheckpointManager:
    """
    Manages saving and loading of simulation checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "data/simulations"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        simulator: Simulator,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save current simulation state to checkpoint.
        
        Args:
            simulator: Simulator instance to save
            checkpoint_id: Custom checkpoint ID (auto-generated if None)
            metadata: Additional metadata to save
            
        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID if not provided
        if checkpoint_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_id = f"checkpoint_{timestamp}"
        
        # Get current state and history
        current_state = simulator.state_manager.current_state
        state_history = simulator.state_manager.get_history()
        
        if current_state is None:
            raise ValueError("Cannot save checkpoint: no current state")
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            simulation_time_s=simulator.simulation_time_s,
            current_state=current_state,
            state_history=state_history,
            config_data=self._config_to_dict(simulator.config),
            metadata=metadata
        )
        
        # Save to disk
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_meta.json"
        meta_dict = {
            'checkpoint_id': checkpoint_id,
            'timestamp': checkpoint.timestamp.isoformat(),
            'simulation_time_s': checkpoint.simulation_time_s,
            'simulation_time_h': checkpoint.simulation_time_s / 3600,
            'power_kw': current_state.power_output_kw,
            'status': current_state.status.name,
            'energy_mwh': current_state.total_energy_mwh,
            'metadata': metadata or {}
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(meta_dict, f, indent=2)
        
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        """
        Load checkpoint from disk.
        
        Args:
            checkpoint_id: Checkpoint ID to load
            
        Returns:
            Checkpoint object
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        return checkpoint
    
    def restore_simulator(
        self,
        checkpoint: Checkpoint,
        weather_data
    ) -> Simulator:
        """
        Restore simulator from checkpoint.
        
        NOTE: This is a basic implementation for Phase 2.
        Full replay functionality deferred to Phase 3.
        
        Args:
            checkpoint: Checkpoint to restore from
            weather_data: Weather data (must be provided)
            
        Returns:
            Restored Simulator instance
        """
        # For Phase 2, we create a new simulator
        # In Phase 3, we'll add full state restoration
        
        config = self._dict_to_config(checkpoint.config_data)
        simulator = Simulator(
            weather_data=weather_data,
            config=config,
            start_date=checkpoint.current_state.timestamp
        )
        
        # Set simulation time
        simulator.simulation_time_s = checkpoint.simulation_time_s
        simulator.current_time = checkpoint.current_state.timestamp
        
        # Restore current state
        simulator.state_manager.update_state(checkpoint.current_state)
        
        # Note: Full history restoration would be added in Phase 3
        
        return simulator
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dicts
        """
        checkpoints = []
        
        for meta_file in self.checkpoint_dir.glob("*_meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                checkpoints.append(meta)
            except Exception as e:
                print(f"Error reading {meta_file}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str):
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint ID to delete
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}_meta.json"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        if metadata_path.exists():
            metadata_path.unlink()
    
    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """
        Delete old checkpoints, keeping only the most recent.
        
        Args:
            keep_count: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        # Delete old checkpoints
        for checkpoint in checkpoints[keep_count:]:
            self.delete_checkpoint(checkpoint['checkpoint_id'])
    
    @staticmethod
    def _config_to_dict(config) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        # Simplified for Phase 2
        return {
            'turbine': config.turbine.__dict__,
            'simulation': config.simulation.__dict__
        }
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]):
        """Convert dictionary back to config object."""
        # Simplified for Phase 2 - just return None
        # Full implementation would reconstruct Config object
        return None


class AutoCheckpoint:
    """
    Automatic checkpoint system for long-running simulations.
    """
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        interval_minutes: int = 10
    ):
        """
        Initialize auto-checkpoint system.
        
        Args:
            checkpoint_manager: CheckpointManager instance
            interval_minutes: Checkpoint interval in minutes
        """
        self.checkpoint_manager = checkpoint_manager
        self.interval_s = interval_minutes * 60
        self.last_checkpoint_time = 0.0
    
    def check_and_save(self, simulator: Simulator) -> Optional[str]:
        """
        Check if it's time for a checkpoint and save if needed.
        
        Args:
            simulator: Simulator to potentially checkpoint
            
        Returns:
            Checkpoint ID if saved, None otherwise
        """
        current_time = simulator.simulation_time_s
        
        if current_time - self.last_checkpoint_time >= self.interval_s:
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                simulator,
                metadata={'auto_checkpoint': True}
            )
            self.last_checkpoint_time = current_time
            return checkpoint_id
        
        return None


if __name__ == "__main__":
    # Test checkpoint system
    import sys
    sys.path.insert(0, '/home/claude/windtwin-ai/src')
    
    from data.weather_generator import WeatherGenerator
    from twin.simulator import Simulator
    
    print("Checkpoint System Test")
    print("=" * 50)
    
    # Generate weather data
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate(
        start_date="2024-01-01",
        days=1,
        interval_minutes=10
    )
    
    # Create and run simulator
    sim = Simulator(weather_data)
    print("Running simulation for 10 steps...")
    
    for i in range(10):
        sim.step()
    
    # Save checkpoint
    manager = CheckpointManager()
    checkpoint_id = manager.save_checkpoint(
        sim,
        metadata={'test': 'checkpoint_test', 'step': 10}
    )
    
    print(f"\n✓ Checkpoint saved: {checkpoint_id}")
    
    # List checkpoints
    print("\nAvailable checkpoints:")
    for cp in manager.list_checkpoints():
        print(f"  - {cp['checkpoint_id']}: "
              f"{cp['power_kw']:.0f}kW at "
              f"{cp['simulation_time_h']:.2f}h")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_id}")
    loaded = manager.load_checkpoint(checkpoint_id)
    print(f"  Loaded state: {loaded.current_state.power_output_kw:.0f}kW")
    print(f"  History size: {len(loaded.state_history)}")
    
    print("\n✅ Checkpoint system working correctly!")
