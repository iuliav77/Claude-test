"""
Event system for turbine simulation.

Handles discrete events that affect the simulation:
- Weather events
- Operational commands
- Fault injection
- System events
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import deque


class EventType(Enum):
    """Types of events in the simulation."""
    
    # Weather events
    WEATHER_CHANGE = "weather_change"
    STORM_APPROACHING = "storm_approaching"
    WIND_GUST = "wind_gust"
    
    # Operational commands
    START_COMMAND = "start_command"
    STOP_COMMAND = "stop_command"
    EMERGENCY_STOP = "emergency_stop"
    MAINTENANCE_MODE = "maintenance_mode"
    RESUME_OPERATION = "resume_operation"
    
    # Fault events
    GENERATOR_OVERHEAT = "generator_overheat"
    GEARBOX_OVERHEAT = "gearbox_overheat"
    EXCESSIVE_VIBRATION = "excessive_vibration"
    ROTOR_OVERSPEED = "rotor_overspeed"
    GRID_DISCONNECTION = "grid_disconnection"
    COMPONENT_FAILURE = "component_failure"
    
    # System events
    SIMULATION_START = "simulation_start"
    SIMULATION_STOP = "simulation_stop"
    CHECKPOINT_SAVE = "checkpoint_save"
    CHECKPOINT_LOAD = "checkpoint_load"
    
    # Generic
    CUSTOM = "custom"


class EventSeverity(Enum):
    """Event severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Event:
    """
    Represents a discrete event in the simulation.
    """
    # Core attributes
    event_id: int
    timestamp: datetime
    simulation_time_s: float
    event_type: EventType
    severity: EventSeverity
    
    # Event data
    message: str
    data: Dict[str, Any]
    
    # Processing
    handled: bool = False
    handler_response: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of event."""
        return (
            f"[{self.severity.value.upper()}] "
            f"{self.timestamp.strftime('%H:%M:%S')} - "
            f"{self.event_type.value}: {self.message}"
        )


class EventHandler:
    """
    Base class for event handlers.
    
    Subclass this to create custom event handlers.
    """
    
    def can_handle(self, event: Event) -> bool:
        """
        Check if this handler can handle the event.
        
        Args:
            event: Event to check
            
        Returns:
            True if handler can process this event
        """
        raise NotImplementedError
    
    def handle(self, event: Event, context: Dict[str, Any]) -> Optional[str]:
        """
        Handle the event.
        
        Args:
            event: Event to handle
            context: Context data (e.g., current state)
            
        Returns:
            Optional response message
        """
        raise NotImplementedError


class EventProcessor:
    """
    Processes events and dispatches to handlers.
    
    Central event management system for the simulation.
    """
    
    def __init__(self, max_log_size: int = 1000):
        """
        Initialize event processor.
        
        Args:
            max_log_size: Maximum number of events to keep in log
        """
        self._handlers: List[EventHandler] = []
        self._event_log: deque = deque(maxlen=max_log_size)
        self._event_counter: int = 0
        
    def register_handler(self, handler: EventHandler):
        """
        Register an event handler.
        
        Args:
            handler: EventHandler instance
        """
        self._handlers.append(handler)
    
    def unregister_handler(self, handler: EventHandler):
        """
        Unregister an event handler.
        
        Args:
            handler: EventHandler instance to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    def create_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        message: str,
        timestamp: datetime,
        simulation_time_s: float,
        data: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Create a new event.
        
        Args:
            event_type: Type of event
            severity: Severity level
            message: Event message
            timestamp: Real timestamp
            simulation_time_s: Simulation time
            data: Additional event data
            
        Returns:
            Created Event object
        """
        self._event_counter += 1
        
        event = Event(
            event_id=self._event_counter,
            timestamp=timestamp,
            simulation_time_s=simulation_time_s,
            event_type=event_type,
            severity=severity,
            message=message,
            data=data or {}
        )
        
        return event
    
    def process_event(
        self,
        event: Event,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Process an event by dispatching to appropriate handlers.
        
        Args:
            event: Event to process
            context: Context data for handlers
            
        Returns:
            True if event was handled
        """
        # Add to log
        self._event_log.append(event)
        
        # Find and execute handlers
        context = context or {}
        handled = False
        
        for handler in self._handlers:
            if handler.can_handle(event):
                try:
                    response = handler.handle(event, context)
                    event.handler_response = response
                    event.handled = True
                    handled = True
                except Exception as e:
                    # Log handler error but continue
                    print(f"Error in handler {handler.__class__.__name__}: {e}")
        
        return handled
    
    def get_event_log(
        self,
        last_n: Optional[int] = None,
        severity: Optional[EventSeverity] = None,
        event_type: Optional[EventType] = None
    ) -> List[Event]:
        """
        Get events from the log.
        
        Args:
            last_n: Get last N events
            severity: Filter by severity
            event_type: Filter by type
            
        Returns:
            List of Event objects
        """
        events = list(self._event_log)
        
        # Apply filters
        if severity is not None:
            events = [e for e in events if e.severity == severity]
        
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]
        
        # Apply limit
        if last_n is not None:
            events = events[-last_n:]
        
        return events
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """
        Get most recent events.
        
        Args:
            count: Number of events to return
            
        Returns:
            List of recent Event objects
        """
        return list(self._event_log)[-count:]
    
    def clear_log(self):
        """Clear the event log."""
        self._event_log.clear()


# Built-in event handlers

class FaultHandler(EventHandler):
    """Handler for fault events."""
    
    def can_handle(self, event: Event) -> bool:
        """Check if this is a fault event."""
        return event.event_type in [
            EventType.GENERATOR_OVERHEAT,
            EventType.GEARBOX_OVERHEAT,
            EventType.EXCESSIVE_VIBRATION,
            EventType.ROTOR_OVERSPEED,
            EventType.COMPONENT_FAILURE
        ]
    
    def handle(self, event: Event, context: Dict[str, Any]) -> str:
        """
        Handle fault by stopping turbine.
        
        Args:
            event: Fault event
            context: Simulation context
            
        Returns:
            Response message
        """
        # In real implementation, would update turbine state
        return f"Turbine stopped due to {event.event_type.value}"


class MaintenanceHandler(EventHandler):
    """Handler for maintenance events."""
    
    def can_handle(self, event: Event) -> bool:
        """Check if this is a maintenance event."""
        return event.event_type == EventType.MAINTENANCE_MODE
    
    def handle(self, event: Event, context: Dict[str, Any]) -> str:
        """
        Handle maintenance mode request.
        
        Args:
            event: Maintenance event
            context: Simulation context
            
        Returns:
            Response message
        """
        return "Turbine entering maintenance mode"


class WeatherHandler(EventHandler):
    """Handler for weather events."""
    
    def can_handle(self, event: Event) -> bool:
        """Check if this is a weather event."""
        return event.event_type in [
            EventType.WEATHER_CHANGE,
            EventType.STORM_APPROACHING,
            EventType.WIND_GUST
        ]
    
    def handle(self, event: Event, context: Dict[str, Any]) -> str:
        """
        Handle weather event.
        
        Args:
            event: Weather event
            context: Simulation context
            
        Returns:
            Response message
        """
        if event.event_type == EventType.STORM_APPROACHING:
            return "Preparing for storm conditions"
        
        return f"Weather changed: {event.message}"


if __name__ == "__main__":
    # Test event system
    from datetime import datetime
    
    print("Event System Test")
    print("=" * 50)
    
    # Create event processor
    processor = EventProcessor()
    
    # Register handlers
    processor.register_handler(FaultHandler())
    processor.register_handler(MaintenanceHandler())
    processor.register_handler(WeatherHandler())
    
    # Create and process some events
    now = datetime.now()
    
    events = [
        processor.create_event(
            EventType.SIMULATION_START,
            EventSeverity.INFO,
            "Simulation started",
            now,
            0.0
        ),
        processor.create_event(
            EventType.WEATHER_CHANGE,
            EventSeverity.INFO,
            "Wind speed increased to 12 m/s",
            now,
            100.0,
            {'wind_speed': 12.0}
        ),
        processor.create_event(
            EventType.GENERATOR_OVERHEAT,
            EventSeverity.ERROR,
            "Generator temperature exceeded 110°C",
            now,
            500.0,
            {'temperature': 112.0}
        ),
        processor.create_event(
            EventType.MAINTENANCE_MODE,
            EventSeverity.WARNING,
            "Entering scheduled maintenance",
            now,
            1000.0
        )
    ]
    
    # Process all events
    for event in events:
        processor.process_event(event)
        print(f"{event}")
        if event.handler_response:
            print(f"  → {event.handler_response}")
    
    # Get recent events
    print("\nRecent events:")
    for event in processor.get_recent_events(3):
        print(f"  {event}")
    
    # Get error events only
    print("\nError events:")
    for event in processor.get_event_log(severity=EventSeverity.ERROR):
        print(f"  {event}")
    
    print("\n✅ Event system working correctly!")
