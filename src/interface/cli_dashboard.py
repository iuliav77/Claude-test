"""
CLI Dashboard for real-time turbine monitoring.

Beautiful terminal-based dashboard using Rich library.
Updates at 1 Hz showing live simulation data.
"""

import time
from datetime import datetime
from typing import Optional, List
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.text import Text
from rich import box

from twin.state import TurbineState
from twin.turbine import OperationalStatus
from twin.simulator import Simulator, SimulationMode
from twin.events import Event, EventSeverity


class DashboardWidget:
    """Base class for dashboard widgets."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render the widget."""
        raise NotImplementedError


class HeaderWidget(DashboardWidget):
    """Header with title and current time."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render header."""
        title = Text("WINDTWIN-AI: Digital Twin Dashboard", style="bold cyan")
        
        # Status line
        status_color = self._get_status_color(state.status)
        status_text = f"Status: [{status_color}]{state.status.name}[/{status_color}]"
        
        time_str = state.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        mode_str = simulator.mode.name
        speed_str = f"{simulator.speed_multiplier}x"
        
        content = Table.grid(padding=(0, 2))
        content.add_column(justify="left")
        content.add_column(justify="right")
        
        content.add_row(title, "")
        content.add_row(
            status_text,
            f"Time: {time_str} | Mode: {mode_str} | Speed: {speed_str}"
        )
        
        return Panel(content, box=box.DOUBLE)
    
    @staticmethod
    def _get_status_color(status: OperationalStatus) -> str:
        """Get color for status."""
        colors = {
            OperationalStatus.RUNNING: "green",
            OperationalStatus.RATED_POWER: "bright_green",
            OperationalStatus.STOPPED: "yellow",
            OperationalStatus.STARTING: "cyan",
            OperationalStatus.STOPPING: "cyan",
            OperationalStatus.FAULT: "red",
            OperationalStatus.EMERGENCY_STOP: "bright_red",
            OperationalStatus.MAINTENANCE: "blue"
        }
        return colors.get(status, "white")


class PowerGaugeWidget(DashboardWidget):
    """Power output gauge."""
    
    def __init__(self, rated_power_kw: float = 2500):
        """Initialize with rated power."""
        self.rated_power_kw = rated_power_kw
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render power gauge."""
        power = state.power_output_kw
        percentage = (power / self.rated_power_kw) * 100
        
        # Create bar
        bar_width = 40
        filled = int((power / self.rated_power_kw) * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Color based on percentage
        if percentage >= 90:
            color = "bright_green"
        elif percentage >= 50:
            color = "green"
        elif percentage >= 10:
            color = "yellow"
        else:
            color = "dim"
        
        content = f"[{color}]{bar}[/{color}]  {power:,.0f} kW  ({percentage:.1f}%)\n"
        content += f"Rated: {self.rated_power_kw:,.0f} kW"
        
        return Panel(
            content,
            title="Power Output",
            border_style=color
        )


class StatusWidget(DashboardWidget):
    """Turbine status parameters."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render status table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        
        # Weather conditions
        table.add_row("Wind Speed", f"{state.wind_speed_ms:.1f} m/s")
        table.add_row("Wind Direction", f"{state.wind_direction_deg:.0f}° ({self._wind_direction_name(state.wind_direction_deg)})")
        table.add_row("Temperature", f"{state.ambient_temp_c:.1f}°C")
        
        # Turbine parameters
        table.add_row("", "")  # Spacer
        table.add_row("Rotor Speed", f"{state.rotor_speed_rpm:.1f} RPM")
        table.add_row("Generator Temp", self._format_temp(state.generator_temp_c, 110))
        table.add_row("Gearbox Temp", self._format_temp(state.gearbox_temp_c, 100))
        table.add_row("Vibration", self._format_vibration(state.vibration_mms))
        
        return Panel(table, title="Turbine Status", border_style="blue")
    
    @staticmethod
    def _wind_direction_name(degrees: float) -> str:
        """Convert degrees to compass direction."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((degrees + 22.5) / 45) % 8
        return directions[index]
    
    @staticmethod
    def _format_temp(temp: float, limit: float) -> str:
        """Format temperature with color based on limit."""
        percentage = (temp / limit) * 100
        if percentage >= 95:
            return f"[red]{temp:.1f}°C[/red]"
        elif percentage >= 85:
            return f"[yellow]{temp:.1f}°C[/yellow]"
        else:
            return f"{temp:.1f}°C"
    
    @staticmethod
    def _format_vibration(vib: float) -> str:
        """Format vibration with color."""
        if vib >= 7:
            return f"[red]{vib:.2f} mm/s[/red]"
        elif vib >= 5:
            return f"[yellow]{vib:.2f} mm/s[/yellow]"
        else:
            return f"{vib:.2f} mm/s"


class MetricsWidget(DashboardWidget):
    """Production metrics."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render metrics table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        # Calculate additional metrics
        capacity_factor = simulator.state_manager.calculate_capacity_factor()
        availability = simulator.state_manager.calculate_availability()
        
        table.add_row("Energy Produced", f"{state.total_energy_mwh:.1f} MWh")
        table.add_row("Operating Hours", f"{state.operating_hours:.1f} h")
        table.add_row("Capacity Factor", f"{capacity_factor:.1%}")
        table.add_row("Availability", f"{availability:.1%}")
        table.add_row("Start Count", str(state.start_count))
        
        if state.fault_count > 0:
            table.add_row("Fault Count", f"[red]{state.fault_count}[/red]")
        
        return Panel(table, title="Production Metrics", border_style="green")


class ForecastWidget(DashboardWidget):
    """Power forecast display (Phase 4)."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render forecast information."""
        if not state.forecast_available or state.forecast_power_1h is None:
            content = "[dim]Forecast unavailable[/dim]\n"
            content += "[dim](Generating forecast...)[/dim]"
            return Panel(content, title="1-Hour Forecast", border_style="magenta")
        
        forecast = state.forecast_power_1h
        current = state.power_output_kw
        delta = forecast - current
        
        # Determine trend
        if abs(delta) < 50:  # Less than 50 kW change
            trend = "→"
            trend_color = "white"
        elif delta > 0:
            trend = "↑"
            trend_color = "green"
        else:
            trend = "↓"
            trend_color = "yellow"
        
        # Format content
        content = f"[bold]Forecasted Power:[/bold] {forecast:.0f} kW\n"
        content += f"[bold]Current Power:[/bold]    {current:.0f} kW\n"
        content += f"[bold]Delta:[/bold]            [{trend_color}]{delta:+.0f} kW {trend}[/{trend_color}]\n"
        
        # Add percentage change if current > 0
        if current > 0:
            pct_change = (delta / current) * 100
            content += f"[bold]Change:[/bold]           [{trend_color}]{pct_change:+.1f}%[/{trend_color}]"
        
        return Panel(content, title="1-Hour Forecast", border_style="magenta")


class EventsWidget(DashboardWidget):
    """Recent events log."""
    
    def __init__(self, max_events: int = 5):
        """Initialize with max events to show."""
        self.max_events = max_events
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render recent events."""
        events = simulator.event_processor.get_recent_events(self.max_events)
        
        if not events:
            content = "[dim]No recent events[/dim]"
        else:
            lines = []
            for event in reversed(events):  # Newest first
                time_str = event.timestamp.strftime("%H:%M:%S")
                severity_color = self._get_severity_color(event.severity)
                severity_str = event.severity.value.upper()
                
                line = (
                    f"[dim]{time_str}[/dim]  "
                    f"[{severity_color}]{severity_str:<8}[/{severity_color}]  "
                    f"{event.message}"
                )
                lines.append(line)
            
            content = "\n".join(lines)
        
        return Panel(content, title="Recent Events", border_style="yellow")
    
    @staticmethod
    def _get_severity_color(severity: EventSeverity) -> str:
        """Get color for severity."""
        colors = {
            EventSeverity.INFO: "blue",
            EventSeverity.WARNING: "yellow",
            EventSeverity.ERROR: "red",
            EventSeverity.CRITICAL: "bright_red"
        }
        return colors.get(severity, "white")


class ControlsWidget(DashboardWidget):
    """Control instructions."""
    
    def render(self, state: TurbineState, simulator: Simulator) -> Panel:
        """Render controls."""
        controls = (
            "[cyan]P[/cyan] Pause/Resume  "
            "[cyan]F[/cyan] Toggle Speed (1x→10x→100x)  "
            "[cyan]S[/cyan] Stop  "
            "[cyan]C[/cyan] Checkpoint  "
            "[cyan]Q[/cyan] Quit"
        )
        
        return Panel(controls, title="Controls", border_style="magenta")


class CLIDashboard:
    """
    Main CLI dashboard coordinator.
    
    Manages widgets and live updates.
    """
    
    def __init__(self, simulator: Simulator, refresh_rate: float = 1.0):
        """
        Initialize dashboard.
        
        Args:
            simulator: Simulator to monitor
            refresh_rate: Update frequency in Hz
        """
        self.simulator = simulator
        self.refresh_rate = refresh_rate
        self.console = Console()
        
        # Create widgets
        rated_power = simulator.config.turbine.rated_power_kw
        self.widgets = {
            'header': HeaderWidget(),
            'power': PowerGaugeWidget(rated_power),
            'status': StatusWidget(),
            'metrics': MetricsWidget(),
            'forecast': ForecastWidget(),  # Phase 4
            'events': EventsWidget(5),
            'controls': ControlsWidget()
        }
        
        # Register as observer
        simulator.add_observer(self._on_state_update)
        
        self._last_state: Optional[TurbineState] = None
    
    def _on_state_update(self, state: TurbineState):
        """Called when simulator state updates."""
        self._last_state = state
    
    def _create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="power", size=5),
            Layout(name="status"),
            Layout(name="metrics")
        )
        
        layout["right"].split_column(
            Layout(name="forecast", size=7),  # Phase 4
            Layout(name="events", ratio=1)
        )
        
        return layout
    
    def _update_layout(self, layout: Layout):
        """Update layout with current data."""
        state = self._last_state or self.simulator.state_manager.current_state
        
        if state is None:
            return
        
        # Update each section
        layout["header"].update(self.widgets['header'].render(state, self.simulator))
        layout["power"].update(self.widgets['power'].render(state, self.simulator))
        layout["status"].update(self.widgets['status'].render(state, self.simulator))
        layout["metrics"].update(self.widgets['metrics'].render(state, self.simulator))
        layout["forecast"].update(self.widgets['forecast'].render(state, self.simulator))  # Phase 4
        layout["events"].update(self.widgets['events'].render(state, self.simulator))
        layout["footer"].update(self.widgets['controls'].render(state, self.simulator))
    
    def run(self, duration_s: Optional[float] = None):
        """
        Run dashboard with live updates.
        
        Args:
            duration_s: Duration to run (None = until stopped)
        """
        layout = self._create_layout()
        
        with Live(
            layout,
            console=self.console,
            refresh_per_second=self.refresh_rate,
            screen=True
        ) as live:
            start_time = time.time()
            
            while self.simulator.is_running:
                # Update display
                self._update_layout(layout)
                live.update(layout)
                
                # Check duration
                if duration_s and (time.time() - start_time) >= duration_s:
                    break
                
                # Small sleep to prevent CPU spinning
                time.sleep(1.0 / self.refresh_rate)
    
    def display_summary(self):
        """Display simulation summary after completion."""
        summary = self.simulator.get_summary()
        
        self.console.print("\n" + "=" * 60)
        self.console.print("[bold cyan]Simulation Complete[/bold cyan]")
        self.console.print("=" * 60)
        
        table = Table(show_header=False, box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Simulation Time", f"{summary['simulation_time_h']:.2f} hours")
        table.add_row("Total Energy", f"{summary['total_energy_mwh']:.1f} MWh")
        table.add_row("Operating Hours", f"{summary['operating_hours']:.1f} h")
        table.add_row("Capacity Factor", f"{summary['capacity_factor']:.1%}")
        table.add_row("Availability", f"{summary['availability']:.1%}")
        table.add_row("Start Count", str(summary['start_count']))
        table.add_row("Fault Count", str(summary['fault_count']))
        
        self.console.print(table)
        self.console.print()


if __name__ == "__main__":
    # Test dashboard (static display)
    from datetime import datetime
    from twin.turbine import OperationalStatus
    from twin.state import TurbineState
    import sys
    sys.path.insert(0, '/home/claude/windtwin-ai/src')
    
    from data.weather_generator import WeatherGenerator
    from twin.simulator import Simulator
    
    print("Dashboard Test (Static Display)")
    print("=" * 50)
    print("Note: Full interactive dashboard requires simulation loop")
    print()
    
    # Generate sample data
    weather_gen = WeatherGenerator(seed=42)
    weather_data = weather_gen.generate(
        start_date="2024-01-01",
        days=1,
        interval_minutes=10
    )
    
    # Create simulator
    sim = Simulator(weather_data)
    
    # Run a few steps
    for _ in range(20):
        sim.step()
    
    # Create dashboard and display once
    dashboard = CLIDashboard(sim, refresh_rate=1.0)
    
    # Display summary
    dashboard.display_summary()
    
    print("✅ Dashboard components working!")
    print("\nTo see live dashboard, run:")
    print("  python main.py simulate --days 1")
