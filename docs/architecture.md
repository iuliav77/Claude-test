# WINDTWIN-AI Architecture

## System Overview

WINDTWIN-AI is a digital twin system for a single wind turbine that combines synthetic data generation, physics-based simulation, and AI-powered analytics.

## Design Principles

1. **Modularity**: Each component can be developed, tested, and replaced independently
2. **Local-First**: No external dependencies for core functionality
3. **Educational**: Clear code structure for learning
4. **Extensible**: Easy to add new features or scale to multiple turbines

## Component Architecture

### 1. Data Layer

#### SCADA Generator (`src/data/scada_generator.py`)
Generates realistic synthetic SCADA (Supervisory Control and Data Acquisition) data:

**Parameters Generated:**
- Power output (kW)
- Rotor speed (RPM)
- Generator temperature (°C)
- Gearbox temperature (°C)
- Nacelle position (°)
- Vibration levels (mm/s)
- Operational status (running, idle, fault)

**Characteristics:**
- 10-minute sampling interval (industry standard)
- Realistic noise and variation
- Correlation with weather conditions
- Occasional anomalies (5% of time)

#### Weather Generator (`src/data/weather_generator.py`)
Generates synthetic weather data affecting turbine performance:

**Parameters Generated:**
- Wind speed at hub height (m/s)
- Wind direction (°)
- Air temperature (°C)
- Atmospheric pressure (hPa)
- Air density (kg/m³)

**Characteristics:**
- Diurnal patterns (day/night variation)
- Weather fronts and transitions
- Seasonal trends
- Realistic turbulence

### 2. Digital Twin Core

#### Turbine Model (`src/twin/turbine.py`)
Represents the physical turbine with key parameters:

**Specifications:**
- Rated power: 2.5 MW
- Cut-in wind speed: 3 m/s
- Rated wind speed: 12 m/s
- Cut-out wind speed: 25 m/s
- Rotor diameter: 90m
- Hub height: 80m

**Power Curve:**
```
P = 0                          if v < 3 m/s
P = k * (v³ - 3³)             if 3 ≤ v < 12 m/s
P = 2500 kW                    if 12 ≤ v < 25 m/s
P = 0                          if v ≥ 25 m/s
```

#### Simulator (`src/twin/simulator.py`)
Time-step simulation engine:

**Features:**
- Configurable time step (default: 10 minutes)
- State persistence
- Event logging
- Real-time or accelerated simulation

**State Variables:**
- Current power output
- Rotor speed
- Component temperatures
- Operational mode
- Accumulated energy production
- Operating hours

### 3. AI Layer

#### Production Forecaster (`src/ai/forecaster.py`)
Predicts turbine power output for the next 1-6 hours:

**Input Features:**
- Current power output
- Recent power trend (1-3 hours)
- Wind speed forecast
- Wind direction
- Air temperature
- Time of day
- Season

**Model:**
- Algorithm: Random Forest Regressor or Gradient Boosting
- Training: Rolling window approach
- Evaluation: MAE, RMSE, R²

**Output:**
- Hourly power forecast (6 points)
- Confidence intervals
- Expected energy production

#### Anomaly Detector (`src/ai/anomaly_detector.py`)
Identifies unusual patterns indicating potential issues:

**Methods:**
1. **Statistical**: Isolation Forest
2. **Threshold-based**: Operating range violations
3. **Pattern-based**: Unusual correlations

**Anomaly Types:**
- Performance degradation
- Temperature spikes
- Vibration anomalies
- Power curve deviations
- Unexpected shutdowns

**Output:**
- Anomaly score (0-100)
- Anomaly type classification
- Severity level (low, medium, high, critical)
- Potential root cause

### 4. Interface Layer

#### CLI Dashboard (`src/interface/cli_dashboard.py`)
Terminal-based real-time monitoring:

**Features:**
- Live turbine status
- Current power output (gauge display)
- Weather conditions
- Recent alerts
- Production statistics
- Forecast preview

**Technology:** Rich library for beautiful terminal UI

#### Data Export (`src/interface/data_export.py`)
Export functionality for analysis:

**Formats:**
- CSV: Time-series data
- JSON: Configuration and metadata
- Parquet: Compressed columnar storage

## Data Flow

```
1. Weather Generator → Weather Data
                           ↓
2. SCADA Generator ← Weather Data → SCADA Data
                           ↓
3. Simulator ← SCADA + Weather → Digital Twin State
                           ↓
4. AI Models ← State + History → Forecasts + Anomalies
                           ↓
5. Dashboard ← All Data → User Interface
```

## Time-Step Simulation Loop

```python
for each time_step:
    1. Get current weather conditions
    2. Calculate expected power output (physics model)
    3. Generate SCADA measurements (with noise)
    4. Update turbine state
    5. Check for anomalies
    6. Generate forecast for next period
    7. Log all data
    8. Update dashboard
```

## Storage Strategy

### Generated Data
- **Location**: `data/scada/` and `data/weather/`
- **Format**: CSV files with timestamps
- **Naming**: `scada_YYYYMMDD.csv`, `weather_YYYYMMDD.csv`
- **Retention**: Keep last 90 days (configurable)

### Models
- **Location**: `models/`
- **Format**: scikit-learn pickle or joblib
- **Versioning**: Include training date in filename
- **Metadata**: JSON file with model parameters

### Logs
- **Location**: `logs/`
- **Format**: Structured JSON logs
- **Levels**: INFO, WARNING, ERROR, CRITICAL

## Configuration

All system parameters stored in `config.py`:

```python
TURBINE_CONFIG = {
    'rated_power': 2500,  # kW
    'rotor_diameter': 90,  # m
    'hub_height': 80,  # m
    ...
}

SIMULATION_CONFIG = {
    'time_step': 600,  # seconds (10 minutes)
    'start_date': '2024-01-01',
    'duration_days': 365,
    ...
}

AI_CONFIG = {
    'forecast_horizon': 6,  # hours
    'anomaly_threshold': 0.75,
    ...
}
```

## Extension Points

### Easy to Add:
- Multiple turbines (wind farm)
- Additional sensors (blade pitch, yaw angle)
- Weather API integration (replace synthetic data)
- Web dashboard (Flask/FastAPI)
- Database storage (PostgreSQL, InfluxDB)
- Predictive maintenance models
- Grid integration simulation

### Modular Replacement:
- Swap physics model (detailed to simple)
- Change AI algorithms (tree-based to neural nets)
- Replace data source (synthetic to real)
- Switch UI (CLI to web)

## Performance Considerations

- **Memory**: ~100MB for 1 year of 10-min data
- **CPU**: Minimal (simulation runs in real-time or faster)
- **Disk**: ~500MB per year (uncompressed CSV)
- **AI Training**: <1 minute on standard laptop

## Security Notes

Since this is local-first:
- No authentication needed initially
- No network exposure
- No sensitive data (all synthetic)

For future deployment:
- Add API authentication
- Encrypt data at rest
- Use HTTPS for web access
- Implement role-based access control

---

**Design Status:** Phase 1 Complete  
**Last Updated:** January 27, 2026
