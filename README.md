# WindTwin AI - Real-Time Wind Turbine Digital Twin

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Phase](https://img.shields.io/badge/phase-5.1%20Complete-green)
![License](https://img.shields.io/badge/license-MIT-blue)

A production-ready digital twin for wind turbine monitoring and predictive maintenance, featuring real-time simulation, power forecasting, and SCADA-like web interface.

---

## 🎯 Project Overview

WindTwin AI is a comprehensive digital twin system that simulates and monitors wind turbine operations in real-time. The system provides:

- **Real-time simulation** with physics-based turbine modeling
- **Live monitoring dashboard** with professional SCADA-like interface
- **Power output forecasting** (1-hour ahead predictions)
- **Predictive maintenance** recommendations based on sensor trends
- **Event logging** for operational changes and faults
- **Historical data analysis** with configurable time windows

**Current Status:** Phase 5.1 Complete - Production-ready with v1.0 REALISM enhancements

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Claude-test.git
cd Claude-test

# Install dependencies
pip install -r requirements.txt
```

### Run the System

```bash
# Start API server + web dashboard
python api.py --days 1 --forecast

# Server starts on http://localhost:8000
# - Web Dashboard: http://localhost:8000/
# - API Docs: http://localhost:8000/docs
```

**Expected Output:**
```
🔄 v1.0 Simulation loop started (drift-safe, 5-minute intervals)
Starting API server on http://127.0.0.1:8000
```

---

## ✨ Current Capabilities

### Real-Time Simulation
- Time behavior: 1 real second = 1 simulated second
- Update frequency: Every 5 minutes
- Physics: Thermal inertia, mechanical lag, realistic power curve

### API Endpoints
| Endpoint | Description |
|----------|-------------|
| `/api/state/current` | Current turbine state (20+ fields) |
| `/api/state/history` | Historical data (1h/6h/24h/all) |
| `/api/forecast` | 1-hour power forecast + trend |
| `/api/events` | Recent event log |
| `/api/maintenance` | Maintenance recommendations |
| `/docs` | Interactive API documentation |

### Web Dashboard
- Live updates every 2 seconds
- Color-coded status badges
- Power history chart
- Event log
- Maintenance panel

---

## 📊 Usage Examples

### Access Dashboard
Open `http://localhost:8000/` in your browser

### API Examples

```bash
# Get current state
curl http://localhost:8000/api/state/current

# Get 1-hour history
curl http://localhost:8000/api/state/history?window=1h

# Get forecast
curl http://localhost:8000/api/forecast
```

---

## 📁 Project Structure

```
windtwin-ai/
├── api.py                    # FastAPI backend + simulation loop
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── PHASE_STATUS.md          # Detailed phase documentation
├── config/                   # Configuration files
├── src/twin/                 # Core simulation engine
├── src/data/                 # Weather generation
├── web/index.html           # Dashboard UI
├── models/                   # ML models (optional)
└── tests/                    # Test suite
```

---

## 🔧 Configuration

Edit `config/default.yaml`:

```yaml
turbine:
  rated_power_kw: 2500
  cut_in_speed_ms: 4.0
  cut_out_speed_ms: 25.0

simulation:
  time_step_seconds: 300    # 5 minutes
  history_buffer_size: 1000
```

---

## 🧪 Testing

```bash
# Manual test
python api.py --days 1

# Automated test
python tests/test_realtime_loop.py

# Acceptance test: Run for 2-6 hours
# Verify: No crashes, data evolves, UI updates
```

---

## 📈 Development Status

### ✅ Complete
- Phase 1-4: Core twin, physics, data pipeline, forecasting
- Phase 5: Web API + Dashboard
- Phase 5.1: Real-time simulation loop
- v1.0 REALISM: Production enhancements

### 🔄 Planned
- WebSocket support
- Multi-turbine fleet
- Advanced analytics
- Control actions

---

## 🐛 Known Limitations

1. No persistence (in-memory only)
2. No authentication (local dev only)
3. Read-only monitoring (no control)
4. Single turbine (no fleet)
5. Pre-generated weather data

### Recently Fixed
- ✅ Enum validation errors
- ✅ Time step alignment (10s → 300s)
- ✅ State coherence (RPM/power)
- ✅ Maintenance trending

---

## 📝 Documentation

- `PHASE_STATUS.md` - Detailed phase breakdown
- `/docs` - Interactive API documentation (when server running)
- Code comments - Inline documentation

---

## 📧 Support

- GitHub Issues: Report bugs or request features
- API Docs: `http://localhost:8000/docs` when running

---

**Last Updated:** January 28, 2026  
**Version:** 1.0.0 (v1.0 REALISM)  
**Status:** Production Ready ✅
