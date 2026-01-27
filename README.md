# WINDTWIN-AI 🌀⚡

A learning project for building a **Single Wind Turbine Digital Twin** with synthetic data, physics simulation, and AI-powered forecasting and anomaly detection.

## 🎯 Project Goals

- Understand full software development lifecycle with AI assistance
- Build a modular, extensible digital twin system
- Learn AI/ML integration in industrial IoT context
- Practice clean code architecture and documentation

## 🏗️ Architecture

```
Data Generation → Digital Twin Core → AI Layer → User Interface
     ↓                   ↓               ↓            ↓
  SCADA+Weather      Simulation      Forecast     CLI Dashboard
                                    +Anomaly
```

## 📦 Features

### Phase 1: Foundation (Current)
- ✅ Project structure
- 🚧 Synthetic SCADA data generation
- 🚧 Synthetic weather data generation
- 🚧 Basic CLI interface

### Phase 2: Digital Twin Core (Upcoming)
- Turbine physics model (power curve)
- Real-time time-step simulator
- State tracking and logging

### Phase 3-4: AI Layer (Future)
- Production forecasting (1-6 hours ahead)
- Anomaly detection and alerting

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd windtwin-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run the main application
python main.py

# Generate synthetic data
python main.py generate --days 30

# View dashboard (coming in Phase 2)
python main.py dashboard
```

## 📁 Project Structure

```
windtwin-ai/
├── src/
│   ├── data/           # Data generation modules
│   ├── twin/           # Digital twin simulation
│   ├── ai/             # AI/ML models
│   └── interface/      # User interfaces
├── data/               # Generated datasets
├── models/             # Trained AI models
├── notebooks/          # Jupyter notebooks for exploration
├── docs/               # Documentation
└── main.py            # Entry point
```

## 📚 Documentation

- [Architecture Details](docs/architecture.md)
- [Setup Guide](docs/setup.md)
- [Project Roadmap](docs/roadmap.md)

## 🛠️ Technology Stack

- **Language:** Python 3.11+
- **Data:** NumPy, Pandas
- **ML:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **CLI:** Rich, Click
- **Testing:** pytest

## 🤝 Development Workflow

This project follows a learning-focused workflow:
1. Outcome-based requirements (no code from human)
2. AI proposes implementation
3. Human reviews and approves
4. Iterate and extend

## 📈 Current Status

**Phase:** 1 - Foundation  
**Progress:** 15%  
**Next Milestone:** Complete data generators

## 📝 License

This is a learning project. Feel free to use and modify as needed.

## 🙏 Acknowledgments

Built as a hands-on learning project to understand AI-assisted software development and digital twin concepts.

---

**Last Updated:** January 27, 2026
