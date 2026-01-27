# WINDTWIN-AI Setup Guide

## Prerequisites

### Required Software
- **Python**: 3.11 or higher ([Download](https://www.python.org/downloads/))
- **Git**: Latest version ([Download](https://git-scm.com/downloads))
- **pip**: Comes with Python

### Recommended Tools
- **VS Code**: For code editing ([Download](https://code.visualstudio.com/))
- **Jupyter**: For notebook exploration (included in requirements)

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 2GB free space
- **CPU**: Any modern processor (simulation is lightweight)

## Installation Steps

### 1. Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/yourusername/windtwin-ai.git

# Or clone via SSH
git clone git@github.com:yourusername/windtwin-ai.git

# Navigate to project directory
cd windtwin-ai
```

### 2. Create Virtual Environment

**Why?** Isolates project dependencies from your system Python.

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Verify activation:**  
Your terminal prompt should now start with `(venv)`.

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

**Expected time:** 2-5 minutes depending on internet speed.

### 4. Verify Installation

```bash
# Check Python version
python --version
# Should show: Python 3.11.x or higher

# Check installed packages
pip list
# Should show: numpy, pandas, scikit-learn, etc.

# Run quick test (coming soon)
python -c "import numpy, pandas, sklearn; print('All imports successful!')"
```

### 5. Configure Project (Optional)

Create a `.env` file for custom settings:

```bash
# Copy example config
cp .env.example .env

# Edit with your preferred editor
# nano .env  # or vim, code, notepad++, etc.
```

**Example `.env` contents:**
```
# Simulation settings
TURBINE_RATED_POWER=2500
SIMULATION_TIMESTEP=600
DATA_RETENTION_DAYS=90

# AI settings
FORECAST_HORIZON_HOURS=6
ANOMALY_THRESHOLD=0.75

# Logging
LOG_LEVEL=INFO
```

## First Run

### Generate Sample Data

```bash
# Generate 7 days of synthetic data
python main.py generate --days 7

# Expected output:
# ✓ Generated weather data: 1,008 samples
# ✓ Generated SCADA data: 1,008 samples
# ✓ Saved to: data/weather/weather_20260127.csv
# ✓ Saved to: data/scada/scada_20260127.csv
```

### View Generated Data

```bash
# Quick stats
python main.py stats

# Open in Jupyter (optional)
jupyter notebook notebooks/exploration.ipynb
```

## Development Setup

### Install Development Tools

```bash
# Install additional dev dependencies
pip install pytest black pylint ipython

# Format code
black src/

# Lint code
pylint src/

# Run tests
pytest
```

### IDE Configuration

#### VS Code
1. Install Python extension
2. Select virtual environment:
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Choose the `venv` interpreter
3. Install recommended extensions:
   - Python
   - Pylance
   - Jupyter

#### PyCharm
1. Open project folder
2. File → Settings → Project → Python Interpreter
3. Add interpreter → Existing environment
4. Select `venv/bin/python` (or `venv\Scripts\python.exe` on Windows)

## Troubleshooting

### Issue: `pip install` fails

**Solution:**
```bash
# Try upgrading pip
pip install --upgrade pip setuptools wheel

# Or use --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

### Issue: Virtual environment not activating

**Windows:**
```powershell
# If execution policy blocks scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
# Ensure script has execute permissions
chmod +x venv/bin/activate
```

### Issue: Import errors

**Solution:**
```bash
# Ensure you're in the virtual environment
which python  # Should point to venv/bin/python

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: "Module not found" when running scripts

**Solution:**
```bash
# Run from project root directory
cd /path/to/windtwin-ai

# Or add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;%CD%\src          # Windows
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows

# Remove generated data (optional)
rm -rf data/scada data/weather models/*.pkl
```

## Next Steps

After successful setup:

1. ✅ Read [Architecture Documentation](architecture.md)
2. ✅ Review [Project Roadmap](roadmap.md)
3. ✅ Explore `notebooks/exploration.ipynb`
4. ✅ Run `python main.py --help` to see available commands
5. ✅ Start generating data and building the digital twin!

## Getting Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `notebooks/` for usage examples
- **Issues**: Open a GitHub issue
- **Code**: Read inline comments in `src/`

---

**Setup Version:** 1.0  
**Last Updated:** January 27, 2026  
**Tested On:** Windows 11, macOS 14, Ubuntu 22.04
