# 🚀 NeuroInsight Quick Start Guide

## Installation (Choose One Method)

### Method 1: Automated Installation (Recommended)
```bash
./install.sh
```

### Method 2: Manual Installation
```bash
pip install -r requirements.txt
pip install -e .
```

## Test Installation
```bash
python3 test_installation.py
```

## Run Demo
```bash
python3 demo_eeg_simulation.py
```

## Quick Code Example

```python
from neuroinsight.eeg_simulator import EEGSimulator

# Create simulator
simulator = EEGSimulator(sampling_rate=256, duration=10.0)

# Generate EEG data for focus state
eeg_data = simulator.generate_cognitive_state_eeg('focus', n_channels=8)

# Plot the results
simulator.plot_sample(eeg_data)
simulator.plot_power_spectrum(eeg_data)

# Generate complete dataset
dataset = simulator.generate_dataset(samples_per_state=100, n_channels=8)
```

## Cognitive States

- **Focus**: High beta waves (13-30 Hz)
- **Stress**: High beta + gamma waves (30-100 Hz)  
- **Relaxation**: High alpha waves (8-13 Hz)

## Files Overview

- `neuroinsight/eeg_simulator.py` - Main EEG simulation engine
- `demo_eeg_simulation.py` - Interactive demo with plots
- `test_installation.py` - Verify everything works
- `requirements.txt` - Python dependencies
- `README.md` - Complete documentation

## Next Steps

1. Run the demo to see synthetic EEG data
2. Experiment with different parameters
3. Generate datasets for machine learning
4. Build your own BCI classification models

---

**Need help?** Check the full `README.md` for detailed documentation! 