# 🧠 NeuroInsight Usage Guide

## Quick Start Examples

### 1. Basic Usage - Generate Single EEG Sample

```python
from neuroinsight.eeg_simulator import EEGSimulator

# Create simulator
simulator = EEGSimulator(sampling_rate=256, duration=10.0)

# Generate EEG for focus state
eeg_data = simulator.generate_cognitive_state_eeg('focus', n_channels=8)

# Access the data
print(f"EEG shape: {eeg_data['data'].shape}")  # (channels, time_points)
print(f"Duration: {eeg_data['duration']}s")
print(f"Sampling rate: {eeg_data['sampling_rate']} Hz")
print(f"Cognitive state: {eeg_data['state']}")
```

### 2. Generate Complete Dataset for Machine Learning

```python
# Generate training dataset
dataset = simulator.generate_dataset(samples_per_state=100, n_channels=8)

# Access dataset components
X = dataset['data']  # List of EEG arrays
y = dataset['labels']  # List of state labels
metadata = dataset['metadata']  # List of metadata dicts

print(f"Total samples: {len(X)}")
print(f"States: {set(y)}")
print(f"Sample shape: {X[0].shape}")
```

### 3. Real-time Data Generation

```python
import time
import numpy as np

# Generate data in real-time simulation
simulator = EEGSimulator(sampling_rate=256, duration=1.0)

for i in range(10):
    # Generate 1-second chunks
    eeg_chunk = simulator.generate_cognitive_state_eeg('focus', n_channels=4)
    
    # Process the chunk (e.g., feature extraction, classification)
    print(f"Chunk {i+1}: {eeg_chunk['data'].shape}")
    
    time.sleep(0.1)  # Simulate real-time processing
```

## Data Analysis Examples

### 4. Feature Extraction

```python
import numpy as np
from scipy import signal
from scipy.stats import entropy

def extract_eeg_features(eeg_data):
    """Extract common EEG features"""
    features = {}
    
    # Time domain features
    features['mean'] = np.mean(eeg_data, axis=1)
    features['std'] = np.std(eeg_data, axis=1)
    features['variance'] = np.var(eeg_data, axis=1)
    features['rms'] = np.sqrt(np.mean(eeg_data**2, axis=1))
    
    # Frequency domain features
    for ch in range(eeg_data.shape[0]):
        freqs, psd = signal.welch(eeg_data[ch, :], fs=256)
        
        # Band powers
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        theta_mask = (freqs >= 4) & (freqs <= 8)
        
        features[f'alpha_power_ch{ch}'] = np.mean(psd[alpha_mask])
        features[f'beta_power_ch{ch}'] = np.mean(psd[beta_mask])
        features[f'theta_power_ch{ch}'] = np.mean(psd[theta_mask])
    
    return features

# Use the feature extractor
simulator = EEGSimulator()
eeg_sample = simulator.generate_cognitive_state_eeg('stress', n_channels=4)
features = extract_eeg_features(eeg_sample['data'])
print("Extracted features:", features)
```

### 5. Machine Learning Pipeline

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np

# Generate dataset
simulator = EEGSimulator(sampling_rate=256, duration=5.0)
dataset = simulator.generate_dataset(samples_per_state=50, n_channels=4)

# Extract features for all samples
X_features = []
for eeg_data in dataset['data']:
    features = extract_eeg_features(eeg_data)
    X_features.append(list(features.values()))

X = np.array(X_features)
y = np.array(dataset['labels'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 6. Real-time Classification System

```python
import time
import numpy as np
from collections import deque

class RealTimeBCI:
    def __init__(self, model, feature_extractor, window_size=5):
        self.model = model
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def process_chunk(self, eeg_chunk):
        """Process a new EEG chunk and return classification"""
        self.buffer.append(eeg_chunk)
        
        if len(self.buffer) == self.window_size:
            # Combine chunks and extract features
            combined_data = np.concatenate(list(self.buffer), axis=1)
            features = self.feature_extractor(combined_data)
            
            # Predict
            prediction = self.model.predict([list(features.values())])[0]
            confidence = np.max(self.model.predict_proba([list(features.values())]))
            
            return prediction, confidence
        
        return None, None

# Usage example
simulator = EEGSimulator(sampling_rate=256, duration=1.0)
bci_system = RealTimeBCI(clf, extract_eeg_features)

for i in range(10):
    eeg_chunk = simulator.generate_cognitive_state_eeg('focus', n_channels=4)
    prediction, confidence = bci_system.process_chunk(eeg_chunk['data'])
    
    if prediction:
        print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
    
    time.sleep(0.1)
```

## Advanced Usage

### 7. Custom Cognitive States

```python
# Define custom cognitive state
simulator.cognitive_states['meditation'] = {
    'description': 'Deep meditation state',
    'dominant_bands': ['theta', 'alpha'],
    'amplitudes': {'alpha': 0.7, 'beta': 0.1, 'theta': 0.8, 'delta': 0.3, 'gamma': 0.05},
    'noise_level': 0.05
}

# Generate custom state data
meditation_eeg = simulator.generate_cognitive_state_eeg('meditation', n_channels=8)
```

### 8. Multi-Subject Simulation

```python
def simulate_subject_variability(simulator, n_subjects=10):
    """Simulate different subjects with varying characteristics"""
    subjects_data = []
    
    for subject_id in range(n_subjects):
        # Modify simulator parameters for each subject
        subject_simulator = EEGSimulator(sampling_rate=256, duration=10.0)
        
        # Add subject-specific variations
        for state in subject_simulator.cognitive_states:
            # Random amplitude variations
            for band in subject_simulator.cognitive_states[state]['amplitudes']:
                variation = np.random.normal(1.0, 0.2)  # ±20% variation
                subject_simulator.cognitive_states[state]['amplitudes'][band] *= variation
        
        # Generate data for this subject
        subject_dataset = subject_simulator.generate_dataset(samples_per_state=20, n_channels=8)
        
        # Add subject ID to metadata
        for metadata in subject_dataset['metadata']:
            metadata['subject_id'] = subject_id
        
        subjects_data.append(subject_dataset)
    
    return subjects_data

# Generate multi-subject dataset
subjects = simulate_subject_variability(simulator, n_subjects=5)
```

### 9. Data Export and Import

```python
import pickle
import json

# Save dataset
def save_dataset(dataset, filename):
    """Save dataset to file"""
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(filename):
    """Load dataset from file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Generate and save dataset
simulator = EEGSimulator()
dataset = simulator.generate_dataset(samples_per_state=100, n_channels=8)
save_dataset(dataset, 'eeg_dataset.pkl')

# Load dataset later
loaded_dataset = load_dataset('eeg_dataset.pkl')
```

### 10. Visualization and Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compare states
states = ['focus', 'stress', 'relaxation']
alpha_powers = []

for state in states:
    eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=4)
    freqs, psd = signal.welch(eeg_data['data'][0, :], fs=256)
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_powers.append(np.mean(psd[alpha_mask]))

# Plot comparison
plt.figure(figsize=(10, 6))
plt.bar(states, alpha_powers)
plt.title('Alpha Power Comparison Across States')
plt.ylabel('Alpha Power')
plt.show()

# Correlation matrix
dataset = simulator.generate_dataset(samples_per_state=30, n_channels=8)
X_features = []
for eeg_data in dataset['data']:
    features = extract_eeg_features(eeg_data)
    X_features.append(list(features.values()))

correlation_matrix = np.corrcoef(np.array(X_features).T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()
```

## Integration with Other Tools

### 11. Export to MNE-Python Format

```python
import mne
import numpy as np

def create_mne_raw(eeg_data, ch_names=None):
    """Convert EEG data to MNE Raw object"""
    if ch_names is None:
        ch_names = [f'EEG{i+1:03d}' for i in range(eeg_data['data'].shape[0])]
    
    # Create MNE info
    info = mne.create_info(ch_names=ch_names, sfreq=eeg_data['sampling_rate'], ch_types=['eeg'] * len(ch_names))
    
    # Create Raw object
    raw = mne.io.RawArray(eeg_data['data'], info)
    
    return raw

# Convert to MNE format
simulator = EEGSimulator()
eeg_data = simulator.generate_cognitive_state_eeg('focus', n_channels=8)
mne_raw = create_mne_raw(eeg_data)

# Now you can use MNE functions
mne_raw.plot_psd()
```

### 12. Export to CSV for External Analysis

```python
import pandas as pd

def export_to_csv(dataset, filename):
    """Export dataset to CSV format"""
    rows = []
    
    for i, (eeg_data, label, metadata) in enumerate(zip(dataset['data'], dataset['labels'], dataset['metadata'])):
        # Flatten EEG data
        eeg_flat = eeg_data.flatten()
        
        # Create row
        row = {
            'sample_id': i,
            'state': label,
            'duration': metadata.get('duration', 0),
            'sampling_rate': metadata.get('sampling_rate', 0)
        }
        
        # Add EEG data columns
        for ch in range(eeg_data.shape[0]):
            for t in range(eeg_data.shape[1]):
                row[f'ch{ch}_t{t}'] = eeg_data[ch, t]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df

# Export dataset
dataset = simulator.generate_dataset(samples_per_state=10, n_channels=4)
df = export_to_csv(dataset, 'eeg_data.csv')
print(f"Exported {len(df)} samples to CSV")
```

## Tips for Best Results

1. **Sampling Rate**: Use 256 Hz or higher for realistic EEG simulation
2. **Duration**: 5-10 seconds per sample works well for most applications
3. **Channels**: 4-8 channels are sufficient for most BCI applications
4. **Dataset Size**: 50-100 samples per state for initial testing, 500+ for robust models
5. **Feature Engineering**: Combine time and frequency domain features for best results
6. **Validation**: Always use cross-validation when training machine learning models

## Common Use Cases

- **Research**: Prototype BCI algorithms without expensive hardware
- **Education**: Teach EEG signal processing and BCI concepts
- **Development**: Test real-time processing pipelines
- **Validation**: Benchmark new classification algorithms
- **Simulation**: Study the effects of different parameters on BCI performance

This simulator gives you a solid foundation for BCI research and development without the need for expensive EEG hardware! 