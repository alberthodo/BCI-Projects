#!/usr/bin/env python3
"""
Practical Examples for NeuroInsight BCI Simulator
This script demonstrates the most common use cases
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from neuroinsight.eeg_simulator import EEGSimulator

def extract_eeg_features(eeg_data):
    """Extract common EEG features for classification"""
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

def example_1_basic_usage():
    """Example 1: Basic EEG generation and visualization"""
    print("=" * 60)
    print("Example 1: Basic EEG Generation and Visualization")
    print("=" * 60)
    
    # Create simulator
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)
    
    # Generate EEG for each cognitive state
    states = ['focus', 'stress', 'relaxation']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    for i, state in enumerate(states):
        eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=4)
        
        # Plot time series
        time = eeg_data['time']
        data = eeg_data['data']
        
        for ch in range(data.shape[0]):
            axes[i].plot(time, data[ch, :], label=f'CH{ch+1}', alpha=0.8, linewidth=0.8)
        
        axes[i].set_title(f'{state.capitalize()} State EEG', fontweight='bold')
        axes[i].set_ylabel('Amplitude (μV)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        print(f"✅ Generated {state} state: {data.shape[0]} channels, {data.shape[1]} time points")
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()

def example_2_feature_extraction():
    """Example 2: Feature extraction and analysis"""
    print("\n" + "=" * 60)
    print("Example 2: Feature Extraction and Analysis")
    print("=" * 60)
    
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)
    
    # Generate samples and extract features
    states = ['focus', 'stress', 'relaxation']
    all_features = []
    all_labels = []
    
    for state in states:
        print(f"Processing {state} state...")
        
        for i in range(10):  # 10 samples per state
            eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=4)
            features = extract_eeg_features(eeg_data['data'])
            
            all_features.append(list(features.values()))
            all_labels.append(state)
    
    # Convert to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"✅ Extracted features from {len(X)} samples")
    print(f"   Feature vector shape: {X.shape}")
    print(f"   States: {set(y)}")
    
    # Analyze feature distributions
    feature_names = list(extract_eeg_features(np.zeros((4, 1280))).keys())
    
    # Plot feature distributions by state
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature_name in enumerate(['alpha_power_ch0', 'beta_power_ch0', 'theta_power_ch0', 'std']):
        if i < len(axes):
            for state in states:
                state_mask = y == state
                axes[i].hist(X[state_mask, feature_names.index(feature_name)], 
                           alpha=0.7, label=state, bins=10)
            
            axes[i].set_title(f'{feature_name} Distribution')
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def example_3_machine_learning():
    """Example 3: Machine learning classification"""
    print("\n" + "=" * 60)
    print("Example 3: Machine Learning Classification")
    print("=" * 60)
    
    # Generate larger dataset
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)
    dataset = simulator.generate_dataset(samples_per_state=50, n_channels=4)
    
    print(f"Generated dataset: {len(dataset['data'])} samples")
    
    # Extract features
    X_features = []
    for eeg_data in dataset['data']:
        features = extract_eeg_features(eeg_data)
        X_features.append(list(features.values()))
    
    X = np.array(X_features)
    y = np.array(dataset['labels'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Feature importance
    feature_names = list(extract_eeg_features(np.zeros((4, 1280))).keys())
    importances = clf.feature_importances_
    
    # Plot top 10 features
    top_indices = np.argsort(importances)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_indices)), importances[top_indices])
    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.show()

def example_4_real_time_simulation():
    """Example 4: Real-time BCI simulation"""
    print("\n" + "=" * 60)
    print("Example 4: Real-time BCI Simulation")
    print("=" * 60)
    
    # Train a model first
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)
    dataset = simulator.generate_dataset(samples_per_state=30, n_channels=4)
    
    X_features = []
    for eeg_data in dataset['data']:
        features = extract_eeg_features(eeg_data)
        X_features.append(list(features.values()))
    
    X = np.array(X_features)
    y = np.array(dataset['labels'])
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    # Simulate real-time processing
    print("Simulating real-time BCI processing...")
    print("Generating 1-second chunks and classifying...")
    
    real_time_simulator = EEGSimulator(sampling_rate=256, duration=1.0)
    
    for i in range(15):
        # Generate chunk
        eeg_chunk = real_time_simulator.generate_cognitive_state_eeg('focus', n_channels=4)
        
        # Extract features
        features = extract_eeg_features(eeg_chunk['data'])
        
        # Predict
        prediction = clf.predict([list(features.values())])[0]
        confidence = np.max(clf.predict_proba([list(features.values())]))
        
        print(f"Chunk {i+1:2d}: Predicted: {prediction:10s}, Confidence: {confidence:.2f}")
    
    print("✅ Real-time simulation completed!")

def example_5_custom_states():
    """Example 5: Creating custom cognitive states"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Cognitive States")
    print("=" * 60)
    
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)
    
    # Add custom cognitive state
    simulator.cognitive_states['meditation'] = {
        'description': 'Deep meditation state',
        'dominant_bands': ['theta', 'alpha'],
        'amplitudes': {'alpha': 0.7, 'beta': 0.1, 'theta': 0.8, 'delta': 0.3, 'gamma': 0.05},
        'noise_level': 0.05
    }
    
    # Generate custom state data
    meditation_eeg = simulator.generate_cognitive_state_eeg('meditation', n_channels=4)
    
    print(f"✅ Generated custom meditation state")
    print(f"   Description: {meditation_eeg['description']}")
    print(f"   Dominant bands: {meditation_eeg['metadata']['dominant_bands']}")
    
    # Compare with standard states
    states = ['focus', 'meditation', 'relaxation']
    alpha_powers = []
    
    for state in states:
        eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=4)
        freqs, psd = signal.welch(eeg_data['data'][0, :], fs=256)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_powers.append(np.mean(psd[alpha_mask]))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(states, alpha_powers, color=['red', 'purple', 'blue'])
    plt.title('Alpha Power Comparison')
    plt.ylabel('Alpha Power')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Run all examples"""
    print("🧠 NeuroInsight BCI Simulator - Practical Examples")
    print("=" * 60)
    
    # Run examples
    example_1_basic_usage()
    example_2_feature_extraction()
    example_3_machine_learning()
    example_4_real_time_simulation()
    example_5_custom_states()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Modify parameters to experiment with different settings")
    print("2. Try different machine learning algorithms")
    print("3. Add more custom cognitive states")
    print("4. Integrate with your own BCI projects")
    print("\nCheck USAGE_GUIDE.md for more advanced examples!")

if __name__ == "__main__":
    main() 