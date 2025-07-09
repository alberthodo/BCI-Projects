#!/usr/bin/env python3
"""
Demo script for NeuroInsight EEG Simulator
Run this script to generate and visualize synthetic EEG data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neuroinsight.eeg_simulator import EEGSimulator
import matplotlib.pyplot as plt

def main():
    """Demo the EEG simulator"""
    print("🧠 NeuroInsight EEG Simulator Demo")
    print("=" * 50)
    
    # Initialize the simulator
    print("Initializing EEG simulator...")
    simulator = EEGSimulator(sampling_rate=256, duration=5.0)  # 5 seconds for demo
    
    # Test each cognitive state
    states = ['focus', 'stress', 'relaxation']
    
    print("\nGenerating sample EEG data for each cognitive state...")
    print("This will create plots showing the time series and power spectrum.")
    
    for i, state in enumerate(states):
        print(f"\n[{i+1}/3] Generating {state} state...")
        
        # Generate EEG data
        eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=4)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot time series
        time = eeg_data['time']
        data = eeg_data['data']
        
        for ch in range(data.shape[0]):
            ax1.plot(time, data[ch, :], label=f'CH{ch+1}', alpha=0.8, linewidth=0.8)
        
        ax1.set_title(f'EEG Time Series - {state.capitalize()} State', fontweight='bold')
        ax1.set_ylabel('Amplitude (μV)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot power spectrum
        from scipy import signal
        for ch in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch, :], fs=simulator.sampling_rate, nperseg=512)
            ax2.semilogy(freqs, psd, label=f'CH{ch+1}', alpha=0.8)
        
        ax2.set_title(f'Power Spectrum - {state.capitalize()} State', fontweight='bold')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (μV²/Hz)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 50)
        
        # Highlight frequency bands
        colors = ['purple', 'blue', 'green', 'orange', 'red']
        for j, (band, (low, high)) in enumerate(simulator.freq_bands.items()):
            if high <= 50:
                ax2.axvspan(low, high, alpha=0.2, color=colors[j], label=f'{band} band')
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Generated {state} state EEG with {len(eeg_data['channel_names'])} channels")
        print(f"   Duration: {eeg_data['duration']}s, Sampling rate: {eeg_data['sampling_rate']} Hz")
        print(f"   Dominant bands: {', '.join(eeg_data['metadata']['dominant_bands'])}")
    
    # Generate a small dataset
    print("\n" + "="*50)
    print("Generating complete dataset for machine learning...")
    
    dataset = simulator.generate_dataset(samples_per_state=20, n_channels=4)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(dataset['data'])}")
    print(f"   Channels: {len(dataset['channel_names'])}")
    print(f"   Duration per sample: {dataset['duration']}s")
    print(f"   Sampling rate: {dataset['sampling_rate']} Hz")
    
    # Count samples per state
    import numpy as np
    unique_labels, counts = np.unique(dataset['labels'], return_counts=True)
    print(f"\n📈 Samples per state:")
    for label, count in zip(unique_labels, counts):
        print(f"   {label.capitalize()}: {count} samples")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"   The synthetic EEG data is ready for cognitive state classification.")
    print(f"   Next steps: Feature extraction and machine learning model training.")

if __name__ == "__main__":
    main() 