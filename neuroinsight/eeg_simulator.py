"""
EEG Signal Simulator for NeuroInsight BCI Project
Generates synthetic EEG data for cognitive state classification
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from typing import Dict, List, Tuple
import seaborn as sns

class EEGSimulator:
    """
    Simulates realistic EEG signals for different cognitive states:
    - Focus: High beta waves (13-30 Hz)
    - Stress: High beta + gamma waves (30-100 Hz)
    - Relaxation: High alpha waves (8-13 Hz)
    """
    
    def __init__(self, sampling_rate: int = 256, duration: float = 10.0):
        """
        Initialize EEG simulator
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 256 Hz)
            duration: Duration of each recording in seconds (default: 10.0s)
        """
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.n_samples = int(sampling_rate * duration)
        self.time = np.linspace(0, duration, self.n_samples)
        
        # Define frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),      # Deep sleep
            'theta': (4, 8),        # Drowsiness, meditation
            'alpha': (8, 13),       # Relaxation, eyes closed
            'beta': (13, 30),       # Active thinking, focus
            'gamma': (30, 100)      # High-level processing, stress
        }
        
        # Define cognitive states and their characteristics
        self.cognitive_states = {
            'focus': {
                'description': 'High concentration and active thinking',
                'dominant_bands': ['beta'],
                'amplitudes': {'alpha': 0.1, 'beta': 0.8, 'theta': 0.2, 'delta': 0.05, 'gamma': 0.3},
                'noise_level': 0.1
            },
            'stress': {
                'description': 'High stress and anxiety',
                'dominant_bands': ['beta', 'gamma'],
                'amplitudes': {'alpha': 0.05, 'beta': 0.9, 'theta': 0.1, 'delta': 0.02, 'gamma': 0.7},
                'noise_level': 0.15
            },
            'relaxation': {
                'description': 'Calm and relaxed state',
                'dominant_bands': ['alpha'],
                'amplitudes': {'alpha': 0.9, 'beta': 0.1, 'theta': 0.4, 'delta': 0.2, 'gamma': 0.05},
                'noise_level': 0.08
            }
        }
    
    def generate_sine_wave(self, frequency: float, amplitude: float, phase: float = 0) -> np.ndarray:
        """Generate a sine wave with given parameters"""
        return amplitude * np.sin(2 * np.pi * frequency * self.time + phase)
    
    def generate_band_noise(self, freq_range: Tuple[float, float], amplitude: float) -> np.ndarray:
        """Generate band-limited noise for a specific frequency range"""
        low_freq, high_freq = freq_range
        
        # Generate white noise
        white_noise = np.random.normal(0, 1, self.n_samples)
        
        # Design bandpass filter
        nyquist = self.sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Create bandpass filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter
        band_noise = signal.filtfilt(b, a, white_noise)
        
        # Normalize and scale
        band_noise = band_noise / np.std(band_noise) * amplitude
        
        return band_noise
    
    def generate_cognitive_state_eeg(self, state: str, n_channels: int = 8) -> Dict:
        """
        Generate synthetic EEG data for a specific cognitive state
        
        Args:
            state: Cognitive state ('focus', 'stress', 'relaxation')
            n_channels: Number of EEG channels to simulate
            
        Returns:
            Dictionary containing EEG data and metadata
        """
        if state not in self.cognitive_states:
            raise ValueError(f"Unknown cognitive state: {state}")
        
        state_config = self.cognitive_states[state]
        
        # Generate EEG signal for each channel
        eeg_data = np.zeros((n_channels, self.n_samples))
        channel_names = [f'CH{i+1}' for i in range(n_channels)]
        
        for ch in range(n_channels):
            # Initialize signal
            signal_components = []
            
            # Add frequency band components
            for band, freq_range in self.freq_bands.items():
                amplitude = state_config['amplitudes'][band]
                
                if amplitude > 0:
                    # Generate multiple frequencies within the band
                    n_freqs = 3  # Number of frequencies per band
                    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
                    
                    for freq in freqs:
                        # Add some randomness to frequency and phase
                        freq_variation = np.random.normal(freq, freq * 0.1)
                        phase = np.random.uniform(0, 2 * np.pi)
                        
                        # Add sine wave component
                        sine_component = self.generate_sine_wave(
                            freq_variation, 
                            amplitude / n_freqs, 
                            phase
                        )
                        signal_components.append(sine_component)
                        
                        # Add band-limited noise
                        noise_component = self.generate_band_noise(
                            freq_range, 
                            amplitude * 0.3 / n_freqs
                        )
                        signal_components.append(noise_component)
            
            # Combine all components
            combined_signal = np.sum(signal_components, axis=0)
            
            # Add overall noise
            noise_level = state_config['noise_level']
            noise = np.random.normal(0, noise_level, self.n_samples)
            combined_signal += noise
            
            # Add channel-specific variations
            channel_variation = np.random.normal(1, 0.2)
            combined_signal *= channel_variation
            
            # Apply slight smoothing to make it more realistic
            polyorder = 2
            window_size = int(0.01 * self.sampling_rate)  # 10ms window
            # Ensure window_size is odd and > polyorder
            if window_size <= polyorder:
                window_size = polyorder + 3  # minimum odd > polyorder
            if window_size % 2 == 0:
                window_size += 1
            if window_size < self.n_samples:
                combined_signal = signal.savgol_filter(combined_signal, window_size, polyorder)
            
            eeg_data[ch, :] = combined_signal
        
        return {
            'data': eeg_data,
            'channel_names': channel_names,
            'sampling_rate': self.sampling_rate,
            'duration': self.duration,
            'time': self.time,
            'state': state,
            'description': state_config['description'],
            'metadata': {
                'dominant_bands': state_config['dominant_bands'],
                'amplitudes': state_config['amplitudes'],
                'noise_level': state_config['noise_level']
            }
        }
    
    def generate_dataset(self, samples_per_state: int = 100, n_channels: int = 8) -> Dict:
        """
        Generate a complete dataset with all cognitive states
        
        Args:
            samples_per_state: Number of samples to generate for each state
            n_channels: Number of EEG channels
            
        Returns:
            Dictionary containing the complete dataset
        """
        dataset = {
            'data': [],
            'labels': [],
            'metadata': [],
            'channel_names': [f'CH{i+1}' for i in range(n_channels)],
            'sampling_rate': self.sampling_rate,
            'duration': self.duration
        }
        
        states = list(self.cognitive_states.keys())
        
        for state in states:
            print(f"Generating {samples_per_state} samples for {state} state...")
            
            for i in range(samples_per_state):
                eeg_sample = self.generate_cognitive_state_eeg(state, n_channels)
                
                dataset['data'].append(eeg_sample['data'])
                dataset['labels'].append(state)
                dataset['metadata'].append(eeg_sample['metadata'])
        
        return dataset
    
    def plot_sample(self, eeg_data: Dict, save_path: str = None):
        """
        Plot a sample EEG recording
        
        Args:
            eeg_data: EEG data dictionary from generate_cognitive_state_eeg
            save_path: Optional path to save the plot
        """
        data = eeg_data['data']
        channel_names = eeg_data['channel_names']
        time = eeg_data['time']
        state = eeg_data['state']
        
        n_channels = len(channel_names)
        
        # Create subplots
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels))
        if n_channels == 1:
            axes = [axes]
        
        # Plot each channel
        for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            ax.plot(time, data[i, :], linewidth=0.8, color='blue', alpha=0.8)
            ax.set_ylabel(f'{ch_name}\n(μV)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.duration)
            
            # Only show x-label for bottom plot
            if i == n_channels - 1:
                ax.set_xlabel('Time (s)', fontsize=12)
        
        # Add title
        fig.suptitle(f'Synthetic EEG - {state.capitalize()} State\n{eeg_data["description"]}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_power_spectrum(self, eeg_data: Dict, save_path: str = None):
        """
        Plot power spectrum of EEG data
        
        Args:
            eeg_data: EEG data dictionary
            save_path: Optional path to save the plot
        """
        data = eeg_data['data']
        channel_names = eeg_data['channel_names']
        state = eeg_data['state']
        
        n_channels = len(channel_names)
        
        # Create subplots
        fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2*n_channels))
        if n_channels == 1:
            axes = [axes]
        
        # Calculate and plot power spectrum for each channel
        for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            # Calculate power spectral density
            freqs, psd = signal.welch(data[i, :], fs=self.sampling_rate, nperseg=1024)
            
            # Plot power spectrum
            ax.semilogy(freqs, psd, linewidth=1.5, color='red', alpha=0.8)
            ax.set_ylabel(f'{ch_name}\nPower (μV²/Hz)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)  # Show up to 50 Hz
            
            # Highlight frequency bands
            colors = ['purple', 'blue', 'green', 'orange', 'red']
            for j, (band, (low, high)) in enumerate(self.freq_bands.items()):
                if high <= 50:  # Only show bands up to 50 Hz
                    ax.axvspan(low, high, alpha=0.2, color=colors[j], label=band)
            
            # Only show x-label for bottom plot
            if i == n_channels - 1:
                ax.set_xlabel('Frequency (Hz)', fontsize=12)
        
        # Add title
        fig.suptitle(f'Power Spectrum - {state.capitalize()} State', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Main function to demonstrate EEG simulation"""
    print("NeuroInsight EEG Simulator")
    print("=" * 40)
    
    # Initialize simulator
    simulator = EEGSimulator(sampling_rate=256, duration=10.0)
    
    # Generate samples for each cognitive state
    states = ['focus', 'stress', 'relaxation']
    
    for state in states:
        print(f"\nGenerating {state} state EEG...")
        
        # Generate EEG data
        eeg_data = simulator.generate_cognitive_state_eeg(state, n_channels=8)
        
        # Plot time series
        simulator.plot_sample(eeg_data, save_path=f'eeg_{state}_time.png')
        
        # Plot power spectrum
        simulator.plot_power_spectrum(eeg_data, save_path=f'eeg_{state}_spectrum.png')
        
        print(f"Generated {state} state EEG with {len(eeg_data['channel_names'])} channels")
        print(f"Duration: {eeg_data['duration']}s, Sampling rate: {eeg_data['sampling_rate']} Hz")
    
    # Generate complete dataset
    print("\nGenerating complete dataset...")
    dataset = simulator.generate_dataset(samples_per_state=50, n_channels=8)
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(dataset['data'])}")
    print(f"Channels: {len(dataset['channel_names'])}")
    print(f"Duration per sample: {dataset['duration']}s")
    print(f"Sampling rate: {dataset['sampling_rate']} Hz")
    
    # Count samples per state
    unique_labels, counts = np.unique(dataset['labels'], return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"{label.capitalize()}: {count} samples")


if __name__ == "__main__":
    main() 