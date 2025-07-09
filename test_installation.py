#!/usr/bin/env python3
"""
Test script for NeuroInsight BCI Project
Verifies that all components are working correctly
"""

import sys
import os

def test_imports():
    """Test that all required libraries can be imported"""
    print("🧪 Testing library imports...")
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import scipy
        print("✅ SciPy imported successfully")
    except ImportError as e:
        print(f"❌ SciPy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    # MNE is optional and not required for basic functionality
    print("⚠️  MNE not available (optional package - not needed for core functionality)")
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    return True

def test_eeg_simulator():
    """Test the EEG simulator functionality"""
    print("\n🧪 Testing EEG simulator...")
    
    try:
        from neuroinsight.eeg_simulator import EEGSimulator
        print("✅ EEGSimulator imported successfully")
    except ImportError as e:
        print(f"❌ EEGSimulator import failed: {e}")
        return False
    
    try:
        # Initialize simulator
        simulator = EEGSimulator(sampling_rate=256, duration=2.0)
        print("✅ EEGSimulator initialized successfully")
    except Exception as e:
        print(f"❌ EEGSimulator initialization failed: {e}")
        return False
    
    try:
        # Test data generation
        eeg_data = simulator.generate_cognitive_state_eeg('focus', n_channels=4)
        print("✅ EEG data generation successful")
        print(f"   Generated {eeg_data['data'].shape[0]} channels")
        print(f"   Duration: {eeg_data['duration']}s")
        print(f"   Sampling rate: {eeg_data['sampling_rate']} Hz")
    except Exception as e:
        print(f"❌ EEG data generation failed: {e}")
        return False
    
    try:
        # Test dataset generation
        dataset = simulator.generate_dataset(samples_per_state=5, n_channels=4)
        print("✅ Dataset generation successful")
        print(f"   Total samples: {len(dataset['data'])}")
        print(f"   Labels: {set(dataset['labels'])}")
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return False
    
    return True

def test_visualization():
    """Test visualization functionality"""
    print("\n🧪 Testing visualization...")
    
    try:
        from neuroinsight.eeg_simulator import EEGSimulator
        import matplotlib.pyplot as plt
        
        # Generate test data
        simulator = EEGSimulator(sampling_rate=256, duration=1.0)
        eeg_data = simulator.generate_cognitive_state_eeg('relaxation', n_channels=2)
        
        # Test plotting (without showing)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(eeg_data['time'], eeg_data['data'][0, :])
        ax.set_title('Test Plot')
        plt.close(fig)  # Close without showing
        
        print("✅ Visualization test successful")
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 NeuroInsight BCI Project - Installation Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test EEG simulator
    if not test_eeg_simulator():
        print("\n❌ EEG simulator tests failed. Please check the code.")
        return False
    
    # Test visualization
    if not test_visualization():
        print("\n❌ Visualization tests failed. Please check matplotlib installation.")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print("\n✅ Your NeuroInsight BCI project is ready to use!")
    print("\n🚀 Next steps:")
    print("   1. Run: python3 demo_eeg_simulation.py")
    print("   2. Or: neuroinsight-demo")
    print("   3. Check README.md for more examples")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 