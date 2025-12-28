"""
Quick test script to verify all imports work correctly after refactoring
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing Simple-FL imports...")

try:
    # Test core imports
    print("✓ Testing core imports...")
    from simplefl.core import Server, Client, init_clients, Data_init
    print("  ✓ Core imports successful")
    
    # Test methods imports
    print("✓ Testing methods imports...")
    from simplefl.methods import fl_methods
    print("  ✓ Methods imports successful")
    
    # Test models imports
    print("✓ Testing models imports...")
    from simplefl.models import CNN_FEMNIST, ResNet20, DIN
    print("  ✓ Models imports successful")
    
    # Test datasets imports
    print("✓ Testing datasets imports...")
    from simplefl.datasets.femnist import FEMNIST
    from simplefl.datasets.cifar import CIFAR
    print("  ✓ Datasets imports successful")
    
    # Test utils imports
    print("✓ Testing utils imports...")
    from simplefl.utils import init_args, setup_seed, save_results
    print("  ✓ Utils imports successful")
    
    # Test main package imports
    print("✓ Testing main package imports...")
    import simplefl
    print(f"  ✓ Simple-FL version: {simplefl.__version__}")
    
    print("\n✅ All imports successful! Refactoring completed correctly.")
    print("\nYou can now run:")
    print("  python scripts/train_fl.py          # For federated learning")
    print("  python scripts/train_centralized.py # For centralized baseline")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("Please check the module structure and imports.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
