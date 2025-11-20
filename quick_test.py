#!/usr/bin/env python3
"""
Quick test script to verify all dependencies and imports work
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    missing = []
    
    modules = [
        'torch',
        'yaml',
        'numpy',
        'scipy',
        'pandas',
        'sklearn',
        'transformers',
        'nltk',
        'rouge_score',
        'bert_score',
        'tqdm',
        'matplotlib',
        'seaborn'
    ]
    
    print("Testing imports...")
    for module in modules:
        try:
            if module == 'yaml':
                import yaml
            elif module == 'sklearn':
                import sklearn
            elif module == 'rouge_score':
                import rouge_score
            elif module == 'bert_score':
                import bert_score
            else:
                __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing modules: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def test_codebase_imports():
    """Test if project modules can be imported"""
    print("\nTesting project modules...")
    try:
        from models import GraphEnhancedEEG2Text
        print("  ✓ models")
    except Exception as e:
        print(f"  ✗ models - {e}")
        return False
    
    try:
        from preprocessing.preprocessing import ZuCoDataset
        print("  ✓ preprocessing")
    except Exception as e:
        print(f"  ✗ preprocessing - {e}")
        return False
    
    try:
        from utils.metrics import evaluate_predictions
        print("  ✓ utils.metrics")
    except Exception as e:
        print(f"  ✗ utils.metrics - {e}")
        return False
    
    try:
        from utils.losses import CompositeLoss
        print("  ✓ utils.losses")
    except Exception as e:
        print(f"  ✗ utils.losses - {e}")
        return False
    
    try:
        from utils.visualization import save_adjacency_heatmap
        print("  ✓ utils.visualization")
    except Exception as e:
        print(f"  ✗ utils.visualization - {e}")
        return False
    
    try:
        from utils.statistics import compute_statistics
        print("  ✓ utils.statistics")
    except Exception as e:
        print(f"  ✗ utils.statistics - {e}")
        return False
    
    print("\n✅ All project modules importable!")
    return True

def test_config():
    """Test if config file can be loaded"""
    print("\nTesting config...")
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("  ✓ config/config.yaml loaded")
        return True
    except Exception as e:
        print(f"  ✗ config/config.yaml - {e}")
        return False

if __name__ == '__main__':
    print("=" * 60)
    print("Graph-Enhanced EEG-to-Text - Setup Verification")
    print("=" * 60)
    
    all_ok = True
    
    # Test dependencies
    if not test_imports():
        all_ok = False
    
    # Test project modules
    if not test_codebase_imports():
        all_ok = False
    
    # Test config
    if not test_config():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ SETUP COMPLETE - Ready to run!")
        print("\nNext steps:")
        print("  1. Ensure data is in correct format (see SETUP.md)")
        print("  2. Run: python3 train.py --config config/config.yaml --data_dir data")
    else:
        print("❌ SETUP INCOMPLETE - Please fix errors above")
        print("\nSee SETUP.md for installation instructions")
    print("=" * 60)

