# test_data_loading.py
"""Test data loading functionality"""
import os
import pytest


def test_load_sample_data():
    """Test if the sample data can be loaded"""
    import torch
    import urllib.request
    
    home = os.environ['ITHEMAL_HOME']
    data_file = os.path.join(home, 'learning/pytorch/inputs/data/haswell_sample1000.data')
    
    if not os.path.exists(data_file):
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/ithemal/Ithemal-models/master/paper/data/haswell_sample1000.data',
            data_file
        )
    
    data = torch.load(data_file)
    
    assert len(data) == 1000
    code_id, timing, code_intel, code_xml = data[0]
    assert isinstance(timing, float) and timing > 0
    
    print(f"✓ Loaded {len(data)} training samples")
    print(f"  Example timing: {timing} cycles")


def test_load_bhive_data():
    """Test loading BHive training data if available"""
    import torch
    
    home = os.environ['ITHEMAL_HOME']
    data_dir = os.path.join(home, 'learning/pytorch/inputs/data')
    
    bhive_files = [
        'bhive_training_ivb.data',
        'bhive_training_hsw.data', 
        'bhive_training_skl.data'
    ]
    
    found_any = False
    for fname in bhive_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            found_any = True
            data = torch.load(fpath)
            print(f"✓ {fname}: {len(data)} samples")
            
            # Verify structure
            if len(data) > 0:
                sample = data[0]
                assert len(sample) == 4, f"Expected 4 elements per sample, got {len(sample)}"
    
    if not found_any:
        pytest.skip("No BHive training data found. Run bhive_preprocess.py first.")
