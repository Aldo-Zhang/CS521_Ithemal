# test_model_components.py
"""Test model component imports and initialization"""
import os
import sys
import pytest


def test_data_cost_module():
    """Test DataInstructionEmbedding module can be imported"""
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from data import data_cost as dt
    
    data = dt.DataInstructionEmbedding()
    print("✓ DataInstructionEmbedding module works")


def test_model_creation():
    """Test model modules can be imported"""
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from models import graph_models as gm
    
    print("✓ Model modules can be imported")


def test_rnn_model_structure():
    """Test RNN model structure"""
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from models import rnn_models
    
    # Check model classes exist
    assert hasattr(rnn_models, 'RNN')
    print("✓ RNN model class available")


def test_loss_functions():
    """Test loss function modules"""
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from models import losses
    
    print("✓ Loss function module available")
