# test_model_components.py
import os
def test_data_cost_module():
    import sys
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from data import data_cost as dt
    
    data = dt.DataInstructionEmbedding()
    print("✓ DataInstructionEmbedding module works")

def test_model_creation():
    import sys
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    from models import graph_models as gm
    
    print("✓ Model modules can be imported")