"""
Pytest configuration for BHive pipeline tests
"""
import os
import sys

# Set ITHEMAL_HOME before imports
if 'ITHEMAL_HOME' not in os.environ:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ithemal_home = os.path.dirname(test_dir)
    os.environ['ITHEMAL_HOME'] = ithemal_home

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
