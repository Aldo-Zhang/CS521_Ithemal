"""
Pytest configuration for CS521_Ithemal tests

This configuration automatically sets up the ITHEMAL_HOME environment variable
for all test sessions, making tests work seamlessly without manual setup.

IMPORTANT: The environment variable must be set at MODULE LEVEL (not in a fixture)
because test files import modules (like graphs.py, utilities.py) at import time,
and those imports need ITHEMAL_HOME to be set before they load.
"""
import os
import pytest

# Set ITHEMAL_HOME at module level, before any imports in test files occur
# This executes when pytest loads conftest.py, before importing test modules
if 'ITHEMAL_HOME' not in os.environ:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    ithemal_home = os.path.dirname(test_dir)  # Go up from tests/ to project root
    os.environ['ITHEMAL_HOME'] = ithemal_home

