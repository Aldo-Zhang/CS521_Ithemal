"""
Pytest configuration for CS521_Ithemal tests

This configuration automatically sets up the ITHEMAL_HOME environment variable
for all test sessions, making tests work seamlessly without manual setup.
"""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_ithemal_home():
    """
    Auto-set ITHEMAL_HOME for test sessions if not already set.
    
    This fixture runs once per test session and is automatically used by all tests.
    It sets ITHEMAL_HOME to point to the CS521_Ithemal directory, which is the
    parent directory of the tests/ folder.
    
    The fixture allows tests to import modules that depend on ITHEMAL_HOME
    without requiring users to manually set the environment variable.
    """
    if 'ITHEMAL_HOME' not in os.environ:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        ithemal_home = os.path.dirname(test_dir)  # Go up from tests/ to project root
        os.environ['ITHEMAL_HOME'] = ithemal_home
        print(f"\n✓ Auto-configured ITHEMAL_HOME={ithemal_home}")
    
    yield
    
    # No cleanup needed - environment variables persist for the test session

