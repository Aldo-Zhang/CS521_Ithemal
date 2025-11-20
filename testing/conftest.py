import pytest
import os
import sys
import subprocess
import glob
import re
from shutil import copyfile

if 'ITHEMAL_HOME' not in os.environ:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['ITHEMAL_HOME'] = os.path.dirname(test_dir)

sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'common'))

if 'DYNAMORIO_HOME' not in os.environ: # If Dynamorio is not installed, skip the tests that require it
    os.environ['DYNAMORIO_HOME'] = '/tmp/dynamorio_not_installed'

dynamorio = pytest.mark.skipif('DYNAMORIO_HOME' not in os.environ.keys(),
                                reason="DYNAMORIO_HOME not set")

ithemal = pytest.mark.skipif('ITHEMAL_HOME' not in os.environ.keys(),
                                reason="ITHEMAL_HOME not set")

@pytest.fixture(scope="module")
def db_config():
    # Get the path to the test data directory and the example configuration file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(test_dir, 'test_data')
    example_config = os.path.join(test_data_dir, 'example_config.cfg')
    db_config_file = os.path.join(test_data_dir, 'db_config.cfg')
    
    if not os.path.exists(db_config_file):
        if os.path.exists(example_config):
            copyfile(example_config, db_config_file)
        else:
            pytest.skip(f"Missing test configuration file: {example_config}")
    
    config = dict()
    with open(db_config_file, 'r') as f:
        for line in f:
            found = re.search(r'([a-zA-Z\-]+) *= *\"*([a-zA-Z0-9#\./]+)\"*', line)
            if found:
                config[found.group(1)] = found.group(2)
    
    return config




