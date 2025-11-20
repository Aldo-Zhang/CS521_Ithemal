import pytest
import os
import subprocess
import glob
import re
from shutil import copyfile

if 'ITHEMAL_HOME' not in os.environ:
    test_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ['ITHEMAL_HOME'] = os.path.dirname(test_dir)

sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'common', 'common_libs'))

dynamorio = pytest.mark.skipif('DYNAMORIO_HOME' not in os.environ.keys(),
                                reason="DYNAMORIO_HOME not set")

ithemal = pytest.mark.skipif('ITHEMAL_HOME' not in os.environ.keys(),
                                reason="ITHEMAL_HOME not set")

@pytest.fixture(scope="module")
def db_config():

    if not os.path.exists('test_data/db_config.cfg'):
        copyfile('test_data/example_config.cfg','test_data/db_config.cfg')

    config = dict()
    with open('test_data/db_config.cfg','r') as f:
        for line in f:
            found = re.search('([a-zA-Z\-]+) *= *\"*([a-zA-Z0-9#\./]+)\"*', line)
            if found:
                config[found.group(1)] = found.group(2)

    return config




