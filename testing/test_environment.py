# test_environment.py
import pytest
import torch
import mysql.connector
import os

def test_python_version():
    import sys
    assert sys.version_info >= (3, 11)
    print(f"✓ Python {sys.version}")

def test_pytorch_installation():
    assert torch.__version__.startswith('2.')
    print(f"✓ PyTorch {torch.__version__}")
    
def test_cuda_available():
    print(f"CUDA available: {torch.cuda.is_available()}")

def test_mysql_connection():
    cnx = mysql.connector.connect(
        host='db',
        user='root',
        password='ithemal'
    )
    assert cnx.is_connected()
    cnx.close()
    print("✓ MySQL connection works")

def test_ithemal_home():
    assert 'ITHEMAL_HOME' in os.environ
    print(f"✓ ITHEMAL_HOME={os.environ['ITHEMAL_HOME']}")