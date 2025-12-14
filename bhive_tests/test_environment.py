"""
Environment verification tests for BHive pipeline
"""
import os
import sys
import unittest


class TestPythonEnvironment(unittest.TestCase):
    """Test Python environment setup"""

    def test_python_version(self):
        """Test Python version is 3.8+"""
        self.assertGreaterEqual(sys.version_info.major, 3)
        self.assertGreaterEqual(sys.version_info.minor, 8)

    def test_required_packages(self):
        """Test required packages are installed"""
        required = ['torch', 'numpy', 'pandas', 'matplotlib', 'scipy', 'tqdm']
        
        for package in required:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Required package '{package}' not installed")

    def test_pytorch_version(self):
        """Test PyTorch is version 2.x"""
        import torch
        major = int(torch.__version__.split('.')[0])
        self.assertGreaterEqual(major, 2, "PyTorch 2.x required")


class TestProjectStructure(unittest.TestCase):
    """Test project structure is correct"""

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_bhive_preprocess_exists(self):
        """Test bhive_preprocess.py exists"""
        path = os.path.join(self.project_root, 'bhive_preprocess.py')
        self.assertTrue(os.path.exists(path), f"bhive_preprocess.py not found at {path}")

    def test_analyze_script_exists(self):
        """Test analyze_results.py exists"""
        path = os.path.join(self.project_root, 'analysis', 'analyze_results.py')
        self.assertTrue(os.path.exists(path), f"analyze_results.py not found at {path}")

    def test_training_script_exists(self):
        """Test run_ithemal.py exists"""
        path = os.path.join(self.project_root, 'learning', 'pytorch', 'ithemal', 'run_ithemal.py')
        self.assertTrue(os.path.exists(path), f"run_ithemal.py not found at {path}")

    def test_data_directory_exists(self):
        """Test data directory exists"""
        path = os.path.join(self.project_root, 'learning', 'pytorch', 'inputs', 'data')
        self.assertTrue(os.path.exists(path), f"Data directory not found at {path}")

    def test_analysis_directory_exists(self):
        """Test analysis directory exists"""
        path = os.path.join(self.project_root, 'analysis')
        self.assertTrue(os.path.exists(path), f"Analysis directory not found at {path}")


class TestTokenizerSetup(unittest.TestCase):
    """Test tokenizer setup (optional - may not be built on all systems)"""

    def setUp(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_tokenizer_path_defined(self):
        """Test tokenizer path is correctly defined in preprocess script"""
        import bhive_preprocess as bp
        self.assertIn('tokenizer', bp.TOKENIZER.lower())

    @unittest.skipUnless(
        os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data_collection', 'build', 'bin', 'tokenizer'
        )),
        "Tokenizer not built"
    )
    def test_tokenizer_executable(self):
        """Test tokenizer is executable (only if built)"""
        path = os.path.join(self.project_root, 'data_collection', 'build', 'bin', 'tokenizer')
        self.assertTrue(os.access(path, os.X_OK), "Tokenizer is not executable")


if __name__ == '__main__':
    unittest.main()
