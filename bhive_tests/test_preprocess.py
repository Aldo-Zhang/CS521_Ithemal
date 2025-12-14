"""
Tests for bhive_preprocess.py
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPreprocessConfig(unittest.TestCase):
    """Test preprocessing configuration"""

    def test_config_imports(self):
        """Test that config variables exist"""
        import bhive_preprocess as bp
        
        self.assertIsNotNone(bp.PROJECT_ROOT)
        self.assertIsNotNone(bp.BHIVE_ROOT)
        self.assertIsNotNone(bp.INPUT_FILE_DICT)
        self.assertIsNotNone(bp.OUTPUT_FILE_DICT)
        self.assertIsNotNone(bp.TOKENIZER)

    def test_input_output_architectures(self):
        """Test that all architectures are configured"""
        import bhive_preprocess as bp
        
        expected_archs = ['hsw', 'ivb', 'skl']
        
        for arch in expected_archs:
            self.assertIn(arch, bp.INPUT_FILE_DICT)
            self.assertIn(arch, bp.OUTPUT_FILE_DICT)

    def test_paths_are_absolute(self):
        """Test that paths are absolute"""
        import bhive_preprocess as bp
        
        self.assertTrue(os.path.isabs(bp.PROJECT_ROOT))
        self.assertTrue(os.path.isabs(bp.BHIVE_ROOT))
        self.assertTrue(os.path.isabs(bp.TOKENIZER))


class TestProcessRow(unittest.TestCase):
    """Test the process_row function"""

    @patch('bhive_preprocess.subprocess.check_output')
    def test_process_row_success(self, mock_subprocess):
        """Test successful processing of a row"""
        import bhive_preprocess as bp
        
        mock_subprocess.side_effect = [
            '<token>test</token>',  # --token output
            'mov rax, rbx'          # --intel output
        ]
        
        result = bp.process_row('48890000', '10.5')
        
        self.assertIsNotNone(result)
        self.assertEqual(result[0], '48890000')  # hex_code
        self.assertEqual(result[1], 10.5)         # timing (float)
        self.assertEqual(result[2], 'mov rax, rbx')  # intel
        self.assertEqual(result[3], '<token>test</token>')  # xml

    @patch('bhive_preprocess.subprocess.check_output')
    def test_process_row_failure(self, mock_subprocess):
        """Test handling of processing failure"""
        import bhive_preprocess as bp
        
        mock_subprocess.side_effect = Exception("Tokenizer failed")
        
        result = bp.process_row('invalid', '10.5')
        
        self.assertIsNone(result)


class TestTokenizerPath(unittest.TestCase):
    """Test tokenizer path configuration"""

    def test_tokenizer_path_format(self):
        """Test tokenizer path is in expected location"""
        import bhive_preprocess as bp
        
        self.assertIn('data_collection', bp.TOKENIZER)
        self.assertIn('build', bp.TOKENIZER)
        self.assertIn('bin', bp.TOKENIZER)
        self.assertIn('tokenizer', bp.TOKENIZER)


if __name__ == '__main__':
    unittest.main()
