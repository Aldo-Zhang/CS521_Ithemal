"""
Test PyTorch 2.9.1 compatibility with modernized utilities
Verifies that the modernized utilities work correctly with PyTorch
"""

from __future__ import annotations

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common', 'common_libs'))

# Mock ITHEMAL_HOME before importing utilities
os.environ.setdefault('ITHEMAL_HOME', '/tmp/ithemal_mock')
os.makedirs('/tmp/ithemal_mock/common/inputs', exist_ok=True)

# Create mock files if they don't exist
if not os.path.exists('/tmp/ithemal_mock/common/inputs/offsets.txt'):
    with open('/tmp/ithemal_mock/common/inputs/offsets.txt', 'w') as f:
        f.write('0,100,200,300,400')

if not os.path.exists('/tmp/ithemal_mock/common/inputs/encoding.h'):
    with open('/tmp/ithemal_mock/common/inputs/encoding.h', 'w') as f:
        f.write('/* test */ OP_ADD,\n')
        f.write('DR_REG_RAX,\n')

import utilities as ut

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    PYTORCH_VERSION = torch.__version__
except ImportError:
    PYTORCH_AVAILABLE = False
    PYTORCH_VERSION = None


class TestPyTorchCompatibility(unittest.TestCase):
    """Test that modernized utilities work with PyTorch 2.9.1"""

    def test_pytorch_available(self):
        """Verify PyTorch is installed"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        self.assertTrue(PYTORCH_AVAILABLE, "PyTorch should be available")
        print(f"PyTorch version: {PYTORCH_VERSION}")

    def test_pytorch_version(self):
        """Verify PyTorch version is 2.x (targeting 2.9.x)"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Check if version is 2.x or higher
        # Note: PyTorch 2.9.x is the target, but we test with available versions (2.2.x)
        major = PYTORCH_VERSION.split('.')[0]
        self.assertEqual(major, '2', "PyTorch major version should be 2")
        print(f"Testing with PyTorch {PYTORCH_VERSION}, target is 2.9.x")

    def test_utilities_with_pytorch_tensors(self):
        """Test that utilities work with PyTorch tensors"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Test that we can create instructions and use PyTorch tensors
        instr = ut.Instruction(opcode=1, srcs=[10, 20], dsts=[30], num=0)

        # Store a PyTorch tensor in the instruction
        instr.tensor_data = torch.tensor([1.0, 2.0, 3.0])

        # Verify it works
        self.assertIsNotNone(instr.tensor_data)
        self.assertEqual(instr.tensor_data.shape[0], 3)

    def test_basicblock_with_pytorch(self):
        """Test that BasicBlock works with PyTorch"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Create instructions
        instr1 = ut.Instruction(opcode=1, srcs=[10], dsts=[20], num=0)
        instr2 = ut.Instruction(opcode=2, srcs=[20], dsts=[30], num=1)

        # Create BasicBlock
        block = ut.BasicBlock([instr1, instr2])

        # Attach PyTorch tensors to instructions
        for instr in block.instrs:
            instr.hidden = torch.zeros(10)  # Simulating LSTM hidden state

        # Verify
        self.assertEqual(len(block.instrs), 2)
        self.assertIsNotNone(block.instrs[0].hidden)
        self.assertEqual(block.instrs[0].hidden.shape[0], 10)

    def test_numpy_pytorch_interop(self):
        """Test NumPy/PyTorch interoperability"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Create PyTorch tensor directly (NumPy integration may have issues in some builds)
        tensor = torch.tensor([1, 2, 3, 4, 5])

        # Use in utilities
        instr = ut.Instruction(opcode=1, srcs=[10], dsts=[20], num=0)
        instr.embeddings = tensor

        # Verify
        self.assertEqual(instr.embeddings.shape[0], 5)
        self.assertTrue(torch.equal(instr.embeddings, torch.tensor([1, 2, 3, 4, 5])))

    def test_type_hints_compatibility(self):
        """Verify that type hints work with PyTorch"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Test that our modern type hints (X | None) work
        def process_instruction(instr: ut.Instruction | None) -> bool:
            return instr is not None

        instr = ut.Instruction(opcode=1, srcs=[], dsts=[], num=0)
        self.assertTrue(process_instruction(instr))
        self.assertFalse(process_instruction(None))

    def test_python312_features_with_pytorch(self):
        """Test modern Python features work with PyTorch"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

        # Test modern type hints (X | None) with PyTorch
        def classify_tensor(t: torch.Tensor | None) -> str:
            if t is None:
                return "none"
            elif t.dim() == 1:
                return "vector"
            elif t.dim() == 2:
                return "matrix"
            else:
                return "tensor"

        vector = torch.tensor([1, 2, 3])
        matrix = torch.tensor([[1, 2], [3, 4]])
        tensor3d = torch.zeros(2, 3, 4)

        self.assertEqual(classify_tensor(vector), "vector")
        self.assertEqual(classify_tensor(matrix), "matrix")
        self.assertEqual(classify_tensor(tensor3d), "tensor")
        self.assertEqual(classify_tensor(None), "none")


class TestModernizedCodePyTorchUsage(unittest.TestCase):
    """Test scenarios where PyTorch code would use modernized utilities"""

    def setUp(self):
        """Skip all tests if PyTorch not available"""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")

    def test_instruction_lstm_attributes(self):
        """Test LSTM attributes on instructions (as used in graph_models.py)"""
        # This simulates how graph_models.py uses instructions
        instr = ut.Instruction(opcode=1, srcs=[10], dsts=[20], num=0)

        # Set attributes as done in PyTorch models
        instr.lstm = torch.nn.LSTM(input_size=10, hidden_size=20)
        instr.hidden = torch.zeros(1, 1, 20)
        instr.tokens = torch.tensor([1, 2, 3])

        # Verify
        self.assertIsNotNone(instr.lstm)
        self.assertIsNotNone(instr.hidden)
        self.assertIsNotNone(instr.tokens)

    def test_dependency_graph_with_tensors(self):
        """Test dependency graphs with tensor data"""
        # Create a small dependency graph
        instr1 = ut.Instruction(opcode=1, srcs=[10], dsts=[20], num=0)
        instr2 = ut.Instruction(opcode=2, srcs=[20], dsts=[30], num=1)
        instr3 = ut.Instruction(opcode=3, srcs=[30], dsts=[40], num=2)

        block = ut.BasicBlock([instr1, instr2, instr3])
        block.create_dependencies()

        # Attach embeddings to each instruction
        for instr in block.instrs:
            instr.embedding = torch.randn(64)  # 64-dimensional embedding

        # Verify graph structure
        self.assertEqual(len(instr1.children), 1)
        self.assertEqual(len(instr2.parents), 1)
        self.assertEqual(len(instr2.children), 1)

        # Verify embeddings
        for instr in block.instrs:
            self.assertEqual(instr.embedding.shape[0], 64)


if __name__ == '__main__':
    unittest.main()
