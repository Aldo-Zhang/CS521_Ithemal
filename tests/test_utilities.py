"""
Test suite for modernized utilities.py
Tests basic functionality without requiring external dependencies like MySQL or ITHEMAL_HOME
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common', 'common_libs'))

import utilities as ut


class TestInstructionClass(unittest.TestCase):
    """Test the Instruction class"""

    def setUp(self):
        """Set up test fixtures"""
        self.instr = ut.Instruction(opcode=1, srcs=[10, 20], dsts=[30], num=0)

    def test_instruction_creation(self):
        """Test that an instruction can be created"""
        self.assertEqual(self.instr.opcode, 1)
        self.assertEqual(self.instr.srcs, [10, 20])
        self.assertEqual(self.instr.dsts, [30])
        self.assertEqual(self.instr.num, 0)
        self.assertEqual(self.instr.parents, [])
        self.assertEqual(self.instr.children, [])

    def test_instruction_clone(self):
        """Test that an instruction can be cloned"""
        cloned = self.instr.clone()
        self.assertEqual(cloned.opcode, self.instr.opcode)
        self.assertEqual(cloned.srcs, self.instr.srcs)
        self.assertEqual(cloned.dsts, self.instr.dsts)
        # Verify it's a new object
        self.assertIsNot(cloned, self.instr)
        # Verify lists are new objects
        self.assertIsNot(cloned.srcs, self.instr.srcs)
        self.assertIsNot(cloned.dsts, self.instr.dsts)

    def test_is_idempotent(self):
        """Test idempotency check"""
        # Non-overlapping srcs and dsts - should be idempotent
        self.assertTrue(self.instr.is_idempotent())

        # Overlapping srcs and dsts - should not be idempotent
        instr2 = ut.Instruction(opcode=1, srcs=[10, 20], dsts=[10], num=0)
        self.assertFalse(instr2.is_idempotent())


class TestBasicBlockClass(unittest.TestCase):
    """Test the BasicBlock class"""

    def setUp(self):
        """Set up test fixtures"""
        self.instr1 = ut.Instruction(opcode=1, srcs=[10], dsts=[20], num=0)
        self.instr2 = ut.Instruction(opcode=2, srcs=[20], dsts=[30], num=1)
        self.instr3 = ut.Instruction(opcode=3, srcs=[30], dsts=[40], num=2)
        self.instrs = [self.instr1, self.instr2, self.instr3]
        self.block = ut.BasicBlock(self.instrs)

    def test_basic_block_creation(self):
        """Test that a basic block can be created"""
        self.assertEqual(len(self.block.instrs), 3)
        self.assertEqual(self.block.num_instrs(), 3)

    def test_find_roots(self):
        """Test finding root instructions"""
        # Before dependencies are created, all instructions should be roots
        roots = self.block.find_roots()
        self.assertEqual(len(roots), 3)

        # After creating dependencies
        self.block.create_dependencies()
        roots = self.block.find_roots()
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0], self.instr1)

    def test_find_leaves(self):
        """Test finding leaf instructions"""
        # Before dependencies are created, all instructions should be leaves
        leaves = self.block.find_leaves()
        self.assertEqual(len(leaves), 3)

        # After creating dependencies
        self.block.create_dependencies()
        leaves = self.block.find_leaves()
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0], self.instr3)

    def test_create_dependencies(self):
        """Test dependency creation"""
        self.block.create_dependencies()

        # Check that dependencies are created correctly
        # instr1 should have instr2 as a child
        self.assertIn(self.instr2, self.instr1.children)
        # instr2 should have instr1 as a parent and instr3 as a child
        self.assertIn(self.instr1, self.instr2.parents)
        self.assertIn(self.instr3, self.instr2.children)
        # instr3 should have instr2 as a parent
        self.assertIn(self.instr2, self.instr3.parents)

    def test_remove_edges(self):
        """Test edge removal"""
        self.block.create_dependencies()
        self.block.remove_edges()

        # All instructions should have no parents or children
        for instr in self.block.instrs:
            self.assertEqual(len(instr.parents), 0)
            self.assertEqual(len(instr.children), 0)

    def test_linearize_edges(self):
        """Test edge linearization"""
        self.block.remove_edges()
        self.block.linearize_edges()

        # Check linear dependencies
        self.assertIn(self.instr2, self.instr1.children)
        self.assertIn(self.instr3, self.instr2.children)
        self.assertIn(self.instr1, self.instr2.parents)
        self.assertIn(self.instr2, self.instr3.parents)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""

    def test_get_percentage_error(self):
        """Test percentage error calculation"""
        predicted = [10, 20, 30]
        actual = [10, 25, 40]
        errors = ut.get_percentage_error(predicted, actual)

        self.assertEqual(len(errors), 3)
        self.assertEqual(errors[0], 0.0)  # (10-10)/10 * 100 = 0
        self.assertEqual(errors[1], 20.0)  # (20-25)/25 * 100 = 20
        self.assertAlmostEqual(errors[2], 25.0)  # (30-40)/40 * 100 = 25

    def test_get_percentage_error_with_lists(self):
        """Test percentage error calculation with list inputs"""
        predicted = [[5, 10], [15, 20]]
        actual = [[5, 10], [15, 25]]
        errors = ut.get_percentage_error(predicted, actual)

        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0], 0.0)  # (10-10)/10 * 100 = 0
        self.assertEqual(errors[1], 20.0)  # (20-25)/25 * 100 = 20

    def test_get_name(self):
        """Test get_name function"""
        sym_dict = {0: 'OP_ADD', 1: 'OP_SUB', 2: 'REG_RAX'}
        mem_offset = 1000

        # Regular symbol
        self.assertEqual(ut.get_name(0, sym_dict, mem_offset), 'OP_ADD')
        self.assertEqual(ut.get_name(1, sym_dict, mem_offset), 'OP_SUB')

        # Memory reference
        self.assertEqual(ut.get_name(1005, sym_dict, mem_offset), 'mem_5')

        # Delimiter
        self.assertEqual(ut.get_name(-1, sym_dict, mem_offset), 'delim')

    def test_get_mysql_config(self):
        """Test MySQL config parsing"""
        with patch('builtins.open', unittest.mock.mock_open(read_data='user = "testuser"\npassword = "testpass"\nport = 3306\n')):
            config = ut.get_mysql_config('dummy.cnf')
            self.assertEqual(config['user'], 'testuser')
            self.assertEqual(config['password'], 'testpass')
            self.assertEqual(config['port'], '3306')


class TestCreateBasicBlock(unittest.TestCase):
    """Test create_basicblock function"""

    def test_create_basicblock(self):
        """Test basic block creation from tokens"""
        # Token format: [opcode, -1, src1, src2, -1, dst1, -1, ...]
        tokens = [1, -1, 10, 20, -1, 30, -1, 2, -1, 30, -1, 40, -1]
        block = ut.create_basicblock(tokens)

        self.assertEqual(len(block.instrs), 2)
        self.assertEqual(block.instrs[0].opcode, 1)
        self.assertEqual(block.instrs[0].srcs, [10, 20])
        self.assertEqual(block.instrs[0].dsts, [30])
        self.assertEqual(block.instrs[1].opcode, 2)
        self.assertEqual(block.instrs[1].srcs, [30])
        self.assertEqual(block.instrs[1].dsts, [40])


class TestRegisterClasses(unittest.TestCase):
    """Test register classification functions"""

    def test_get_register_class(self):
        """Test register class identification"""
        # These tests depend on _global_sym_dict being initialized
        # We'll test the logic with string inputs
        # REG_RAX is a 64-bit register
        result = ut.get_register_class('REG_RAX')
        self.assertIsNotNone(result)
        self.assertIn('REG_RAX', result)
        # REG_RAX class includes other 64-bit registers
        self.assertIn('REG_RBX', result)
        self.assertIn('REG_RCX', result)

        # Test with unknown register
        result = ut.get_register_class('UNKNOWN_REG')
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
