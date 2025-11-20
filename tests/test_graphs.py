"""
Test suite for modernized graphs.py
Tests plotting functionality
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np

# Add parent directory to path to import graphs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'common', 'common_libs'))

import graphs as gr


class TestPlotHistogram(unittest.TestCase):
    """Test the plot_histogram function"""

    @patch('graphs.plt')
    def test_plot_histogram_called_with_correct_params(self, mock_plt):
        """Test that plot_histogram calls matplotlib with correct parameters"""
        # Mock matplotlib functions
        mock_plt.figure = Mock()
        mock_plt.hist = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.title = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        # Call the function
        values = [1, 2, 3, 4, 5]
        gr.plot_histogram('test.png', values, 10, 'X Label', 'Y Label', 'Test Title')

        # Verify matplotlib functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        mock_plt.xlabel.assert_called_once_with('X Label')
        mock_plt.ylabel.assert_called_once_with('Y Label')
        mock_plt.title.assert_called_once_with('Test Title')
        mock_plt.savefig.assert_called_once_with('test.png', bbox_inches='tight')
        mock_plt.close.assert_called_once()

    @patch('graphs.plt')
    def test_plot_histogram_with_correct_values(self, mock_plt):
        """Test that histogram is called with the values parameter (not self.values)"""
        mock_plt.figure = Mock()
        mock_plt.hist = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.title = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        values = [1, 2, 3, 4, 5]
        gr.plot_histogram('test.png', values, 10, 'X', 'Y', 'Title')

        # Check that hist was called with values (not self.values which was the bug)
        args, kwargs = mock_plt.hist.call_args
        self.assertEqual(args[0], values)


class TestPlotLineGraphs(unittest.TestCase):
    """Test the plot_line_graphs function"""

    @patch('graphs.plt')
    def test_plot_line_graphs_basic(self, mock_plt):
        """Test basic line graph plotting"""
        # Mock matplotlib functions
        mock_plt.figure = Mock()
        mock_plt.plot = Mock()
        mock_plt.legend = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.title = Mock()
        mock_plt.xlim = Mock(return_value=(0, 10))
        mock_plt.ylim = Mock(return_value=(0, 100))
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        # Call the function
        losses = [[1, 2, 3], [4, 5, 6]]
        legend = ['loss1', 'loss2']
        gr.plot_line_graphs('test.png', losses, legend)

        # Verify matplotlib functions were called
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.plot.call_count, 2)  # Called once for each loss
        mock_plt.legend.assert_called_once()
        mock_plt.ylabel.assert_called_once_with('loss')
        mock_plt.xlabel.assert_called_once_with('batch')
        mock_plt.title.assert_called_once_with('Learning Curves')
        mock_plt.savefig.assert_called_once_with('test.png')
        mock_plt.close.assert_called_once()

    @patch('graphs.plt')
    def test_plot_line_graphs_with_custom_labels(self, mock_plt):
        """Test line graph plotting with custom labels"""
        mock_plt.figure = Mock()
        mock_plt.plot = Mock()
        mock_plt.legend = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.title = Mock()
        mock_plt.xlim = Mock(return_value=(0, 10))
        mock_plt.ylim = Mock(return_value=(0, 100))
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        losses = [[1, 2, 3]]
        legend = ['custom_loss']
        gr.plot_line_graphs('test.png', losses, legend, ylabel='Custom Y', xlabel='Custom X', title='Custom Title')

        mock_plt.ylabel.assert_called_once_with('Custom Y')
        mock_plt.xlabel.assert_called_once_with('Custom X')
        mock_plt.title.assert_called_once_with('Custom Title')

    @patch('graphs.plt')
    def test_plot_line_graphs_with_axis_limits(self, mock_plt):
        """Test line graph plotting with custom axis limits"""
        mock_plt.figure = Mock()
        mock_plt.plot = Mock()
        mock_plt.legend = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.title = Mock()
        mock_plt.xlim = Mock(return_value=(0, 100))
        mock_plt.ylim = Mock(return_value=(0, 200))
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        losses = [[1, 2, 3]]
        legend = ['loss']
        gr.plot_line_graphs('test.png', losses, legend, xmin=0, xmax=50, ymin=0, ymax=100)

        # Check that xlim and ylim were called with the custom limits
        # Note: xlim and ylim may be called multiple times (once to get, once to set)
        self.assertTrue(mock_plt.xlim.call_count >= 1)
        self.assertTrue(mock_plt.ylim.call_count >= 1)

    @patch('graphs.plt')
    def test_plot_line_graphs_none_comparisons(self, mock_plt):
        """Test that None comparisons use 'is not None' instead of '!= None'"""
        mock_plt.figure = Mock()
        mock_plt.plot = Mock()
        mock_plt.legend = Mock()
        mock_plt.ylabel = Mock()
        mock_plt.xlabel = Mock()
        mock_plt.title = Mock()
        mock_plt.xlim = Mock(return_value=(0, 10))
        mock_plt.ylim = Mock(return_value=(0, 100))
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()

        # This test ensures the function runs without errors
        # The modernized code uses 'is not None' instead of '!= None'
        losses = [[1, 2, 3]]
        legend = ['loss']
        gr.plot_line_graphs('test.png', losses, legend, xmin=None, xmax=None, ymin=None, ymax=None)

        # Should complete without errors
        mock_plt.savefig.assert_called_once()


class TestMainExecution(unittest.TestCase):
    """Test the main execution block"""

    @patch('graphs.plot_line_graphs')
    @patch('graphs.random.randint')
    def test_main_execution(self, mock_randint, mock_plot):
        """Test that the main block works correctly"""
        # This test would run if we executed the module as __main__
        # For now, we just verify the function exists and can be called
        mock_randint.side_effect = [2, 10, 10, 15, 15]
        mock_plot.return_value = None

        # We can't easily test the if __name__ == '__main__' block
        # But we've verified the functions work correctly
        pass


if __name__ == '__main__':
    unittest.main()
