"""
Tests for analyze_results.py
"""
import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add analysis directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis'))

import analyze_results as ar


class TestParseValidationResults(unittest.TestCase):
    """Test parse_validation_results function"""

    def test_parse_basic(self):
        """Test basic parsing of validation results"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("10.5,10.0\n")
            f.write("20.3,21.0\n")
            f.write("30.1,30.5\n")
            f.write("loss - 0.123456\n")
            f.write("1000,250\n")
            temp_path = f.name

        try:
            df, loss = ar.parse_validation_results(temp_path)
            
            self.assertEqual(len(df), 3)
            self.assertAlmostEqual(loss, 0.123456, places=5)
            self.assertEqual(df.iloc[0]['predicted'], 10.5)
            self.assertEqual(df.iloc[0]['actual'], 10.0)
        finally:
            os.unlink(temp_path)

    def test_parse_no_loss(self):
        """Test parsing when loss line is missing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("10.5,10.0\n")
            f.write("20.3,21.0\n")
            temp_path = f.name

        try:
            df, loss = ar.parse_validation_results(temp_path)
            
            self.assertEqual(len(df), 2)
            self.assertIsNone(loss)
        finally:
            os.unlink(temp_path)


class TestParseLossReport(unittest.TestCase):
    """Test parse_loss_report function"""

    def test_parse_basic(self):
        """Test basic parsing of loss report"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write("1\t100.5\t0.500\t4\n")
            f.write("2\t200.3\t0.400\t4\n")
            f.write("3\t300.1\t0.350\t4\n")
            temp_path = f.name

        try:
            df = ar.parse_loss_report(temp_path)
            
            self.assertEqual(len(df), 3)
            self.assertEqual(df.iloc[0]['epoch'], 1)
            self.assertAlmostEqual(df.iloc[0]['loss'], 0.500, places=3)
            self.assertEqual(df.iloc[2]['epoch'], 3)
        finally:
            os.unlink(temp_path)


class TestCalculateMetrics(unittest.TestCase):
    """Test calculate_metrics function"""

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions"""
        df = pd.DataFrame({
            'predicted': [10.0, 20.0, 30.0, 40.0, 50.0],
            'actual': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        metrics = ar.calculate_metrics(df)
        
        self.assertAlmostEqual(metrics['pearson_r'], 1.0, places=5)
        self.assertAlmostEqual(metrics['spearman_r'], 1.0, places=5)
        self.assertAlmostEqual(metrics['mape'], 0.0, places=5)
        self.assertAlmostEqual(metrics['mae'], 0.0, places=5)
        self.assertAlmostEqual(metrics['rmse'], 0.0, places=5)
        self.assertEqual(metrics['within_10_pct'], 100.0)
        self.assertEqual(metrics['within_20_pct'], 100.0)

    def test_with_errors(self):
        """Test metrics with prediction errors"""
        df = pd.DataFrame({
            'predicted': [11.0, 22.0, 27.0, 44.0, 55.0],
            'actual': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        metrics = ar.calculate_metrics(df)
        
        self.assertGreater(metrics['pearson_r'], 0.9)
        self.assertGreater(metrics['spearman_r'], 0.9)
        self.assertGreater(metrics['mape'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['rmse'], 0)

    def test_filtered_pearson(self):
        """Test that filtered Pearson correlations are calculated"""
        # Create data with some high values
        df = pd.DataFrame({
            'predicted': list(range(10, 110, 10)) + [5000],
            'actual': list(range(10, 110, 10)) + [10000]
        })
        
        metrics = ar.calculate_metrics(df)
        
        # Filtered Pearson should be higher than raw (because outlier is removed)
        self.assertIn('pearson_r_1000', metrics)
        self.assertIn('pearson_r_500', metrics)
        self.assertIn('n_filtered_1000', metrics)
        self.assertIn('n_filtered_500', metrics)


class TestMetricsFormulas(unittest.TestCase):
    """Test that metrics formulas are correct"""

    def test_mape_formula(self):
        """Test MAPE calculation formula"""
        df = pd.DataFrame({
            'predicted': [110.0, 90.0],
            'actual': [100.0, 100.0]
        })
        
        metrics = ar.calculate_metrics(df)
        
        # MAPE = mean(|pred - actual| / actual) * 100
        # = mean(|110-100|/100, |90-100|/100) * 100
        # = mean(0.1, 0.1) * 100 = 10%
        self.assertAlmostEqual(metrics['mape'], 10.0, places=3)

    def test_mae_formula(self):
        """Test MAE calculation formula"""
        df = pd.DataFrame({
            'predicted': [15.0, 25.0],
            'actual': [10.0, 20.0]
        })
        
        metrics = ar.calculate_metrics(df)
        
        # MAE = mean(|15-10|, |25-20|) = mean(5, 5) = 5
        self.assertAlmostEqual(metrics['mae'], 5.0, places=5)

    def test_rmse_formula(self):
        """Test RMSE calculation formula"""
        df = pd.DataFrame({
            'predicted': [13.0, 17.0],
            'actual': [10.0, 20.0]
        })
        
        metrics = ar.calculate_metrics(df)
        
        # RMSE = sqrt(mean((13-10)^2, (17-20)^2)) = sqrt(mean(9, 9)) = sqrt(9) = 3
        self.assertAlmostEqual(metrics['rmse'], 3.0, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def test_empty_file(self):
        """Test handling of empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            df, loss = ar.parse_validation_results(temp_path)
            self.assertEqual(len(df), 0)
        finally:
            os.unlink(temp_path)

    def test_single_sample(self):
        """Test with single sample"""
        df = pd.DataFrame({
            'predicted': [10.0],
            'actual': [10.0]
        })
        
        # Should not crash
        metrics = ar.calculate_metrics(df)
        self.assertEqual(metrics['n_samples'], 1)


if __name__ == '__main__':
    unittest.main()
