# test_predict_pipline.py
"""
This file test the predict pipeline by using a pre-trained model (haswell) and a binary file.
Models are loaded from https://github.com/ithemal/Ithemal-models/tree/master/paper/haswell
"""
import pytest
import os
import subprocess
import tempfile
import shutil
import glob
import sys

@pytest.fixture(scope="module")
def predict_script():
    """Get path to predict.py script"""
    home = os.environ['ITHEMAL_HOME']
    script = os.path.join(home, 'learning/pytorch/ithemal/predict.py')
    assert os.path.exists(script), f"predict.py not found: {script}"
    return script

@pytest.fixture(scope="module")
def trained_model():
    """Get trained model files from models/ directory"""
    home = os.environ['ITHEMAL_HOME']
    models_dir = os.path.join(home, 'learning/pytorch/inputs/models')
    predictor_dump = os.path.join(models_dir, 'predictor.dump')
    trained_mdl = os.path.join(models_dir, 'trained.mdl')
    
    if not os.path.exists(predictor_dump):
        pytest.skip(f"predictor.dump not found: {predictor_dump}")
    if not os.path.exists(trained_mdl):
        pytest.skip(f"trained.mdl not found: {trained_mdl}")
    
    print(f"✓ Using model files from: {models_dir}")
    print(f"  - predictor.dump: {os.path.getsize(predictor_dump)} bytes")
    print(f"  - trained.mdl: {os.path.getsize(trained_mdl)} bytes")
    
    return (predictor_dump, trained_mdl)

@pytest.fixture(scope="module")
def example_binary():
    """Compile example.c to create a test binary with IACA markers"""
    home = os.environ['ITHEMAL_HOME']
    example_c = os.path.join(home, 'learning/pytorch/examples/example.c')
    example_h = os.path.join(home, 'learning/pytorch/examples/iacaMarks.h')
    
    if not os.path.exists(example_c):
        pytest.skip(f"example.c not found: {example_c}")
    if not os.path.exists(example_h):
        pytest.skip(f"iacaMarks.h not found: {example_h}")
    
    # Create temporary directory for compiled binary
    temp_dir = tempfile.mkdtemp(prefix='ithemal_test_')
    binary_path = os.path.join(temp_dir, 'example_binary')
    
    try:
        # Compile example.c
        compile_cmd = [
            'gcc',
            '-o', binary_path,
            example_c,
            '-I', os.path.dirname(example_h)
        ]
        
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            pytest.skip(f"Failed to compile example.c: {result.stderr}")
        
        if not os.path.exists(binary_path):
            pytest.skip(f"Compiled binary not found: {binary_path}")
        
        yield binary_path
        
    finally:
        # Cleanup
        if os.path.exists(binary_path):
            os.remove(binary_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@pytest.mark.slow
def test_predict_file_mode(predict_script, trained_model, example_binary):
    """Test prediction with binary file input (--files mode)"""
    predictor_dump, trained_mdl = trained_model
    
    args = [
        'python', predict_script,
        '--model', predictor_dump,
        '--model-data', trained_mdl,
        '--files', example_binary,
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing file-based prediction...")
    print(f"Command: {' '.join(args)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        pytest.fail(f"Prediction failed with return code {result.returncode}")
    
    # Parse output - should be a single float (predicted cycles)
    output_lines = result.stdout.strip().split('\n')
    assert len(output_lines) > 0, "No output from prediction"
    
    try:
        prediction = float(output_lines[-1])  # Last line should be the prediction
        assert prediction > 0, f"Prediction should be positive, got {prediction}"
        assert prediction < 1000, f"Prediction seems unreasonably high: {prediction}"
        print(f"✓ Prediction successful: {prediction} cycles")
    except ValueError:
        pytest.fail(f"Could not parse prediction as float. Output: {result.stdout}")

@pytest.mark.slow
def test_predict_file_mode_verbose(predict_script, trained_model, example_binary):
    """Test prediction with verbose output (shows assembly code)"""
    predictor_dump, trained_mdl = trained_model
    
    args = [
        'python', predict_script,
        '--model', predictor_dump,
        '--model-data', trained_mdl,
        '--files', example_binary,
        '--verbose',
    ]
    
    print(f"\n{'='*60}")
    print(f"Testing verbose prediction...")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        pytest.fail(f"Verbose prediction failed with return code {result.returncode}")
    
    # Check that verbose output contains assembly code markers
    assert '=' * 40 in result.stdout, "Verbose output should contain separator lines"
    
    # Parse prediction
    output_lines = result.stdout.strip().split('\n')
    prediction_line = output_lines[-1]
    
    try:
        prediction = float(prediction_line)
        assert prediction > 0, f"Prediction should be positive, got {prediction}"
        print(f"✓ Verbose prediction successful: {prediction} cycles")
        print(f"  Assembly code displayed: {len([l for l in output_lines if '=' not in l and l.strip()])} lines")
    except ValueError:
        pytest.fail(f"Could not parse prediction. Output: {result.stdout}")
