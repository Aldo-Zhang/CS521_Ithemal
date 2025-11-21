# test_training_pipeline.py
import pytest
import os
import subprocess
import shutil
import time

@pytest.mark.slow 
def test_mini_training():
    """Test a mini training run to verify the entire pipeline works"""
    
    home = os.environ['ITHEMAL_HOME']
    
    data_file = os.path.join(home, 'learning/pytorch/inputs/data/haswell_sample1000.data')
    assert os.path.exists(data_file), f"Data file not found: {data_file}. Run test_data_loading first."
    
    print(f"✓ Using data file: {data_file}")
    
    experiment_name = "test_mini_training"
    experiment_time = str(int(time.time()))
    result_dir = os.path.join(home, 'learning/pytorch/saved', experiment_name, experiment_time)
    
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    
    # run the training
    script = os.path.join(home, 'learning/pytorch/ithemal/run_ithemal.py')
    
    args = [
        'python', script,
        '--data', data_file,
        '--use-rnn',
        'train',
        '--experiment-name', experiment_name,
        '--experiment-time', experiment_time,
        '--sgd',
        '--threads', '2',
        '--trainers', '2',
        '--epochs', '1',
        '--batch-size', '4',
    ]
    
    print(f"\n{'='*60}")
    print(f"Running mini training test...")
    print(f"Command: {' '.join(args)}")
    print(f"{'='*60}\n")
    

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    

    for line in proc.stdout:
        print(line, end='')
    
    proc.wait()
    
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        pytest.fail(f"Training failed with return code {proc.returncode}\nSTDERR: {stderr}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")
    
    assert os.path.exists(result_dir), f"Result directory not created: {result_dir}"
    
    trained_model = os.path.join(result_dir, 'trained.mdl')
    predictor_dump = os.path.join(result_dir, 'predictor.dump')
    loss_report = os.path.join(result_dir, 'loss_report.log')
    validation_results = os.path.join(result_dir, 'validation_results.txt')
    
    files_to_check = {
        'trained.mdl': trained_model,
        'predictor.dump': predictor_dump,
        'loss_report.log': loss_report,
    }
    
    print("Checking output files:")
    for name, path in files_to_check.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {name}: {size} bytes")
        else:
            print(f"  ⚠ {name}: missing")
    
    assert os.path.exists(trained_model), f"Trained model not created: {trained_model}"
    assert os.path.getsize(trained_model) > 0, "Trained model file is empty"
    
    print(f"\n✓ Mini training test PASSED!")
    print(f"  Model saved to: {result_dir}")
    
    return result_dir


@pytest.mark.slow
def test_model_prediction():
    """Test that we can use the trained model for prediction"""
    
    home = os.environ['ITHEMAL_HOME']
    
    pytest.skip("Prediction test requires test code samples - implement later")


def test_training_components_available():
    """Quick smoke test that training components can be imported"""
    import sys
    sys.path.insert(0, os.path.join(os.environ['ITHEMAL_HOME'], 'learning/pytorch'))
    
    try:
        from data import data_cost
        from models import graph_models
        from models import train
        import ithemal.run_ithemal as run_ithemal
        
        print("✓ All training modules can be imported")
        
    except ImportError as e:
        pytest.fail(f"Failed to import training modules: {e}")