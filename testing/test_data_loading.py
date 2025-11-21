# test_data_loading.py
def test_load_sample_data():
    """Test if the sample data can be loaded"""
    import torch
    import urllib.request
    
    home = os.environ['ITHEMAL_HOME']
    data_file = os.path.join(home, 'learning/pytorch/inputs/data/haswell_sample1000.data')
    
    if not os.path.exists(data_file):
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/ithemal/Ithemal-models/master/paper/data/haswell_sample1000.data',
            data_file
        )
    
    data = torch.load(data_file)
    
    assert len(data) == 1000
    code_id, timing, code_intel, code_xml = data[0]
    assert isinstance(timing, float) and timing > 0
    
    print(f"✓ Loaded {len(data)} training samples")
    print(f"  Example timing: {timing} cycles")