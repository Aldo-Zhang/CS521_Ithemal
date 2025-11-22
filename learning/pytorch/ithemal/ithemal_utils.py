import sys
import os
sys.path.append(os.path.join(os.environ['ITHEMAL_HOME'], 'learning', 'pytorch'))

from enum import Enum
import torch
from typing import Any, Callable, List, Optional, Iterator, Tuple, NamedTuple, Union

import data.data_cost as dt
import models.graph_models as md
import models.train as tr

class EdgeAblationType(Enum):
    TRANSITIVE_REDUCTION = 'transitive-reduction'
    TRANSITIVE_CLOSURE = 'transitive-closure'
    ADD_LINEAR_EDGES = 'add-linear-edges'
    ONLY_LINEAR_EDGES = 'only-linear-edges'
    NO_EDGES = 'no-edges'

BaseParameters = NamedTuple('BaseParameters', [
    ('data', str),
    ('embed_mode', str),
    ('embed_file', str),
    ('random_edge_freq', float),
    ('predict_log', bool),
    ('no_residual', bool),
    ('no_dag_rnn', bool),
    ('dag_reduction', md.ReductionType),
    ('edge_ablation_types', List[EdgeAblationType]),
    ('embed_size', int),
    ('hidden_size', int),
    ('linear_embeddings', bool),
    ('use_rnn', bool),
    ('rnn_type', md.RnnType),
    ('rnn_hierarchy_type', md.RnnHierarchyType),
    ('rnn_connect_tokens', bool),
    ('rnn_skip_connections', bool),
    ('rnn_learn_init', bool),
    ('no_mem', bool),
    ('linear_dependencies', bool),
    ('flat_dependencies', bool),
    ('dag_nonlinearity', md.NonlinearityType),
    ('dag_nonlinearity_width', int),
    ('dag_nonlinear_before_max', bool),
])

TrainParameters = NamedTuple('TrainParameters', [
    ('experiment_name', str),
    ('experiment_time', str),
    ('load_file', Optional[str]),
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('decay_trainers', bool),
    ('weight_decay', float),
    ('initial_lr', float),
    ('decay_lr', bool),
    ('epochs', int),
    ('split', Union[int, List[float]]),
    ('optimizer', tr.OptimizerType),
    ('momentum', float),
    ('nesterov', bool),
    ('weird_lr', bool),
    ('lr_decay_rate', float),
])

BenchmarkParameters = NamedTuple('BenchmarkParameters', [
    ('batch_size', int),
    ('trainers', int),
    ('threads', int),
    ('examples', int),
])

PredictorDump = NamedTuple('PredictorDump', [
    ('model', md.AbstractGraphModule),
    ('dataset_params', Any),
])


def ablate_data(data, edge_ablation_types, random_edge_freq):
    # type: (dt.DataCost, List[EdgeAblationType], float) -> None

    for edge_ablation_type in edge_ablation_types:
        if edge_ablation_type == EdgeAblationType.TRANSITIVE_REDUCTION:
            for data_item in data.data:
                data_item.block.transitive_reduction()
        elif edge_ablation_type == EdgeAblationType.TRANSITIVE_CLOSURE:
            for data_item in data.data:
                data_item.block.transitive_closure()
        elif edge_ablation_type == EdgeAblationType.ADD_LINEAR_EDGES:
            for data_item in data.data:
                data_item.block.linearize_edges()
        elif edge_ablation_type == EdgeAblationType.ONLY_LINEAR_EDGES:
            for data_item in data.data:
                data_item.block.remove_edges()
                data_item.block.linearize_edges()
        elif edge_ablation_type == EdgeAblationType.NO_EDGES:
            for data_item in data.data:
                data_item.block.remove_edges()

    if random_edge_freq > 0:
        for data_item in data.data:
            data_item.block.random_forward_edges(random_edge_freq / len(data_item.block.instrs))

def load_data(params):
    # type: (BaseParameters) -> dt.DataCost
    data = dt.load_dataset(params.data)

    def filter_data(filt):
        # type: (Callable[[dt.DataItem], bool]) -> None
        data.data = [d for d in data.data if filt(d)]
        data.train = [d for d in data.train if filt(d)]
        data.test = [d for d in data.test if filt(d)]

    if params.no_mem:
        filter_data(lambda d: not d.block.has_mem())

    ablate_data(data, params.edge_ablation_types, params.random_edge_freq)

    if params.linear_dependencies:
        filter_data(lambda d: d.block.has_linear_dependencies())

    if params.flat_dependencies:
        filter_data(lambda d: d.block.has_no_dependencies())

    return data

def load_model(params, data):
    # type: (BaseParameters, dt.DataCost) -> md.AbstractGraphModule

    if params.use_rnn:
        rnn_params = md.RnnParameters(
            embedding_size=params.embed_size,
            hidden_size=params.hidden_size,
            num_classes=1,
            connect_tokens=params.rnn_connect_tokens,
            skip_connections=params.rnn_skip_connections,
            hierarchy_type=params.rnn_hierarchy_type,
            rnn_type=params.rnn_type,
            learn_init=params.rnn_learn_init,
        )
        model = md.RNN(rnn_params)
    else:
        model = md.GraphNN(embedding_size=params.embed_size, hidden_size=params.hidden_size, num_classes=1,
                           use_residual=not params.no_residual, linear_embed=params.linear_embeddings,
                           use_dag_rnn=not params.no_dag_rnn, reduction=params.dag_reduction,
                           nonlinear_type=params.dag_nonlinearity, nonlinear_width=params.dag_nonlinearity_width,
                           nonlinear_before_max=params.dag_nonlinear_before_max,
        )

    model.set_learnable_embedding(mode=params.embed_mode, dictsize=628 or max(data.hot_idx_to_token) + 1)

    return model

def dump_model_and_data(model, data, fname):
    # type: (md.AbstractGraphMode, dt.DataCost, str) -> None
    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass
    torch.save(PredictorDump(
        model=model,
        dataset_params=data.dump_dataset_params(),
    ), fname)

def load_model_and_data(fname):
    # type: (str) -> (md.AbstractGraphMode, dt.DataCost)
    # PyTorch 2.x compatibility fix for LSTM _flat_weights issue
    # Models saved with PyTorch 1.x use _flat_weights, PyTorch 2.x uses _all_weights
    
    import torch.nn.modules.rnn as rnn_module
    import torch.nn as nn
    
    # Save original __setstate__
    original_lstm_setstate = rnn_module.LSTM.__setstate__
    
    def compatible_lstm_setstate(self, d):
        """Completely rewrite __setstate__ to handle PyTorch 1.x -> 2.x conversion"""
        # Convert _flat_weights to _all_weights before calling Module.__setstate__
        state_dict = d.copy()
        if '_flat_weights' in state_dict:
            state_dict['_all_weights'] = state_dict['_flat_weights']
            del state_dict['_flat_weights']
        
        # Call nn.Module.__setstate__ first to initialize module internals
        # This will set up _state_dict_pre_hooks, _parameters, etc.
        nn.Module.__setstate__(self, state_dict)
        
        # Ensure _all_weights is properly set (PyTorch 2.x requirement)
        if '_all_weights' in state_dict:
            self._all_weights = state_dict['_all_weights']
    
    # Apply patch
    rnn_module.LSTM.__setstate__ = compatible_lstm_setstate
    
    try:
        dump = torch.load(fname, weights_only=False)
    finally:
        # Restore original
        rnn_module.LSTM.__setstate__ = original_lstm_setstate
    
    # Post-load fix: Ensure all LSTM modules have PyTorch 2.x required attributes
    def fix_lstm_module(module):
        """Recursively fix LSTM modules to have PyTorch 2.x required attributes"""
        for child in module.children():
            if isinstance(child, nn.LSTM):
                # PyTorch 2.x requires _flat_weight_refs and _flat_weights_names
                if not hasattr(child, '_flat_weight_refs') or child._flat_weight_refs is None:
                    # Try to call _update_flat_weights if it exists
                    if hasattr(child, '_update_flat_weights'):
                        try:
                            child._update_flat_weights()
                        except (AttributeError, RuntimeError, TypeError):
                            # If _update_flat_weights fails, extract from _all_weights
                            # _all_weights might be a nested list, so we need to flatten it
                            if hasattr(child, '_all_weights') and child._all_weights is not None:
                                import weakref
                                # Flatten _all_weights - handle both flat lists and nested lists
                                flat_weights = []
                                def extract_tensors(item):
                                    if isinstance(item, torch.Tensor):
                                        flat_weights.append(item)
                                    elif isinstance(item, (list, tuple)):
                                        for subitem in item:
                                            extract_tensors(subitem)
                                
                                for item in child._all_weights:
                                    extract_tensors(item)
                                
                                if flat_weights:
                                    try:
                                        child._flat_weight_refs = [weakref.ref(w) for w in flat_weights]
                                        # Generate flat weight names
                                        num_directions = 2 if child.bidirectional else 1
                                        weight_names = []
                                        for layer in range(child.num_layers):
                                            for direction in range(num_directions):
                                                suffix = '_reverse' if direction == 1 else ''
                                                weight_names.extend([
                                                    f'weight_ih_l{layer}{suffix}',
                                                    f'weight_hh_l{layer}{suffix}',
                                                    f'bias_ih_l{layer}{suffix}',
                                                    f'bias_hh_l{layer}{suffix}',
                                                ])
                                        # Match the number of actual weights
                                        child._flat_weights_names = weight_names[:len(flat_weights)]
                                    except TypeError:
                                        # If we can't create weakrefs, PyTorch will handle it during forward
                                        pass
            else:
                # Recursively fix child modules
                fix_lstm_module(child)
    
    # Fix all LSTM modules in the loaded model
    if hasattr(dump, 'model'):
        fix_lstm_module(dump.model)
    
    data = dt.DataInstructionEmbedding()
    data.read_meta_data()
    data.load_dataset_params(dump.dataset_params)
    return (dump.model, data)
