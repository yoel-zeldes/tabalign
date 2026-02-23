import torch
from tabpfn import TabPFNClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backup_caches(classifier):
    backup = []
    for model in classifier.executor_.models:
        model_cache = []
        for layer in model.transformer_encoder.layers:
            attn = layer.self_attn_between_items
            if attn._kv_cache is None:
                raise ValueError('Not implemented for attn._k_cache and attn._v_cache')
            model_cache.append(attn._kv_cache.clone())
        backup.append(model_cache)
    return backup


def _restore_caches(classifier, backup):
    assert len(classifier.executor_.models) == len(backup)
    for model, model_backup in zip(classifier.executor_.models, backup):
        assert len(model.transformer_encoder.layers) == len(model_backup)
        for layer, layer_backup in zip(model.transformer_encoder.layers, model_backup):
            layer.self_attn_between_items._kv_cache = layer_backup.clone()


def _modify_cache_tensor(cache_tensor, indices_to_remove):
    if not indices_to_remove:
        return cache_tensor

    seq_dim = 1
    dim_size = cache_tensor.shape[seq_dim]
    for idx in indices_to_remove:
        if idx < 0 or idx >= dim_size:
            raise ValueError(f"Index {idx} is out of bounds for cache tensor of size {dim_size}")
    indices_to_keep = [i for i in range(dim_size) if i not in set(indices_to_remove)]
    
    return torch.index_select(cache_tensor, seq_dim, torch.tensor(indices_to_keep, device=cache_tensor.device))
    

def _modify_kv_caches(classifier, pruning_config):
    for model_config in classifier.configs_:
        if model_config.num_thinking_rows > 0:
            raise ValueError("Implementation hasn't been tested yet")
    
    for i, model in enumerate(classifier.executor_.models):
        for layer_idx, indices in pruning_config.items():
            attn = model.transformer_encoder.layers[layer_idx].self_attn_between_items
            if attn._k_cache is not None and attn._v_cache is not None:
                raise ValueError('Not implemented for attn._k_cache and attn._v_cache')
            elif attn._kv_cache is not None:
                attn._kv_cache = _modify_cache_tensor(attn._kv_cache, indices)
            else:
                raise ValueError("No KV cache found in layer")


def create_pruning_config(classifier, num_examples_to_prune, same_across_layers, seed=0):
    layers = classifier.executor_.models[0].transformer_encoder.layers
    kv_cache_size = layers[0].self_attn_between_items._kv_cache.shape[1]
    
    return {
        layer_idx: list(np.random.RandomState(
            seed * 10000 + (0 if same_across_layers else layer_idx * 1000) + num_examples_to_prune
        ).choice(
            kv_cache_size,
            num_examples_to_prune,
            replace=False,
        ))
        for layer_idx in range(len(layers))
    }


def load_data(dataset_name):
    is_synthetic = dataset_name.endswith("[synthetic]")
    dataset_name = dataset_name.replace("[synthetic]", "")
    if dataset_name == "breast_cancer":
        data = datasets.load_breast_cancer()
        n_test = 100
    elif dataset_name == "wine":
        data = datasets.load_wine()
        n_test = 70
    elif dataset_name == "iris":
        data = datasets.load_iris()
        n_test = 50
    elif dataset_name == "digits":
        data = datasets.load_digits()
        n_test = 200
    elif dataset_name == "segment":
        # 2310 examples, 19 features, 7 classes (image segmentation).
        # TabPFN v2 scores ~75% accuracy with small subsets → good difficulty.
        from sklearn.preprocessing import LabelEncoder
        data = datasets.fetch_openml(name="segment", version=1, as_frame=False)
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 300
    elif dataset_name == "mfeat-factors":
        # 2000 examples, 216 features, 10 classes. TabPFN v2 scores ~82% accuracy.
        from sklearn.preprocessing import LabelEncoder
        data = datasets.fetch_openml(name="mfeat-factors", version=1, as_frame=False)
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 200
    elif dataset_name == "vehicle":
        # 846 examples, 18 features, 4 classes.
        from sklearn.preprocessing import LabelEncoder
        data = datasets.fetch_openml(name="vehicle", version=1, as_frame=False)
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 200
    elif dataset_name == "ilpd":
        # 583 examples, 10 features, 2 classes. Accuracy ~73.7%
        from sklearn.preprocessing import LabelEncoder
        data = datasets.fetch_openml(data_id=1480, as_frame=False)
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 150
    elif dataset_name == "credit-g":
        # 1000 examples, 20 features, 2 classes. Accuracy ~73.3%
        from sklearn.preprocessing import LabelEncoder
        data = datasets.fetch_openml(data_id=31, as_frame=False)
        # credit-g has categorical features which fetch_openml(as_frame=False) returns as object/strings. 
        # TabPFN handles them if numeric, but here we get mixed types. 
        # For simplicity in this script, we'll OrdinalEncode the features if needed, 
        # but actually TabPFN V2 handles strings. Let's just ensure target is encoded.
        # However, random_subset_uncertainty uses standard numpy arrays. 
        # We should encode categorical features to integers for simplicity.
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        data.data = enc.fit_transform(data.data)
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 200
    elif dataset_name == "sa-heart":
        # 462 examples, 9 features, 2 classes. Accuracy ~75.5%
        from sklearn.preprocessing import LabelEncoder
        # sa-heart has 'famhist' column as string (Present/Absent).
        data = datasets.fetch_openml(data_id=1498, as_frame=False)
        # Encode string features
        try:
             # Fast check if any column is object/string
             if data.data.dtype == object:
                 from sklearn.preprocessing import OrdinalEncoder
                 data.data = OrdinalEncoder().fit_transform(data.data)
        except:
             pass 
        data.target = LabelEncoder().fit_transform(data.target.astype(str))
        n_test = 100
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=n_test, random_state=42)
    
    if is_synthetic:
        synthetic_data_path = f"results/synthetic_data/{dataset_name}.csv"
        X_test = pd.read_csv(synthetic_data_path).values
        y_test = None

    if get_device().type != "cpu" and len(X_train) > 1000:
        raise ValueError("Only CPU is supported for now, because we have to limit the number of samples to 1000. "
                         "We don't want to accidentally mix results from experiments ran on CPU and GPU, since the "
                         "number of samples would be higher on GPU.")
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    return X_train, X_test, y_train, y_test


def fit_model(X_train, y_train, n_estimators=32, fingerprint=True):
    # tabpfn-v2-classifier.ckpt is a model with num_thinking_rows configured to 0, which is what's tested in this repo
    classifier = TabPFNClassifier(
        device=get_device(),
        n_estimators=n_estimators,
        fit_mode="fit_with_cache",
        model_path='tabpfn-v2-classifier.ckpt',
        inference_config={'FINGERPRINT_FEATURE': fingerprint},
    )
    classifier.fit(X_train, y_train)
    return classifier


def calculate_accuracy(classifier, X, y, pruning_config=None, cache_backup=None):
    if pruning_config is not None:
        _modify_kv_caches(classifier, pruning_config)
    y_pred = classifier.predict(X)
    if cache_backup is not None:
        _restore_caches(classifier, cache_backup)
    return accuracy_score(y, y_pred)
