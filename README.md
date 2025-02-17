# VWAP Execution with Signature-Enhanced Transformers: A Multi-Asset Learning Approach

This repository contains code for deep-learning–based optimal VWAP execution using signature-enhanced transformers, as described in the paper [VWAP Execution with Signature-Enhanced Transformers: A Multi-Asset Learning Approach](). Package is made in Keras3 and is checked to work with any backend (TensorFlow, PyTorch, JAX).

---

## Installation

### Using Poetry

Clone the repository and install using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

Alternatively, you can install the package via pip:

```bash
pip install .
```

The package requires Python 3.9–3.12 and depends on:
- `keras` (version 3.0.0 or above)
- `tkan` (for TKAN layers)
- `keras_sig` (for signature-based features)

For development and testing, additional dependencies such as TensorFlow, PyTorch, JAX, Pandas, and NumPy are required.

---

## Usage

### Importing the Models and Loss Functions

You can import the available models and custom VWAP loss functions as follows:

```python
from sig_transformer_vwap import (
    DynamicVWAPTransformer,
    DynamicSigVWAPTransformer,
    quadratic_vwap_loss,
    absolute_vwap_loss,
    volume_curve_loss
)
from sig_transformer_vwap.data_formater import full_generate  # if using the provided data formatter
```

### Data Preparation

The models expect input features as a NumPy array with shape `(num_samples, sequence_length, num_features)`, where `sequence_length` is `lookback + n_ahead - 1`. The targets should be a NumPy array of shape `(num_samples, n_ahead, 2)`, with the first element along the last axis representing volume allocations and the second representing prices.

A helper function (`full_generate`) is provided to create training and testing datasets. For example:

```python
import pandas as pd

# Load your data (e.g., from Parquet files)
volumes = pd.read_parquet('path_to_your_volume_data.parquet')
notionals = pd.read_parquet('path_to_your_notionals_data.parquet')

# For the non-signature model, use an appropriate lookback:
X_train, X_test, y_train, y_test = full_generate(
    volumes,
    notionals,
    target_asset='AAPL',
    lookback=120,  # Number of past steps for input features
    n_ahead=12     # Number of future steps to predict
)
```

For the signature-enhanced model, you might use a longer lookback for signature computation:

```python
X_train, X_test, y_train, y_test = full_generate(
    volumes,
    notionals,
    target_asset='AAPL',
    lookback=30,   # sig_lookback parameter for signature features
    n_ahead=12
)
```

### Training and Prediction

#### Example for DynamicVWAPTransformer (Non-Signature)

```python
model = DynamicVWAPTransformer(
    lookback=15,
    n_ahead=6,
    hidden_size=10,
    hidden_rnn_layer=2,
    num_heads=3,
    num_embedding=3
)

model.compile(optimizer='adam', loss=quadratic_vwap_loss)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=3,
    validation_split=0.2,
    shuffle=True,
    verbose=False
)

predictions = model.predict(X_test, verbose=False)
```

#### Example for DynamicSigVWAPTransformer (Signature-Enhanced)

```python
model = DynamicSigVWAPTransformer(
    lookback=15,
    sig_lookback=30,
    n_ahead=6,
    hidden_size=10,
    hidden_rnn_layer=2,
    signature_depth=3,
    num_heads=3
)

model.compile(optimizer='adam', loss=quadratic_vwap_loss)

history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=3,
    validation_split=0.2,
    shuffle=True,
    verbose=False
)

predictions = model.predict(X_test, verbose=False)
```

---

## Model Parameters

- **lookback**: Number of past time steps to consider as input.
- **sig_lookback**: *(Signature-enhanced model only)* Number of time steps used for computing signature features.
- **n_ahead**: Number of future time steps for which the volume curve is predicted.
- **hidden_size**: Dimensionality of the hidden layers in the transformer.
- **hidden_rnn_layer**: Number of TKAN layers in the internal RNN component.
- **num_heads**: Number of attention heads in the multi-head attention layer.
- **num_embedding**: Dimensionality for the embedding layer (applies to the non-signature model).
- **signature_depth**: Depth of the signature feature computation (applies to the signature-enhanced model).

---

## Loss Functions

The repository includes several custom loss functions to optimize the VWAP execution objective:

- **quadratic_vwap_loss**: Measures the mean squared deviation between the achieved VWAP and the market VWAP.
- **absolute_vwap_loss**: Uses the mean absolute deviation.
- **volume_curve_loss**: Computes the deviation between the predicted volume curve and the market volume curve.

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

This work is licensed under a  
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/  
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png  