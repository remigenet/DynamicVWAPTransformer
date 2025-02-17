"""
Example function to prepare data for the model
This is not necessary, you can change the features used if you want to, but is given as starting example.
"""

import numpy as np
import pandas as pd
 
def prepare_data_with_ahead_inputs(vwaps: np.ndarray, volumes: np.ndarray, features: np.ndarray, lookback: int, n_ahead: int, autoscale_target: bool = True):
    """
    Prepare the data using volumes, features and vwaps as input including the necessary ahead inputs for the model training. 
    """
    n_row = vwaps.shape[0]
    X = np.zeros((n_row - lookback - n_ahead, lookback + n_ahead - 1, features.shape[1]))
    y = {'prices': np.zeros((n_row - lookback - n_ahead, n_ahead, 1)), 'volumes': np.zeros((n_row - lookback - n_ahead, n_ahead, 1))}
    offset=0
    for row in range(lookback, n_row - n_ahead):
        X[row - lookback-offset] = features[row - lookback:row + n_ahead - 1]
        if autoscale_target:
            s=np.sum(volumes[row - lookback:row])
            if s>0 and np.sum(volumes[row:row + n_ahead])>0:
                y['prices'][row - lookback-offset] = np.expand_dims(vwaps[row:row + n_ahead] / vwaps[row - lookback], axis=-1)
                y['volumes'][row - lookback-offset] = np.expand_dims(volumes[row:row + n_ahead] / s, axis=-1)
            else:
                offset+=1
        else:
            y['prices'][row - lookback] = np.expand_dims(vwaps[row:row + n_ahead], axis=-1)
            y['volumes'][row - lookback] = np.expand_dims(volumes[row:row + n_ahead], axis=-1)
    y = np.concatenate([y['volumes'], y['prices']], axis=-1)
    return X[:-offset-1], y[:-offset-1]

def full_generate(volumes: pd.DataFrame, notionals: pd.DataFrame, target_asset, lookback = 120, n_ahead = 12, test_split = 0.2, autoscale_target=True):
    """
    Generate train and tests sets with the basic feature configuration from the paper.
    volumes and notionals are supposed to be pandas DataFrame with date as index at a fixed frequency and assets as columns.
    This only use the target asset column.
    If you want to test for other features or including other asset information to features this is what you want to change.
    """
    assets = [target_asset]
    
    assert target_asset in assets
    volumes = volumes[assets].dropna()
    notionals = notionals[assets].dropna()
    notionals = notionals.loc[volumes.index]
    volumes = volumes.loc[notionals.index]
    vwaps = pd.DataFrame(notionals.values / volumes.values, index=volumes.index, columns = volumes.columns)
    vwaps = vwaps.ffill().dropna()
    notionals = notionals.loc[vwaps.index]
    volumes = volumes.loc[vwaps.index]
    #Create feature
    features = volumes / volumes.shift(lookback + n_ahead).rolling(24 * 7 * 2).mean()
    features['hour'] = volumes.index.hour
    features['dow'] = volumes.index.dayofweek
    for asset in assets:
        features[f'returns {asset}'] = vwaps[asset] / vwaps[asset].shift() - 1.
    
    #Create volume and prices
    volumes = volumes[target_asset]
    vwaps = vwaps[target_asset]
    
    #Remove NaN and align
    features = features.loc[volumes.index].dropna()
    volumes = volumes.loc[features.index]
    vwaps = vwaps.ffill().loc[volumes.index]

    
    X, y = prepare_data_with_ahead_inputs(vwaps.values, volumes.values, features.values, lookback, n_ahead, autoscale_target =  autoscale_target)

    
    test_row = int(len(y) * (1 - test_split))
    
    X_train = X[:test_row]
    X_test = X[test_row:]
    
    y_train, y_test = y[:test_row], y[test_row:]
    
    return X_train, X_test, y_train, y_test