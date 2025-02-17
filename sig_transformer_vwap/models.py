import keras
from keras import ops
from keras.initializers import Initializer
from keras.constraints import Constraint
from keras.models import Model, Sequential
from keras.layers import (
    Layer,
    Add,
    LayerNormalization,
    Dense,
    Multiply,
    Reshape,
    Activation,
    MultiHeadAttention
)
from tkan import TKAN
from keras_sig import SigLayer

@keras.utils.register_keras_serializable(name="EqualInitializer")
class EqualInitializer(Initializer):
    """Initializes weights to 1/n_ahead."""
    
    def __init__(self, n_ahead):
        self.n_ahead = n_ahead
        
    def __call__(self, shape, dtype=None):
        return ops.ones(shape, dtype=dtype) / self.n_ahead

        
    def get_config(self):
        return {'n_ahead': self.n_ahead}


@keras.utils.register_keras_serializable(name="PositiveSumToOneConstraint")
class PositiveSumToOneConstraint(keras.constraints.Constraint):
    """Constrains the weights to be positive and sum to 1."""
    
    def __call__(self, w):
        # First ensure values are positive
        w = keras.ops.maximum(w, 0)
        # Then normalize to sum to 1
        return w / (keras.ops.sum(w) + keras.backend.epsilon())

    def get_config(self):
        return {}

@keras.utils.register_keras_serializable(name="AddAndNorm")
class AddAndNorm(Layer):
    def __init__(self, **kwargs):
        super(AddAndNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        self.add_layer = Add()
        self.add_layer.build(input_shape)
        self.norm_layer = LayerNormalization()
        self.norm_layer.build(self.add_layer.compute_output_shape(input_shape))
    
    def call(self, inputs):
        tmp = self.add_layer(inputs)
        tmp = self.norm_layer(tmp)
        return tmp

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Assuming all input shapes are the same

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable(name="GRN")
class Gate(Layer):
    def __init__(self, hidden_layer_size = None, **kwargs):
        super(Gate, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        

    def build(self, input_shape):
        if self.hidden_layer_size is None:
            self.hidden_layer_size = input_shape[-1]
        self.dense_layer = Dense(self.hidden_layer_size)
        self.gated_layer = Dense(self.hidden_layer_size, activation='sigmoid')
        self.dense_layer.build(input_shape)
        self.gated_layer.build(input_shape)
        self.multiply = Multiply()

    def call(self, inputs):
        dense_output = self.dense_layer(inputs)
        gated_output = self.gated_layer(inputs)
        return ops.multiply(dense_output, gated_output)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.hidden_layer_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
        })
        return config


@keras.utils.register_keras_serializable(name="GRN")
class GRN(Layer):
    def __init__(self, hidden_layer_size, output_size=None, **kwargs):
        super(GRN, self).__init__(**kwargs)
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

    def build(self, input_shape):
        if self.output_size is None:
            self.output_size = self.hidden_layer_size
        self.skip_layer = Dense(self.output_size)
        self.skip_layer.build(input_shape)
        
        self.hidden_layer_1 = Dense(self.hidden_layer_size, activation='elu')
        self.hidden_layer_1.build(input_shape)
        self.hidden_layer_2 = Dense(self.hidden_layer_size)
        self.hidden_layer_2.build((*input_shape[:2], self.hidden_layer_size))
        self.gate_layer = Gate(self.output_size)
        self.gate_layer.build((*input_shape[:2], self.hidden_layer_size))
        self.add_and_norm_layer = AddAndNorm()
        self.add_and_norm_layer.build([(*input_shape[:2], self.output_size),(*input_shape[:2], self.output_size)])

    def call(self, inputs):
        skip = self.skip_layer(inputs)
        hidden = self.hidden_layer_1(inputs)
        hidden = self.hidden_layer_2(hidden)
        gating_output = self.gate_layer(hidden)
        return self.add_and_norm_layer([skip, gating_output])

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_size,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_layer_size': self.hidden_layer_size,
            'output_size': self.output_size,
        })
        return config


@keras.utils.register_keras_serializable(name="VariableSelectionNetwork")
class VariableSelectionNetwork(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        batch_size, time_steps, embedding_dim, num_inputs = input_shape
        self.softmax = Activation('softmax')
        self.num_inputs = num_inputs
        self.flatten_dim = time_steps * embedding_dim * num_inputs
        self.reshape_layer = Reshape(target_shape=[time_steps, embedding_dim * num_inputs])
        self.reshape_layer.build(input_shape)
        self.mlp_dense = GRN(hidden_layer_size = self.num_hidden, output_size=num_inputs)
        self.mlp_dense.build((batch_size, time_steps, embedding_dim * num_inputs))
        self.grn_layers = [GRN(self.num_hidden) for _ in range(num_inputs)]
        for i in range(num_inputs):
            self.grn_layers[i].build(input_shape[:3])
        super(VariableSelectionNetwork, self).build(input_shape)

    def call(self, inputs):
        _, time_steps, embedding_dim, num_inputs = inputs.shape
        flatten = self.reshape_layer(inputs)
        # Variable selection weights
        mlp_outputs = self.mlp_dense(flatten)
        sparse_weights = keras.activations.softmax(mlp_outputs)
        sparse_weights = ops.expand_dims(sparse_weights, axis=2)
        
        # Non-linear Processing & weight application
        trans_emb_list = []
        for i in range(num_inputs):
            grn_output = self.grn_layers[i](inputs[:, :, :, i])
            trans_emb_list.append(grn_output)
        
        transformed_embedding = ops.stack(trans_emb_list, axis=-1)
        combined = ops.multiply(sparse_weights, transformed_embedding)
        temporal_ctx = ops.sum(combined, axis=-1)
        
        return temporal_ctx

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config


@keras.utils.register_keras_serializable(name="EmbeddingLayer")
class EmbeddingLayer(Layer):
    def __init__(self, num_hidden, **kwargs):
        super(EmbeddingLayer, self).__init__(**kwargs)
        self.num_hidden = num_hidden

    def build(self, input_shape):
        self.dense_layers = [
            Dense(self.num_hidden) for _ in range(input_shape[-1])
        ]
        for i in range(input_shape[-1]):
            self.dense_layers[i].build((*input_shape[:2], 1))
        super(EmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        embeddings = [dense_layer(inputs[:, :, i:i+1]) for i, dense_layer in enumerate(self.dense_layers)]
        return ops.stack(embeddings, axis=-1)

    def compute_output_shape(self, input_shape):
        return list(input_shape[:-1]) + [self.num_hidden, input_shape[-1]]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_hidden': self.num_hidden,
        })
        return config

@keras.utils.register_keras_serializable(name="DynamicVWAPTransformer")
class DynamicVWAPTransformer(Model):
    def __init__(self, lookback, n_ahead, hidden_size=100, hidden_rnn_layer=2, num_heads=3, num_embedding=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookback = lookback
        self.n_ahead = n_ahead
        self.hidden_size = hidden_size
        self.hidden_rnn_layer = hidden_rnn_layer
        self.num_embedding = num_embedding
        self.num_heads = num_heads
        
    def build(self, input_shape):
        feature_shape = input_shape
        assert feature_shape[1] == self.lookback + self.n_ahead - 1

        self.embedding = EmbeddingLayer(self.num_embedding)
        self.embedding.build(input_shape)
        embedding_output_shape = self.embedding.compute_output_shape(input_shape)
        self.vsn = VariableSelectionNetwork(self.hidden_size)
        self.vsn.build(embedding_output_shape)
        vsn_output_shape = (input_shape[0], input_shape[1], self.hidden_size)
        
        # RNN layers
        self.internal_rnn = Sequential([
            TKAN(self.hidden_size, return_sequences=True)
            for _ in range(self.hidden_rnn_layer)
        ])
        self.internal_rnn.build(vsn_output_shape)
        self.gate = Gate()
        self.gate.build(vsn_output_shape)
        self.addnorm = AddAndNorm()
        self.addnorm.build([vsn_output_shape,vsn_output_shape])
        self.grn = GRN(self.hidden_size)
        self.grn.build(vsn_output_shape)
        # Multi-head attention layer
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads,
            value_dim=self.hidden_size // self.num_heads,
            use_bias=True
        )
        self.attention.build(vsn_output_shape, vsn_output_shape, vsn_output_shape)#as the tkan do not changes shapes
        
        # Dense layers for volume prediction
        self.internal_hidden_to_volume = [
            Sequential([
                Dense(self.hidden_size, activation='relu'),
                Dense(self.hidden_size, activation='relu'),
                Dense(1, activation='tanh')
            ])
            for _ in range(self.n_ahead - 1)
        ]
        
        for i in range(self.n_ahead - 1):
            self.internal_hidden_to_volume[i].build((feature_shape[0], self.hidden_size + i))
            
        # Base volume curve
        self.base_volume_curve = self.add_weight(
            shape=(self.n_ahead,),
            name="base_curve",
            initializer=EqualInitializer(self.n_ahead),
            constraint=PositiveSumToOneConstraint(),
            trainable=True
        )
        super(DynamicVWAPTransformer, self).build(input_shape)
        
    def call(self, inputs):
        embedded_features = self.embedding(inputs)
        selected = self.vsn(embedded_features)
        # Get RNN hidden states
        rnn_hidden = self.internal_rnn(selected)
        all_context = self.addnorm([self.gate(rnn_hidden), selected])
        enriched = self.grn(all_context)
        
        # Apply causal self-attention
        attended_hidden = self.attention(
            query=enriched,
            value=enriched,
            key=enriched,
            use_causal_mask=True
        )
        
        total_volume = ops.zeros((ops.shape(inputs)[0], 1))
        
        for t in range(0, self.n_ahead - 1):
            if t > 0:
                current_hidden = ops.concatenate([attended_hidden[:, self.lookback + t, :], volume_curve], axis=1)
            else:
                current_hidden = attended_hidden[:, self.lookback + t, :]
                
            estimated_factor = 1. + self.internal_hidden_to_volume[t](current_hidden)
            estimated_volume = self.base_volume_curve[t] * estimated_factor
            estimated_volume = keras.ops.clip(estimated_volume, 0., 1. - total_volume)
            total_volume += estimated_volume
            
            if t == 0:
                volume_curve = estimated_volume
            else:
                volume_curve = ops.concatenate([volume_curve, estimated_volume], axis=1)
        
        estimated_volume = 1. - total_volume
        volume_curve = ops.concatenate([volume_curve, estimated_volume], axis=1)
        volume_curve = ops.expand_dims(volume_curve, axis=2)
        results = keras.ops.concatenate([volume_curve, keras.ops.zeros_like(volume_curve)], axis=-1)
        
        return results
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lookback': self.lookback,
            'n_ahead': self.n_ahead,
            'hidden_size': self.hidden_size,
            'hidden_rnn_layer': self.hidden_rnn_layer,
            'num_embedding': self.num_embedding,
            'num_heads': self.num_heads
        })
        return config
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2, self.n_ahead)
    

@keras.utils.register_keras_serializable(name="DynamicSigVWAPTransformer")
class DynamicSigVWAPTransformer(Model):
    def __init__(self, lookback, sig_lookback, n_ahead, hidden_size=100, 
                 hidden_rnn_layer=2, signature_depth=3, num_heads=3, *args, **kwargs):
        super(DynamicSigVWAPTransformer, self).__init__(*args, **kwargs)
        self.lookback = lookback
        self.n_ahead = n_ahead
        self.hidden_size = hidden_size
        self.hidden_rnn_layer = hidden_rnn_layer
        self.signature_depth = signature_depth
        self.sig_lookback = sig_lookback
        
    def build(self, input_shape):
        assert input_shape[1] == self.sig_lookback + self.n_ahead - 1
        self.sig_layer_gate = SigLayer(depth=self.signature_depth, stream=False)
        self.sig_layer_gate.build((input_shape[0], self.sig_lookback, 2))
        self.sig_learnable_kernel = self.add_weight(
            shape=(1, self.sig_lookback, 2),
            name="sig_kernel",
            initializer=keras.initializers.Ones(),
            trainable=True
        )
        sig_size = sum([2 ** i for i in range(1, self.signature_depth + 1)])
        sig_output_shape = (input_shape[0], sig_size)
        self.sig_to_weights = Sequential([
            keras.layers.BatchNormalization(),
            keras.layers.RepeatVector(self.lookback + self.n_ahead - 1),
        ])
        self.sig_to_weights.build(sig_output_shape)
        vwap_input_shape = (input_shape[0], self.lookback + self.n_ahead - 1, input_shape[2]+sig_size)
        self.dynamic_vwap = DynamicVWAPTransformer(self.lookback, self.n_ahead, hidden_size=self.hidden_size, hidden_rnn_layer=self.hidden_rnn_layer)
        self.dynamic_vwap.build(vwap_input_shape)

    def call(self, inputs):
        sig_features = self.sig_learnable_kernel * inputs[:,:self.sig_lookback,:2]
        features = inputs[:,-self.lookback-self.n_ahead+1:]
        signature_gate = self.sig_layer_gate(sig_features)
        weights = self.sig_to_weights(signature_gate)
        return self.dynamic_vwap(ops.concatenate([features, weights], axis=2))

    def get_config(self):
        config = super().get_config()
        config.update({
            'lookback': self.lookback,
            'n_ahead': self.n_ahead,
            'hidden_size': self.hidden_size,
            'hidden_rnn_layer': self.hidden_rnn_layer,
            'signature_depth': self.signature_depth,
            'sig_lookback': self.sig_lookback,
        })
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2, self.n_ahead)