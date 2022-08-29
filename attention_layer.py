import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import (
    l2, 
    l1, 
    l1_l2
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import (
    activations, 
    initializers, 
    regularizers, 
    constraints
)
class Attention(Layer):
    def __init__(
        self,
        hidden,
        init="glorot_uniform",
        activation="linear",
        W_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        **kwargs
    ):
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.hidden = hidden
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.add_weight(
            name="{}_W1".format(self.name),
            shape=(input_dim, self.hidden),
            initializer="glorot_uniform",
            trainable=True,
        )  # Keras 2 API
        self.W = self.add_weight(
            name="{}_W".format(self.name),
            shape=(self.hidden, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b0 = K.zeros((self.hidden,), name="{}_b0".format(self.name))
        self.b = K.zeros((1,), name="{}_b".format(self.name))
        # AttributeError: Can't set the attribute "trainable_weights",
        # likely because it conflicts with an existing read-only @property of the object.
        # Please choose a different name.
        # https://issueexplorer.com/issue/wenguanwang/ASNet/8
        self._trainable_weights = [self.W0, self.W, self.b, self.b0]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W0] = self.W_constraint
            self.constraints[self.W] = self.W_constraint

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        attmap = self.activation(K.dot(x, self.W0) + self.b0)
        attmap = K.dot(attmap, self.W) + self.b
        attmap = K.reshape(
            attmap, (-1, self.input_length)
        )  # Softmax needs one dimension
        attmap = K.softmax(attmap)
        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
        out = K.concatenate(
            [dense_representation, attmap]
        )  # Output the attention maps but do not pass it to the next layer by DIY flatten layer
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1] + input_shape[1])

    def get_config(self):
        config = {
            "init": "glorot_uniform",
            "activation": self.activation.__name__,
            "W_constraint": self.W_constraint.get_config()
            if self.W_constraint
            else None,
            "W_regularizer": self.W_regularizer.get_config()
            if self.W_regularizer
            else None,
            "b_regularizer": self.b_regularizer.get_config()
            if self.b_regularizer
            else None,
            "hidden": self.hidden if self.hidden else None,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer):  # Based on the source code of Keras flatten
    def __init__(self, keep_dim, **kwargs):
        self.keep_dim = keep_dim # 64
        super(attention_flatten, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception(
                'The shape of the input to "Flatten" '
                "is not fully defined "
                "(got " + str(input_shape[2:]) + ". "
                'Make sure to pass a complete "input_shape" '
                'or "batch_input_shape" argument to the first '
                "layer in your model."
            )
        return (input_shape[0], self.keep_dim)  # Remove the attention map

    def call(self, x, mask=None):
        x = x[:, : self.keep_dim]
        #return K.batch_flatten(x)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        return config