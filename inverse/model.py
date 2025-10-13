import tensorflow as tf

def DNN_builder(in_shape=3, out_shape=1, n_hidden_layers=5, neuron_per_layer=32, actfn="tanh"):
    input_layer = tf.keras.layers.Input(shape=(in_shape,))
    x = input_layer
    for _ in range(n_hidden_layers):
        x = tf.keras.layers.Dense(neuron_per_layer, activation=actfn)(x)
    output_layer = tf.keras.layers.Dense(out_shape)(x)
    model = tf.keras.Model(input_layer, output_layer)
    return model