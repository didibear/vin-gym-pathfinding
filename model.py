from keras.models import Model
from keras.layers import merge
import keras.backend as K

from keras.models import Model
from keras.layers import merge, Input, Conv2D, Reshape, Dense, Lambda, Concatenate

CHANNEL_AXIS = 3

def vin_model(n, k=20, conv_filters=150, Q_size=4):
    """
    Input1 = (n, n), grid_state
    Input2 = (n, n), grid_goal
    Input3 = (1, 1), start
    
    Output = (Q_size), action probabilities
    """

    image = Input(shape=(n, n), name="image")
    reward = Input(shape=(n, n), name="reward")
    position = Input(shape=(2,), dtype='int32', name="position")
    
    reshaper = Reshape((n,n,1))
    H = Concatenate(axis=CHANNEL_AXIS)([reshaper(image), reshaper(reward)])
    H = Conv2D(conv_filters, (3, 3), padding="same", activation="relu")(H)
    
    R = Conv2D(1, (1, 1), use_bias=False, name="predicted_reward")(H)

    Q = conv2D_Q(Q_size, "Q0")(R)
    V = maximum(n, "value0", axis=CHANNEL_AXIS)(Q)

    for i in range(1, k):
        Q = Concatenate(axis=CHANNEL_AXIS)([R, V])
        Q = conv2D_Q(Q_size, "Q{}".format(i))(Q)
        V = maximum(n, "value{}".format(i), axis=CHANNEL_AXIS)(Q)

    Q = Concatenate(axis=CHANNEL_AXIS)([R, V])
    Q = conv2D_Q(Q_size, "Q{}".format(k))(Q)

    Q_out = extract_Q_channels(Q, position, n, Q_size)
    out = Dense(Q_size, activation='softmax', use_bias=False)(Q_out)

    return Model(inputs=[image, reward, position], outputs=out)

def maximum(n, name, *, axis):
    return Lambda(lambda x: K.max(x, axis=axis, keepdims=True), output_shape=(n, n, 1), name=name)

def conv2D_Q(Q_size, name):
    return Conv2D(Q_size, (3, 3), use_bias=False, padding="same", name=name)

def extract_Q_channels(Q, position, n, Q_size):
    """ Return values at position for each channels

    Example
    -------
    >>> Q channels = [
        [[1,2,3],  [[9,8,7],
         [4,5,6],   [6,5,4], 
         [7,8,9]],  [3,2,1]],
    ]
    >>> position = (0, 1)
    >>> result = [2, 8]
    """
    def extract_position(inputs):
        Q, pos = inputs
        w = K.one_hot(pos[:, 0] + n * pos[:, 1], n * n) # (None, n * n)
        return K.transpose(K.sum(w * K.permute_dimensions(Q, (1, 0, 2)), axis=2))

    Q = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)), output_shape=(Q_size, n, n))(Q)
    Q = Reshape((Q_size, n * n))(Q)
    return Lambda(extract_position, output_shape=(Q_size,))([Q, position])

def main():
    from keras.utils import plot_model
    model = vin_model(9, k=1)
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

if __name__ == "__main__":
    main()
