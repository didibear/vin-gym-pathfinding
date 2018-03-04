import numpy as np
import tensorflow as tf

def attention(tensor, params):
    """ Attention model for grid world domain """
    
    S1, S2 = params
    # Flatten
    s1 = tf.reshape(S1, [-1])
    s2 = tf.reshape(S2, [-1])
    
    # Indices for slicing
    N = tf.shape(tensor)[0]
    idx = tf.stack([tf.range(N), s1, s2], axis=1)
    # Slicing values
    q_out = tf.gather_nd(tensor, idx, name='q_out')    
    
    return q_out

def VIN(X, S1, S2, *, k, ch_i, ch_h, ch_q):
    """
    X : (?, m, n, 2) - stack of gridworld and goal
        gridworld (m, n) = grid with 1 and 0 ;
        goal (m, n) = grid with 10 at goal position
    S1 : list of vertical position of the player
    S2 : list of horizontal position of the player

    k: Number of Value Iteration computations
    ch_i: Channels in input layer
    ch_h: Channels in initial hidden layer
    ch_q: Channels in q layer (~actions)
    """
    h = conv2d(inputs=X, filters=ch_h, name='h0', use_bias=True)
    r = conv2d(inputs=h, filters=1, name='r')
    
    # Add collection of reward image
    tf.add_to_collection('r', r)
    
    # Initialize value map (zero everywhere)
    v = tf.zeros_like(r)

    rv = tf.concat([r, v], axis=3)
    q = conv2d(inputs=rv, filters=ch_q, name='q', reuse=None)  # Initial set before sharing weights
    v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

    # K iterations of VI module
    for i in range(0, k - 1):
        rv = tf.concat([r, v], axis=3)
        q = conv2d(inputs=rv, filters=ch_q, name='q', reuse=True) # Sharing weights
        v = tf.reduce_max(q, axis=3, keep_dims=True, name='v')

        
    # Add collection of value images
    tf.add_to_collection('v', v)
        
    # Do one last convolution
    rv = tf.concat([r, v], axis=3)
    q = conv2d(inputs=rv, filters=ch_q, name='q', reuse=True) # Sharing weights

    # Attention model
    q_out = attention(tensor=q, params=[S1, S2])

    # Final Fully Connected layer
    logits = tf.layers.dense(
        inputs=q_out, 
        units=8, 
        activation=None, 
        use_bias=False, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),  
        name='logits'
    )
    prob_actions = tf.nn.softmax(logits, name='probability_actions')
    
    return logits, prob_actions

def conv2d(*, inputs, filters, name, use_bias=False, reuse=False):
    return tf.layers.conv2d(
        inputs=inputs, 
        filters=filters, 
        kernel_size=[3, 3], 
        strides=[1, 1], 
        padding='same', 
        activation=None, 
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), 
        use_bias=use_bias,
        bias_initializer=tf.zeros_initializer() if use_bias else None,
        name=name,
        reuse=reuse
    )
