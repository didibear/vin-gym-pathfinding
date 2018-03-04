import time
import argparse

import numpy as np
import tensorflow as tf

import os
from model import VIN
from dataset import Dataset

def main():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str, default='./data/dataset.pkl', help='Path to data file')
    parser.add_argument('--imsize', type=int, default=9, help='Size of image')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--k', type=int, default=10, help='Number of Value Iterations')
    parser.add_argument('--ch_i', type=int, default=2, help='Number of channels in input layer')
    parser.add_argument('--ch_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--ch_q', type=int, default=4, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--use_log', type=bool, default=False, help='True to enable TensorBoard summary')
    parser.add_argument('--logdir', type=str, default='./log/', help='Directory to store TensorBoard summary')
    parser.add_argument('--save', type=str, default="./model/weights.ckpt", help='File to save the weights')
    parser.add_argument('--load', type=str, default="./model/weights.ckpt", help='File to load the weights')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.save)):
        print("Error : file cannot be created (need folders) : " + args.save)
        return

    # Define placeholders

    # Input tensor: Stack obstacle image and goal image, i.e. ch_i = 2
    X = tf.placeholder(tf.float32, shape=[None, args.imsize, args.imsize, args.ch_i], name='X')
    S1 = tf.placeholder(tf.int32, shape=[None], name='S1') # vertical positions
    S2 = tf.placeholder(tf.int32, shape=[None], name='S2') # horizontal positions
    y = tf.placeholder(tf.int64, shape=[None], name='y') # labels : actions {0,1,2,3}

    # VIN model
    logits, prob_actions = VIN(X, S1, S2, k=args.k, ch_i=args.ch_i, ch_h=args.ch_h, ch_q=args.ch_q)

    # Training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')
    loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
    train_step = tf.train.RMSPropOptimizer(args.lr, epsilon=1e-6, centered=True).minimize(loss)

    # Testing
    actions = tf.argmax(prob_actions, 1)
    num_err = tf.reduce_sum(tf.to_float(tf.not_equal(actions, y))) # Number of wrongly selected actions

    print("loading trainset...")
    trainset = Dataset(args.datafile, mode='train')
    print("loading testset...")
    testset = Dataset(args.datafile, mode='test')
    print("loaded")

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Intialize all variables / weights
        if loadfile_exists(args.load):
            saver.restore(sess, args.load)
            print("weights loaded")
        else:
            sess.run(tf.global_variables_initializer())

        print("start training...")
        for epoch in range(args.epochs):
            start_time = time.time()
            
            # Train for one step and evaluate error rate and mean loss
            mean_err, mean_loss = train_or_eval(sess, trainset, args,
                feed_ops=[X, S1, S2, y], 
                eval_ops=[train_step, num_err, loss]
            )

            time_duration = time.time() - start_time
            print('Epoch: {:3d} ({:.1f} s):'.format(epoch, time_duration))
            print('\t Train Loss: {:.5f} \t Train Err: {:.5f}'.format(mean_loss, mean_err))

            saver.save(sess, args.save)

        print('\n finished training...\n ')
        
        # Testing
        print('\n testing...\n')
        
        mean_err, mean_loss = train_or_eval(sess, testset, args, 
            feed_ops=[X, S1, S2, y], 
            eval_ops=[num_err, loss]
        )
        print('Test Accuracy: {:.2f}%'.format(100*(1 - mean_err)))


def train_or_eval(sess, dataset, args, feed_ops, eval_ops):
    num_batches = dataset.num_examples//args.batch_size
    total_examples = num_batches*args.batch_size
    
    assert len(eval_ops) == 2 or len(eval_ops) == 3
    if len(eval_ops) == 3: # [train_step, num_err, loss]
        train_mode = True
    else: # test mode: [num_err, loss]
        train_mode = False

    total_err = 0.0
    total_loss = 0.0
    
    for batch in range(num_batches):
        X, S1, S2, y = feed_ops
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(args.batch_size)
        
        feed_dict = {X: X_batch,
                     S1: S1_batch, 
                     S2: S2_batch, 
                     y: y_batch}
        
        if train_mode:
            _, err, loss = sess.run(eval_ops, feed_dict)
        else:
            err, loss = sess.run(eval_ops, feed_dict)
            
        total_err += err
        total_loss += loss
        
    return total_err/total_examples, total_loss/total_examples
        
def loadfile_exists(filepath):
    filename = os.path.basename(filepath)
    for file in os.listdir(os.path.dirname(filepath)):
        if file.startswith(filename):
            return True
    return False

if __name__ == "__main__":
    main()