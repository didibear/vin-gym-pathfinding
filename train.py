import argparse
import joblib
import argparse
from model import vin_model
import numpy as np
import os.path

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def main():
    parser = argparse.ArgumentParser(description='Train value iteration network model')
    parser.add_argument('--data', '-d', type=str, default='./data/training_data.pkl', help='Path of the training data (states, goals, starts, actions)')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save', '-s', type=str, default="./model/weights-checkpoint.h5", help='File to save the weights')
    parser.add_argument('--load', '-l', type=str, default="./model/weights-checkpoint.h5", help='File to load the weights')
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.save)):
        print("Error : file cannot be created (need folders) : " + args.save)
        return

    print("loading training data from " + args.data + " ...")
    states, goals, starts, actions = joblib.load(args.data)

    print("create model...")
    model = vin_model(states[0].shape[0], k=20, Q_size=actions[0].shape[0])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.isfile(args.load):
        print("loading weights from " + args.load + " ...")
        model.load_weights(args.load)
    checkpoint = ModelCheckpoint(args.save, monitor='val_acc', period=5, save_best_only=True)
    
    print("start training...")
    model.fit([np.array(states), np.array(goals), np.array(starts)], np.array(actions),
        validation_split=0.2,
        batch_size=args.batchsize,
        epochs=args.epoch,
        callbacks=[checkpoint]
    )

    print("saving weights into " + args.save + " ...")
    model.save_weights(args.save, overwrite=True)

    print("done")

if __name__ == "__main__":
    main()
