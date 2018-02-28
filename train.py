import argparse
import joblib
import argparse
from model import vin_model
import numpy as np

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def main():
    parser = argparse.ArgumentParser(description='train vin model')
    parser.add_argument('--data', '-d', type=str, default='./data/training_data.pkl', help='Path of the training data (states, goals, starts, actions)')
    parser.add_argument('--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30, help='Number of sweeps over the dataset to train')
    args = parser.parse_args()

    states, goals, starts, actions = joblib.load(args.data)

    print(type(states))
    print(type(goals))
    print(type(starts))
    print(type(actions))

    model = vin_model(states[0].shape[0], k=20, Q_size=actions[0].shape[0])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    filepath = "weights-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    model.fit([np.array(states), np.array(goals), np.array(starts)], np.array(actions),
        validation_split=0.2,
        batch_size=args.batchsize,
        epochs=args.epoch,
        callbacks=[checkpoint]
    )

    with open('vin_model_structure.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights('vin_model_weights.h5', overwrite=True)

if __name__ == "__main__":
    main()
