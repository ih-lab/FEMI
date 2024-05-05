from tensorflow import keras
import tensorflow as tf
import os
import argparse
from pathlib import Path


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import random
import os
from keras.callbacks import EarlyStopping
from transformers import TFViTMAEForPreTraining
import random

from utils.helpers import WarmUpCosine

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--buffer_size', default=1024, type=int,
                        help='Buffer size')
    parser.add_argument('--epochs', default=800, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--data_augmentation', type=float, default=False, metavar='DATA_AUG',
                        help='Use data augmentation, if true')
    
    # Model Parameters
    parser.add_argument('--mask_ratio', type=float, default=0.75,
                        help='mask ratio (default: 0.75)')
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='early stopping patience (default: 25)')
    parser.add_argument('--save_every_epoch', type=int, default=True,
                        help='save model every epoch (default: True)')

    # * Finetuning params
    parser.add_argument('--femi_model_path', default='',type=str,
                        help='FEMI model path')
    parser.add_argument('--validation_split', default=0.2, type=float,
                        help='validation split percentage')

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_model_name', default='model', type=str,
                        help='output model name')
    parser.add_argument('--device', default='gpu', type=str,
                        help='Either gpu or cpu, default is gpu')
    parser.add_argument('--GPUs', default=None, type=str,
                        help='List of gpus to use, e.g. 0,1,2,3')

    return parser


def main():
    if args.devide == 'gpu':
        if args.GPUs is None:
            raise ValueError('GPUs must be specified when using GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
        NUM_GPUS = len(args.GPUs.split(','))
        NUM_DEVICES = NUM_GPUS
        if len(NUM_GPUS) > 1:
            USE_MULTIPROCESSING = 1
        else:
            USE_MULTIPROCESSING = 0
    else:
        NUM_GPUS = 0
        NUM_DEVICES = 1
        USE_MULTIPROCESSING = 0
    

    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * NUM_DEVICES
    EPOCHS = args.epochs
    AUTO = tf.data.AUTOTUNE
    BASE_LEARNING_RATE = args.blr
    LEARNING_RATE = BASE_LEARNING_RATE * NUM_DEVICES
    WEIGHT_DECAY = args.weight_decay
    NUM_EPOCHS_WARMUP = args.warmup_epochs
    beta_1=0.9
    beta_2=0.95

    if args.data_augmentation:
        USE_DATA_AUG = 1
    else:
        USE_DATA_AUG = 0

    INPUT_SHAPE = (224, 224, 3)

    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    mean = tf.reshape(imagenet_mean, [1, 1, 3])
    std = tf.reshape(imagenet_std, [1, 1, 3])


    def process_path(file_path):
        image = tf.io.read_file(file_path)
        img = tf.io.decode_image(image, channels=3)
        img.set_shape([None, None, 3])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [224, 224])
        normalized_image = (img - mean) / std
        input = tf.transpose(normalized_image, [2, 0, 1]) 
        input.set_shape([3, 224, 224])
        return {'pixel_values': input}

    def process_path_aug(file_path):
        image = tf.io.read_file(file_path)
        img = tf.io.decode_image(image, channels=3)
        img.set_shape([None, None, 3])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [224, 224])

        # Data Augmentation Steps
        # Resize the image to a slightly larger size
        resized_img = tf.image.resize(img, [INPUT_SHAPE[0] + 20, INPUT_SHAPE[1] + 20])
        # Randomly crop the image back to the original input size
        cropped_img = tf.image.random_crop(resized_img, size=INPUT_SHAPE)
        # Randomly flip the image horizontally
        final_img = tf.image.random_flip_left_right(cropped_img)

        normalized_image = (final_img - mean) / std
        input = tf.transpose(normalized_image, [2, 0, 1]) 
        input.set_shape([3, 224, 224])
        return {'pixel_values': input}

    data_directory = args.data_path
    file_paths = os.listdir(data_directory)
    file_paths = [data_directory + file_path for file_path in file_paths]
    file_paths = random.sample(file_paths, len(file_paths))
    train_percentage = 1 - args.validation_split
    print("Number of files: ", len(file_paths))

    # Create a dataset from file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices(file_paths).cache()

    # Split the dataset into train and val
    num_val_files = int(len(file_paths) * (1 - train_percentage))
    val_dataset = dataset.take(num_val_files)
    train_dataset = dataset.skip(num_val_files)
    train_dataset_size = len(file_paths) - num_val_files
    print("Number of training files: ", train_dataset_size)
    print("Number of validation files: ", len(file_paths) - train_dataset_size)

    print("Loading training dataset...")
    if USE_DATA_AUG:
        train_ds = train_dataset.shuffle(BUFFER_SIZE).map(process_path_aug, num_parallel_calls=AUTO).batch(GLOBAL_BATCH_SIZE).prefetch(AUTO)
        val_ds = val_dataset.map(process_path_aug, num_parallel_calls=AUTO).batch(GLOBAL_BATCH_SIZE).prefetch(AUTO)
    else:
        train_ds = train_dataset.shuffle(BUFFER_SIZE).map(process_path, num_parallel_calls=AUTO).batch(GLOBAL_BATCH_SIZE).prefetch(AUTO)
        val_ds = val_dataset.map(process_path, num_parallel_calls=AUTO).batch(GLOBAL_BATCH_SIZE).prefetch(AUTO)
    print("Mapping done...")
        
    print("Size of training dataset: ", train_dataset_size)
    total_steps = int((train_dataset_size / GLOBAL_BATCH_SIZE) * EPOCHS)
    print(f"Total steps: {total_steps}")
    warmup_steps = int((train_dataset_size / GLOBAL_BATCH_SIZE) * NUM_EPOCHS_WARMUP)
    scheduled_lrs = WarmUpCosine(
        learning_rate_base=LEARNING_RATE,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    ### COMPILATION AND TRAINING
    print("Compiling the model...")

    path_to_FEMI = args.femi_model_path

    if USE_MULTIPROCESSING:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = TFViTMAEForPreTraining.from_pretrained(path_to_FEMI)
            model.config.mask_ratio = args.mask_ratio
            optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY, beta_1=beta_1, beta_2=beta_2)
            model.compile(optimizer=optimizer, loss='auto', metrics=['mae'])
    else:
        model = TFViTMAEForPreTraining.from_pretrained(path_to_FEMI)
        model.config.mask_ratio = args.mask_ratio
        optimizer = tf.keras.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY, beta_1=beta_1, beta_2=beta_2)
        model.compile(optimizer=optimizer, loss='auto', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, restore_best_weights=True)

    if args.save_every_epoch:
        filepath = args.output_dir + '/' + args.output_model_name + "-{epoch:02d}" + "/ckpt"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=False)
        train_callbacks = [
            early_stopping,
            model_checkpoint_callback
        ]
    else:
        train_callbacks = [
            early_stopping
        ]

    print("Training the model...")
    history = model.fit(
        train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1, callbacks=train_callbacks
    )

    # Save model
    filepath = args.output_dir + '/' + args.output_model_name + "-final" + "/ckpt"
    model.save_weights(filepath)
    print("Training done...")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)