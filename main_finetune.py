import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import random
import math
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from transformers import TFViTMAEForPreTraining

from utils.helpers import set_trainable_recursively, get_lr_metric, CustomReduceLROnPlateau, AttentionWithContext, Addition

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--buffer_size', default=1024, type=int,
                        help='Buffer size')
    parser.add_argument('--epochs', default=500, type=int)

    # Optimizer parameters
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate')
    
    # Model Parameters
    parser.add_argument('--early_stopping_patience', type=int, default=25,
                        help='early stopping patience (default: 25)')
    parser.add_argument('--do_binary_classification', type=int, default=False,
                        help='do binary classification (default: True)')
    parser.add_argument('--do_regression', type=int, default=False,
                        help='do regression (default: False)')
    parser.add_argument('--use_age', type=int, default=False,
                        help='use age as an added input (default: False)')
    parser.add_argument('--num_transformer_blocks_unfreeze', type=int, default=1,
                        help='number of transformer blocks to unfreeze (default: 1)')
    parser.add_argument('--video_input', type=int, default=False,
                        help='use video instead of image input (default: False)')
    parser.add_argument('--num_frames_in_vid', type=int, default=18,
                        help='number of frames in video (default: 1)')
    parser.add_argument('--use_reduce_lr_on_plateau', type=int, default=True,
                        help='use reduce lr on plateau (default: True)')

    # * Finetuning params
    parser.add_argument('--femi_path', default='',type=str,
                        help='FEMI model path')
    parser.add_argument('--validation_split', default=0.2, type=float,
                        help='validation split percentage')

    # Dataset parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path')
    parser.add_argument('--csv_path', default=None, type=str,
                        help='csv path with labels and subject numbers')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_model_name', default='model', type=str,
                        help='output model name')
    parser.add_argument('--device', default='gpu', type=str,
                        help='Either gpu or cpu, default is gpu')
    parser.add_argument('--GPUs', default=None, type=str,
                        help='List of gpus to use, e.g. 0,1,2,3')

    return parser


def main(args):
    if args.device == 'gpu':
        if args.GPUs is None:
            raise ValueError('GPUs must be specified when using GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.GPUs
        NUM_GPUS = len(args.GPUs.split(','))
        NUM_DEVICES = NUM_GPUS
        if NUM_GPUS > 1:
            USE_MULTIPROCESSING = 1
        else:
            USE_MULTIPROCESSING = 0
    else:
        NUM_GPUS = 0
        NUM_DEVICES = 1
        USE_MULTIPROCESSING = 0

    # Set all possible random seeds
    seed_value= 42
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    if args.do_binary_classification:
        binary_classification_task = 1
        regression_task = 0
    elif args.do_regression:
        binary_classification_task = 0
        regression_task = 1
    if not args.do_binary_classification and not args.do_regression:
        raise ValueError("Either binary classification or regression task must be selected")

    if not args.use_age:
        exclude_age = 1
    else:
        exclude_age = 0

    if args.num_transformer_blocks_unfreeze == 0:
        freeze_encoder = 1
    else:
        freeze_encoder = 0

    if args.video_input:
        do_image_classification = 0
        num_frames_in_vid = args.num_frames_in_vid
    else:
        do_image_classification = 1
        num_frames_in_vid = 1

    train_dataset_fraction = 1 - args.validation_split
    val_dataset_fraction = args.validation_split
    num_transformer_blocks_unfreeze = args.num_transformer_blocks_unfreeze
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    BASE_LEARNING_RATE = args.blr
    LEARNING_RATE = BASE_LEARNING_RATE * NUM_DEVICES
    PATIENCE = 10
    RESTORE_BEST_WEIGHTS = True
    if args.use_reduce_lr_on_plateau:
        REDUCE_LR_PATIENCE = 5
        REDUCE_LR_FACTOR = 0.1


    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    mean = tf.reshape(imagenet_mean, [1, 1, 1, 3])
    std = tf.reshape(imagenet_std, [1, 1, 1, 3])


    class IVFDataset(Sequence):
        def __init__(self, folders, batch_size, df_labels, n_frames=18, target_size=(224, 224), 
                    val_subsample_rate=1.0, batch_size_frac=None):
            self.folders = folders
            if batch_size_frac == 0: batch_size_frac = None
            if batch_size_frac is None:
                self.batch_size = batch_size
            self.n_frames = n_frames
            self.target_size = target_size
            self.val_subsample_rate = val_subsample_rate
            
            # Pre-compute labels
            label_dicts = []
            all_samples = []
            for i, folder in enumerate(self.folders):
                label_dict = df_labels[i].set_index('SUBJECT_NO').to_dict(orient='index')
                label_dicts.append(label_dict)
                samples = [os.path.join(folder, dir_name) for dir_name in os.listdir(folder) 
                            if dir_name in label_dict]
                all_samples.extend(samples)
            self.samples = all_samples
            if batch_size_frac is not None:
                self.batch_size = int(len(self.samples) * batch_size_frac)
            combined_dict = {}
            for d in label_dicts:
                combined_dict.update(d)
            self.label_dict = combined_dict

        def __len__(self):
            return math.ceil(len(self.samples) / self.batch_size)
        
        def __getitem__(self, idx):
            batch_samples = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_images = []
            final_labels = []
            age_input = []

            for sample in batch_samples:
                images = []
                # Simplify file selection
                files = [f for f in sorted(os.listdir(sample)) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for i in range(self.n_frames):
                    img_path = os.path.join(sample, files[min(i, len(files) - 1)])
                    img_bytes = tf.io.read_file(img_path)
                    img = tf.image.decode_image(img_bytes, channels=3)
                    img = tf.image.resize(img, self.target_size)

                    if do_image_classification:
                        batch_images.append(img)
                    else:
                        images.append(img)            
                        vid = tf.stack(images, axis=0)     
                        batch_images.append(vid)

                # Use pre-computed labels
                labels = self.label_dict[os.path.basename(sample)]
                final_labels.append(labels.get('LABEL', 0))
                if not exclude_age:
                    age_input.append(labels['AGE_AT_RET'] / 50)

            if not exclude_age:
                x = [np.array(batch_images), np.array(age_input)]
            else:
                x = [np.array(batch_images)]
        
            y = [np.array(final_labels)]

            return x, y


    def create_model_linear_probe():
        he_initializer = tf.keras.initializers.HeNormal()
        normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        inp = Input(shape=(None, 224, 224, 3))
        maternal_age = Input(shape=(1,))
        age_features = layers.Dense(10, activation='relu', name='age_dense_1', kernel_initializer=he_initializer)(maternal_age)
        if do_image_classification:
            inp = Input(shape=(224, 224, 3))
        
        model = TFViTMAEForPreTraining.from_pretrained(args.femi_data_path)
        num_hidden_layers = 24
        model.config.mask_ratio = 0
        model.config.hidden_dropout_prob = 0.2
            
        image_input = Input(shape=(224, 224, 3))
        img = tf.cast(image_input, tf.float32) / 255.0
        normalized_image = (img - mean) / std
        input = tf.transpose(normalized_image, [0, 3, 1, 2])

        patch_embeddings, mask, ids_restore = model.vit.embeddings(pixel_values=input)

        head_mask = [None] * num_hidden_layers
        output_attentions = None
        output_hidden_states = None
        return_dict = None

        encoder_outputs = model.vit.encoder(
                patch_embeddings,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = encoder_outputs[0]
        sequence_output = model.vit.layernorm(inputs=sequence_output)

        sequence_output = sequence_output[:, 1:, :] #Removing CLS token
        final_output = layers.GlobalAveragePooling1D()(sequence_output)
        
        feature_extractor = Model(inputs=image_input, outputs=final_output)

        for layer in feature_extractor.layers:
            layer.trainable = True
            if freeze_encoder:
                layer.trainable = False

        if not freeze_encoder:
            model.vit.embeddings.patch_embeddings.trainable = False
            for i, layer in enumerate(model.vit.encoder.layer):
                if i < num_hidden_layers - num_transformer_blocks_unfreeze:
                    set_trainable_recursively(layer, False)
                
        if not do_image_classification:
            features = TimeDistributed(feature_extractor)(inp)
        else:
            features = feature_extractor(inp)

        if not do_image_classification:
            lstm_features = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                            name='td_lstm_1', kernel_initializer=normal_initializer))(features)
            lstm_features, attention_context_weights = AttentionWithContext()(lstm_features)
            lstm_features = Addition()(lstm_features)
        
            if not exclude_age:
                lstm_features = layers.Dense(10, activation='relu', kernel_initializer=normal_initializer)(lstm_features)
                features = layers.BatchNormalization()(features)
                age_features = layers.BatchNormalization()(age_features)
                feats = keras.layers.concatenate([lstm_features, age_features])
            
            if exclude_age:
                feats = lstm_features
        
        else:
            if not exclude_age:
                features = layers.Dense(10, activation='relu', kernel_initializer=normal_initializer)(features)
                features = layers.BatchNormalization()(features)
                age_features = layers.BatchNormalization()(age_features)
                feats = keras.layers.concatenate([features, age_features])
            if exclude_age:
                feats = features

        binary_model_output = keras.layers.Dense(1, activation="sigmoid", name='binary_output', kernel_initializer=normal_initializer)(feats)
        regression_model_output = keras.layers.Dense(1, activation="linear", name='regression_output', kernel_initializer=normal_initializer)(feats)

        if freeze_encoder:
            opt = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0)
        else:
            opt = keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.05, beta_1=0.9, beta_2=0.999)
        
        lr_metric = get_lr_metric(opt)

        if not exclude_age:
            final_inputs = [inp, maternal_age]
        else:
            final_inputs = [inp]

        if binary_classification_task:
            final_outputs = [binary_model_output]
            classification_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
            final_loss = {'binary_output':classification_loss}
            final_metrics = {'binary_output':[tf.keras.metrics.AUC(name='auc'), lr_metric]}
            final_loss_weights = {'binary_output':1}
        elif regression_task:
            final_outputs = [regression_model_output]
            final_loss = {'regression_output':'logcosh'}
            final_metrics = {'regression_output':['mean_absolute_error', lr_metric]}
            final_loss_weights = {'regression_output':1}

        # Compile the model
        model = Model(inputs=final_inputs, outputs=final_outputs)

        model.compile(optimizer=opt,
                                    loss=final_loss,
                                    metrics=final_metrics,
                                    loss_weights=final_loss_weights
        )
        return model


    internal_df = pd.read_csv(args.csv_path)
    # Should have SUBJECT_NO, LABEL, and AGE columns. AGE only needs to be present if users want to include it in the model
    assert 'SUBJECT_NO' in internal_df.columns, "SUBJECT_NO column not found in the dataframe"
    assert 'LABEL' in internal_df.columns, "LABEL column not found in the dataframe"
    if not exclude_age:
        assert 'AGE' in internal_df.columns, "AGE column not found in the dataframe"
    internal_df = internal_df.reset_index(drop=True)
    internal_df['SUBJECT_NO'] = internal_df['SUBJECT_NO'].astype(str)
    internal_df['LABEL'] = internal_df['LABEL'].astype(int)


    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=RESTORE_BEST_WEIGHTS)

    print("Using the following parameters: ")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")

    if args.use_reduce_lr_on_plateau:
        reduce_lr = CustomReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE)
        callbacks = [early_stopping, reduce_lr]
    else:
        callbacks = [early_stopping]

    print("Training the model...")

    # Train the model
    data_dir = args.data_path
    if not data_dir.endswith('/'):
        data_dir += '/'
    dir = [data_dir]
    df_data = internal_df.copy()
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    val_df = df_data.sample(frac=val_dataset_fraction, random_state=42)
    train_all_df = df_data.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_df = train_all_df.reset_index(drop=True)
    train_dfs = [train_df]
    val_dfs = [val_df]
    source_train_gen = IVFDataset(dir, BATCH_SIZE, train_dfs, n_frames=num_frames_in_vid)
    source_val_gen = IVFDataset(dir, BATCH_SIZE, val_dfs, n_frames=num_frames_in_vid)

    if USE_MULTIPROCESSING:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
           task_model = create_model_linear_probe()
    else:
        task_model = create_model_linear_probe()

    history = task_model.fit(source_train_gen,
                        epochs=EPOCHS,
                        validation_data=source_val_gen,
                        callbacks=callbacks, verbose=1)
    filepath = args.output_dir + '/' + args.output_model_name + "-final" + "/ckpt"
    task_model.save_weights(filepath)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)