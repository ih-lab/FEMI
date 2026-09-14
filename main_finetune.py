# --------------------------------------------------------------------------
# main_finetune.py
#
# Entry point for SUPERVISED FINE-TUNING of the FEMI foundation model on a
# labeled downstream task (binary classification or regression), using
# either single still images or short videos (sequences of frames) per
# embryo. Requires a folder of per-embryo image subfolders plus a CSV of
# labels (see README.md for the exact expected format).
#
# High-level flow of `main()`:
#   1. Parse CLI args; determine task type (classification/regression),
#      input type (image vs. video), whether to use maternal age as an
#      extra input feature, and how much of the pretrained encoder to
#      unfreeze for fine-tuning.
#   2. Define `IVFDataset`, a Keras `Sequence` that lazily loads batches of
#      images/videos (+ optional age) and labels from disk on demand.
#   3. Define `create_model_linear_probe()`, which builds the downstream
#      model: FEMI's ViT-MAE encoder (frozen or partially unfrozen) as a
#      feature extractor, optionally wrapped in `TimeDistributed` + a
#      BiLSTM + attention pooling for video input, followed by a small
#      classification/regression head (with optional age fusion).
#   4. Load and split labels from the CSV, build train/val data generators.
#   5. Build (optionally multi-GPU) and train the model with early
#      stopping, a warmup schedule, and optional ReduceLROnPlateau.
#   6. Save the final fine-tuned weights to `--output_dir`.
# --------------------------------------------------------------------------

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

# set_trainable_recursively: unfreeze specific transformer blocks for
#   fine-tuning.
# get_lr_metric: exposes the optimizer's current LR as a trackable metric.
# CustomReduceLROnPlateau / WarmupScheduler: custom LR schedule callbacks.
# AttentionWithContext / Addition: attention-pooling layers used to combine
#   per-frame video features into a single embryo-level representation.
from utils.helpers import set_trainable_recursively, get_lr_metric, CustomReduceLROnPlateau, AttentionWithContext, Addition, WarmupScheduler

import warnings
warnings.filterwarnings("ignore")


def get_args_parser():
    """Define and return the argparse parser for all fine-tuning CLI flags:
    task type, input type (image/video), age usage, freezing strategy,
    dataset paths, device config, etc. See README.md for example
    commands."""
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
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.2,
                        help='factor by which the learning rate will be reduced')

    # * Finetuning params
    parser.add_argument('--femi_model_path', default='',type=str,
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
    # ------------------------------------------------------------------
    # Device setup: same pattern as main_pretrain.py - restrict visible
    # GPUs and decide whether multi-GPU MirroredStrategy is needed.
    # ------------------------------------------------------------------
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
    # Fix every relevant RNG (Python hash seed, random, numpy, TF) so that
    # dataset shuffling/splitting and model initialization are reproducible
    # across runs.
    seed_value= 42
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Determine which downstream task head to build/train. Exactly one of
    # --do_binary_classification / --do_regression must be set.
    if args.do_binary_classification:
        binary_classification_task = 1
        regression_task = 0
    elif args.do_regression:
        binary_classification_task = 0
        regression_task = 1
    if not args.do_binary_classification and not args.do_regression:
        raise ValueError("Either binary classification or regression task must be selected")

    # Whether to fuse maternal age as an extra scalar feature alongside the
    # image/video embedding.
    if not args.use_age:
        exclude_age = 1
    else:
        exclude_age = 0

    # If 0 transformer blocks are requested to be unfrozen, the entire ViT
    # encoder stays frozen (pure "linear probe" on top of frozen features).
    # Otherwise, the last `num_transformer_blocks_unfreeze` blocks are
    # unfrozen for fine-tuning (see set_trainable_recursively usage below).
    if args.num_transformer_blocks_unfreeze == 0:
        freeze_encoder = 1
    else:
        freeze_encoder = 0

    # Video input mode processes a fixed number of frames per embryo
    # through the encoder (via TimeDistributed) followed by temporal
    # pooling; image mode treats each embryo as a single still frame.
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
    NUM_EPOCHS_WARMUP = args.warmup_epochs
    REDUCE_LR_FACTOR = args.reduce_lr_factor
    if args.use_reduce_lr_on_plateau:
        REDUCE_LR_PATIENCE = 5
        REDUCE_LR_FACTOR = 0.1


    imagenet_mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    imagenet_std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    mean = tf.reshape(imagenet_mean, [1, 1, 1, 3])
    std = tf.reshape(imagenet_std, [1, 1, 1, 3])


    class IVFDataset(Sequence):
        """
        Keras `Sequence` (lazy, index-based data generator) that yields
        batches of (images_or_videos[, age]) -> labels for training/
        validation. Using a `Sequence` (rather than loading everything into
        memory up front) means only one batch's worth of images needs to be
        decoded and held in memory at a time.

        Each "sample" corresponds to one embryo folder. In image mode, a
        single representative frame is used per embryo; in video mode,
        `n_frames` frames are sampled (padding by repeating the last frame
        if the folder has fewer than `n_frames` images - see `min(i,
        len(files) - 1)` below).
        """
        def __init__(self, data_path, batch_size, df_labels, n_frames=18, target_size=(224, 224)):
            self.data_path = data_path
            self.batch_size = batch_size
            self.n_frames = n_frames
            self.target_size = target_size

            # Pre-compute labels
            # This dict can contains entries which folder is non-existent.
            self.label_dict = df_labels.set_index('SUBJECT_NO').to_dict(orient='index')

            # Only keep embryo folders that (a) have a label in the CSV and
            # (b) contain enough frames to satisfy `n_frames` (relevant for
            # video mode; in image mode n_frames=1 so this just checks the
            # folder is non-empty).
            samples = []
            for dir_name in os.listdir(data_path):
                embryo_path = os.path.join(data_path, dir_name)
                # Include existing entries with a sufficient number of frames only
                if dir_name in self.label_dict and len(os.listdir(embryo_path)) >= self.n_frames:
                    samples.append(embryo_path)
            self.samples = samples

        def __len__(self):
            # Number of batches per epoch (Keras Sequence requirement).
            return math.ceil(len(self.samples) / self.batch_size)
        
        def __getitem__(self, idx):
            # Return the `idx`-th batch: reads and decodes each sample's
            # image(s) from disk on demand (i.e. this is where the actual
            # file I/O + decode cost is paid, once per epoch per sample).
            batch_samples = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch_images = []
            final_labels = []
            age_input = []

            for sample in batch_samples:
                images = []
                # Simplify file selection
                files = [f for f in sorted(os.listdir(sample)) if f.endswith(('.jpg', '.png', '.jpeg'))]
                for i in range(self.n_frames):
                    # min(i, len(files)-1) clamps to the last available
                    # frame if fewer than n_frames files exist for this
                    # embryo (in practice, __init__ already filters out
                    # folders with too few frames, so this mostly guards
                    # against exact-boundary cases).
                    img_path = os.path.join(sample, files[min(i, len(files) - 1)])
                    img_bytes = tf.io.read_file(img_path)
                    img = tf.image.decode_image(img_bytes, channels=3)
                    img = tf.image.resize(img, self.target_size)

                    if do_image_classification:
                        # Image mode: n_frames=1, so this loop runs once
                        # and appends a single (224, 224, 3) image directly.
                        batch_images.append(img)
                    else:
                        images.append(img)            

                if not do_image_classification:
                    # Video mode: stack all frames into a single
                    # (n_frames, 224, 224, 3) tensor per sample.
                    vid = tf.stack(images, axis=0)
                    batch_images.append(vid)
                # Use pre-computed labels
                labels = self.label_dict[os.path.basename(sample)]
                final_labels.append(labels.get('LABEL', 0))
                if not exclude_age:
                    # Maternal age is scaled by /50 as a simple
                    # normalization heuristic (roughly maps typical
                    # maternal ages into a [0, 1]-ish range for the network).
                    age_input.append(labels['AGE'] / 50)

            y = np.array(final_labels)

            if not exclude_age:
                return [np.array(batch_images), np.array(age_input)], y

            return np.array(batch_images), y


    def create_model_linear_probe():
        """
        Build the downstream fine-tuning model on top of FEMI's pretrained
        ViT-MAE encoder. Despite the name ("linear probe"), this supports
        both a frozen-encoder linear-probe setup AND a partially-unfrozen
        fine-tuning setup, controlled by `freeze_encoder` /
        `num_transformer_blocks_unfreeze`.

        Overview:
          - `inp`: the model's image/video input. Shape is (224,224,3) for
            single images, or (n_frames, 224, 224, 3) for video (the `None`
            placeholder is overwritten below when do_image_classification).
          - `feature_extractor`: a sub-model that runs FEMI's ViT encoder
            (manually re-implementing its forward pass: patch embed ->
            transformer encoder -> layernorm -> drop CLS token -> global
            average pool) on a single 224x224x3 image, producing one
            feature vector per image.
          - For video input, `feature_extractor` is wrapped in
            `TimeDistributed` so it runs independently on every frame, and
            the resulting sequence of per-frame features is summarized by a
            bidirectional LSTM + learned attention pooling (see
            AttentionWithContext/Addition in utils/helpers.py) into a
            single embryo-level feature vector.
          - Optionally, maternal age is embedded via a small Dense layer
            and concatenated with the visual features before the final
            head.
          - Finally, a small Dense head produces either a sigmoid
            (binary classification) or linear (regression) output,
            depending on `binary_classification_task` / `regression_task`.
        """
        he_initializer = tf.keras.initializers.HeNormal()
        normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        # Default input assumes a variable-length sequence of frames
        # (video mode); overridden below to a single image shape when
        # do_image_classification is true.
        inp = Input(shape=(None, 224, 224, 3))

        maternal_age = Input(shape=(1,))
        age_features = layers.Dense(10, activation='relu', name='age_dense_1', kernel_initializer=he_initializer)(maternal_age)
        if do_image_classification:
            inp = Input(shape=(224, 224, 3))
        
        # Load the pretrained FEMI MAE model purely to reuse its trained
        # ViT weights (`model.vit`) - the MAE decoder/reconstruction head
        # itself is not used for fine-tuning.
        model = TFViTMAEForPreTraining.from_pretrained(args.femi_model_path)
        num_hidden_layers = 24
        model.config.mask_ratio = 0  # disable masking - we want to encode the full image, not reconstruct it
        model.config.hidden_dropout_prob = 0.2
            
        # ---- Build a standalone single-image feature extractor by
        # ---- manually replaying the ViT encoder's forward pass ----
        image_input = Input(shape=(224, 224, 3))
        img = tf.cast(image_input, tf.float32) / 255.0
        normalized_image = (img - mean) / std
        # HuggingFace's ViT implementation expects channels-first
        # (batch, channels, H, W) input.
        input = tf.transpose(normalized_image, [0, 3, 1, 2])

        # Patchify + linearly project + add position embeddings (mask is
        # unused since mask_ratio=0 means nothing is masked out).
        patch_embeddings, mask, ids_restore = model.vit.embeddings(pixel_values=input)

        head_mask = [None] * num_hidden_layers
        output_attentions = None
        output_hidden_states = None
        return_dict = None

        # Run the transformer encoder stack over the patch embeddings.
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
        # Average-pool the remaining patch tokens into a single feature
        # vector per image (rather than using the CLS token directly).
        final_output = layers.GlobalAveragePooling1D()(sequence_output)
        
        feature_extractor = Model(inputs=image_input, outputs=final_output)

        # Freezing strategy: start with everything trainable, then either
        # freeze the whole encoder (linear probe) or selectively unfreeze
        # just the last N transformer blocks (partial fine-tuning) while
        # keeping the patch embedding layer frozen either way.
        for layer in feature_extractor.layers:
            layer.trainable = True
            if freeze_encoder:
                layer.trainable = False

        if not freeze_encoder:
            model.vit.embeddings.patch_embeddings.trainable = False
            for i, layer in enumerate(model.vit.encoder.layer):
                # Only the last `num_transformer_blocks_unfreeze` of the 24
                # encoder blocks remain trainable; earlier blocks (closer
                # to the input) stay frozen to preserve general low-level
                # features learned during pretraining.
                if i < num_hidden_layers - num_transformer_blocks_unfreeze:
                    set_trainable_recursively(layer, False)
                
        if not do_image_classification:
            # Video mode: apply the same feature extractor independently to
            # every frame in the sequence (shared weights across time).
            features = TimeDistributed(feature_extractor)(inp)
        else:
            features = feature_extractor(inp)

        if not do_image_classification:
            # Temporal modeling: a bidirectional LSTM processes the
            # per-frame feature sequence, then a learned attention
            # mechanism (AttentionWithContext) weights each timestep's
            # contribution before Addition() sums them into one vector.
            lstm_features = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                            name='td_lstm_1', kernel_initializer=normal_initializer))(features)
            lstm_features, attention_context_weights = AttentionWithContext()(lstm_features)
            lstm_features = Addition()(lstm_features)
        
            if not exclude_age:
                lstm_features = layers.Dense(10, activation='relu', kernel_initializer=normal_initializer)(lstm_features)
                features = layers.BatchNormalization()(features)
                age_features = layers.BatchNormalization()(age_features)
                feats = layers.concatenate([lstm_features, age_features])
            
            if exclude_age:
                feats = lstm_features
        
        else:
            if not exclude_age:
                features = layers.Dense(10, activation='relu', kernel_initializer=normal_initializer)(features)
                features = layers.BatchNormalization()(features)
                age_features = layers.BatchNormalization()(age_features)
                feats = layers.concatenate([features, age_features])
            if exclude_age:
                feats = features

        # Both heads are always constructed; only one is actually wired
        # into the final model's outputs below, based on the task type.
        binary_model_output = layers.Dense(1, activation="sigmoid", name='binary_output', kernel_initializer=normal_initializer)(feats)
        regression_model_output = layers.Dense(1, activation="linear", name='regression_output', kernel_initializer=normal_initializer)(feats)

        # Use a smaller/no weight decay when only training the head
        # (frozen encoder), and a fuller AdamW config when also fine-tuning
        # unfrozen transformer blocks.
        if freeze_encoder:
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0)
        else:
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.05, beta_1=0.9, beta_2=0.999)
        
        lr_metric = get_lr_metric(opt)

        if not exclude_age:
            final_inputs = [inp, maternal_age]
        else:
            final_inputs = inp

        if binary_classification_task:
            final_outputs = binary_model_output
            classification_loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
            final_loss = {'binary_output':classification_loss}
            final_metrics = {'binary_output':[tf.keras.metrics.AUC(name='auc'), lr_metric]}
            final_loss_weights = {'binary_output':1}
        elif regression_task:
            final_outputs = regression_model_output
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


    # Load and validate the labels CSV (see README.md for the expected
    # SUBJECT_NO / LABEL / AGE column format).
    internal_df = pd.read_csv(args.csv_path)
    # Should have SUBJECT_NO, LABEL, and AGE columns. AGE only needs to be present if users want to include it in the model
    assert 'SUBJECT_NO' in internal_df.columns, "SUBJECT_NO column not found in the dataframe"
    assert 'LABEL' in internal_df.columns, "LABEL column not found in the dataframe"
    if not exclude_age:
        assert 'AGE' in internal_df.columns, "AGE column not found in the dataframe"
    internal_df = internal_df.reset_index(drop=True)
    internal_df['SUBJECT_NO'] = internal_df['SUBJECT_NO'].astype(str)
    internal_df['LABEL'] = internal_df['LABEL'].astype(int)

    # Define warmup scheduler callback
    warmup_callback = WarmupScheduler(warmup_epochs=NUM_EPOCHS_WARMUP, target_lr=LEARNING_RATE)

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=RESTORE_BEST_WEIGHTS)

    print("Using the following parameters: ")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")

    # ReduceLROnPlateau (custom implementation) additionally restores the
    # best-seen weights whenever it reduces the LR, acting like a mini
    # early-restore mechanism at each plateau, not just at the very end.
    if args.use_reduce_lr_on_plateau:
        reduce_lr = CustomReduceLROnPlateau(monitor='val_loss', factor=REDUCE_LR_FACTOR, patience=REDUCE_LR_PATIENCE)
        callbacks = [early_stopping, reduce_lr, warmup_callback]
    else:
        callbacks = [early_stopping, warmup_callback]

    print("Training the model...")

    # Train the model
    # Shuffle all labeled rows, then carve off a random validation fraction
    # (fixed random_state=42 for reproducibility) and use the remainder for
    # training.
    df_data = internal_df.copy()
    df_data = df_data.sample(frac=1).reset_index(drop=True)
    val_df = df_data.sample(frac=val_dataset_fraction, random_state=42)
    train_all_df = df_data.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    train_df = train_all_df.reset_index(drop=True)
    source_train_gen = IVFDataset(args.data_path, BATCH_SIZE, train_df, n_frames=num_frames_in_vid)
    source_val_gen = IVFDataset(args.data_path, BATCH_SIZE, val_df, n_frames=num_frames_in_vid)

    # Build the model inside a MirroredStrategy scope when training across
    # multiple GPUs, same pattern as main_pretrain.py.
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
    # Save the final fine-tuned weights (prefixed with "ft-" to distinguish
    # from pretraining checkpoints produced by main_pretrain.py).
    filepath = args.output_dir + '/' + 'ft-' + args.output_model_name + "-final" + "/ckpt"
    task_model.save_weights(filepath)


if __name__ == '__main__':
    # Script entry point: parse CLI args, ensure the output directory
    # exists, then kick off fine-tuning.
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
