# --------------------------------------------------------------------------
# utils/helpers.py
#
# Shared TensorFlow/Keras utility classes used by both main_pretrain.py and
# main_finetune.py: custom learning-rate schedules/callbacks, a small
# metric helper, and custom Keras layers used for the video/temporal
# fine-tuning path (attention-based pooling over per-frame features).
# --------------------------------------------------------------------------

import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras import initializers, regularizers
from keras import constraints
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule combining linear warmup with cosine decay.

    - For the first `warmup_steps` steps, the LR ramps up linearly from
      `warmup_learning_rate` to `learning_rate_base`.
    - After warmup, the LR follows a cosine curve down from
      `learning_rate_base` towards 0 as `step` approaches `total_steps`.
    - Beyond `total_steps`, the LR is clamped to 0.

    Used as the `learning_rate` argument to an optimizer (e.g. AdamW) in
    main_pretrain.py, where TensorFlow calls this schedule's `__call__`
    once per training step, passing in the current global step.
    """
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.pi = tf.constant(np.pi)
        self.warmup_steps = warmup_steps
        

    def __call__(self, step):
        # Called by the optimizer/TF graph once per training step with the
        # current global step count; must return the LR to use for that step.
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        # Standard cosine annealing formula: oscillates from
        # learning_rate_base (at step == warmup_steps) down to 0 (at
        # step == total_steps).
        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            # Use the linear warmup rate while step < warmup_steps, and the
            # cosine-annealed rate afterwards.
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        # Force LR to exactly 0 once training has run past total_steps
        # (shouldn't normally happen, but guards against edge cases).
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
    
class WarmupScheduler(tf.keras.callbacks.Callback):
    """
    Simple EPOCH-based (rather than step-based) linear warmup callback,
    used in main_finetune.py. During the first `warmup_epochs` epochs, the
    optimizer's learning rate is linearly ramped from ~0 up to `target_lr`;
    afterwards this callback no longer touches the LR (any subsequent decay
    is handled separately, e.g. by CustomReduceLROnPlateau below).
    """
    def __init__(self, warmup_epochs, target_lr):
        super(WarmupScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr

    def on_epoch_begin(self, epoch, logs=None):
        # Keras calls this automatically before each epoch starts.
        if epoch < self.warmup_epochs:
            new_lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f"\nWarmup: Learning rate set to {new_lr:.6f}")

class CustomReduceLROnPlateau(tf.keras.callbacks.Callback):
    """
    Custom reimplementation of Keras' built-in ReduceLROnPlateau, with one
    key difference: whenever the LR is reduced due to a plateau, this
    callback ALSO restores the model's weights to the best ones seen so
    far (like a mini "restore_best_weights" applied at every LR drop, not
    just once at the very end of training). This helps avoid continuing to
    train from a degraded state before the LR reduction kicks in.
    """
    def __init__(self, monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min', min_lr=1e-6):
        super(CustomReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_lr = min_lr
        self.best_weights = None  # To store the weights of the best performing model
        self.wait = 0  # How many epochs to wait after the last time the monitored metric improved
        self.best = np.Inf if self.mode == 'min' else -np.Inf
        self.last_lr_reduction_epoch = 0  # Track the last epoch at which the LR was reduced

    def on_train_begin(self, logs=None):
        # Reset the wait counter if the training is starting
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        # Keras calls this automatically after each epoch, with `logs`
        # containing the epoch's metrics (including the monitored one).
        current = logs.get(self.monitor)
        if self.mode == 'min':
            comparison = np.less
        else:
            comparison = np.greater

        if comparison(current, self.best):
            self.best = current
            self.wait = 0
            # Save the best weights if current results is better (less if 'min', greater if 'max')
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                self.last_lr_reduction_epoch = epoch
                # Reduce learning rate
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                new_lr = max(old_lr * self.factor, self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                # Load the best model weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)


def get_lr_metric(optimizer):
    """
    Returns a Keras-metric-compatible function that reports the
    optimizer's CURRENT learning rate at each step/epoch. This is a
    common trick to make the live LR value show up in Keras' training logs
    and `history` object alongside loss/accuracy, purely for monitoring
    (it doesn't affect training in any way).
    """
    def lr(y_true, y_pred):
        curLR = optimizer.lr
        return curLR
    return lr


def dot_product(x, kernel):
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights
	Returns:
	"""
	if K.backend() == 'tensorflow':
		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
		return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    follows these equations:

    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.

    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(name='b',
                                     shape=(input_shape[-1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(name='u',
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # x: (batch, timesteps, features) - one feature vector per frame.
        # Step 1: project each timestep through a learned Dense-like
        # transform (W, b) and squash with tanh -> u_t = tanh(W h_t + b).
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        # Step 2: score each timestep's relevance against a learned context
        # vector `u` via dot product, then softmax-normalize (exp + divide
        # by sum below) across the time axis to get attention weights.
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        # Step 3: scale each timestep's original feature vector by its
        # attention weight (v_t = alpha_t * h_t). The actual summation
        # across time (producing the final pooled vector) happens in the
        # separate Addition layer below, so callers can access the
        # per-timestep weighted features/weights if needed.
        weighted_input = x * a

        return weighted_input, a

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
	"""
	This layer is supposed to add of all activation weight.
	We split this from AttentionWithContext to help us getting the activation weights

	follows this equation:

	(1) v = \sum_t(\alpha_t * h_t)
	
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		2D tensor with shape: `(samples, features)`.
	"""

	def __init__(self, **kwargs):
		super(Addition, self).__init__(**kwargs)

	def build(self, input_shape):
		self.output_dim = input_shape[-1]
		super(Addition, self).build(input_shape)

	def call(self, x):
		return K.sum(x, axis=1)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)


def set_trainable_recursively(layer, trainable, depth=0, max_depth=None):
    """
    Recursively set the trainable attribute of a layer and its sub-layers.

    Parameters:
    layer (tf.keras.layers.Layer): The TensorFlow layer to process.
    trainable (bool): The desired trainable state.
    depth (int): The current depth of recursion.
    max_depth (int or None): The maximum depth to recurse, or None for no limit.
    """
    # Set the trainable attribute of the current layer
    layer.trainable = trainable

    # Check if maximum depth is reached, if specified
    if max_depth is not None and depth >= max_depth:
        return
    # Recursively set for sub-layers (if any)
    if hasattr(layer, 'submodules'):
        for sublayer in layer.submodules:
            set_trainable_recursively(sublayer, trainable, depth + 1, max_depth)