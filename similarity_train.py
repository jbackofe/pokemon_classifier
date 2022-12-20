import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

import gc
from typing import Tuple

# INFO messages are not printed.
# This must be run before loading other modules.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
import tensorflow_similarity as tfsim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tfsim.utils.tf_cap_memory()  # Avoid GPU memory blow up

# Clear out any old model state.
gc.collect()
tf.keras.backend.clear_session()

print("TensorFlow:", tf.__version__)
print("TensorFlow Similarity", tfsim.__version__)

def import_train_data(path, dim):
    x = []
    y = []
    class_names=[]
    i = 0
    for folder in os.listdir(path):
        class_names.append(folder)
        for filepath in os.listdir(path+folder):
            # import images
            stream = open(u'{0}{1}/{2}'.format(path, folder, filepath), "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            bgrImage = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            # resize
            bgrImage = cv2.resize(bgrImage, dim, interpolation = cv2.INTER_AREA)

            # convert bgr to rgb
            rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)

            # convert to float32
            rgbImage = np.asarray(rgbImage).astype('float32')

            # Check height and width
            if rgbImage.shape[0] == dim[1] and rgbImage.shape[1] == dim[0]:
                x.append(rgbImage / 255)
            else:
                print('Dim Not Matching!')

            y.append(i)
        i += 1

    x = np.asarray(x)
    y = np.asarray(y)

    return x, y, class_names

# Import training data
path = '../pokemon_data/xy12_images/'
dim = (120, 185)
x, y, class_names = import_train_data(path, dim)

# Save the class_names to csv
np.save('class_names.npy', class_names)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=42)
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape, 'y_test shape:', y_test.shape)

# Define data preperation parameters
CLASSES = np.unique(y)
NUM_CLASSES = len(CLASSES)
CLASSES_PER_BATCH = NUM_CLASSES 
EXAMPLES_PER_CLASS = 4 # min number of examples per class
STEPS_PER_EPOCH = 100  # @param {type:"integer"}

# Save the classes to csv
np.save('classes.npy', CLASSES)

# Define data augmentation layers
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, fill_mode='constant'),
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.05, 0.2), fill_mode='constant'),
    ]
)

def augmenter(
    x: tfsim.types.FloatTensor, y: tfsim.types.IntTensor, examples_per_class: int, is_warmup: bool, stddev=0.08
) -> Tuple[tfsim.types.FloatTensor, tfsim.types.IntTensor]:
    """Image augmentation function.

    Args:
        X: FloatTensor representing the example features.
        y: IntTensor representing the class id. In this case
           the example index will be used as the class id.
        examples_per_class: The number of examples per class.
           Not used here.
        is_warmup: If True, the training is still in a warm
           up state. Not used here.
        stddev: Sets the amount of gaussian noise added to
           the image.
    """
    _ = examples_per_class
    _ = is_warmup

    aug = tf.squeeze(data_augmentation(x))
    aug = aug + tf.random.normal(tf.shape(aug), stddev=stddev)
    x = tf.concat((x, aug), axis=0)
    y = tf.concat((y, y), axis=0)
    idxs = tf.range(start=0, limit=tf.shape(x)[0])
    idxs = tf.random.shuffle(idxs)
    x = tf.gather(x, idxs)
    y = tf.gather(y, idxs)
    return x, y

# Create the sampler
sampler = tfsim.samplers.MultiShotMemorySampler(
    x_train,
    y_train,
    augmenter=augmenter,
    classes_per_batch=CLASSES_PER_BATCH,
    examples_per_class_per_batch=EXAMPLES_PER_CLASS,
    class_list=list(CLASSES[:NUM_CLASSES]),
    steps_per_epoch=STEPS_PER_EPOCH,
)

print(f"The sampler contains {len(sampler)} steps per epoch.")
print(f"The sampler is using {sampler.num_examples} examples out of the original {len(x_train)}.")
print(f"Each examples has the following shape: {sampler.example_shape}.")

# building model
model = tfsim.architectures.ResNet50Sim(
    (dim[1], dim[0], 3),
    embedding_size=128, # embedding_size
    trainable='full',
    pooling="gem",    # Can change to use `gem` -> GeneralizedMeanPooling2D
    gem_p=3.0,        # Increase the contrast between activations in the feature map.
)

# Define the loss function
distance = "cosine"  # @param ["cosine", "L2", "L1"]{allow-input: false}
loss = tfsim.losses.MultiSimilarityLoss(distance=distance)

# Compile the model
LR = 0.0001  # @param {type:"number"}
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss=loss)

# Train the model
EPOCHS = 4  # @param {type:"integer"}
history = model.fit(sampler, epochs=EPOCHS, validation_data=(x_test, y_test))

# Index the model
x_index, y_index = tfsim.samplers.select_examples(x_train, y_train, CLASSES, 20)
model.reset_index()
model.index(x_index, y_index, data=x_index)

# Calibrate the model (find optimal threshold)
num_calibration_samples = x_train.shape[0]  # @param {type:"integer"}
calibration = model.calibrate(
    x_train[:num_calibration_samples],
    y_train[:num_calibration_samples],
    extra_metrics=["precision", "recall", "binary_accuracy"],
    verbose=1,
)

# save the model and the index
save_path = "../models/pokemon_similarity"  # @param {type:"string"}
model.save(save_path, save_index=True)