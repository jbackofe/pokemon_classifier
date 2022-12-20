import cv2
import os

import numpy as np
from tabulate import tabulate

import gc

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

def import_inf_data(path, dim):
    x = []
    for filepath in os.listdir(path):
        # import images
        stream = open(u'{0}/{1}'.format(path, filepath), "rb")
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

    x = np.asarray(x)

    return x

save_path = "../models/pokemon_similarity"  # @param {type:"string"}

# reload the model
reloaded_model = tf.keras.models.load_model(
    save_path,
    custom_objects={"SimilarityModel": tfsim.models.SimilarityModel},
)
# reload the index
reloaded_model.load_index(save_path)

# Import the data
path = '../test_images/xy12_test/Clefairy'
dim = (120, 185)
x_inf = import_inf_data(path, dim)

# Import class_names
with open('class_names.npy', 'rb') as f:
    class_names = np.load(f)

# Import classes
with open('classes.npy', 'rb') as f:
    CLASSES = np.load(f)

# Save labels, cutpoint from training
labels = np.append(CLASSES, "Unknown")
print(len(labels))
output = reloaded_model.match(x_inf, cutpoint="optimal", no_match_label=len(labels)-1)
labels[output]
for val in output:
    if val > len(class_names)-1:
        print('Unknown')
    else:
        print(class_names[val])
