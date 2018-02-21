import tensorflow as tf
import os
from tensorflow.python import pywrap_tensorflow
# checkpoint_path = os.path.join(model_dir, "model.ckpt-9999")
logdir = './collect_dqn/'
ckpt = tf.train.get_checkpoint_state(logdir)
reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key, reader.get_tensor(key).shape)
    print(reader.get_tensor(key))
