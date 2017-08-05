from keras.models import load_model
from keras.utils import plot_model
import keras.backend as K

K.clear_session()

model = load_model('face_model_5120_b64_e50_s80-reorder-v0.h5')

# plot_model(model, to_file='model.png')

plot_model(model, to_file='faces.png', show_shapes=False, show_layer_names=True)