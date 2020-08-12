from keras import models
from keras import layers
from keras import regularizers


#
# 150x150x3 => catdog_p150 => probability
# - convbase VGG16 imagenet
# - flatten
# - dense 256 relu
#
def model_catdog_p150():
    model = models.Sequential()
    model.add(VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
