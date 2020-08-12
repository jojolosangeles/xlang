#
# 64x64x3 => image_p262 => 10 probabilities
# with conv=separableconv
# - conv 32 3x3 relu
# - conv 64
# - conv 128
# - maxpool 2x2
# - conv 64
# - conv 128
# - global_average_pool
# - dense 32 relu
#
def model_image_p262():
    model = models.Sequential()
    model.add(layers.SeparableConv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
    model.add(layers.SeparableConv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.SeparableConv2D(64, 3, activation='relu'))
    model.add(layers.SeparableConv2D(8, 3, activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

