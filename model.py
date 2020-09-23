from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras import Sequential


input_shape = 48, 48, 1


def facial_emotion_model_block(model, filters, kernel_size, input_shape=None):

    if input_shape:
        model.add(Conv2D(filters, kernel_size, padding="same",
                         input_shape=input_shape))
    else:
        model.add(Conv2D(filters, kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(filters * 1.5, kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(filters * 2, kernel_size, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D())


def facial_emotion_model():
    model = Sequential()
    # block 1
    facial_emotion_model_block(model, 32, 3, input_shape=input_shape)
    # block 2
    facial_emotion_model_block(model, 64, 5)
    # block 3
    facial_emotion_model_block(model, 128, 3)
    # block 4
    facial_emotion_model_block(model, 256, 3)

    # final block
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation="softmax"))

    return model




    
