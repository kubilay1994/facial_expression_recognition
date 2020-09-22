from tensorflow.keras.utils import to_categorical

import csv
import numpy as np


input_shape = 48, 48
num_of_classes = 7


X_train = []
y_train = []

X_test = []
y_test = []

X_val = []
y_val = []

with open("data/fer2013.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Usage"] == "Training":
            X_train.extend((int(pixel) for pixel in row["pixels"].split()))
            y_train.append(int(row["emotion"]))
        elif row["Usage"] == "PublicTest":
            X_val.extend((int(pixel) for pixel in row["pixels"].split()))
            y_val.append(int(row["emotion"]))
        else:
            X_test.extend((int(pixel) for pixel in row["pixels"].split()))
            y_test.append(int(row["emotion"]))


X_train = np.asarray(X_train, dtype=np.uint8).reshape(-1, *input_shape)
X_test = np.asarray(X_test, dtype=np.uint8).reshape(-1, *input_shape)
X_val = np.asarray(X_val, dtype=np.uint8).reshape(-1, *input_shape)


y_train = to_categorical(y_train, num_classes=num_of_classes)
y_test = to_categorical(y_test, num_classes=num_of_classes)
y_val = to_categorical(y_val, num_classes=num_of_classes)


np.savez_compressed("data/fer2013_train.npz",
                    X_train=X_train, y_train=y_train)
np.savez_compressed("data/fer2013_test.npz",
                    X_test=X_test, y_test=y_test)
np.savez_compressed("data/fer2013_val.npz",
                    X_val=X_val, y_val=y_val)
