import tensorflow as tf
import matplotlib.pyplot as plt

# Load and prepare MNIST for CNN
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_full = X_train.astype("float32") / 255.0
X_train = X_train_full[:55000]
X_valid = X_train_full[55000:60000]

# Add channel dimension
X_train = X_train[..., tf.newaxis]
X_valid = X_valid[..., tf.newaxis]
X_test = X_test.astype("float32")[..., tf.newaxis] / 255.0

# Labels
y_train_cat = tf.keras.utils.to_categorical(y_train[:55000], 10)
y_valid_cat = tf.keras.utils.to_categorical(y_train[55000:60000], 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(inputs)
a1 = x

x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(x)
a2 = x

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
a3 = outputs

model = tf.keras.Model(inputs=inputs, outputs=outputs)
intermediate_model = tf.keras.Model(inputs=inputs, outputs=[a1, a2, a3])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train_cat, batch_size=64, epochs=10, validation_data=(X_valid, y_valid_cat))
model.save("mnist_cnn_model.h5")