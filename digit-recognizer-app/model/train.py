import argparse
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

# Create the parser to accept cnn architecture type
parser = argparse.ArgumentParser(description="Train a CNN model on MNIST dataset")
parser.add_argument("--arch", type=str, default='simple', help="CNN Architecture")
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--optimizer", type=str, default='adam', help="Optimizer for training")
args = parser.parse_args()

# Validate architecture arguments
if args.arch not in ['lenet', 'vgg', 'resnet', 'simple']:
    raise ValueError("Invalid architecture type. Choose from 'lenet', 'vgg', 'resnet', or 'simple'.")
if args.optimizer not in ['sgd', 'adam', 'rmsprop']:
    raise ValueError("Invalid optimizer type. Choose from 'sgd', 'adam', or 'rmsprop'.")
if args.batch_size <= 0:
    raise ValueError("Batch size must be a positive integer.")
if args.epochs <= 0:
    raise ValueError("Number of epochs must be a positive integer.")

# Define the CNN model
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.ZeroPadding2D(padding=(2, 2))(inputs)

if args.arch == "lenet":

    x = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120, activation='relu')(x)
    x = tf.keras.layers.Dense(84, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

elif args.arch == "vgg":
    
    x = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(x)
    x = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(x)
    x = tf.keras.layers.Conv2D(32,(3,3),activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)

elif args.arch == "resnet":

    shortcut = tf.keras.layers.Conv2D(16, (1, 1), padding='same')(x)  # Adjust channels to match main path
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

else:
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()

if args.optimizer == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
elif args.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
elif args.optimizer == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train_cat, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, y_valid_cat))
model.save(f"mnist_{args.arch}_model.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Accuracy plot
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.savefig(f"mnist_{args.arch}_acc_plot.png")

# Loss plot
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig(f"mnist_{args.arch}_loss_plot.png")
