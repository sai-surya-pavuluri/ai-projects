# train.py — FER-2013 VGG-style variant (48→40 crops) + 10-crop TTA
# Exports: emotion_model.h5 (40x40x1 input), class_names.json

import os, json, math, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ---------------------------
# Argparse
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Root with train/ and test/ folders")
    p.add_argument("--out_dir", type=str, default="./fer2013_runs", help="Output dir for logs/models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--fine_tune_epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_lr", type=float, default=1e-2)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--mixed_precision", action="store_true", default=False)
    return p.parse_args()

# ---------------------------
# Globals filled from args
# ---------------------------
IMG_H = IMG_W = 48
CROP = 40
NUM_CLASSES = 7
SCALE_RANGE = 0.20
SHIFT_RANGE = 0.20
ROT_DEG = 10.0
AUTOTUNE = tf.data.AUTOTUNE

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Repro + determinism
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    if args.mixed_precision:
        try:
            from tensorflow.keras.mixed_precision import set_global_policy
            set_global_policy("mixed_float16")
            print("✓ Mixed precision enabled")
        except Exception as e:
            print("! Mixed precision not available:", e)

    TRAIN_DIR = os.path.join(args.data_dir, "train")
    TEST_DIR  = os.path.join(args.data_dir, "test")

    # ---------------------------
    # Augment & utils
    # ---------------------------
    def ensure_grayscale(x):
        x = tf.image.convert_image_dtype(x, tf.float32)  # to [0,1]
        c = tf.shape(x)[-1]
        return tf.cond(tf.equal(c, 1), lambda: x, lambda: tf.reduce_mean(x, axis=-1, keepdims=True))

    ROT_FACTOR = ROT_DEG / 180.0
    _rr = layers.RandomRotation(factor=(-ROT_FACTOR, ROT_FACTOR), fill_mode="reflect", dtype="float32")
    _rt = layers.RandomTranslation(height_factor=(-SHIFT_RANGE, SHIFT_RANGE),
                                   width_factor=(-SHIFT_RANGE, SHIFT_RANGE),
                                   fill_mode="reflect", dtype="float32")
    _rz = layers.RandomZoom(height_factor=(-SCALE_RANGE, SCALE_RANGE),
                            width_factor=(-SCALE_RANGE, SCALE_RANGE),
                            fill_mode="reflect", dtype="float32")

    def _maybe_apply(layer, x, p=0.5):
        x_dtype = x.dtype
        def apply():
            y = tf.squeeze(layer(tf.expand_dims(x, 0), training=True), 0)
            return tf.cast(y, x_dtype)
        def keep():
            return tf.identity(x)
        return tf.cond(tf.random.uniform([]) < p, apply, keep)

    def random_affine(x):
        x = _maybe_apply(_rz, x, 0.5)
        x = _maybe_apply(_rt, x, 0.5)
        x = _maybe_apply(_rr, x, 0.5)
        return x

    def ten_crop_48_to_40(img48):
        H = W = 48; ch = cw = CROP
        crops = [
            tf.image.crop_to_bounding_box(img48, 0, 0, ch, cw),
            tf.image.crop_to_bounding_box(img48, 0, W-cw, ch, cw),
            tf.image.crop_to_bounding_box(img48, H-ch, 0, ch, cw),
            tf.image.crop_to_bounding_box(img48, H-ch, W-cw, ch, cw),
            tf.image.crop_to_bounding_box(img48, (H-ch)//2, (W-cw)//2, ch, cw),
        ]
        flipped = tf.image.flip_left_right(img48)
        crops += [
            tf.image.crop_to_bounding_box(flipped, 0, 0, ch, cw),
            tf.image.crop_to_bounding_box(flipped, 0, W-cw, ch, cw),
            tf.image.crop_to_bounding_box(flipped, H-ch, 0, ch, cw),
            tf.image.crop_to_bounding_box(flipped, H-ch, W-cw, ch, cw),
            tf.image.crop_to_bounding_box(flipped, (H-ch)//2, (W-cw)//2, ch, cw),
        ]
        return tf.stack(crops, axis=0)  # [10,40,40,1]

    def random_erasing(x, p=0.5, sl=0.02, sh=0.3, r1=0.3):
        def _erase(z):
            H = tf.shape(z)[0]; W = tf.shape(z)[1]
            area = tf.cast(H*W, tf.float32)
            target = tf.random.uniform([], sl, sh) * area
            aspect = tf.random.uniform([], r1, 1.0/r1)
            h = tf.cast(tf.round(tf.sqrt(target * aspect)), tf.int32)
            w = tf.cast(tf.round(tf.sqrt(target / aspect)), tf.int32)
            h = tf.minimum(h, H); w = tf.minimum(w, W)
            y1 = tf.random.uniform([], 0, H - h + 1, dtype=tf.int32)
            x1 = tf.random.uniform([], 0, W - w + 1, dtype=tf.int32)
            box = tf.ones([h, w, 1], dtype=z.dtype)
            box = tf.pad(box, [[y1, H - y1 - h], [x1, W - x1 - w], [0, 0]], constant_values=0.0)
            return z * (1.0 - box)
        return tf.cond(tf.random.uniform([]) < p, lambda: _erase(x), lambda: x)

    # ---------------------------
    # Data
    # ---------------------------
    def load_train_val_from_dir(directory, val_split=args.val_split, seed=args.seed):
        train_raw = tf.keras.utils.image_dataset_from_directory(
            directory, labels='inferred', label_mode='int', color_mode='grayscale',
            batch_size=None, image_size=(IMG_H, IMG_W), shuffle=True,
            seed=seed, validation_split=val_split, subset='training'
        )
        val_raw = tf.keras.utils.image_dataset_from_directory(
            directory, labels='inferred', label_mode='int', color_mode='grayscale',
            batch_size=None, image_size=(IMG_H, IMG_W), shuffle=False,
            seed=seed, validation_split=val_split, subset='validation'
        )
        return train_raw, val_raw

    def load_test_from_dir(directory):
        return tf.keras.utils.image_dataset_from_directory(
            directory, labels='inferred', label_mode='int', color_mode='grayscale',
            batch_size=None, image_size=(IMG_H, IMG_W), shuffle=False, seed=args.seed
        )

    # create once to capture class_names order
    tmp_for_labels = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels='inferred', label_mode='int', color_mode='grayscale',
        batch_size=32, image_size=(IMG_H, IMG_W), shuffle=True, seed=args.seed,
        validation_split=args.val_split, subset='training'
    )
    class_names = tmp_for_labels.class_names  # alphabetical by folder
    print("Class names (label order):", class_names)
    with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    train_raw, val_raw = load_train_val_from_dir(TRAIN_DIR)
    test_raw = load_test_from_dir(TEST_DIR)

    # (Optional) Cache because 48×48 is tiny
    train_raw = train_raw.cache()
    val_raw   = val_raw.cache()
    test_raw  = test_raw.cache()

    def map_train(example, label):
        x = ensure_grayscale(example)
        x = tf.cast(x, tf.float32)
        x = random_affine(x)
        crops = ten_crop_48_to_40(x)  # [10,40,40,1]
        idx = tf.random.uniform([], 0, 10, dtype=tf.int32)
        x = tf.gather(crops, idx)     # [40,40,1]
        x = random_erasing(x, p=0.5)
        return x, label

    def map_val(example, label):
        x = ensure_grayscale(example)
        x = tf.image.crop_to_bounding_box(
            x, (IMG_H - CROP)//2, (IMG_W - CROP)//2, CROP, CROP
        )
        x = tf.cast(x, tf.float32)
        return x, label

    train_ds = (train_raw.map(map_train, num_parallel_calls=AUTOTUNE)
                .shuffle(8192, seed=args.seed)
                .batch(args.batch)
                .prefetch(AUTOTUNE))
    val_ds = (val_raw.map(map_val, num_parallel_calls=AUTOTUNE)
              .batch(args.batch)
              .prefetch(AUTOTUNE))

    # speed opts
    opts = tf.data.Options(); opts.experimental_deterministic = False
    train_ds = train_ds.with_options(opts)
    val_ds   = val_ds.with_options(opts)
    test_raw = test_raw.with_options(opts)

    # ---------------------------
    # Model
    # ---------------------------
    def build_model(weight_decay=args.weight_decay, dropout1=0.5, dropout2=0.5):
        L2 = regularizers.l2(weight_decay)
        inp = keras.Input(shape=(CROP, CROP, 1), dtype='float32')

        def conv_block(x, f):
            x = layers.Conv2D(f, 3, padding='same', use_bias=False, kernel_regularizer=L2)(x)
            x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
            x = layers.Conv2D(f, 3, padding='same', use_bias=False, kernel_regularizer=L2)(x)
            x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
            return layers.MaxPooling2D(2)(x)

        x = conv_block(inp, 64)
        x = conv_block(x, 128)
        x = conv_block(x, 256)
        x = conv_block(x, 512)

        x = layers.Flatten()(x)
        x = layers.Dense(512, kernel_regularizer=L2)(x); x = layers.ReLU()(x); x = layers.Dropout(dropout1)(x)
        x = layers.Dense(256, kernel_regularizer=L2)(x); x = layers.ReLU()(x); x = layers.Dropout(dropout2)(x)
        logits = layers.Dense(NUM_CLASSES, kernel_regularizer=L2)(x)
        out = layers.Softmax(dtype='float32')(logits)  # stable under AMP

        return keras.Model(inp, out, name="FER2013_VGG_variant")

    def compile_model(model, lr=args.init_lr):
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=args.momentum, nesterov=True)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model = build_model()
    compile_model(model)
    model.summary()

    # ---------------------------
    # Train (stage 1)
    # ---------------------------
    ckpt_path = os.path.join(args.out_dir, "best.weights.h5")
    cbs = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True,
            save_weights_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.75, patience=5, verbose=1, min_lr=1e-6
        ),
        keras.callbacks.CSVLogger(os.path.join(args.out_dir, "train_log.csv")),
    ]
    print("Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)
    model.load_weights(ckpt_path)

    # ---------------------------
    # Ten-crop TTA evaluation
    # ---------------------------
    def make_test_10crop_ds(test_raw_ds, batch_size=128):
        def to_crops(img, label):
            x = ensure_grayscale(img)
            crops = ten_crop_48_to_40(x)   # [10,40,40,1]
            return crops, label
        return (test_raw_ds.map(to_crops, num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(AUTOTUNE))

    def ten_crop_predict_acc(model, test_raw_ds):
        ds = make_test_10crop_ds(test_raw_ds, batch_size=128)
        correct = 0; total = 0
        for crops, labels in ds:
            B = tf.shape(crops)[0]
            crops_flat = tf.reshape(crops, (-1, CROP, CROP, 1))      # [B*10,40,40,1]
            probs_flat = model.predict(crops_flat, verbose=0)        # [B*10,7]
            probs = tf.reshape(probs_flat, (B, 10, NUM_CLASSES))     # [B,10,7]
            mean_probs = tf.reduce_mean(probs, axis=1)               # [B,7]
            preds = tf.argmax(mean_probs, axis=-1, output_type=tf.int32)
            correct += tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
            total   += tf.shape(labels)[0]
        return (tf.cast(correct, tf.float32) / tf.cast(total, tf.float32)).numpy()

    print("Validation-best reloaded. Ten-crop test evaluation...")
    test_raw_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels='inferred', label_mode='int', color_mode='grayscale',
        batch_size=None, image_size=(IMG_H, IMG_W), shuffle=False, seed=args.seed
    ).cache().with_options(opts)

    test_acc = ten_crop_predict_acc(model, test_raw_ds)
    print(f"Test (10-crop) accuracy: {test_acc*100:.2f}%")

    # ---------------------------
    # Fine-tune (stage 2)
    # ---------------------------
    print(f"Fine-tuning (+{args.fine_tune_epochs} epochs @ {args.finetune_lr})...")
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    total_steps = max(1, steps_per_epoch) * args.fine_tune_epochs
    cosine = keras.optimizers.schedules.CosineDecay(initial_learning_rate=args.finetune_lr,
                                                    decay_steps=total_steps)
    opt_ft = keras.optimizers.SGD(learning_rate=cosine, momentum=args.momentum, nesterov=True)
    model.compile(optimizer=opt_ft, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
              callbacks=[keras.callbacks.CSVLogger(os.path.join(args.out_dir, "finetune_log.csv"))])

    final_weights = os.path.join(args.out_dir, "finetuned.weights.h5")
    model.save_weights(final_weights)

    print("Final 10-crop test evaluation (after fine-tune)...")
    test_acc2 = ten_crop_predict_acc(model, test_raw_ds)
    print(f"Final Test (10-crop) accuracy: {test_acc2*100:.2f}%")

    # ---------------------------
    # Export full inference model (40x40x1 input)
    # ---------------------------
    export_path = os.path.join(args.out_dir, "emotion_model.h5")
    model.save(export_path)  # full model (architecture + weights)
    print("Saved full model to:", export_path)

    # Save class_names.json (already saved), print reminder
    print("Saved class names to:", os.path.join(args.out_dir, "class_names.json"))
    print("DONE.")

if __name__ == "__main__":
    main()
