import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

INIT_LR = 1e-4
NUM_EPOCHS = 50
BATCH_SIZE = 32


def load_and_prepare_BCCD_data():
    dataset, info = tfds.load('bccd', with_info=True)

    def prepare_datapoint(datapoint):
        input_image = datapoint['image']
        input_objects = datapoint['objects']
        return input_image, input_objects

    train_data = dataset['train'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)
    val_data = dataset['validation'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = dataset['test'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)

    print("Number of training images:", info.splits['train'].num_examples)
    print("Number of val images:", info.splits['validation'].num_examples)
    print("Number of testing images:", info.splits['test'].num_examples)

    return train_data, val_data, test_data


def visualize_image(image, bbox, label):
    fig, axs = plt.subplots(1, figsize=(5, 5))

    axs.imshow(image)
    axs.axis('off')

    for j, box in enumerate(bbox):
        ymin, xmin, ymax, xmax = box
        rect = plt.Rectangle(
            (xmin * image.shape[1], ymin * image.shape[0]),
            (xmax - xmin) * image.shape[1],
            (ymax - ymin) * image.shape[0],
            fill=False,
            edgecolor='red',
            linewidth=2
        )
        axs.add_patch(rect)

    plt.show()


def get_data(data):
    images, boxes = [], []
    for i, sample in data:
        image = i
        label = tfds.as_numpy(sample['label'])
        bbox = tfds.as_numpy(sample['bbox'])

        image1 = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)

        image1 = cv2.resize(image1, (224, 224), interpolation=cv2.INTER_LINEAR)
        images.append(image1 / 255.0)

        if label[label == 1].shape[0] > 0:

            t = 0
            for ii, j in zip(label, bbox):
                if ii == 1:
                    t += 1
                    boxes.append(j)
                    if t == 1:
                        break

        else:
            boxes.append(np.array([0, 0, 0, 0]))

        # if count == 172:
        # visualize_image(image, bbox, label)

    return np.array(images), np.array(boxes)


def get_model():
    pre_trained = tf.keras.applications.VGG16(weights="imagenet", include_top=False,
      input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)))

    pre_trained.trainable = False

    flatten = pre_trained.output
    flatten = tf.keras.layers.Flatten()(flatten)

    bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
    bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
    bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
    bboxHead = tf.keras.layers.Dense(4, activation="sigmoid")(bboxHead)

    model = tf.keras.Model(inputs=pre_trained.input, outputs=bboxHead)

    opt = tf.keras.optimizers.Adam(lr=INIT_LR)
    model.compile(loss="mse", optimizer=opt)

    print(model.summary())

    return model


train, val, test = load_and_prepare_BCCD_data()

images, boxes = get_data(train)
val_images, val_boxes = get_data(val)
test_images, test_boxes = get_data(test)

print(images.shape, boxes.shape)
print(val_images.shape, val_boxes.shape)
print(test_images.shape, test_boxes.shape)

images = np.concatenate((images, val_images), axis=0)
boxes = np.concatenate((boxes, val_boxes), axis=0)

print(images.shape, boxes.shape)

model_path = "wbc_model.h5"

# for i , j in zip(images, boxes):
#     print(j)
#     visualize_image(i, [j], 1)


model = get_model()

checkpoint = tf.keras.callbacks.ModelCheckpoint("wbc_model_checkpoint.h5", monitor="val_loss", save_best_only=True,
                                                verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

print("[INFO] training bounding box regressor...")
H = model.fit(images, boxes,
              validation_data=(test_images, test_boxes),
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              verbose=1,
              callbacks=[reduce_lr, early_stopping, checkpoint])

print("[INFO] saving object detector model...")
model.save(model_path, save_format="h5")
