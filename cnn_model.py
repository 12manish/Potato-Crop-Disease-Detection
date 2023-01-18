import tensorflow as tf
import numpy as np
import pathlib
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras import models, layers 
from tensorflow.python.keras.layers import Dense
from keras.models import Sequential
from keras.layers.convolutional import Conv2D 
from keras.layers import Dense, Activation, Dropout, Flatten, MaxPooling2D 

BATCH_SIZE = 32
IMAGE_SIZE = 256
CHANNELS=3
EPOCHS=5

ds = tf.keras.preprocessing.image_dataset_from_directory(
    pathlib.Path('potato-disease'),
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

cn = ds.class_names
train_ds = ds.take(54)
test_ds = ds.skip(54)
val_ds = test_ds.take(6)    

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

scaling = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])

aug = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

train_ds = train_ds.map(
    lambda x, y: (aug(x, training=True), y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3

model = models.Sequential([
    scaling,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

training = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=5,
)

def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = cn[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {
        predicted_class,
        confidence
    }
----------------------------------------------------------------
------------------------------------------------------------------------
--------------------------------------------------------------------------------
def load_image(img_path, show = False) :
    img = image.load_img(img_path, target_size = (256, 256))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis = 0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show :
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

def get_labels(test_path) : 
    # getting class labels
    from glob import glob

    class_names = []
    test_path = test_path + '/*'
    for i in glob(test_path) :  # Reads all the folders in which images are present
        class_names.append(i.split('/')[-1])

    # return dict(zip(class_names, range(len(class_names))))    # return dictionary containing class name and numeric label.
    return sorted(class_names)

if __name__ == "__main__":
    # load model
    model = load_model("/content/drive/MyDrive/Potato Leaf Disease Project/final_model.h5", compile = False)

    # image path
    img1 = '/content/Potato/Test/Potato___Late_blight/00695906-210d-4a9d-822e-986a17384115___RS_LB 4026.JPG'   
    img2 = '/content/Potato/Test/Potato___Early_blight/109730cd-03f3-4139-a464-5f9151483e8c___RS_Early.B 6738.JPG'
    img3 = '/content/Potato/Test/Potato___healthy/Potato_healthy-28-_0_8545.JPG'
    img4 = '/content/Potato/Test/Potato___Late_blight/815516f8-6fb1-4f92-bdff-63349e5ee83f___RS_LB 3237.JPG'
    img5 = '/content/Potato/Test/Potato___healthy/Potato_healthy-35-_0_3642.JPG'
    img6 = '/content/Potato/Test/Potato___Late_blight/9631fd8f-244c-4047-98e4-aecc907624c1___RS_LB 4573.JPG'
    img7 = '/content/Potato/Test/Potato___healthy/Potato_healthy-30-_0_7912.JPG'
    img8 = '/content/Potato/Test/Potato___Early_blight/9125d133-5b86-4363-8fbe-79c813ac8795___RS_Early.B 6748.JPG'
    img9 = '/content/Potato/Test/Potato___Early_blight/9846eead-9fc1-4c35-8b63-1adfbdf0b118___RS_Early.B 8325.JPG'
    
    class_names = get_labels('/content/Potato/Test')
    for i in [img1, img2, img3, img4, img5, img6, img7, img8, img9] : 
        new_image = load_image(i, show = True)
        y_proba = model.predict(new_image)
        confidence = round(100 * (np.max(y_proba[0])), 2)
        print('Predicted Class : ', class_names[np.argmax(y_proba)])
        print('Actual Class : ', i.split('/')[-2])
        print('Confidence : ', confidence, '%')
        print('_____________________________________________________________')
