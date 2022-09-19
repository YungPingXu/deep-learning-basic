import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # no warning and info

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

(train_feature, train_label), (test_feature, test_label) = mnist.load_data()
print(len(train_feature), len(train_label)) # 60000 60000
print(train_feature.shape, train_label.shape) # (60000, 28, 28) (60000,)

def show_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

def show_images_labels_predictions(images, labels, predictions, start_id, num=10):
    plt.gcf().set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[start_id], cmap='binary')
        if len(predictions) > 0:
            title = 'ai = ' + str(predictions[start_id])
            title += ('(o)' if predictions[start_id] == labels[start_id] else '(x)')
            title += '\nlabel = ' + str(labels[start_id])
        else:
            title = 'label = ' + str(labels[start_id])
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
    plt.show()

# print(train_feature[0]) # 28*28 matrix
# show_image(train_feature[0])
# print(train_label[0])
# show_images_labels_predictions(train_feature, train_label, [], 0, 10)

train_feature_vector = train_feature.reshape(len(train_feature), 28 * 28).astype('float32')
test_feature_vector = test_feature.reshape(len(test_feature), 28 * 28).astype('float32')
print(train_feature_vector.shape, test_feature_vector.shape) # (60000, 784) (10000, 784)
# print(train_feature_vector[0])

train_feature_normalize = train_feature_vector / 255
test_feature_normalize = test_feature_vector / 255
# print(train_feature_normalize[0])

#print(train_label[0:5])
train_label_onehot = np_utils.to_categorical(train_label)
test_label_onehot = np_utils.to_categorical(test_label)
#print(train_label_onehot[0:5])

model = Sequential()
model.add(Dense(units=256, input_dim=28*28, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(
    x=train_feature_normalize, y=train_label_onehot,
    validation_split=0.2, epochs=10, batch_size=200, verbose=2
)

scores = model.evaluate(test_feature_normalize, test_label_onehot)
print('\n損失函數誤差值 =', scores[0])
print('準確率 =', scores[1])

prediction = np.argmax(model.predict(test_feature_normalize), axis=1)
# print(test_label)
show_images_labels_predictions(test_feature, test_label, prediction, 0)
model.save("Mnist_mlp_model.h5")
