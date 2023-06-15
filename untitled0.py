"""
Created on Wed Jun 14 18:01:18 2023
coding: utf-8

"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# データセットの読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの前処理
x_train = x_train / 255.0
x_test = x_test / 255.0

# モデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルのトレーニング
model.fit(x_train, y_train, epochs=5, batch_size=32)

# モデルの評価
_, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_accuracy)
