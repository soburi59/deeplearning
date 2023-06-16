import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

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

# 損失と正解率のログを保存するリスト
loss_history = []
accuracy_history = []

# モデルのトレーニング
for epoch in range(5):
    history = model.fit(x_train, y_train, epochs=1, batch_size=32)
    loss_history.append(history.history['loss'][0])
    accuracy_history.append(history.history['accuracy'][0])

    # グラフの描画
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title('loss_graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.title('Accuracy_graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

# モデルの評価
_, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_accuracy)
