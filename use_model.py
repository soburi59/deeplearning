import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mnist import MNIST
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image):
    # mnistのデータセットは背景が黒・グレースケール・28*28のデータなので、色反転させ、グレースケールに変換し、リサイズして入力する
    image = ImageOps.invert(image.convert('L')).resize((28, 28))
    
    # 画素値の平均を計算
    average = np.mean(np.array(image))
    if average >= 127.5:#背景が白か黒か判定
        image = ImageOps.invert(image)
    transform = transforms.Compose([
        transforms.ToTensor(),  # テンソルに変換
        #transforms.Normalize((0.5,), (0.5,))  # 正規化
    ])
    image = transform(image).unsqueeze(0)
    return image

def predict_image(image, model):
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
    return probabilities.squeeze().tolist()

def display_and_save_image(original_image, input_image, probabilities, save_path):
    # PIL画像をMatplotlibで表示
    plt.subplot(121)  # 1行2列のプロットの1番目
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(122)  # 1行2列のプロットの2番目
    plt.imshow(input_image.squeeze().numpy(), cmap='gray')
    plt.title('Input Image')

    plt.tight_layout()  # プロット間のスペースを調整
    plt.savefig(save_path)  # 画像をファイルに保存
    plt.show()

def result_show(probabilities,path):
    # 最も確率の高いクラスを見つける
    max_probability = 0.0
    max_class = None
    for class_index, probability in enumerate(probabilities):
        print(f"Class {class_index}: Probability {probability:.4f}")
        if probability > max_probability:
            max_probability = probability
            max_class = class_index
    print(f"Most probable class: {max_class}")

    display_and_save_image(image, input_image, probabilities, path)
if __name__ == '__main__':
    model_path = "model/model_batch128_epoch7_Adam.pth"
    state_dict = torch.load(model_path)
    model = MNIST()
    model.load_state_dict(state_dict)
    model.eval()
    for i in range(10):
        image_path = f"data_image/digital{i}.png"
        image = Image.open(image_path)
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_digital{i}.png"
        result_show(probabilities,path)
        print("----------------")
    for i in range(10):
        image_path = f"data_image/hand{i}.png"
        image = Image.open(image_path)
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_hand{i}.png"
        result_show(probabilities,path)
        print("----------------")
    for i in range(10):
        image_path = f"data_image/test{i}.png"
        image = Image.open(image_path)
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_test{i}.png"
        result_show(probabilities,path)
        print("----------------")