# C:\Users\59195\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\mpl-data\matplotlibrc →font.family: MS Gothic,Agency FB
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
        #transforms.Normalize((0.5,), (0.5,))  # 正規化 #←なぜかここをコメントアウトすると精度が良くなる
    ])
    image = transform(image).unsqueeze(0) #テンソルの0階目に要素数1の次元を挿入する
    return image

def predict_image(image, model):
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
    return probabilities.squeeze().tolist()

def display_and_save_image(original_image, input_image, save_path,title):
    # PIL画像をMatplotlibで表示
    plt.figure()
    plt.suptitle(title)
    
    plt.subplot(121)  # 1行2列のプロットの1番目
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(122)  # 1行2列のプロットの2番目
    plt.imshow(input_image.squeeze().numpy(), cmap='gray')
    plt.title('Input Image')

    plt.tight_layout()  # プロット間のスペースを調整
    plt.savefig(save_path)  # 画像をファイルに保存
    plt.show()

def result_show(probabilities,correct):
    # 最も確率の高いクラスを見つける
    max_probability = 0.0
    max_class = None
    for class_index, probability in enumerate(probabilities):
        print(f"Class {class_index}: Probability {probability:.4f}")
        if probability > max_probability:
            max_probability = probability
            max_class = class_index
    print(f"Most probable class: {max_class}")
    if i==max_class:
        print("Correct!!")
    else:
        print(f"Wrong(correct is {correct})")
    
if __name__ == '__main__':
    model_path = "model/model_batch128_epoch7_Adam.pth"
    state_dict = torch.load(model_path)
    model = MNIST()
    model.load_state_dict(state_dict)
    model.eval()
    for i in range(10):
        title="case1:PC上で手書きした数字"
        print(title)
        image = Image.open(f"data_image/digital{i}.png")
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_digital{i}.png"
        result_show(probabilities,i)
        display_and_save_image(image, input_image, path, title)
        print("----------------")
    for i in range(10):
        title="case2:手書きした数字"
        print(title)
        image = Image.open(f"data_image/hand{i}.png")
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_hand{i}.png"
        result_show(probabilities,i)
        display_and_save_image(image, input_image, path, title)
        print("----------------")
    for i in range(10):
        title="case3:Mnistデータセット"
        print(title)
        image = Image.open(f"data_image/test{i}.png")
        input_image = preprocess_image(image)
        probabilities = predict_image(input_image, model)
        path=f"data_fig/fig_test{i}.png"
        result_show(probabilities,i)
        display_and_save_image(image, input_image, path, title)
        print("----------------")