import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mnist import MNIST
from PIL import Image, ImageOps

def load_and_predict(image_path,model_path):
    # モデルの読み込み
    state_dict = torch.load(model_path)
    model = MNIST()  # モデルのインスタンスを作成する
    model.load_state_dict(state_dict)
    model.eval()

    # 画像の前処理
    image = Image.open(image_path)
    image = ImageOps.invert(image.convert('L')).resize((28,28))
    transform = transforms.Compose([
        transforms.ToTensor(),  # テンソルに変換
        transforms.Normalize((0.0,), (1.0,))  # 正規化
    ])
    input_image = transform(image).unsqueeze(0)

    # 予測の実行
    with torch.no_grad():
        output = model(input_image)
        probabilities = F.softmax(output, dim=1)

    # クラスごとの確率を返す
    return zip(range(10), probabilities.squeeze().tolist())

if __name__ == '__main__':
    image_path = "data_image/1.png"
    model_path = "model_batch256_SGD.pth"
    class_probabilities = load_and_predict(image_path,model_path)
    
    # 結果の表示
    for class_index, probability in class_probabilities:
        print(f"Class {class_index}: Probability {probability:.4f}")
