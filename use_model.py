import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def load_and_predict(image_path,model_path):
    # モデルの読み込み
    model = MyModelDefinition(args)
    model.load_state_dict(torch.load(model_path)
    model.eval()

    # 画像の前処理
    image = Image.open(image_path).convert("L")  # グレースケールに変換
    transform = transforms.ToTensor()
    input_image = transform(image).unsqueeze(0)

    # 予測の実行
    with torch.no_grad():
        output = model(input_image)
        probabilities = F.softmax(output, dim=1)

    # クラスごとの確率を返す
    return zip(range(10), probabilities.squeeze().tolist())

if __name__ == '__main__':
    image_path = "test_image.png"
    model_path = "mode.pth"
    class_probabilities = load_and_predict(image_path,model_path)
    
    # 結果の表示
    for class_index, probability in class_probabilities:
        print(f"Class {class_index}: Probability {probability:.4f}")
