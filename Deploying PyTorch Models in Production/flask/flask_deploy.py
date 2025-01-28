from flask import Flask,jsonify,request
import json
import io
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image

# FLASK_ENV=development FLASK_APP=flask_deploy.py flask run
# 生产环境部署flask: https://flask.palletsprojects.com/en/stable/tutorial/deploy/
# UI: https://github.com/avinassh/pytorch-flask-api-heroku
model = models.densenet121(weights="IMAGENET1K_V1")
model.eval()
with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/Deploying PyTorch Models in Production/imagenet_class_index.json","r") as fp:
    imagenet_class_idx = json.load(fp)
print(f"name: {__name__}")
app = Flask(__name__)
print(app)


def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image)


def get_prediction(image_bytes):
    image = transform_image(image_bytes)
    pred = model.forward(image)
    _,y_hat = pred.max(1)
    pred_idx = str(y_hat.item())
    return imagenet_class_idx[pred_idx]


# 讲路径名映射为函数
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        image_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes)
        # 一个 Python 字典转换为 JSON 格式的响应对象，并返回给客户端
        return jsonify({"id":class_id,"name":class_name})
    
if __name__ == "__main__":
    app.run()