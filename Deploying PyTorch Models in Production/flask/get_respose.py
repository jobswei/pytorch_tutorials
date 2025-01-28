import requests

# resp = requests.post("http://127.0.0.1:5000/predict",files={"file":open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/teaser.png","rb")})
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/pytorch_tutorials/teaser.png','rb')})
print(resp)