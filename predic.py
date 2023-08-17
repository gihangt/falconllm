import requests

data = {"prompt": "Hi there, how's it going?"}
res = requests.post("http://127.0.0.1:8081/v1/models/model:predict", json=data)
print(res.json())