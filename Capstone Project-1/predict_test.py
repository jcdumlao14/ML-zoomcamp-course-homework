import requests

url = 'http://0.0.0.0:8888/predict'

filepath ='./content/grapevine-leaves/test/'

response = requests.post(url,json=filepath).json()
print(response)  