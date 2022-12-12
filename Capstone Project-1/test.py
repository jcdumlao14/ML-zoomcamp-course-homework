import requests

url = 'http://localhost:8888/2022-'

data = {'url':'https://'}

result = requests.post(url,json=data).json()


result = requests.post(url, json=data).json()
print(result)
