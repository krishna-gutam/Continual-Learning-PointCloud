import requests
import zipfile
url = "http://modelnet.cs.princeton.edu/ModelNet10.zip"
response = requests.get(url)
with open("ModelNet10.zip", "wb") as f:
    f.write(response.content)


import zipfile
with zipfile.ZipFile("ModelNet10.zip", 'r') as zip_ref:
    zip_ref.extractall("ModelNet10")


