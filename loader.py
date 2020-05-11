import requests

data_tranform = "https://raw.githubusercontent.com/ashvinijangid/covid-19-x-ray/master/data_tranform.py"
dataset = "https://raw.githubusercontent.com/ashvinijangid/covid-19-x-ray/master/dataset.py"
model = "https://raw.githubusercontent.com/ashvinijangid/covid-19-x-ray/master/model.py"
train = "https://raw.githubusercontent.com/ashvinijangid/covid-19-x-ray/master/train.py"


libs = [data_tranform, dataset, model, train]


for lib in libs:
    data = requests.get(lib, allow_redirects=True).content
    open(lib.split("/")[-1], 'wb').write(data)


