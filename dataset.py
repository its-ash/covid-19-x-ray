from zipfile import ZipFile 
import subprocess
from pathlib import Path
import pandas as pd
import os
import shutil


repo_path = Path(os.getcwd()) / 'covid-chestxray-dataset'
normal_data = Path(os.getcwd()) / 'chest_xray'



metadata = pd.read_csv(repo_path / 'metadata.csv')
x_ray_data = metadata[metadata.modality == 'X-ray']
x_ray = x_ray_data[x_ray_data.view=='PA']
final_data = x_ray[['sex','age','finding','filename','clinical_notes']]
illness = final_data.finding.unique()



def copy(old_file_name,old_file_dir_path, new_file_name, new_file_dir_path):
    old_file_location = old_file_dir_path / old_file_name
    if old_file_location.is_file():
        if new_file_dir_path.is_dir():
            with open(old_file_location, 'rb') as c:
                data = c.read()
                with open(new_file_dir_path/new_file_name, 'wb') as p:
                    p.write(data)
        else:
            raise FileNotFoundError(new_file_dir_path)
    else:
        raise FileNotFoundError(old_file_location)

images_dir = repo_path / 'images'
print("Image Dir Found", images_dir.is_dir())


path = Path(os.getcwd())
dataset_path = path / 'dataset'
shutil.rmtree(dataset_path) if dataset_path.is_dir() else None
os.mkdir(dataset_path) if not dataset_path.is_dir() else None   

print("Creating Dataset")
for disease in illness:
    for x,data in enumerate(final_data[final_data.finding == disease].values):
        sex, age,_, image_name, _ = data 
        fileType = image_name.split('.')[-1]
        temp_path = dataset_path / disease.replace(', ','_')
        os.mkdir(temp_path) if not temp_path.is_dir() else None
        copy(image_name, images_dir,f"{x}__{sex}__{age}.{fileType}",temp_path)

covid_path = dataset_path / 'class_covid'
os.mkdir(covid_path) if not covid_path.is_dir() else None   
non_covid_path = dataset_path / 'class_non-covid'
os.mkdir(non_covid_path) if not non_covid_path.is_dir() else None   
pneumocystis_path = dataset_path / 'class_pneumocystis'
os.mkdir(pneumocystis_path) if not pneumocystis_path.is_dir() else None   




file_name = "ChestXRay2017.zip"
with ZipFile(file_name, 'r') as zip: 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!')

normal_path = normal_data / 'train' / 'NORMAL'
for item in os.listdir(normal_path):
    copy(item, normal_path, item, non_covid_path)
    

pn_path = normal_data / 'train' / 'PNEUMONIA'
for item in os.listdir(pn_path):
    copy(item, pn_path, item, pneumocystis_path)



for folder in os.listdir(dataset_path):
    folder_path = dataset_path / folder
    if not 'class' in folder.lwer():
        if 'covid' in folder.lower():
            for item in os.listdir(folder_path):
                copy(item, folder_path, item, covid_path)
        elif 'pneum' in folder.lower():
            for item in os.listdir(folder_path):
                copy(item, folder_path, item, pneumocystis_path)
        else:
            for item in os.listdir(folder_path):
                copy(item, folder_path, item, non_covid_path)

    
for folder in os.listdir(dataset_path):
  if 'class' not in folder:
    folder_path = dataset_path / folder
    shutil.rmtree(folder_path)



max_item = 0
for item in os.listdir(dataset_path):
    size = len(os.listdir(dataset_path / item))
    max_item = size if size > max_item else max_item
    print(f"{item:<20}{size}")