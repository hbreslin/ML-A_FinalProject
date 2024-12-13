import pandas as pd
import os
from PIL import Image


votes_file = "filtered_file.csv"

liked_folder = "filtered_data/liked"
disliked_folder = "filtered_data/disliked"


votes = pd.read_csv('filtered_file.csv', header=0)
print(votes.head())


for _, row in votes.iterrows():
    value = row['Vote']
    img_path = row['Image']
    img_path = img_path[1:]
    
    if not os.path.isabs(img_path):  # If relative path
        img_path = os.path.join(os.getcwd(), img_path)
    
    with Image.open(img_path) as img:
        
        if value == 0:
            output_path = os.path.join('filtered_data/disliked', os.path.basename(img_path))
            img.save(output_path)
        elif value ==1:
            output_path = os.path.join('filtered_data/liked', os.path.basename(img_path))
            img.save(output_path)
        else: print(error)
    