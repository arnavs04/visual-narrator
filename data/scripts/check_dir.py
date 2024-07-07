import os

directory = "visual-narrator/data/flickr8k"

if os.path.exists(directory):
    print("Directory Exists")
else:
    print("Directory doesnt exist")