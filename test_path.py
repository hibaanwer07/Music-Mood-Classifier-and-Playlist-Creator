import os

data_dir = r"C:\Users\ASUS\Downloads\archive\Data\genres_original\train"
print("Path exists:", os.path.exists(data_dir))
print("Is dir:", os.path.isdir(data_dir))
if os.path.exists(data_dir):
    print("Contents:", os.listdir(data_dir))
else:
    print("Path not found")
