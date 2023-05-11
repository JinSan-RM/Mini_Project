import os
import shutil

a_folder_path = "C:/Users/leeyo/Project/test/labels"
b_folder_path = "C:/Users/leeyo/Project/test/images"
output_folder_path = "C:/Users/leeyo/Project/test/images_val"


if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

json_files = [f[:-5] for f in os.listdir(a_folder_path) if f.endswith('.json')]
image_files = [f for f in os.listdir(b_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
out_files = [f for f in os.listdir(output_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

for img in image_files:
    img_name = os.path.splitext(img)[0]
    if img_name in json_files:
        shutil.move(os.path.join(b_folder_path, img), os.path.join(output_folder_path, img))
