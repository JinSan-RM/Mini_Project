import json
import os


def process_json_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)


    for file in os.listdir(src_folder):
        if file.endswith('.json'):
            src_file_path = os.path.join(src_folder, file)
            dest_file_path = os.path.join(dest_folder, file.replace('.json', '.txt'))


            try:
                with open(src_file_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                    if not content:
                        print(f"Skipping empty file: {src_file_path}")
                        continue
                    data = json.loads(content)
            except ValueError as e:
                print(f"Error while processing {src_file_path}: {e}")
                continue


            valid_classes = ['01', '02', '05', '06', '07', '08']
            output_lines = []


            for annotation in data["annotations"]:
                if annotation["class"] in valid_classes:
                    if annotation["class"] == '01':
                        annotation["class"] = '0'
                    elif annotation["class"] == '02':
                        annotation["class"] = '1'
                    elif annotation["class"] == '05':
                        annotation["class"] = '2'
                    elif annotation["class"] == '06':
                        annotation["class"] = '3'
                    elif annotation["class"] == '07':
                        annotation["class"] = '4'
                    else:
                        annotation["class"] = '5'
                    box = annotation["box"]
                    line = f'{annotation["class"]} {box[0] / 1920} {box[3] / 1080} {(int(box[2])-int(box[0])) / 1920 } {(int(box[3])-int(box[1])) / 1080}'
                    output_lines.append(line)


            with open(dest_file_path, 'w', encoding='utf-8') as dest_file:
                dest_file.write('\n'.join(output_lines))

src_folder = 'C:/Users/leeyo/Project/test/labels'  # 원본 폴더
dest_folder = 'C:/Users/leeyo/Project/test/label_text'  # 새로운 폴더
process_json_files(src_folder, dest_folder)
