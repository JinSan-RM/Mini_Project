# Import FastAPI
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from urllib.request import urlopen
from io import BytesIO
from ultralytics import YOLO
from PIL import Image
import tensorflow as tf
import os
import json
import time
import aiofiles
from pathlib import Path
import requests
import cv2
import numpy as np
from collections import Counter
from scipy.spatial import distance
#create the app
app = FastAPI()

modelV8s = YOLO('/code/app/best_v8s.pt')
modelV8m = YOLO('/code/app/best_v8m.pt')
model = modelV8m

model3 = YOLO('/code/app/step3_last.pt')

# step3 현재 사용되는 클래스 ( 임시 )
step3_cls_names = {
    0: 'typo_text',
    1: 'typo_price_num',
    2: 'typo_price_dollar',
    3: 'typo_price_W',
    4: 'typo_price_won_en',
    5: 'typo_price_won_kr',
    6: 'image_photo',
    7: 'image_colorBG', #교체
    8: 'image_removeBG',
    9: 'icon_arrow_left',
    10: 'icon_arrow_top',
    11: 'icon_arrow_bottom', #추가
    12: 'icon_arrow_right',
    13: 'icon_video_play', #추가
    14: 'icon_SNS_insta',
    15: 'icon_SNS_youtube',
    16: 'btn_radius',
    17: 'btn_ellipse',
    18: 'btn_square'
}



    
def getModelResult( results ):
    ROOT_PATH = os.getcwd() + "\\";
    data = {}
    dataList = []
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        names = result.names
        origImg = result.orig_shape
        imgWidth = origImg[1]
        imgHeight = origImg[0]

        data['img_width'] = imgWidth
        data['img_height'] = imgHeight
        print( origImg )
        # masks = result.masks  # Masks object for segmenation masks outputs
    #     probs = result.probs  # Class probabilities

    #     # print(boxes.xywhn)
    #     # print(boxes.cls)
    #     # print(masks)
    #     # print(probs)
        boxList = []
        for box in boxes:

            boxData = {}
            # box.xyxy[0]
            
            boxResult = box.xywhn.tolist()[0]
            classNum = int(box.cls.tolist()[0])
            className = names[classNum]
            conf = box.conf.tolist()[0]
            centerX = boxResult[0]
            centerY = boxResult[1]
            width = boxResult[2]
            height = boxResult[3]

            left = centerX - ( width / 2 )
            top = centerY - ( height / 2 )

            minX = left
            minY = top
            maxX = left + width
            maxY = top + height

            boxData['class_name'] = className
            boxData['class_id'] = classNum
            boxData['confidence'] = conf
            boxData['center_x'] = centerX
            boxData['center_y'] = centerY
            boxData['left'] = left
            boxData['top'] = top
            boxData['width'] = width
            boxData['height'] = height
            boxData['minX'] = minX
            boxData['minY'] = minY
            boxData['maxX'] = maxX
            boxData['maxY'] = maxY


            boxData['origin_center_x'] = centerX * imgWidth
            boxData['origin_center_y'] = centerY * imgHeight
            boxData['origin_left'] = left * imgWidth
            boxData['origin_top'] = top * imgHeight
            boxData['origin_width'] = width * imgWidth
            boxData['origin_height'] = height * imgHeight
            boxData['origin_minX'] = minX * imgWidth
            boxData['origin_minY'] = minY * imgHeight
            boxData['origin_maxX'] = maxX * imgWidth
            boxData['origin_maxY'] = maxY * imgHeight


            # if( boxData['confidence'] * 100 > 60 ):
            boxList.append( boxData )
        
        dataList.append( boxList )
    data['data'] = dataList
    # jsonStr = json.dumps( dataList )
    # print( jsonStr )
    return data

def imageUrlToPixels( source ):
    res = urlopen( source ).read()
    # Image open
    test_image = Image.open(BytesIO(res))

    test_image = tf.image.resize( test_image, size=(224,224))
    test_image = tf.expand_dims(test_image, axis=0)

    # test_image = Image.open( "upload/"+imagePath )
    # pixels = np.array(test_image)/255.0
    pixels = np.array(test_image)

    return pixels

@app.get('/getapi')
def getapi():
    modelName = "step1_section"
    # source = "D://WEVEN/htdocs/tfjs/ultralytics/sample/nike.jpeg"
    source = "/code/app/giordano.jpeg"
    # data = getModelResult( source, modelName )

    # model = YOLO('/code/app/best_v8s.pt')
    results = model.predict(source=source)
    image_bytes = results[0].plot()
    image = Image.fromarray(image_bytes)
    image.save('prediction.png')
    return FileResponse('prediction.png')
    
@app.post('/models/step1')
def postapi(source: str):

    ROOT_PATH = os.getcwd() + "\\";

    modelName = "step1_section"
    # model = YOLO('/code/app/best_v8s.pt')  # load a pretrained YOLOv8n detection model
    # model = YOLO('/code/app/best.pt')  # load a pretrained YOLOv8n detection model
    results = model(source)
    
    data = getModelResult( results )
    return data

@app.post('/models/step3')
def postapi(source: str):

    ROOT_PATH = os.getcwd() + "\\";

    modelName = "step3_section"
    # model3 = YOLO('/code/app/step3_last.pt')  # load a pretrained YOLOv8n detection model
    # model = YOLO('/code/app/best.pt')  # load a pretrained YOLOv8n detection model
    results = model3(source)
    
    data = getModelResult( results )
    return data


@app.get("/predict/{source_url:path}")
def predict_source_url(source_url: str):
    # model = YOLO('/code/app/best.pt')
    results = model.predict(source=source_url)
    image_bytes = results[0].plot()
    image = Image.fromarray(image_bytes)
    image.save('prediction.png')
    return FileResponse('prediction.png')


@app.get("/predict_v8s/{source_url:path}")
def predict_source_url(source_url: str):
    # model = YOLO('/code/app/best_v8s.pt')
    results = model.predict(source=source_url)
    image_bytes = results[0].plot()
    image = Image.fromarray(image_bytes)
    image.save('prediction.jpg')
    return FileResponse('prediction.jpg')

@app.post("/models/step2/block")
def step2_block(source: str):
    
    print( 'image path ==> ' + source )
    pixels = imageUrlToPixels( source )
    endpoint = 'http://tfserving:8501/v1/models/block:predict'
    header = {"content-type": "application/json"} 
    batch_json = json.dumps({"instances": pixels.tolist()})

    json_res = requests.post(url=endpoint, data=batch_json, headers=header)
    server_preds = json.loads(json_res.text)

    return server_preds['predictions']

@app.post("/models/step2/header")
def step2_footer(source: str):
    print( 'image path ==> ' + source )
    pixels = imageUrlToPixels( source )
    endpoint = 'http://tfserving:8501/v1/models/header:predict'
    header = {"content-type": "application/json"} 
    batch_json = json.dumps({"instances": pixels.tolist()})

    json_res = requests.post(url=endpoint, data=batch_json, headers=header)
    server_preds = json.loads(json_res.text)
    return server_preds['predictions']



@app.get("/models/step2/header2")
def step2_headerTest(source: str):
    print( 'aaa' )
    # source = "/code/app/giordano.jpeg"
    print( 'image path ==> ' + source )
    pixels = imageUrlToPixels( source )
    endpoint = 'http://tfserving:8501/v1/models/header:predict'
    header = {"content-type": "application/json"} 
    batch_json = json.dumps({"instances": pixels.tolist()})

    json_res = requests.post(url=endpoint, data=batch_json, headers=header)
    server_preds = json.loads(json_res.text)
    return server_preds['predictions']




@app.post("/models/step2/footer")
def step2_footer(source: str):
    print( 'image path ==> ' + source )
    pixels = imageUrlToPixels( source )
    endpoint = 'http://tfserving:8501/v1/models/footer:predict'
    header = {"content-type": "application/json"} 
    batch_json = json.dumps({"instances": pixels.tolist()})

    json_res = requests.post(url=endpoint, data=batch_json, headers=header)
    server_preds = json.loads(json_res.text)
    return server_preds['predictions']


@app.get("/predict/image_url/")
def predict_image_url(url: str):
    # model = YOLO("/code/app/step3_last.pt")
    results = model3.predict(source=url)
    image_bytes = results[0].plot()
    image = Image.fromarray(image_bytes)
    image.save('prediction.png')
    return FileResponse('prediction.png')

@app.get("/predict/image_url/json/")
def predict_image_url_json(url: str):
    # model = YOLO("/code/app/step3_last.pt")
    results = model3.predict(source=url)
    boxes = results[0].boxes 
    boxes_data = boxes.data.tolist()
    for box in boxes_data:
        cls_id = box[5]
        box.append(step3_cls_names[cls_id])
    return {
        'cls': boxes.cls.tolist(),
        'cls_names': [step3_cls_names[cls_id] for cls_id in boxes.cls.tolist()],
        'conf': boxes.conf.tolist(),
        'data': boxes.data.tolist(),
        'data_with_cls_names': boxes_data,
        'xywh': boxes.xywh.tolist(),
        'xywhn': boxes.xywhn.tolist(),
        'xyxy': boxes.xyxy.tolist(),
        'xyxyn': boxes.xyxyn.tolist(),
    }

image_dir = Path('files')
@app.post("/predict/image/")
async def predict_image(file: UploadFile):
    uploaded_image_path = image_dir / file.filename
    async with aiofiles.open(uploaded_image_path, 'wb') as uploaded_image:
        while content := await file.read(1024):
            await uploaded_image.write(content)
    
    return predict_image_url(uploaded_image_path)

@app.post("/predict/image/json/")
async def predict_image_json(file: UploadFile):
    uploaded_image_path = file.filename
    print( "path ===> " + uploaded_image_path )
    async with aiofiles.open(uploaded_image_path, 'wb') as uploaded_image:
        while content := await file.read(1024):
            await uploaded_image.write(content)
    
    return predict_image_url_json(uploaded_image_path)


@app.get('/getColor')
def color(source: str):
    top_n = 3
    print("path ===> " + source)
    # 이미지 다운로드
    response = requests.get(source)
    image_array = np.array(bytearray(response.content), dtype=np.uint8)

    # 이미지를 BGR 형식으로 디코드
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 이미지를 HSV(Hue, Saturation, Value) 색상 공간으로 변환합니다.
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV 이미지에서 채도(Saturation) 값을 추출합니다.
    saturation = hsv_image[:, :, 1]

    # 채도 값의 히스토그램을 계산합니다.
    histogram = cv2.calcHist([saturation], [0], None, [256], [0, 256])

    # 상위 N개의 값과 해당 인덱스를 찾습니다.
    max_values = np.argsort(histogram.flatten())[::-1][:top_n]
    max_colors = []
    max_ratios = []

    total_pixels = saturation.size

    for max_index in max_values:
        # 가장 높은 값의 인덱스에 해당하는 HSV 값을 BGR로 변환합니다.
        hsv_max = np.array([[[max_index, 255, 255]]], dtype=np.uint8)
        bgr_max = cv2.cvtColor(hsv_max, cv2.COLOR_HSV2BGR_FULL)
        hex_max = bgr_max[0][0].tolist()

        max_colors.append('#{:02x}{:02x}{:02x}'.format(*hex_max))
        max_ratios.append(float(histogram[max_index] / total_pixels))

    print('가장 높은 값들:', max_values)
    print('가장 높은 값들의 16진수 컬러:', max_colors)
    print('가장 높은 값들의 비율:', max_ratios)

    data = {
        'max_values': max_values.tolist(),
        'colors': max_colors,
        'ratios': max_ratios
    }
    return data

@app.get('/getMostColor')
def mostColor(source: str):
    top_n = 6
    threshold = 0.04  # 비율이 이 값보다 큰 색상은 제외

    print("path ===> " + source)
    # 이미지 다운로드
    response = requests.get(source)
    image_array = np.array(bytearray(response.content), dtype=np.uint8)

    # 이미지를 BGR 형식으로 디코드
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 이미지를 RGB 또는 BGR 형식으로 변환합니다.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지에서 모든 픽셀의 색상을 추출합니다.
    pixels = image.reshape(-1, 3)

    # 추출한 색상을 카운트하여 각 색상의 빈도를 계산합니다.
    color_counts = Counter(map(tuple, pixels))

    # 전체 픽셀 수 계산
    total_pixels = image.shape[0] * image.shape[1]

    # 가장 많이 사용된 색상의 비율 계산
    most_common_colors = color_counts.most_common(top_n)

    most_common_colors_list = []
    for color, count in most_common_colors:
        color_hex = f'#{"".join(format(c, "02x") for c in color)}'
        color_ratio = count / total_pixels

        # 비율이 40% 이상인 색상은 제외
        if color_ratio <= threshold:
            most_common_colors_list.append([color_hex, count, color_ratio])

    data = {
        'colors': most_common_colors_list
    }

    print(most_common_colors_list)

    return data

def download_image(url):
    response = requests.get(url)
    image_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def resize_image(image, max_size):
    height, width = image.shape[:2]
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def combine_similar_colors(colors, threshold):
    grouped_colors = []
    used = [False] * len(colors)

    for i in range(len(colors)):
        if not used[i]:
            similar_group = [colors[i]]
            used[i] = True

            for j in range(i+1, len(colors)):
                if not used[j] and color_distance(colors[i], colors[j]) <= threshold:
                    similar_group.append(colors[j])
                    used[j] = True

            grouped_colors.append(similar_group)

    return grouped_colors


def color_distance(color1, color2):
    return distance.euclidean(color1, color2)


@app.get('/getMostColorGroup')
def mostColorGroup(source: str):
    top_n = 6
    threshold = 0.04  # 비율이 이 값보다 큰 색상은 제외
    max_image_size = 500  # 이미지의 최대 크기

    print("path ===> " + source)
    # 이미지 다운로드
    image = download_image(source)

    # 이미지 크기 축소
    resized_image = resize_image(image, max_image_size)

    # 이미지를 RGB 형식으로 변환합니다.
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 이미지에서 모든 픽셀의 색상을 추출합니다.
    pixels = rgb_image.reshape(-1, 3)

    # 추출한 색상을 카운트하여 각 색상의 빈도를 계산합니다.
    color_counts = Counter(map(tuple, pixels))

    # 전체 픽셀 수 계산
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]

    # 가장 많이 사용된 색상의 비율 계산
    most_common_colors = color_counts.most_common(top_n)

    most_common_colors_list = []
    for color, count in most_common_colors:
        color_hex = f'#{"".join(format(c, "02x") for c in color)}'
        color_ratio = count / total_pixels

        # 비율이 40% 이상인 색상은 제외
        if color_ratio <= threshold:
            most_common_colors_list.append(color)

    # 비슷한 색상군 합치기
    combined_colors = combine_similar_colors(most_common_colors_list, threshold=5)

    merged_colors = []
    for color_group in combined_colors:
        merged_color = np.mean(color_group, axis=0).astype(int)
        merged_color_hex = f'#{"".join(format(c, "02x") for c in merged_color)}'
        merged_count = sum(color_counts[tuple(color)] for color in color_group)
        merged_ratio = merged_count / total_pixels
        merged_colors.append([merged_color_hex, merged_count, merged_ratio])

    data = {
        'colors': merged_colors
    }

    print(merged_colors)

    return data


target_classes = ['typo_text','image_photo', 'image_colorBG', 'image_removeBG'] # 바운딩 박스를 추출하고자 하는 클래스명들의 리스트
def extract_boxes_by_class(boxList, target_classes):                            #boxList: getModelResult 함수로부터 반환된 data의 'data' 키에 해당하는 값

    target_boxes = []                                                           #target들을 넣을 박스 리스트       
    for box in boxList:                                                         #boxlist에서  클래스 찾아내기
        class_name = box['class_name']
        if class_name in target_classes:
            target_boxes.append(box)                                            #찾아냈으면 리스트에 넣기
    return target_boxes



def get_text_and_image_boxes(results):                                          # image, text_box label을 가져오기 위함
    data = getModelResult(results)
    boxList = data['data']
                                                                                # 텍스트(0번 클래스)와 이미지(6, 7, 8번 클래스)에 해당하는 바운딩 박스 리스트 추출
    text_boxes = extract_boxes_by_class(boxList, ['typo_text'])
    image_boxes = extract_boxes_by_class(boxList, ['image_photo', 'image_colorBG', 'image_removeBG'])

    return text_boxes, image_boxes


def is_box_inside_image(text_boxes, image_boxes):                               # image label안에 text label이 있는지 체크
    

    for text_box in text_boxes:                                                 
        text_x_min, text_y_min, text_x_max, text_y_max = text_box 
        for image_box in image_boxes:
            image_x_min, image_y_min, image_x_max, image_y_max = image_box                                                 
            if not (text_x_min >= image_x_min and text_y_min >= image_y_min and
                    text_x_max <= image_x_max and text_y_max <= image_y_max):
                return False
    
    return True

def text_on_right_image(image_boxes, text_boxes):                               # 텍스트가 image 오른쪽에 있는지 체크

    for text_box in text_boxes:                                                 
        text_x_min, _, _, _ = text_box
        for image_box in image_boxes:
            image_x_max, _, _, _ = image_box   
        if text_x_min >= image_x_max:
            return True

    return False

def text_on_left_image(image_boxes, text_boxes):                                # 텍스트가 image 왼쪽에 있는지 체크
    
    for text_box in text_boxes:
        _, text_x_max, _, _ = text_box
        for image_box in image_boxes:
            image_x_min, _, _, _ = image_box
            if text_x_max <= image_x_min:
                return True
    
    return False


def text_on_bottom_image(image_boxes, text_boxes):                              # 텍스트가 image 아래쪽에 있는지 체크
    
    for text_box in text_boxes:
        _, _, _, text_y_min = text_box
        for image_box in image_boxes:
            _, _, _, image_y_max = image_box
            if text_y_min >= image_y_max:
                return True
    
    return False
                                                                                # 어느방향에 있는지 한번에 체크
def check_text_position(image_boxes, text_boxes):                               
    if is_box_inside_image(image_boxes, text_boxes):
        return "Image Inside Text"
    elif text_on_right_image(image_boxes, text_boxes):
        return "Image Right Text"
    elif text_on_left_image(image_boxes, text_boxes):
        return "Image Left Text"
    elif text_on_bottom_image(image_boxes, text_boxes):
        return "Image Bottom Text"
    else:
        return "Image and Texts are not related"
    
