import openai
import requests
import uuid
import time
import json
import re
import os

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import uvicorn

import numpy as np
import pandas as pd

import boto3

app = FastAPI()

openai.api_key = os.getenv('OPENAI_KEY', '')

api_url = os.getenv('CLOBA_OCR_URL', '')
secret_key = os.getenv('CLOBA_OCK_KEY', '')

s3 = boto3.client('s3')

# 책에서 추출된 글자를 gpt api로 요약해주는 함수
def summarize_article(korean_text):

    model_engine = "text-davinci-003" 
    max_tokens = 2700

    completion = openai.Completion.create(
        engine=model_engine,
        prompt=f'''
        아래 내용을 두 줄로 핵심 요약해줘.

        {korean_text}
        ''',
        max_tokens=max_tokens,
        temperature=0.3,       # creativity
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    summary = completion.choices[0].text
    return summary

# 제목, 줄거리, 인상깊은 내용을 바탕으로 뮤직 벡터를 추출해주는 함수
def calculate_music_vector(title, summary, impressive):

    model_engine = "gpt-3.5-turbo" 
    max_tokens = 2700

    chat_completion = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {"role": "system", "content": '''당신은 각 책의 독특한 감정적 분위기, 테마, 캐릭터의 복잡성, 장면의 세부 묘사, 그리고 작가의 문체를 깊이 이해하고 이를 기반으로 음악을 추천하는 고도로 발전된 시스템입니다. 
                                            책의 제목, 줄거리 요약, 그리고 인상 깊은 문장을 통해 책의 핵심 감정과 테마를 파악하고, 이를 반영한 독특하고 맞춤형의 음악 특성 벡터를 생성하세요. 
                                            각 책의 독특한 요소들을 고려하여 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo' 등의 음악적 특성이 어떻게 변화해야 하는지 분석해주세요. '''},
            {"role": "user", "content": f"{title}책을 읽으면서 음악을 추천 받고 싶어"},
            {"role": "user", "content": f"{title}는 다음과 같은 줄거리를 가지고 있어: {summary}"},
            {"role": "user", "content": f"인상 깊은 문장은 이거야: {impressive}"},
            {"role": "user", "content": "음악의 특성은 다음과 같은 범위 내에서 수치화해야 합니다: 'danceability' [0,1], 'energy' [0,1], 'key' [0,11], 'loudness' [-30,0], 'mode' [0,1], 'speechiness' [0,1], 'acousticness' [0,1], 'instrumentalness' [0,1], 'liveness' [0,1], 'valence' [0,1], 'tempo' [50,250]."},
            {"role": "user", "content": f"이 분석을 바탕으로, 각 책에 대한 음악적 분위기를 가장 잘 반영하는 1차원 실수 벡터 배열을 생성하여 제공해주세요. 배열은 총 11개의 요소를 포함해야 하며, 각 책에 따라 특성 값이 상당히 달라져야 합니다. 추가적인 텍스트 없이 값들만 대괄호 안에 쉼표로 구분되어 표현되어야 합니다."},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # 마지막 메시지에서 음악 벡터 추출
    vector = chat_completion.choices[0].message['content']
    print("vector입니다:", vector)
    return vector

def vector_to_music(vector):
    data = pd.read_csv('music-info.csv')

    list_feature = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    max_feature = [1, 1, 11, 30, 1, 1, 1, 1, 1, 1, 200]

    data['score'] = 0

    print("vector의 길이 및 내용", vector, len(vector))

    for j in range(11):
        if max_feature[j] != 0: 
            rounded_value = round(abs(data[list_feature[j]] - vector[j]) / max_feature[j], 3)
            data['score'] += rounded_value

    # 데이터 결측치 및 형변환
    data['score'] = data['score'].fillna(0)
    data['score'] = data['score'].astype(float)
    data_sort = data.sort_values(by='score')

    print("data?:", data_sort)
    # 가장 score가 높은 index를 반환
    primary_index = data_sort.iloc[0, 0]
    print("데이터 sort 값:", primary_index)

    return primary_index

def load_music(index: int):
    bucket_name = 'readm-bucket'
    object_key = f'{index}.mp3'
    music_url = f'https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{object_key}'

    # 파일 존재 여부 확인
    try:
        s3.head_object(Bucket=bucket_name, Key=object_key)
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"No music file found for index: {index}")

    return music_url

# 책 이미지 촬영 시 요약해주는 api 
@app.post("/ocr-summary/", status_code=200)
async def random_num(image_file: UploadFile = File(...)):
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),  # UUID를 문자열로 변환
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', (image_file.filename, await image_file.read(), image_file.content_type))
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)

    result = response.json()

    with open('result.json', 'w', encoding='utf-8') as make_file:
        json.dump(result, make_file, indent="\t", ensure_ascii=False)

    text = ""
    for field in result['images'][0]['fields']:
        text += field['inferText']

    # 한글 문자만 추출하는 정규 표현식 사용
    korean_text = re.sub('[^가-힣]+', ' ', text)

    summary = summarize_article(korean_text)
    
    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "emotion": summary}

@app.post("/ocr-music-vector/", status_code=201)
async def random_num(title: str, summary: str, image_file: UploadFile = File(...)):
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),  # UUID를 문자열로 변환
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', (image_file.filename, await image_file.read(), image_file.content_type))
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)

    result = response.json()

    with open('result.json', 'w', encoding='utf-8') as make_file:
        json.dump(result, make_file, indent="\t", ensure_ascii=False)

    text = ""
    for field in result['images'][0]['fields']:
        text += field['inferText']

    # 한글 문자만 추출하는 정규 표현식 사용
    korean_text = re.sub('[^가-힣]+', ' ', text)

    impressive = summarize_article(korean_text)

    vector_string = calculate_music_vector(title, summary, impressive)
    
    match = re.search(r'\[(.*?)\]', vector_string)

    # 답변 중 배열만을 추출
    if match:
        array_str = match.group(1)
        vector = [float(x) if '.' in x else int(x) for x in array_str.split(',')]
    else:
        vector = []

    music_category = ['danceability[0,1]', 'energy[0,1]', 'key[0,11]', 'loudness[-30,0]', 'mode[0,1]', 'speechiness[0,1]', 'acousticness[0,1]', 'instrumentalness[0,1]', 'liveness[0,1]', 'valence[0,1]', 'tempo[50,250]']
    music_dict = dict(zip(music_category, vector))

    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "music_vector": music_dict}

@app.post("/ocr-music-url/", status_code=201)
async def image_to_music(
    title: str = Form(...), 
    summary: str = Form(...), 
    image_file: UploadFile = File(...)
):
    request_json = {
        'images': [
            {
                'format': 'jpg',
                'name': 'demo'
            }
        ],
        'requestId': str(uuid.uuid4()),  # UUID를 문자열로 변환
        'version': 'V2',
        'timestamp': int(round(time.time() * 1000))
    } 

    payload = {'message': json.dumps(request_json).encode('UTF-8')}
    files = [
        ('file', (image_file.filename, await image_file.read(), image_file.content_type))
    ]
    headers = {
        'X-OCR-SECRET': secret_key
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=files)

    result = response.json()

    with open('result.json', 'w', encoding='utf-8') as make_file:
        json.dump(result, make_file, indent="\t", ensure_ascii=False)

    text = ""
    for field in result['images'][0]['fields']:
        text += field['inferText']

    # 한글 문자만 추출하는 정규 표현식 사용
    korean_text = re.sub('[^가-힣]+', ' ', text)

    impressive = summarize_article(korean_text)

    # 2/3 길이에 해당하는 부분 문자열 추출
    length = len(summary)
    cut_length = int(length * (2 / 3))
    shortened_summary = summary[:cut_length]

    vector_string = calculate_music_vector(title, shortened_summary, impressive)
    
    match = re.search(r'\[(.*?)\]', vector_string)

    # 답변 중 배열만을 추출
    if match:
        array_str = match.group(1)
        vector = [float(x) if '.' in x else float(x) for x in array_str.split(',')]
    else:
        vector = []

    music_index = int(vector_to_music(vector))
    music = load_music(music_index)

    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "music": music}

if __name__ == "__main__":
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"An error occurred: {e}")