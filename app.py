import openai
import requests
import uuid
import time
import json
import re
import os

from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn

import numpy as np

app = FastAPI()

openai.api_key = os.getenv('OPENAI_KEY', '')

api_url = os.getenv('CLOBA_OCR_URL', '')
secret_key = os.getenv('CLOBA_OCK_KEY', '')

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

@app.post("/uploadfile/", status_code=200)
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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

