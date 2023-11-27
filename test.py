from fastapi import FastAPI, HTTPException, File, UploadFile
import uvicorn

import numpy as np
from random import randint


app = FastAPI()

@app.post("/uploadfile/", status_code=200)
async def random_num(file: UploadFile = File(...)):
    random_number = randint(0, 5)

    return {"statusCode": 201, "success": True, "message":"File uploaded successfully", "emotion": random_number}

if __name__ == "__main__":
    uvicorn.run("test:app", host="0.0.0.0", port=8000)