import os
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from enum import Enum
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import time
from pyngrok import ngrok

app = FastAPI()

class ModelOption(str, Enum):
    fast = "fast"
    balanced = "balanced"
    accurate = "accurate"

class PredictionRequest(BaseModel):
    image: str  # base64 encoded image
    option: ModelOption = ModelOption.balanced

class PredictionResponse(BaseModel):
    species: str
    confidence: float
    inference_time: float

# 해파리 종류 매핑
JELLYFISH_SPECIES = {
    0: "Moon_jellyfish",
    1: "barrel_jellyfish",
    2: "blue_jellyfish",
    3: "compass_jellyfish",
    4: "lions_mane_jellyfish",
    5: "mauve_stinger_jellyfish"
}

# Load models
MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), 'models')

models = {
    ModelOption.fast: tf.keras.models.load_model(os.path.join(MODEL_DIR, 'basemodel_1_kaggle_without_aug.h5')),
    ModelOption.balanced: tf.keras.models.load_model(os.path.join(MODEL_DIR, 'basemodel_2_quality_sort.h5')),
    ModelOption.accurate: tf.keras.models.load_model(os.path.join(MODEL_DIR, 'basemodel_3_classweight80.h5'))
}

async def preprocess_image(image_file: UploadFile, target_size: tuple = (224, 224)) -> np.ndarray:
    contents = await image_file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, 0)
    
    return image_array

@app.post("/predict", response_model=PredictionResponse)
async def predict(image: UploadFile = File(...), option: ModelOption = ModelOption.balanced):
    # 이미지 전처리
    preprocessed_image = await preprocess_image(image)
    
    # 모델 선택 및 예측
    model = models[option]
    
    # 추론 시간 측정 시작
    start_time = time.time()
    
    # 예측 수행
    prediction = model.predict(preprocessed_image)
    
    # 추론 시간 측정 종료
    inference_time = time.time() - start_time
    
    # 예측 결과 해석
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    
    # 해파리 종류 매핑 (예측된 클래스 인덱스를 실제 종 이름으로 변환)
    species = JELLYFISH_SPECIES.get(predicted_class, "Unknown")
    
    return PredictionResponse(
        species=species,
        confidence=confidence,
        inference_time=inference_time
    )

if __name__ == "__main__":
    # ngrok을 사용하여 터널 생성
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)