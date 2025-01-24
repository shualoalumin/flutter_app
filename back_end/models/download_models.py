import wandb
import os

# 현재 디렉토리를 기준으로 모델 저장 경로 설정
MODEL_DIR = os.getcwd()
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize wandb run
run = wandb.init(project="National Oceanographic AI", job_type="model_download")

# List of model artifacts to download
model_artifacts = [
    'james-an/National Oceanographic AI/basemodel_4_augmented:v0',
    'james-an/National Oceanographic AI/basemodel_1_kaggle_without_aug:v0',
    'james-an/National Oceanographic AI/basemodel_3_classweight80:v1',
    'james-an/National Oceanographic AI/basemodel_2_quality_sort:v0'
]

# Download and store artifacts
for artifact_path in model_artifacts:
    artifact = run.use_artifact(artifact_path, type='model')
    artifact_dir = artifact.download(root=MODEL_DIR)
    print(f"Downloaded {artifact_path} to {artifact_dir}")

# Finish the wandb run
run.finish()

print(f"All models have been downloaded to {MODEL_DIR}")