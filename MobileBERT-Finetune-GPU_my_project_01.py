import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 장치: {device}")

# CSV 파일 경로 설정
csv_file = r"/250604_snsd_data_labeling_2000_Into_The_New_World_0513_02_only_comments_english.csv"

# 데이터 로드
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print("CSV 파일 로드 실패:", e)
    exit()

# 필요한 컬럼 확인
if 'comment' not in df.columns or 'label' not in df.columns:
    print("CSV 파일에 'comment' 또는 'label' 컬럼이 없습니다.")
    exit()

# 결측치 제거
df = df.dropna(subset=['comment', 'label'])

# 라벨을 숫자형으로 변환 (float -> int)
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# 텍스트 컬럼을 문자열로 변환
df['comment'] = df['comment'].astype(str)

# 훈련/검증 데이터 분할 (80% 학습, 20% 검증)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# MobileBERT 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=3)
model.to(device)

# 커스텀 Dataset 클래스 정의
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['comment'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)        # (seq_len)
        attention_mask = encoding['attention_mask'].squeeze(0)  # (seq_len)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Dataset 및 DataLoader 생성
train_dataset = SentimentDataset(train_df, tokenizer, max_length=128)
val_dataset = SentimentDataset(val_df, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 옵티마이저 설정 (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 학습 반복
epochs = 3
for epoch in range(epochs):
    model.train()
    total_train_correct = 0
    total_train = 0

    train_loop = tqdm(train_loader, desc=f"훈련 에폭 {epoch+1}")
    for batch in train_loop:
        # 배치를 GPU로 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 순전파 및 손실 계산
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # 역전파 및 파라미터 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 정확도 계산
        preds = torch.argmax(logits, dim=1)
        total_train_correct += (preds == labels).sum().item()
        total_train += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    train_accuracy = total_train_correct / total_train
    print(f"에폭 {epoch+1} 훈련 정확도: {train_accuracy:.4f}")

    # 검증 단계
    model.eval()
    total_val_correct = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            total_val_correct += (preds == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = total_val_correct / total_val
    print(f"에폭 {epoch+1} 검증 정확도: {val_accuracy:.4f}")

# 모델 및 토크나이저 저장
save_path = "./mobilebert_sentiment_model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("모델이 저장되었습니다:", save_path)
