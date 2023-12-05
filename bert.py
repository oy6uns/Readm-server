from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
import numpy as np

# FastAPI 앱 초기화
app = FastAPI()

# 요청 데이터 모델 정의
class Review(BaseModel):
    text: str

# 모델과 토크나이저 설정
device = torch.device("cpu")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer,vocab, max_len,
                 pad, pair):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))


    def __len__(self):
        return (len(self.labels))
    
# 모델 인스턴스 생성 및 상태 로드
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load("kobert_sentiment_model.pt", map_location=device))

# 예측을 위한 경로 연산자
@app.post("/predict/")
async def predict_review(review: Review):
    data = [review.text, '0']  # '0'은 더미 레이블
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tokenizer.tokenize, vocab, 64, True, False)
    test_dataloader = DataLoader(another_test, batch_size=1, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = model(token_ids, valid_length, segment_ids)
            test_eval = []
            logits = out[0]
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 1:
                test_eval.append("당황")
            elif np.argmax(logits) == 2:
                test_eval.append("분노")
            elif np.argmax(logits) == 3:
                test_eval.append("불안")
            elif np.argmax(logits) == 4:
                test_eval.append("상처")
            elif np.argmax(logits) == 5:
                test_eval.append("슬픔")

            return {"statusCode": 201, "success": True, "message": "File uploaded successfully", "emotion": test_eval[0]}

    return {"emotion": test_eval[0]}  # 수정 가능

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)