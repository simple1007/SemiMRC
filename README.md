# SemiMRC
* AIHub에서 공개된 뉴스기사 mrc 데이터셋을 이용해 모델 구축
  + DATASET - AIHub 뉴스기사 기계독해 데이터셋 사용
* 대중화된 언어 모델인 DistilKoBERT 사용
  + https://github.com/monologg/KoBERT-Transformers
* KoBERT가 Subword Tokenizer를 사용함으로 인해
  + 음절기반 index 범위 -> Subword 토큰 기반 index범위 변환 작업 수행 

## 전처리
AIHub의 뉴스기사 mrc 프로젝트 루트경로에 복사후
전처리 코드 실행
```
python preprocessing.py
```

## 학습코드 실행
```
python train.py
```

## 평가
```
python eval.py
```
