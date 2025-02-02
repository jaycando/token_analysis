# Token Importance Analysis with XLM-RoBERTa

텍스트의 각 토큰(단어)이 모델의 예측에 미치는 영향을 분석하는 도구입니다.

## 특징

- XLM-RoBERTa 모델 기반
- 토큰별 중요도(importance) 계산
- 다중 클래스(20개) 지원
- 시각화 기능 포함 (히트맵, 막대 그래프)

## 설치 방법

필요한 패키지 설치:

```bash
pip install torch transformers pandas numpy seaborn matplotlib tqdm
```

## 사용 방법

1. 모델 초기화:
```python
analyzer = MultiClassRobertaAnalyzer(
    num_labels=20,
    class_names=['정치', '경제', '사회', ...],  # 실제 클래스 이름 입력
    output_dir="shap_results"
)
```

2. 분석 실행:
```python
results_df = analyzer.analyze_texts_from_csv(
    analysis_csv_path='your_data.csv',
    data_column='text'
)
```

### 입력 데이터 형식

CSV 파일에는 다음과 같은 형식의 데이터가 필요합니다:

```csv
text
"분석할 텍스트1"
"분석할 텍스트2"
...
```

### 출력 결과

1. **시각화 파일**:
   - `importance_text_{id}.png`: 토큰 중요도 히트맵과 평균 중요도
   - `probs_text_{id}.png`: 클래스별 예측 확률

2. **CSV 결과 파일**:
   - `token_importance_{timestamp}.csv`: 전체 분석 결과

## 주의사항

1. 현재 모델은 fine-tuning되지 않은 상태이므로, 분류(classification) 결과는 신뢰할 수 없습니다.
2. Importance 값은 각 토큰이 예측에 미치는 상대적인 영향을 보여주는 참고 지표입니다.
3. 정확한 분류를 위해서는 fine-tuning된 모델을 사용해야 합니다.

## 코드 구조

```
├── MultiClassRobertaAnalyzer     # 메인 클래스
│   ├── __init__                  # 초기화 (모델, 토크나이저 로드)
│   ├── predict                   # 예측 함수
│   ├── compute_token_attributions # 토큰별 중요도 계산
│   ├── visualize_importance      # 시각화 함수
│   └── analyze_texts_from_csv    # CSV 파일 분석
```

## 시각화 예시

### 1. 토큰 중요도 히트맵
![importance_heatmap](./example_images/importance_heatmap.png)

### 2. 예측 확률
![prediction_probs](./example_images/prediction_probs.png)

## 제한사항

1. 한글 폰트가 설치되어 있어야 시각화가 제대로 동작합니다.
2. 현재 버전에서는 분류 성능이 제한적입니다.
3. 메모리 사용량이 클 수 있으므로 대량의 텍스트 분석 시 주의가 필요합니다.

## 향후 계획

1. Fine-tuning 기능 추가
2. 토큰 간 상호작용 분석 기능 추가
3. 배치 처리 최적화
4. 다양한 사전 학습 모델 지원