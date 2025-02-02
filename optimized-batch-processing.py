import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
from torch.utils.data import Dataset, DataLoader
warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """텍스트 데이터셋"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

def batch_tokenize(batch_texts: List[str], tokenizer, max_length: int = 512):
    """배치 토큰화"""
    return tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

class MultiClassRobertaAnalyzer:
    def __init__(
        self, 
        num_labels=20, 
        class_names=None, 
        output_dir="shap_results",
        batch_size=16,
        max_length=512,
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        model_name = "xlm-roberta-base"
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        self.num_labels = num_labels
        self.output_dir = output_dir
        self.class_names = class_names if class_names else [f'Class {i}' for i in range(num_labels)]
        self.batch_size = batch_size
        self.max_length = max_length
        
        os.makedirs(output_dir, exist_ok=True)

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> np.ndarray:
        """배치 단위 예측"""
        dataset = TextDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=lambda x: x)
        
        all_probs = []
        
        for batch_texts in dataloader:
            inputs = batch_tokenize(batch_texts, self.tokenizer, self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            
        return np.vstack(all_probs)

    def compute_token_importances_batch(
        self,
        texts: List[str],
        batch_size: int = None
    ) -> List[Tuple[np.ndarray, List[str], np.ndarray]]:
        """배치 단위로 토큰 중요도 계산"""
        batch_size = batch_size or self.batch_size
        results = []

        # 텍스트 배치 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 각 텍스트별로 개별 처리
            for text in batch_texts:
                try:
                    # 1. 토큰화
                    encoded = self.tokenizer.encode_plus(
                        text,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
                    
                    # 2. 기본 예측
                    with torch.no_grad():
                        base_output = self.model(**encoded)
                        base_probs = torch.softmax(base_output.logits, dim=-1)[0].cpu().numpy()
                    
                    # 3. 각 토큰별 중요도 계산
                    importance_matrix = np.zeros((len(tokens), self.num_labels))
                    
                    # 각 토큰에 대해 개별적으로 마스킹
                    for token_idx in range(len(tokens)):
                        # 토큰 리스트 복사
                        masked_tokens = tokens.copy()
                        # 현재 토큰을 마스크로 치환
                        masked_tokens[token_idx] = self.tokenizer.mask_token
                        
                        # 마스킹된 텍스트 생성
                        masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
                        
                        # 마스킹된 텍스트에 대한 예측
                        masked_encoded = self.tokenizer.encode_plus(
                            masked_text,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        with torch.no_grad():
                            masked_output = self.model(**masked_encoded)
                            masked_probs = torch.softmax(masked_output.logits, dim=-1)[0].cpu().numpy()
                        
                        # 중요도 계산 (예측 변화량)
                        importance_matrix[token_idx] = np.abs(base_probs - masked_probs)
                    
                    # 특수 토큰 제외
                    valid_tokens = []
                    valid_importance = []
                    for idx, token in enumerate(tokens):
                        if token not in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                            valid_tokens.append(token)
                            valid_importance.append(importance_matrix[idx])
                    
                    results.append((np.array(valid_importance), valid_tokens, base_probs))
                    
                except Exception as e:
                    print(f"Error processing text: {str(e)}")
                    continue

        return results

    def process_results_batch(
        self,
        batch_results: List[Tuple[np.ndarray, List[str], np.ndarray]],
        texts: List[str],
        start_idx: int
    ) -> List[Dict]:
        """배치 결과 처리"""
        processed_results = []
        
        for idx, (importances, tokens, probs) in enumerate(batch_results):
            text_idx = start_idx + idx
            text = texts[text_idx]
            
            # 시각화
            self.visualize_importance(importances, tokens, probs, text_idx)
            
            # 결과 저장
            for token_idx, (token, importance_vector) in enumerate(zip(tokens, importances)):
                result = {
                    'text_id': text_idx,
                    'text': text,
                    'token': token,
                    'position': token_idx,
                    'importance': float(np.mean(importance_vector))
                }
                
                for class_idx in range(self.num_labels):
                    result[f'importance_class_{class_idx}'] = float(importance_vector[class_idx])
                    result[f'prob_class_{class_idx}'] = float(probs[class_idx])
                
                processed_results.append(result)
                
        return processed_results
    
    def visualize_importance(self, importances, tokens, probs, text_id):
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 1. 클래스별 토큰 중요도 히트맵
        plt.figure(figsize=(20, 12))
        plt.subplot(2, 1, 1)
        sns.heatmap(
            importances.T,
            xticklabels=tokens,
            yticklabels=self.class_names,  # 클래스 라벨 사용
            cmap='YlOrRd',
            annot=True,
            fmt='.4f',
            annot_kws={'size': 8}
        )
        plt.title('Token Importance by Class', pad=20, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # 2. 평균 토큰 중요도
        plt.subplot(2, 1, 2)
        mean_importances = importances.mean(axis=1)
        bars = plt.bar(range(len(tokens)), mean_importances)
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right', fontsize=10)
        plt.title('Average Token Importance', pad=20, fontsize=14)
        plt.ylabel('Importance Score', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'importance_text_{text_id}.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()

        # 3. 클래스별 예측 확률
        plt.figure(figsize=(15, 6))
        bars = plt.bar(range(self.num_labels), probs)
        plt.title('Class Probabilities', pad=20, fontsize=14)
        plt.xticks(range(self.num_labels), self.class_names, rotation=45, ha='right')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, f'probs_text_{text_id}.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()


    def analyze_texts_from_csv(
        self,
        analysis_csv_path: str,
        data_column: str = 'data',
        batch_size: int = None
    ) -> pd.DataFrame:
        """CSV 파일의 텍스트들에 대한 배치 분석"""
        batch_size = batch_size or self.batch_size
        df = pd.read_csv(analysis_csv_path)
        texts = df[data_column].dropna().tolist()
        
        all_results = []
        
        # tqdm으로 전체 진행률 표시
        with tqdm(total=len(texts), desc="Analyzing texts") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    # 배치 분석
                    batch_results = self.compute_token_importances_batch(batch_texts)
                    
                    # 결과 처리
                    processed_results = self.process_results_batch(batch_results, texts, i)
                    all_results.extend(processed_results)
                    
                    # 진행률 업데이트
                    pbar.update(len(batch_texts))
                    
                except Exception as e:
                    print(f"\nError processing batch {i//batch_size}: {str(e)}")
                    continue
        
        if not all_results:
            print("No results were generated.")
            return pd.DataFrame()
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(all_results)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f'token_importance_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        # 메모리 효율을 위해 배치별로 결과 출력
        self.print_analysis_summary(results_df)
        
        return results_df

    def print_analysis_summary(self, results_df: pd.DataFrame):
        """분석 결과 요약 출력"""
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        # 각 텍스트별로 중요한 토큰과 예측된 클래스 출력
        for text_id in results_df['text_id'].unique():
            text_results = results_df[results_df['text_id'] == text_id]
            text = text_results.iloc[0]['text']
            
            print(f"\nText {text_id}: {text}")
            
            # 상위 3개 토큰
            importance_data = text_results[['token', 'importance']].sort_values('importance', ascending=False)
            print("\nTop 3 most important tokens:")
            print(importance_data.head(3))
            
            # 상위 3개 클래스
            prob_columns = [col for col in text_results.columns if col.startswith('prob_class_')]
            probs = text_results[prob_columns].iloc[0]
            top_classes = probs.nlargest(3)
            
            print("\nTop 3 predicted classes:")
            for class_name, prob in top_classes.items():
                class_idx = int(class_name.split('_')[-1])
                print(f"{self.class_names[class_idx]}: {prob:.4f}")

def main():
    # 테스트 데이터
    data = {
        'data': [
            "This product is really good. I recommend it!",
            "The quality is not good. I am disappointed.",
            "It’s expensive, but the performance is great.",
            "Delivery is too late. Improvement is needed.",
            "The design is pretty and practical."
        ]
    }
    
    class_names = [
        "정치", "경제", "사회", "문화", "국제",
        "IT", "스포츠", "연예", "교육", "환경",
        "건강", "여행", "음식", "패션", "자동차",
        "과학", "군사", "종교", "부동산", "취미"
    ]
    
    pd.DataFrame(data).to_csv('test_data.csv', index=False, encoding='utf-8')
    
    try:
        analyzer = MultiClassRobertaAnalyzer(
            num_labels=20,
            class_names=class_names,
            output_dir="shap_results",
            batch_size=16  # 배치 크기 설정
        )
        
        results_df = analyzer.analyze_texts_from_csv(
            analysis_csv_path='test_data.csv',
            data_column='data'
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')

if __name__ == "__main__":
    main()