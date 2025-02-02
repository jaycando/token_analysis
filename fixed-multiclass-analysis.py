import os
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
'''
    1.원본 텍스트에 대한 예측 수행
    2.각 토큰을 하나씩 마스킹하면서 예측 변화 관찰
    3.예측 변화의 크기를 해당 토큰의 중요도로 사용

'''
class MultiClassRobertaAnalyzer:
    def __init__(self, num_labels=20, class_names=None, output_dir="shap_results"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        os.makedirs(output_dir, exist_ok=True)
        
        # 클래스 이름 설정
        self.class_names = class_names if class_names else [f'Class {i}' for i in range(num_labels)]

    def predict(self, x):
        if isinstance(x, str):
            x = [x]
        elif isinstance(x, list) and not x:
            return np.zeros((1, self.num_labels))

        inputs = self.tokenizer(
            x,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def compute_token_attributions(self, text):
        tokens = self.tokenizer.tokenize(text)
        encoded = self.tokenizer.encode_plus(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            base_output = self.model(**encoded)
            base_probs = torch.softmax(base_output.logits, dim=-1)[0].cpu().numpy()

        importance_matrix = np.zeros((len(tokens), self.num_labels))
        
        for i in range(len(tokens)):
            masked_text = tokens.copy()
            masked_text[i] = self.tokenizer.mask_token
            masked_input = self.tokenizer.encode(
                ' '.join(masked_text),
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                masked_output = self.model(masked_input)
                masked_probs = torch.softmax(masked_output.logits, dim=-1)[0].cpu().numpy()

            importance_matrix[i] = np.abs(base_probs - masked_probs)

        return importance_matrix, tokens, base_probs

    # 시각화 함수 부분만 수정
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

    def analyze_texts_from_csv(self, analysis_csv_path, data_column='data'):
        df = pd.read_csv(analysis_csv_path)
        texts = df[data_column].dropna().tolist()
        
        results_data = []
        
        for idx, text in enumerate(tqdm(texts, desc="Analyzing texts")):
            try:
                print(f"\nAnalyzing text {idx}: {text}")
                
                # 중요도 계산
                importances, tokens, probs = self.compute_token_attributions(text)
                
                # 시각화
                self.visualize_importance(importances, tokens, probs, idx)
                
                # 결과 저장
                for token_idx, (token, importance_vector) in enumerate(zip(tokens, importances)):
                    result = {
                        'text_id': idx,
                        'text': text,
                        'token': token,
                        'position': token_idx,
                        'importance': float(np.mean(importance_vector))
                    }
                    # 각 클래스별 중요도와 확률 추가
                    for class_idx in range(self.num_labels):
                        result[f'importance_class_{class_idx}'] = float(importance_vector[class_idx])
                        result[f'prob_class_{class_idx}'] = float(probs[class_idx])
                    
                    results_data.append(result)
                
                # 현재 텍스트에 대한 결과 출력
                print(f"\nResults for text {idx}:")
                print("Top 3 most important tokens:")
                token_importances = list(zip(tokens, np.mean(importances, axis=1)))
                token_importances.sort(key=lambda x: x[1], reverse=True)
                for token, imp in token_importances[:3]:
                    print(f"Token: {token}, Importance: {imp:.6f}")
                
                print("\nTop 3 predicted classes:")
                top_classes = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
                for class_idx, prob in top_classes:
                    print(f"{self.class_names[class_idx]}: {prob:.6f}")
                
            except Exception as e:
                print(f"\nWarning: Text {idx} analysis failed: {str(e)}")
                continue
        
        if not results_data:
            print("No results were generated.")
            return pd.DataFrame()
        
        # 결과를 DataFrame으로 변환
        results_df = pd.DataFrame(results_data)
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f'token_importance_{timestamp}.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        
        return results_df

def main():

    class_names = [
        "정치", "경제", "사회", "문화", "국제",
        "IT", "스포츠", "연예", "교육", "환경",
        "건강", "여행", "음식", "패션", "자동차",
        "과학", "군사", "종교", "부동산", "취미"
    ]
    # 테스트 데이터
    data = {
        'data': [
            "이 제품은 정말 좋습니다. 추천합니다!",
            "품질이 좋지 않아요. 실망했습니다.",
            "가격은 비싸지만 성능은 훌륭해요.",
            "배송이 너무 늦어요. 개선이 필요합니다.",
            "디자인이 예쁘고 실용적입니다."
        ]
    }
    
    pd.DataFrame(data).to_csv('test_data.csv', index=False, encoding='utf-8')
    
    try:
        analyzer = MultiClassRobertaAnalyzer(
            num_labels=20, 
            class_names=class_names,  # 클래스 라벨 전달
            output_dir="shap_results"
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