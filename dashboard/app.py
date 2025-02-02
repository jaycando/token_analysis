from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"  # 필요에 따라 변경

# 전역 변수: 업로드된 CSV 파일의 DataFrame 저장
df_global = None

# 실제 클래스 라벨명 (예시: 20개 클래스)
CLASS_NAMES = [
    "정치", "경제", "사회", "문화", "국제",
    "IT", "스포츠", "연예", "교육", "환경",
    "건강", "여행", "음식", "패션", "자동차",
    "과학", "군사", "종교", "부동산", "취미"
]

# 한글 폰트 설정 (시스템에 NanumGothic 등 한글 폰트가 설치되어 있어야 합니다.)
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

def fig_to_base64(fig):
    """Matplotlib Figure를 PNG 이미지로 변환 후 base64 문자열 반환"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return image_base64

@app.route("/", methods=["GET", "POST"])
def index():
    global df_global

    # POST 요청: CSV 파일 업로드 처리
    if request.method == "POST":
        if "csv_file" in request.files:
            file = request.files["csv_file"]
            if file.filename != "":
                try:
                    # 인코딩 옵션 필요 시 "utf-8-sig" 등으로 지정
                    df_global = pd.read_csv(file, encoding="utf-8-sig")
                except Exception as e:
                    return f"CSV 파일 읽기 오류: {e}"
        return redirect(url_for("index"))

    # CSV 파일이 업로드되지 않은 경우 업로드 페이지 표시
    if df_global is None or df_global.empty:
        return render_template("upload.html")

    # CSV 파일이 업로드된 경우, 대시보드를 구성합니다.
    text_ids = sorted(df_global["text_id"].unique())
    selected_text_id = request.args.get("text_id", None)
    if selected_text_id is None and text_ids:
        selected_text_id = text_ids[0]
    else:
        try:
            selected_text_id = int(selected_text_id)
        except:
            selected_text_id = text_ids[0]

    text_data = None
    heatmap_img = None
    bar_chart_img = None
    prob_chart_img = None

    # 선택된 text_id에 해당하는 데이터 처리 (position 기준 정렬)
    text_df = df_global[df_global["text_id"] == selected_text_id].sort_values("position")
    if not text_df.empty:
        text_data = {
            "text": text_df.iloc[0]["text"],
            "tokens": text_df["token"].tolist(),
            "avg_importance": text_df["importance"].tolist()
        }
        # 토큰별 클래스 중요도 행렬 구성
        importance_cols = [col for col in df_global.columns if col.startswith("importance_class_")]
        importance_matrix = []
        for _, row in text_df.iterrows():
            row_vals = [row[col] for col in importance_cols]
            importance_matrix.append(row_vals)
        importance_matrix = np.array(importance_matrix).T  # (num_classes, num_tokens)
        text_data["importance_matrix"] = importance_matrix

        # 클래스 확률 (모든 토큰에 대해 동일하므로 첫 행 사용)
        prob_cols = [col for col in df_global.columns if col.startswith("prob_class_")]
        class_probs = [text_df.iloc[0][col] for col in prob_cols]
        text_data["class_probs"] = class_probs
        text_data["num_labels"] = len(prob_cols)
        text_data["class_names"] = CLASS_NAMES[:len(prob_cols)]

        # 토큰 리스트에서 앞의 언더바 제거 (▁)
        tokens = text_data["tokens"]
        tokens_clean = [token.lstrip("▁") for token in tokens]
        avg_importance = text_data["avg_importance"]
        num_labels = text_data["num_labels"]
        class_names = text_data["class_names"]

        # 1. 토큰별 클래스 중요도 히트맵 생성
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            importance_matrix,
            annot=True,
            fmt=".4f",
            cmap="YlOrRd",
            xticklabels=tokens_clean,
            yticklabels=class_names,
            ax=ax1
        )
        ax1.set_title("토큰별 클래스 중요도")
        heatmap_img = fig_to_base64(fig1)
        plt.close(fig1)

        # 2. 평균 토큰 중요도 바 차트 생성 (각 막대 위에 값 표시, 소수점 4자리)
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        # 상위 3개 토큰의 인덱스 구하기 (평균 importance가 큰 순)
        top3_indices = sorted(range(len(avg_importance)), key=lambda i: avg_importance[i], reverse=True)[:3]
        # 각 토큰에 대해 색상 지정: 상위 3개면 'orange', 아니면 기본 색상 'C0'
        colors_tokens = ['orange' if i in top3_indices else 'C0' for i in range(len(tokens_clean))]
        bars2 = ax2.bar(range(len(tokens_clean)), avg_importance, color=colors_tokens)
        ax2.set_xticks(range(len(tokens_clean)))
        ax2.set_xticklabels(tokens_clean, rotation=45, ha="right")
        ax2.set_xlabel("토큰")
        ax2.set_ylabel("평균 Importance")
        ax2.set_title("평균 토큰 중요도")
        # 각 막대 위에 평균 중요도 값을 소수점 4자리까지 표시
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                     f'{height:.4f}', ha='center', va='bottom')
        bar_chart_img = fig_to_base64(fig2)
        plt.close(fig2)

        # 3. 클래스 예측 확률 바 차트 생성 + 막대 위에 확률값 표시
        fig3, ax3 = plt.subplots(figsize=(12, 4))
        # 상위 3개 클래스를 구함 (예측 확률 기준)
        top3_class_indices = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)[:3]
        colors_classes = ['orange' if i in top3_class_indices else 'C0' for i in range(num_labels)]
        bars = ax3.bar(range(num_labels), class_probs, color=colors_classes)
        ax3.set_xticks(range(num_labels))
        ax3.set_xticklabels(class_names, rotation=45, ha="right")
        ax3.set_xlabel("클래스")
        ax3.set_ylabel("예측 확률")
        ax3.set_title("클래스 예측 확률")
        # 각 막대 위에 확률값(소수점 2자리) 표시
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}', ha='center', va='bottom')
        prob_chart_img = fig_to_base64(fig3)
        plt.close(fig3)

    return render_template("dashboard.html",
                           text_ids=text_ids,
                           selected_text_id=selected_text_id,
                           text_data=text_data,
                           heatmap_img=heatmap_img,
                           bar_chart_img=bar_chart_img,
                           prob_chart_img=prob_chart_img)

@app.route("/reset", methods=["GET"])
def reset():
    """새 CSV 파일 업로드를 위해 전역 변수 초기화 후 업로드 페이지로 이동"""
    global df_global
    df_global = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
