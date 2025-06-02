import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")

# CSV 파일 경로 (서인님 경로에 맞게 설정)
csv_path = "/home/minha/raspberrypi/imagenet/eval.csv"

# 데이터 불러오기
df = pd.read_csv(csv_path)

# 숫자형으로 변환 (혹시 문자열로 저장된 경우 대비)
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ------------------------
# 1. 막대그래프: Top-1 정확도
# ------------------------
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="model", y="top1_accuracy", palette="viridis")
plt.xticks(rotation=45)
plt.title("Top-1 Accuracy per Model")
plt.tight_layout()
plt.savefig("/home/minha/raspberrypi/imagenet/bar_accuracy.png")
plt.close()

# ------------------------
# 2. 레이더차트 (정규화된 값 기준)
# ------------------------
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

norm_df = df.copy()
metrics = ["top1_accuracy", "avg_confidence", "avg_inference_time_ms", "model_size_mb"]
norm_df[metrics] = norm_df[metrics].apply(normalize)

# 레이더차트 준비
labels = metrics
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

plt.figure(figsize=(8, 8))
for i, row in norm_df.iterrows():
    values = row[labels].tolist()
    values += values[:1]  # 닫기용
    plt.polar(angles, values, label=row['model'], alpha=0.4)

plt.xticks(angles[:-1], labels)
plt.title("Radar Chart: Normalized Metrics")
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
plt.savefig("/home/minha/raspberrypi/imagenet/radar_metrics.png")
plt.close()

# ------------------------
# 3. 산점도: 속도 vs 정확도
# ------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="avg_inference_time_ms", y="top1_accuracy", hue="model", s=100)
plt.title("Accuracy vs Inference Time")
plt.xlabel("Inference Time (ms)")
plt.ylabel("Top-1 Accuracy")
plt.tight_layout()
plt.savefig("/home/minha/raspberrypi/imagenet/scatter_accuracy_time.png")
plt.close()

# ------------------------
# 4. 히트맵 (전체 지표 비교)
# ------------------------
heat_df = df.set_index("model")[metrics]
plt.figure(figsize=(10, 6))
sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Metrics Heatmap")
plt.tight_layout()
plt.savefig("/home/minha/raspberrypi/imagenet/heatmap_metrics.png")
plt.close()

import ace_tools as tools; tools.display_dataframe_to_user(name="모델별 성능 데이터", dataframe=df)