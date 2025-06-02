import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# 1) 로그가 저장된 CSV 파일 경로
csv_file = "/home/minha/raspberrypi/log_googlenet.csv"  # 실제 경로로 바꿔주세요

# 2) 첫 줄(skiprows=1)을 건너뛰고 읽기
df = pd.read_csv(csv_file, skiprows=1)

# 3) x축: frame 컬럼 사용
x = df["frame"]
xlabel = "frame"

# 4) 수치형 컬럼만 골라내기
metrics = [
    col for col in df.columns
    if col != xlabel and is_numeric_dtype(df[col])
]

# 5) 순환할 플롯 타입 및 스타일 정의
plot_types = ['line', 'scatter', 'bar']
styles = {
    'line':    {'func': plt.plot,    'kwargs': {}},
    'scatter': {'func': plt.scatter, 'kwargs': {'marker': 'o'}},
    'bar':     {'func': plt.bar,     'kwargs': {'width': 0.4}}
}

# 6) 차트 그리기
plt.figure()
for i, col in enumerate(metrics):
    ptype     = plot_types[i % len(plot_types)]
    plot_func = styles[ptype]['func']
    kwargs    = styles[ptype]['kwargs']

    if ptype == 'bar':
        plot_func(x, df[col], label=col, **kwargs)
    else:
        plot_func(x, df[col], label=col, **kwargs)

# 7) 레이아웃 설정 및 저장
plt.xlabel(xlabel)
plt.ylabel("value")
plt.title("Performance Metrics Over Frame")
plt.legend()
plt.tight_layout()
plt.savefig("performance_plot.png")
# plt.show()

print("Saved plot to performance_plot.png")