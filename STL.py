import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# === 读取数据 ===
stl_df = pd.read_excel
stl_df["Time"] = pd.to_datetime(stl_df["Time"])

# === 分阶段时间点（基于8:1:1比例）===
total_len = len(stl_df)
train_end_idx = int(total_len * 0.8)
val_end_idx = int(total_len * 0.9)
train_end_date = stl_df["Time"].iloc[train_end_idx]
val_end_date = stl_df["Time"].iloc[val_end_idx]

# === 设置绘图字体 ===
plt.rcParams.update({
    'font.size': 26,
    'axes.titlesize': 26,
    'axes.labelsize': 26,
    'xtick.labelsize': 26,
    'ytick.labelsize': 26,
    'legend.fontsize': 22
})

# === 绘图参数 ===
components = ["Seasonal", "Noise", "Trend"]
colors = ["purple", "gray", "brown"]
titles = ["STL Seasonal Component", "STL Noise Component", "STL Trend Component"]
filenames = ["STL_Seasonal.png", "STL_Noise.png", "STL_Trend.png"]  # 可改为 .tif 或 .eps

# === 循环绘制并保存 ===
for comp, color, title, fname in zip(components, colors, titles, filenames):
    fig, ax = plt.subplots(figsize=(16, 6))

    # 背景分区：训练、验证、测试
    ax.axvspan(stl_df["Time"].min(), train_end_date, color='lightblue', alpha=0.3, label='Training')
    ax.axvspan(train_end_date, val_end_date, color='lightyellow', alpha=0.5, label='Validation')
    ax.axvspan(val_end_date, stl_df["Time"].max(), color='lightpink', alpha=0.5, label='Testing')

    # 绘制曲线
    ax.plot(stl_df["Time"], stl_df[comp], label=comp, color=color, linewidth=1.5)

    # 设置标题、坐标轴
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(f"{comp} (m³/s)")
    ax.set_xlim([stl_df["Time"].min(), stl_df["Time"].max()])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)

    # 图例和边框
    ax.legend(loc='upper left')
    for spine in ax.spines.values():
        spine.set_visible(True)

    # 保存图片
    plt.tight_layout(pad=0)
    plt.savefig(fname, dpi=300, bbox_inches='tight')  # 修改dpi或格式可调整精度
    plt.close()
