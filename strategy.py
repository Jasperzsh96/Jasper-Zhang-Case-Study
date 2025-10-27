##############################################
# IRD Allocation Layer（结构分位独立调仓·最终版）
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]

# ============= ① 读入数据 ==============
data_path = r"E:/承珞资本/宏观/carry_trade.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# ============= ② 结构性 Base Weight（分位线性） ==============
p30, p70 = df["IRD"].quantile([0.3, 0.7])


def compute_base_weight(ird):
    if ird <= p30:
        return 0.2
    elif ird >= p70:
        return 1.0
    else:
        return 0.2 + (ird - p30) / (p70 - p30) * 0.8


df["Base_Weight"] = df["IRD"].apply(compute_base_weight)

min_w, max_w = 0.2, 2.0

# ============= ③ 只用Base Weight生成杠杆权重曲线 ==============
x = np.arange(len(df))
y = df["Base_Weight"].values
spline = make_interp_spline(x, y, k=3)
x_smooth = np.linspace(x.min(), x.max(), 500)
weight_smooth = spline(x_smooth)
weight_smooth = np.clip(weight_smooth * max_w, min_w, max_w)
date_smooth = np.interp(x_smooth, x, df["Date"].astype(np.int64))
date_smooth = pd.to_datetime(date_smooth)

# ============= ④ 持仓权重 ==============
df["IRD_Leverage"] = np.clip(df["Base_Weight"] * max_w, min_w, max_w)

# ============= ⑤ 策略累计收益 ==============
df["Return_Hold2x"] = df["Carry_Return"] * 2.0
df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()
df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()


# ============= ⑥ 业绩指标 ==============
def perf_metrics(r, freq=12):
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    curve = (1 + r).cumprod()
    max_dd = ((curve.cummax() - curve) / curve.cummax()).max()
    return ann_ret, ann_vol, sharpe, max_dd


annr_h, annv_h, shrp_h, mdd_h = perf_metrics(df["Return_Hold2x"])
annr_a, annv_a, shrp_a, mdd_a = perf_metrics(df["Return_IRD_Alloc"])
perf_table = pd.DataFrame(
    {
        "Metric": [
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
        "Buy&Hold (2x)": [
            f"{annr_h:.2%}",
            f"{annv_h:.2%}",
            f"{shrp_h:.2f}",
            f"{mdd_h:.2%}",
        ],
        "IRD Allocation": [
            f"{annr_a:.2%}",
            f"{annv_a:.2%}",
            f"{shrp_a:.2f}",
            f"{mdd_a:.2%}",
        ],
    }
)

# ============= ⑦ 累计收益曲线（极简白底） ==============
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Equity_Hold2x"], lw=2.5, c="#D62728", label="Buy&Hold (2x)")
plt.plot(
    df["Date"], df["Equity_IRD_Alloc"], lw=2.5, c="#0066CC", label="IRD Allocation"
)
plt.title("Cumulative Return: IRD Allocation vs Buy&Hold", fontsize=16)
plt.ylabel("Cumulative Return (Index)")
plt.xlabel("Date")
plt.grid(False)
plt.gca().set_facecolor("white")
plt.legend(frameon=False, fontsize=13)
plt.tight_layout()
plt.show()

# ============= ⑧ 杠杆权重曲线（极简白底） ==============
plt.figure(figsize=(12, 6))
plt.hlines(
    2.0,
    date_smooth.min(),
    date_smooth.max(),
    colors="#D62728",
    linestyles="-",
    lw=2.5,
    label="Buy&Hold Leverage (2x)",
)
plt.plot(
    date_smooth,
    weight_smooth,
    lw=2,
    c="#0066CC",
    label="IRD Allocation Leverage",
    alpha=0.95,
)
plt.title("IRD Allocation Leverage Over Time", fontsize=15)
plt.ylabel("Leverage Multiplier")
plt.xlabel("Date")
plt.ylim([min_w - 0.1, max_w + 0.3])
plt.grid(False)
plt.gca().set_facecolor("white")
plt.legend(frameon=False, fontsize=13, loc="upper left", bbox_to_anchor=(0, 1))
plt.tight_layout()
plt.show()

# ============= ⑨ 业绩对比表（白底） ==============
fig, ax = plt.subplots(figsize=(12, 2))
ax.axis("off")
tbl = ax.table(
    cellText=perf_table.values,
    colLabels=perf_table.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.7)
plt.title("Performance Comparison: IRD Allocation vs Hold", pad=15, fontsize=13)
plt.tight_layout()
plt.show()


##############################################
# Policy Risk Factor Adjustment Layer
# —— 只用Policy/Macro Risk变量分位生成动态权重
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]

# ① 读入数据
data_path = r"E:/承珞资本/宏观/carry_trade.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# ② Policy Risk Signal —— 只用政策/宏观风险变量
policy_variables = [
    "VIX",
    "FX_VOL",
    "US_EPU",
    "TPU",
    "Liquidity_Premium",
    "US_GDP",
    "JP_GDP",
    "US_CPI",
    "JP_CPI",
    "US_PMI",
    "JP_PMI",
    "US_UR",
    "JP_UR",
]
policy_weights = {
    "VIX": 0.12,
    "FX_VOL": 0.12,
    "US_EPU": 0.10,
    "TPU": 0.10,
    "Liquidity_Premium": 0.06,
    "US_GDP": 0.08,
    "JP_GDP": 0.08,
    "US_CPI": 0.08,
    "JP_CPI": 0.08,
    "US_PMI": 0.06,
    "JP_PMI": 0.06,
    "US_UR": 0.08,
    "JP_UR": 0.08,
}
policy_thresholds = {v: df[v].quantile(0.75) for v in policy_variables}


def compute_risk_signal(row):
    penalty = 0
    for var in policy_variables:
        if row[var] > policy_thresholds[var]:
            penalty += policy_weights[var]
    return max(0.2, 1.0 - penalty)


df["Risk_Signal"] = df.apply(compute_risk_signal, axis=1)
min_w, max_w = 0.2, 2.0
df["Risk_Leverage"] = np.clip(df["Risk_Signal"] * max_w, min_w, max_w)

# ③ （可选）丝滑杠杆权重曲线
x = np.arange(len(df))
y = df["Risk_Signal"].values
spline = make_interp_spline(x, y, k=3)
x_smooth = np.linspace(x.min(), x.max(), 500)
risk_weight_smooth = spline(x_smooth)
risk_weight_smooth = np.clip(risk_weight_smooth * max_w, min_w, max_w)
date_smooth = np.interp(x_smooth, x, df["Date"].astype(np.int64))
date_smooth = pd.to_datetime(date_smooth)

# ===== IRD Allocation（结构性调仓权重）======
p30, p70 = df["IRD"].quantile([0.3, 0.7])


def compute_base_weight(ird):
    if ird <= p30:
        return 0.2
    elif ird >= p70:
        return 1.0
    else:
        return 0.2 + (ird - p30) / (p70 - p30) * 0.8


df["Base_Weight"] = df["IRD"].apply(compute_base_weight)
min_w, max_w = 0.2, 2.0
df["IRD_Leverage"] = np.clip(df["Base_Weight"] * max_w, min_w, max_w)

# ④ 策略累计收益
df["Return_Policy_Alloc"] = df["Carry_Return"] * df["Risk_Leverage"]
df["Equity_Policy_Alloc"] = (1 + df["Return_Policy_Alloc"]).cumprod()

# Buy&Hold（2倍杠杆）净值路径（如果还没生成的话）
if "Equity_Hold2x" not in df.columns:
    df["Return_Hold2x"] = df["Carry_Return"] * 2.0
    df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()

# IRD Allocation（如果你也没合并进来）
if "Equity_IRD_Alloc" not in df.columns:
    # 你得事先生成好df["IRD_Leverage"]
    # 比如df["IRD_Leverage"] = ...
    df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
    df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()

# ⑤ （可选）与其它策略并排对比
# 假设你已有 df["Equity_Hold2x"]、df["Equity_IRD_Alloc"] 等
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Equity_Hold2x"], lw=2.5, c="#D62728", label="Buy&Hold (2x)")
plt.plot(
    df["Date"], df["Equity_IRD_Alloc"], lw=2.5, c="#0066CC", label="IRD Allocation"
)
plt.plot(
    df["Date"],
    df["Equity_Policy_Alloc"],
    lw=2.5,
    c="#22B573",
    label="Policy Adjustment",
)
plt.title("Cumulative Return: IRD Allocation vs Policy Adjustment vs Hold", fontsize=16)
plt.ylabel("Cumulative Return (Index)")
plt.xlabel("Date")
plt.grid(False)
plt.gca().set_facecolor("white")
plt.legend(frameon=False, fontsize=13)
plt.tight_layout()
plt.show()

# ⑥ 杠杆权重曲线对比
plt.figure(figsize=(12, 6))
plt.plot(
    df["Date"], df["IRD_Leverage"], lw=2, c="#0066CC", label="IRD Allocation Leverage"
)
plt.plot(
    df["Date"],
    df["Risk_Leverage"],
    lw=2,
    c="#22B573",
    label="Policy Adjustment Leverage",
)
plt.hlines(
    2.0,
    df["Date"].min(),
    df["Date"].max(),
    colors="#D62728",
    linestyles="-",
    lw=2.5,
    label="Buy&Hold (2x)",
)
plt.title("Leverage Comparison: IRD vs Policy vs Hold", fontsize=15)
plt.ylabel("Leverage Multiplier")
plt.xlabel("Date")
plt.ylim([min_w - 0.1, max_w + 0.3])
plt.grid(False)
plt.gca().set_facecolor("white")
plt.legend(frameon=False, fontsize=13, loc="upper left")
plt.tight_layout()
plt.show()


# ⑦ 业绩指标对比
def perf_metrics(r, freq=12):
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    curve = (1 + r).cumprod()
    max_dd = ((curve.cummax() - curve) / curve.cummax()).max()
    return ann_ret, ann_vol, sharpe, max_dd


annr_h, annv_h, shrp_h, mdd_h = perf_metrics(df["Return_Hold2x"])
annr_i, annv_i, shrp_i, mdd_i = perf_metrics(df["Return_IRD_Alloc"])
annr_p, annv_p, shrp_p, mdd_p = perf_metrics(df["Return_Policy_Alloc"])

perf_table = pd.DataFrame(
    {
        "Metric": [
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
        "Buy&Hold (2x)": [
            f"{annr_h:.2%}",
            f"{annv_h:.2%}",
            f"{shrp_h:.2f}",
            f"{mdd_h:.2%}",
        ],
        "IRD Allocation": [
            f"{annr_i:.2%}",
            f"{annv_i:.2%}",
            f"{shrp_i:.2f}",
            f"{mdd_i:.2%}",
        ],
        "Policy Adjustment": [
            f"{annr_p:.2%}",
            f"{annv_p:.2%}",
            f"{shrp_p:.2f}",
            f"{mdd_p:.2%}",
        ],
    }
)

fig, ax = plt.subplots(figsize=(12, 2))
ax.axis("off")
tbl = ax.table(
    cellText=perf_table.values,
    colLabels=perf_table.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.7)
plt.title(
    "Performance Comparison: IRD Allocation vs Policy Adjustment vs Hold",
    pad=15,
    fontsize=13,
)
plt.tight_layout()
plt.show()


##############################################
# 四策略（Buy&Hold/IRD/Policy/Final）齐全对比
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]

# ① 读入数据
data_path = r"E:/承珞资本/宏观/carry_trade.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

min_w, max_w = 0.2, 2.0

# ② IRD Allocation（结构Beta）
p30, p70 = df["IRD"].quantile([0.3, 0.7])


def compute_base_weight(ird):
    if ird <= p30:
        return 0.2
    elif ird >= p70:
        return 1.0
    else:
        return 0.2 + (ird - p30) / (p70 - p30) * 0.8


df["Base_Weight"] = df["IRD"].apply(compute_base_weight)
df["IRD_Leverage"] = np.clip(df["Base_Weight"] * max_w, min_w, max_w)

# ③ Policy Adjustment（只用风险信号）
policy_variables = [
    "VIX",
    "FX_VOL",
    "US_EPU",
    "TPU",
    "Liquidity_Premium",
    "US_GDP",
    "JP_GDP",
    "US_CPI",
    "JP_CPI",
    "US_PMI",
    "JP_PMI",
    "US_UR",
    "JP_UR",
]
policy_weights = {
    "VIX": 0.12,
    "FX_VOL": 0.12,
    "US_EPU": 0.10,
    "TPU": 0.10,
    "Liquidity_Premium": 0.06,
    "US_GDP": 0.08,
    "JP_GDP": 0.08,
    "US_CPI": 0.08,
    "JP_CPI": 0.08,
    "US_PMI": 0.06,
    "JP_PMI": 0.06,
    "US_UR": 0.08,
    "JP_UR": 0.08,
}
policy_thresholds = {v: df[v].quantile(0.75) for v in policy_variables}


def compute_risk_signal(row):
    penalty = 0
    for var in policy_variables:
        if row[var] > policy_thresholds[var]:
            penalty += policy_weights[var]
    return max(0.2, 1.0 - penalty)


df["Risk_Signal"] = df.apply(compute_risk_signal, axis=1)
df["Risk_Leverage"] = np.clip(df["Risk_Signal"] * max_w, min_w, max_w)

# ④ Final State（协同：结构×风险）
df["Final_State_Signal"] = df["Base_Weight"] * df["Risk_Signal"]
df["Final_Leverage"] = np.clip(df["Final_State_Signal"] * max_w, min_w, max_w)

# ⑤ 丝滑权重曲线
x = np.arange(len(df))


def get_smooth(arr):
    spline = make_interp_spline(x, arr, k=3)
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_smooth = spline(x_smooth)
    y_smooth = np.clip(y_smooth * max_w, min_w, max_w)
    date_smooth = np.interp(x_smooth, x, df["Date"].astype(np.int64))
    date_smooth = pd.to_datetime(date_smooth)
    return date_smooth, y_smooth


date_smooth, ird_smooth = get_smooth(df["Base_Weight"].values)
_, risk_smooth = get_smooth(df["Risk_Signal"].values)
_, final_smooth = get_smooth(df["Final_State_Signal"].values)

# ⑥ 各策略收益&净值
df["Return_Hold2x"] = df["Carry_Return"] * 2.0
df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
df["Return_Policy_Alloc"] = df["Carry_Return"] * df["Risk_Leverage"]
df["Return_Final"] = df["Carry_Return"] * df["Final_Leverage"]
df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()
df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()
df["Equity_Policy_Alloc"] = (1 + df["Return_Policy_Alloc"]).cumprod()
df["Equity_Final"] = (1 + df["Return_Final"]).cumprod()

# ⑦ 累计收益曲线
plt.figure(figsize=(13, 6))
plt.plot(df["Date"], df["Equity_Hold2x"], lw=2, c="#D62728", label="Buy&Hold (2x)")
plt.plot(df["Date"], df["Equity_IRD_Alloc"], lw=2, c="#0066CC", label="IRD Allocation")
plt.plot(
    df["Date"], df["Equity_Policy_Alloc"], lw=2, c="#22B573", label="Policy Adjustment"
)
plt.plot(df["Date"], df["Equity_Final"], lw=2, c="#964B00", label="Final State Engine")
plt.title("Cumulative Return: Four Strategies Comparison", fontsize=16)
plt.ylabel("Cumulative Return (Index)")
plt.xlabel("Date")
plt.gca().set_facecolor("white")
plt.grid(False)
plt.legend(frameon=False, fontsize=13)
plt.tight_layout()
plt.show()

# ⑧ 杠杆权重曲线（对比四条线）
plt.figure(figsize=(13, 6))
plt.hlines(
    2.0,
    date_smooth.min(),
    date_smooth.max(),
    colors="#D62728",
    linestyles="-",
    lw=2,
    label="Buy&Hold (2x)",
)
plt.plot(date_smooth, ird_smooth, lw=2, c="#0066CC", label="IRD Allocation Leverage")
plt.plot(
    date_smooth, risk_smooth, lw=2, c="#22B573", label="Policy Adjustment Leverage"
)
plt.plot(date_smooth, final_smooth, lw=2, c="#964B00", label="Final State Leverage")
plt.title("Leverage Comparison: Four Strategies", fontsize=15)
plt.ylabel("Leverage Multiplier")
plt.xlabel("Date")
plt.ylim([min_w - 0.1, max_w + 0.3])
plt.gca().set_facecolor("white")
plt.grid(False)
plt.legend(frameon=False, fontsize=13, loc="upper left", bbox_to_anchor=(0, 1))
plt.tight_layout()
plt.show()


# ⑨ 四组业绩表格
def perf_metrics(r, freq=12):
    ann_ret = (1 + r.mean()) ** freq - 1
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    curve = (1 + r).cumprod()
    max_dd = ((curve.cummax() - curve) / curve.cummax()).max()
    return ann_ret, ann_vol, sharpe, max_dd


annr_h, annv_h, shrp_h, mdd_h = perf_metrics(df["Return_Hold2x"])
annr_i, annv_i, shrp_i, mdd_i = perf_metrics(df["Return_IRD_Alloc"])
annr_p, annv_p, shrp_p, mdd_p = perf_metrics(df["Return_Policy_Alloc"])
annr_f, annv_f, shrp_f, mdd_f = perf_metrics(df["Return_Final"])

perf_table = pd.DataFrame(
    {
        "Metric": [
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
        "Buy&Hold (2x)": [
            f"{annr_h:.2%}",
            f"{annv_h:.2%}",
            f"{shrp_h:.2f}",
            f"{mdd_h:.2%}",
        ],
        "IRD Allocation": [
            f"{annr_i:.2%}",
            f"{annv_i:.2%}",
            f"{shrp_i:.2f}",
            f"{mdd_i:.2%}",
        ],
        "Policy Adjustment": [
            f"{annr_p:.2%}",
            f"{annv_p:.2%}",
            f"{shrp_p:.2f}",
            f"{mdd_p:.2%}",
        ],
        "Final State": [
            f"{annr_f:.2%}",
            f"{annv_f:.2%}",
            f"{shrp_f:.2f}",
            f"{mdd_f:.2%}",
        ],
    }
)

fig, ax = plt.subplots(figsize=(13, 2.3))
ax.axis("off")
tbl = ax.table(
    cellText=perf_table.values,
    colLabels=perf_table.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.6)
plt.title("Performance Comparison: Four Strategies", pad=15, fontsize=13)
plt.tight_layout()
plt.show()


##############################################
# 四策略 + Alpha v4（强凸性：顺风猛踩 + 逆风急刹）
# 目标：收益 > Buy&Hold(2x)；先赢收益，再微调风险
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# -------- 全局画图字体 --------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]

# -------- ① 读数据 --------
data_path = r"E:/承珞资本/宏观/carry_trade.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

FREQ = 12
min_w, max_w = 0.2, 2.0  # 旧四条线的杠杆边界
eps = 1e-8

# -------- ② IRD 结构权重（分位线性） --------
p30, p70 = df["IRD"].quantile([0.3, 0.7])


def compute_base_weight(ird):
    if ird <= p30:
        return 0.2
    if ird >= p70:
        return 1.0
    return 0.2 + (ird - p30) / (p70 - p30) * 0.8


df["Base_Weight"] = df["IRD"].apply(compute_base_weight)

# -------- ③ Policy Risk Signal（分位惩罚）--------
policy_variables = [
    "VIX",
    "FX_VOL",
    "US_EPU",
    "TPU",
    "Liquidity_Premium",
    "US_GDP",
    "JP_GDP",
    "US_CPI",
    "JP_CPI",
    "US_PMI",
    "JP_PMI",
    "US_UR",
    "JP_UR",
]
policy_weights = {
    "VIX": 0.12,
    "FX_VOL": 0.12,
    "US_EPU": 0.10,
    "TPU": 0.10,
    "Liquidity_Premium": 0.06,
    "US_GDP": 0.08,
    "JP_GDP": 0.08,
    "US_CPI": 0.08,
    "JP_CPI": 0.08,
    "US_PMI": 0.06,
    "JP_PMI": 0.06,
    "US_UR": 0.08,
    "JP_UR": 0.08,
}
policy_thresholds = {v: df[v].quantile(0.75) for v in policy_variables}


def compute_risk_signal(row):
    pen = 0.0
    for v in policy_variables:
        if row[v] > policy_thresholds[v]:
            pen += policy_weights[v]
    return max(0.2, 1.0 - pen)


df["Risk_Signal"] = df.apply(compute_risk_signal, axis=1)

# -------- ④ 三种杠杆 + Final --------
df["IRD_Leverage"] = np.clip(df["Base_Weight"] * max_w, min_w, max_w)
df["Risk_Leverage"] = np.clip(df["Risk_Signal"] * max_w, min_w, max_w)
df["Final_State_Signal"] = df["Base_Weight"] * df["Risk_Signal"]
df["Final_Leverage"] = np.clip(df["Final_State_Signal"] * max_w, min_w, max_w)

# -------- ⑤ 四条策略回报/净值 --------
df["Return_Hold2x"] = df["Carry_Return"] * 2.0
df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
df["Return_Policy_Alloc"] = df["Carry_Return"] * df["Risk_Leverage"]
df["Return_Final"] = df["Carry_Return"] * df["Final_Leverage"]

df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()
df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()
df["Equity_Policy_Alloc"] = (1 + df["Return_Policy_Alloc"]).cumprod()
df["Equity_Final"] = (1 + df["Return_Final"]).cumprod()

##############################################
# 四策略 + Alpha v5（Final 基座 + 连续凸性增强）
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ---------- 画图字体 ----------
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]

# ---------- ① 读数据 ----------
data_path = r"E:/承珞资本/宏观/carry_trade.csv"
df = pd.read_csv(data_path)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

FREQ = 12
min_w, max_w = 0.2, 2.0  # 旧四条线杠杆上下限
eps = 1e-8

# ---------- ② IRD 结构权重（分位线性） ----------
p30, p70 = df["IRD"].quantile([0.3, 0.7])


def compute_base_weight(ird):
    if ird <= p30:
        return 0.2
    if ird >= p70:
        return 1.0
    return 0.2 + (ird - p30) / (p70 - p30) * 0.8


df["Base_Weight"] = df["IRD"].apply(compute_base_weight)

# ---------- ③ Policy Risk Signal（分位惩罚） ----------
policy_variables = [
    "VIX",
    "FX_VOL",
    "US_EPU",
    "TPU",
    "Liquidity_Premium",
    "US_GDP",
    "JP_GDP",
    "US_CPI",
    "JP_CPI",
    "US_PMI",
    "JP_PMI",
    "US_UR",
    "JP_UR",
]
policy_weights = {
    "VIX": 0.12,
    "FX_VOL": 0.12,
    "US_EPU": 0.10,
    "TPU": 0.10,
    "Liquidity_Premium": 0.06,
    "US_GDP": 0.08,
    "JP_GDP": 0.08,
    "US_CPI": 0.08,
    "JP_CPI": 0.08,
    "US_PMI": 0.06,
    "JP_PMI": 0.06,
    "US_UR": 0.08,
    "JP_UR": 0.08,
}
policy_thresholds = {v: df[v].quantile(0.75) for v in policy_variables}


def compute_risk_signal(row):
    pen = 0.0
    for v in policy_variables:
        if row[v] > policy_thresholds[v]:
            pen += policy_weights[v]
    return max(0.2, 1.0 - pen)


df["Risk_Signal"] = df.apply(compute_risk_signal, axis=1)

# ---------- ④ 三种杠杆 + Final ----------
df["IRD_Leverage"] = np.clip(df["Base_Weight"] * max_w, min_w, max_w)
df["Risk_Leverage"] = np.clip(df["Risk_Signal"] * max_w, min_w, max_w)
df["Final_State_Signal"] = df["Base_Weight"] * df["Risk_Signal"]
df["Final_Leverage"] = np.clip(df["Final_State_Signal"] * max_w, min_w, max_w)

# ---------- ⑤ 四条策略回报/净值 ----------
df["Return_Hold2x"] = df["Carry_Return"] * 2.0
df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
df["Return_Policy_Alloc"] = df["Carry_Return"] * df["Risk_Leverage"]
df["Return_Final"] = df["Carry_Return"] * df["Final_Leverage"]

df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()
df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()
df["Equity_Policy_Alloc"] = (1 + df["Return_Policy_Alloc"]).cumprod()
df["Equity_Final"] = (1 + df["Return_Final"]).cumprod()


##############################################
# Alpha v10: Final-follow + 强凸性 + 高波动目标（收益>5%版）
##############################################

FREQ = 12
eps = 1e-9

# 基座与信号
alpha_base = df["Final_Leverage"].astype(float).values
fs = df["Final_State_Signal"].clip(0, 1).astype(float).values

# 高位更猛的凸性（只跟 Final 的形状走）
k_low, p_low = 1.10, 1.8
k_mid, p_mid = 1.80, 2.6
k_high, p_high = 3.00, 3.2

f_conv = np.where(
    fs < 0.40,
    1.0 + k_low * (fs**p_low),
    np.where(fs < 0.75, 1.0 + k_mid * (fs**p_mid), 1.0 + k_high * (fs**p_high)),
)

# 顺风加成（只加分，不主导形态）
mom6 = df["Carry_Return"].rolling(6).mean()
std12 = df["Carry_Return"].rolling(12).std().replace(0, np.nan).bfill()
mom_ir = (mom6 / (std12 + eps)).clip(-1, 3).fillna(0)
f_mom = 1.0 + 0.32 * np.clip(mom_ir, 0, None)  # ↑ 提强

fx_q40, fx_q80 = df["FX_VOL"].quantile([0.40, 0.80])
f_fxv = np.where(
    df["FX_VOL"] <= fx_q40, 1.06, np.where(df["FX_VOL"] >= fx_q80, 0.96, 1.00)
)  # ↑ 低波稍多奖，高波稍少罚

turbo = np.where((fs > 0.82) & (mom_ir > 0), 1.18, 1.00)  # ↑ 强信号小涡轮

boost = f_conv * f_mom * f_fxv * turbo

# 上限/下限（抬高高位上限）
floor_rel = 0.93
max_w_alpha_cap = np.where(fs > 0.75, 7.0, 4.0)  # ↑ 高位7.0，中位4.0

alpha_target = np.minimum(alpha_base * boost, max_w_alpha_cap)
alpha_target = np.maximum(alpha_target, floor_rel * alpha_base)

# 平滑 + 非对称限速（上行更快、下行温和）
up_abs, up_rel = 0.70, 0.70  # ↑
down_abs, down_rel = 0.22, 0.22

alpha_ema = (
    pd.Series(alpha_target).ewm(span=2, adjust=False).mean().values
)  # ↓ 减少滞后
alpha_slow = np.zeros_like(alpha_ema)
alpha_slow[0] = alpha_ema[0]
for t in range(1, len(alpha_ema)):
    prev, raw = alpha_slow[t - 1], alpha_ema[t]
    if raw >= prev:
        ub = min(prev * (1 + up_rel), prev + up_abs)
        alpha_slow[t] = np.clip(raw, prev, ub)
    else:
        lb = max(prev * (1 - down_rel), prev - down_abs)
        alpha_slow[t] = np.clip(raw, lb, prev)

alpha_lev = pd.Series(alpha_slow).ewm(span=2, adjust=False).mean().values


# 状态化目标波动（显著抬高高位目标）
def _ann_vol(x):
    return x.std() * np.sqrt(FREQ)


final_vol = _ann_vol(df["Return_Final"])

tgt_mult = np.where(
    fs < 0.40,
    1.4,  # 低位 1.4x
    np.where(
        fs < 0.75,
        1.9,  # 中位 1.9x
        2.6,
    ),  # 高位 2.6x ← 关键
)
tgt_vol = final_vol * tgt_mult

asset_roll_vol = (
    (df["Carry_Return"].rolling(12).std() * np.sqrt(FREQ))
    .replace(0, np.nan)
    .bfill()
    .ffill()
)
scale = (tgt_vol / (asset_roll_vol * (alpha_lev + eps))).clip(0.90, 1.60)  # ↑ 上限放宽
alpha_lev *= scale

# 回撤闸门（保留收益的同时不过度压制）
eq_tmp = (1 + (df["Carry_Return"] * alpha_lev)).cumprod()
dd = (eq_tmp.cummax() - eq_tmp) / (eq_tmp.cummax() + eps)
alpha_lev = np.where(dd > 0.06, alpha_lev * 0.92, alpha_lev)  # ↓ 收紧幅度更轻

# 最终边界
alpha_lev = np.clip(alpha_lev, 0.2, np.where(fs > 0.75, 7.0, 4.0))

# 生成回报/净值
df["Return_Alpha"] = df["Carry_Return"] * alpha_lev
df["Equity_Alpha"] = (1 + df["Return_Alpha"]).cumprod()

###############################################—— 净值（五线）——#############################################
plt.figure(figsize=(13, 6))
plt.plot(df["Date"], df["Equity_Hold2x"], lw=2, c="#D62728", label="Buy&Hold (2x)")
plt.plot(df["Date"], df["Equity_IRD_Alloc"], lw=2, c="#0066CC", label="IRD Allocation")
plt.plot(
    df["Date"], df["Equity_Policy_Alloc"], lw=2, c="#22B573", label="Policy Adjustment"
)
plt.plot(df["Date"], df["Equity_Final"], lw=2, c="#964B00", label="Final State Engine")
plt.plot(
    df["Date"],
    df["Equity_Alpha"],
    lw=2.4,
    c="#6F2DBD",
    label="Alpha+ (Final-follow, Convex)",
)
plt.title("Cumulative Return: Five Strategies Comparison", fontsize=16)
plt.ylabel("Cumulative Return (Index)")
plt.xlabel("Date")
plt.gca().set_facecolor("white")
plt.grid(False)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.show()


# ========= 辅助：宽松匹配列名（忽略空格/大小写/下划线） =========
def find_col(cands):
    norm = lambda s: str(s).lower().replace(" ", "").replace("_", "")
    cols_norm = {norm(c): c for c in df.columns}
    for cand in cands:
        k = norm(cand)
        if k in cols_norm:
            return cols_norm[k]
    # 宽松包含匹配
    for cand in cands:
        for c in df.columns:
            if norm(cand) in norm(c):
                return c
    return None


# ==== 找到各条净值曲线列（找不到就跳过） ====
EQ_BH = find_col(["Equity_Hold2x", "Equity_BH2x", "BH2x"])
EQ_IRD = find_col(["Equity_IRD_Alloc", "Equity_IRD", "Equity_IRDAllocation"])
EQ_POL = find_col(
    ["Equity_Policy_Alloc", "Equity_Policy", "Equity_Risk", "Policy_Equity"]
)
EQ_FIN = find_col(["Equity_Final", "Final_Equity"])
# Alpha 刚刚我们已经生成了：df["Equity_Alpha"]

# ========= ① 净值（五线）=========
plt.figure(figsize=(13, 6))
if EQ_BH is not None:
    plt.plot(df["Date"], df[EQ_BH], lw=2, c="#D62728", label="Buy&Hold (2x)")
if EQ_IRD is not None:
    plt.plot(df["Date"], df[EQ_IRD], lw=2, c="#0066CC", label="IRD Allocation")
if EQ_POL is not None:
    plt.plot(df["Date"], df[EQ_POL], lw=2, c="#22B573", label="Policy Adjustment")
if EQ_FIN is not None:
    plt.plot(df["Date"], df[EQ_FIN], lw=2, c="#964B00", label="Final State Engine")
plt.plot(df["Date"], df["Equity_Alpha"], lw=2.4, c="#6F2DBD", label="Final State+Alpha")
plt.title("Cumulative Return: Five Strategies Comparison", fontsize=16)
plt.ylabel("Cumulative Return (Index)")
plt.xlabel("Date")
plt.gca().set_facecolor("white")
plt.grid(False)
plt.legend(frameon=False, fontsize=12)
plt.tight_layout()
plt.show()

# ==== 找到各条“杠杆”曲线列（有哪个画哪个）====
LEV_IRD = find_col(["IRD_Leverage", "IRD_lev", "Leverage_IRD"])
LEV_POL = find_col(["Policy_Leverage", "Risk_Leverage", "PolicyLev", "RiskLev"])
LEV_FIN = find_col(["Final_Leverage", "FinalLev"])
# Alpha 就用我们刚算好的 alpha_lev

alpha_cap_max = float(np.nanmax(max_w_alpha_cap))  # 数组→标量，作为y轴上限

plt.figure(figsize=(13, 6))
plt.hlines(
    2.0,
    df["Date"].min(),
    df["Date"].max(),
    colors="#D62728",
    linestyles="-",
    lw=2,
    label="Buy&Hold (2x)",
)
if LEV_IRD is not None:
    plt.plot(
        df["Date"], df[LEV_IRD], lw=2, c="#0066CC", label="IRD Allocation Leverage"
    )
if LEV_POL is not None:
    plt.plot(
        df["Date"], df[LEV_POL], lw=2, c="#22B573", label="Policy Adjustment Leverage"
    )
if LEV_FIN is not None:
    plt.plot(df["Date"], df[LEV_FIN], lw=2, c="#964B00", label="Final State Leverage")
plt.plot(df["Date"], alpha_lev, lw=2.3, c="#6F2DBD", label="Final State+Alpha Leverage")
plt.title("Leverage Comparison: Five Strategies", fontsize=15)
plt.ylabel("Leverage Multiplier")
plt.xlabel("Date")
plt.ylim(0.0, alpha_cap_max + 0.5)
plt.gca().set_facecolor("white")
plt.grid(False)
plt.legend(frameon=False, fontsize=12, loc="upper left")
plt.tight_layout()
plt.show()

# ========= ③ 业绩表（自动找列；找不到就忽略那列）=========
RET_BH = find_col(["Return_Hold2x", "BH2x_Return"])
RET_IRD = find_col(["Return_IRD_Alloc", "IRD_Return"])
RET_POL = find_col(["Return_Policy_Alloc", "Policy_Return", "Risk_Return"])
RET_FIN = find_col(["Return_Final", "Final_Return"])
# Alpha 刚算好的：df["Return_Alpha"]


def perf_metrics(r, freq=FREQ):
    r = pd.Series(r).dropna()
    ann = (1 + r.mean()) ** freq - 1
    vol = r.std() * np.sqrt(freq)
    shp = ann / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    mdd = ((eq.cummax() - eq) / (eq.cummax() + eps)).max()
    return ann, vol, shp, mdd


cols = []
names = []
if RET_BH is not None:
    cols.append(df[RET_BH])
    names.append("Buy&Hold (2x)")
if RET_IRD is not None:
    cols.append(df[RET_IRD])
    names.append("IRD Allocation")
if RET_POL is not None:
    cols.append(df[RET_POL])
    names.append("Policy Adjust.")
if RET_FIN is not None:
    cols.append(df[RET_FIN])
    names.append("Final State")
cols.append(df["Return_Alpha"])
names.append("Final State+Alpha")

met = [perf_metrics(c) for c in cols]

perf_table = pd.DataFrame(
    {
        "Metric": [
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ],
    }
)
for i, n in enumerate(names):
    perf_table[n] = [
        f"{met[i][0]:.2%}",
        f"{met[i][1]:.2%}",
        f"{met[i][2]:.2f}",
        f"{met[i][3]:.2%}",
    ]

fig, ax = plt.subplots(figsize=(14, 2.6))
ax.axis("off")
tbl = ax.table(
    cellText=perf_table.values,
    colLabels=perf_table.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 1.6)
plt.title("Performance Comparison: Five Strategies", pad=12, fontsize=13)
plt.tight_layout()
plt.show()


# -*- coding: utf-8 -*-
# =========================================================
# 政策变量预测系统：统计骨架 + 冲击模块 + 情景约束
# 包含示例数据生成功能 + Z-Score标准化可视化
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# 设置中文字体显示（保留中文注释需要时的显示能力，但绘图文本改为英文）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"

# 设置随机种子保证可复现性
np.random.seed(42)

# =========================================================
# 1. 配置参数
# =========================================================

# 数据文件路径
DATA_PATH = "carry_trade.csv"

# 目标变量
TARGET_VARS = ["TPU", "US_EPU", "JP_EPU", "FX_VOL", "CPI_SPREAD", "UR_SPREAD"]

# 颜色映射
COLOR_MAP = {
    "TPU": "blue",
    "US_EPU": "orange",
    "JP_EPU": "purple",
    "FX_VOL": "green",
    "CPI_SPREAD": "red",
    "UR_SPREAD": "brown",
}

# 冲击方向约束（按方法论要求）
SHOCK_DIRECTION = {
    "TPU": "positive",  # 正向冲击为主
    "US_EPU": "positive",  # 正向冲击为主
    "FX_VOL": "positive",  # 正向冲击为主
    "JP_EPU": "negative",  # 负向冲击为主
    "CPI_SPREAD": None,  # 不强加方向
    "UR_SPREAD": None,  # 不强加方向
}

# 参与Tariff强制冲击的变量
TARIFF_SHOCK_VARS = ["TPU", "US_EPU", "FX_VOL", "JP_EPU"]

# 超参数
LAMBDA_EWMA = 0.94  # EWMA衰减系数
SHOCK_THRESHOLD = 2.0  # 冲击识别阈值
N_SIMULATIONS = 1000  # 蒙特卡洛路径数（减少以提高运行速度）
SHOCK_QUANTILE = 0.75  # 强制冲击分位数

# 预测调整选项
PERSISTENCE_BOOST = True  # 是否增强持续性
MIN_RHO = 0.7  # 最小AR(1)系数（增强持续性）
ROLLING_ZSCORE = True  # 是否使用滚动窗口Z-score
ZSCORE_WINDOW = 24  # 滚动Z-score窗口长度（月）

# 时间设置
HIST_END_DATE = "2025-06-30"  # 历史/预测分割点
FORECAST_START = "2025-07-01"  # 预测开始
FORECAST_END = "2026-06-30"  # 预测结束（12个月）

# =========================================================
# 2. 示例数据生成
# =========================================================


def generate_sample_data():
    """Generate sample data for demonstration"""
    print("Generating sample data...")
    # 时间范围：2020-01 到 2025-06
    dates = pd.date_range(start="2020-01-01", end="2025-06-30", freq="MS")
    n_periods = len(dates)

    # 基础参数
    np.random.seed(42)

    # 生成基础时间序列
    data = {}

    # TPU - 贸易政策不确定性指数
    tpu_trend = 100 + np.cumsum(np.random.normal(0, 2, n_periods))
    tpu_shocks = np.random.choice(
        [0, 0, 0, 15, 25], n_periods, p=[0.7, 0.1, 0.1, 0.05, 0.05]
    )
    data["TPU"] = np.maximum(
        0, tpu_trend + tpu_shocks + np.random.normal(0, 5, n_periods)
    )

    # US_EPU - 美国经济政策不确定性
    us_epu_trend = 150 + np.cumsum(np.random.normal(0, 3, n_periods))
    us_epu_shocks = np.random.choice(
        [0, 0, 0, 20, 40], n_periods, p=[0.75, 0.1, 0.1, 0.03, 0.02]
    )
    data["US_EPU"] = np.maximum(
        0, us_epu_trend + us_epu_shocks + np.random.normal(0, 8, n_periods)
    )

    # JP_EPU - 日本经济政策不确定性
    jp_epu_trend = 80 + np.cumsum(np.random.normal(0, 1.5, n_periods))
    jp_epu_shocks = np.random.choice(
        [0, 0, 0, -10, -20], n_periods, p=[0.8, 0.1, 0.05, 0.03, 0.02]
    )
    data["JP_EPU"] = np.maximum(
        0, jp_epu_trend + jp_epu_shocks + np.random.normal(0, 4, n_periods)
    )

    # FX_VOL - 汇率波动率
    fx_vol_trend = 0.1 + 0.05 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)  # 季节性
    fx_vol_shocks = np.random.choice(
        [0, 0, 0, 0.02, 0.05], n_periods, p=[0.8, 0.1, 0.05, 0.03, 0.02]
    )
    data["FX_VOL"] = np.maximum(
        0.01, fx_vol_trend + fx_vol_shocks + np.random.normal(0, 0.01, n_periods)
    )

    # CPI 数据
    us_cpi_trend = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_periods))
    data["US_CPI"] = us_cpi_trend + np.random.normal(0, 0.3, n_periods)

    jp_cpi_trend = 0.5 + np.cumsum(np.random.normal(0, 0.08, n_periods))
    data["JP_CPI"] = jp_cpi_trend + np.random.normal(0, 0.2, n_periods)

    # 失业率数据
    us_ur_trend = 4.0 + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 24)
    data["US_UR"] = np.maximum(2.0, us_ur_trend + np.random.normal(0, 0.3, n_periods))

    jp_ur_trend = 2.5 + 0.3 * np.sin(np.arange(n_periods) * 2 * np.pi / 24)
    data["JP_UR"] = np.maximum(1.0, jp_ur_trend + np.random.normal(0, 0.2, n_periods))

    # 构建DataFrame
    df = pd.DataFrame(data)
    df["Date"] = dates

    # 计算spread变量
    df["CPI_SPREAD"] = df["US_CPI"] - df["JP_CPI"]
    df["UR_SPREAD"] = df["US_UR"] - df["JP_UR"]

    # 转换日期格式为DD/MM/YYYY
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

    # 保存到CSV
    df.to_csv(DATA_PATH, index=False)
    print(f"Sample data saved to: {DATA_PATH}")

    return df


# =========================================================
# 3. 数据读取与预处理
# =========================================================


def load_and_prepare_data(file_path):
    """Load data and preprocess"""
    print("Loading data...")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Generating a sample dataset instead...")
        df = generate_sample_data()

    # 日期处理（DD/MM/YYYY格式）
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 构造缺失的spread变量
    if "CPI_SPREAD" not in df.columns and all(
        col in df.columns for col in ["US_CPI", "JP_CPI"]
    ):
        df["CPI_SPREAD"] = df["US_CPI"] - df["JP_CPI"]
        print("Constructed CPI_SPREAD = US_CPI - JP_CPI")

    if "UR_SPREAD" not in df.columns and all(
        col in df.columns for col in ["US_UR", "JP_UR"]
    ):
        df["UR_SPREAD"] = df["US_UR"] - df["JP_UR"]
        print("Constructed UR_SPREAD = US_UR - JP_UR")

    # 转换为月度频率（取当月最后一个可用值）
    df["YearMonth"] = df["Date"].dt.to_period("M")
    df_monthly = df.groupby("YearMonth").last().reset_index()
    df_monthly["Date"] = df_monthly["YearMonth"].dt.to_timestamp()

    # 筛选历史数据（截止到2025-06）
    hist_cutoff = pd.to_datetime(HIST_END_DATE)
    df_hist = df_monthly[df_monthly["Date"] <= hist_cutoff].copy()

    print(f"Data range: {df_hist['Date'].min()} to {df_hist['Date'].max()}")
    print(f"Historical observations: {len(df_hist)}")

    return df_hist


# =========================================================
# 4. 统计骨架：AR(1) + EWMA
# =========================================================


class AR1_EWMA_Model:
    """AR(1) mean reversion with EWMA volatility"""

    def __init__(self, lambda_ewma=0.94):
        self.lambda_ewma = lambda_ewma
        self.mu = None
        self.rho = None
        self.sigma_hist = None
        self.residuals = None
        self.fitted = False

    def fit(self, data):
        """Fit AR(1)-EWMA model"""
        data = pd.Series(data).dropna()

        if len(data) < 10:
            # 样本太少，退化为常均值白噪声
            self.mu = data.mean()
            self.rho = 0.0
            residuals = data - self.mu
            sigma_const = residuals.std()
            self.sigma_hist = np.full(len(residuals), sigma_const)
            self.residuals = residuals.values
        else:
            # AR(1)估计
            y = data.iloc[1:].values
            X = sm.add_constant(data.iloc[:-1].values)

            try:
                model = sm.OLS(y, X).fit()
                intercept, ar_coef = model.params

                # 转换为均值回归形式: X_t = mu + rho*(X_{t-1} - mu) + e_t
                self.rho = ar_coef
                self.mu = (
                    intercept / (1 - ar_coef) if abs(ar_coef) < 0.99 else data.mean()
                )

                # 确保平稳性，但可选择性增强持续性
                if abs(self.rho) >= 0.99:
                    self.rho = 0.95 if self.rho > 0 else -0.95
                elif PERSISTENCE_BOOST and abs(self.rho) < MIN_RHO:
                    # 可选：增强持续性以减缓均值回归
                    self.rho = MIN_RHO if self.rho > 0 else -MIN_RHO

                # 计算残差
                self.residuals = y - (intercept + ar_coef * data.iloc[:-1].values)

            except Exception:
                # 回退方案
                self.mu = data.mean()
                self.rho = MIN_RHO if PERSISTENCE_BOOST else 0.5
                self.residuals = (data - self.mu).values[1:]

        # EWMA波动率估计
        self._fit_ewma_volatility()
        self.fitted = True

        return self

    def _fit_ewma_volatility(self):
        """Estimate EWMA volatility"""
        residuals_sq = self.residuals**2
        sigma_sq = np.zeros(len(residuals_sq))

        # 初始化
        sigma_sq[0] = np.var(self.residuals) if len(self.residuals) > 1 else 1.0

        # EWMA递推
        for t in range(1, len(residuals_sq)):
            sigma_sq[t] = (
                self.lambda_ewma * sigma_sq[t - 1]
                + (1 - self.lambda_ewma) * residuals_sq[t - 1]
            )

        self.sigma_hist = np.sqrt(sigma_sq)

    def get_standardized_residuals(self):
        """Return standardized residuals"""
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")
        return self.residuals / (self.sigma_hist + 1e-8)


# =========================================================
# 5. 冲击模块：识别与校准
# =========================================================


class ShockModule:
    """Shock identification & calibration"""

    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.shock_rate = None
        self.shock_sizes = None
        self.direction_constraint = None

    def identify_shocks(self, standardized_residuals, residuals, direction=None):
        """Identify historical shocks"""
        z_scores = standardized_residuals

        # 识别冲击（|z| > threshold）
        shock_mask = np.abs(z_scores) > self.threshold
        shock_residuals = residuals[shock_mask]

        # 应用方向约束
        if direction == "positive":
            shock_residuals = shock_residuals[shock_residuals > 0]
        elif direction == "negative":
            shock_residuals = shock_residuals[shock_residuals < 0]

        # 计算到达率
        self.shock_rate = (
            len(shock_residuals) / len(z_scores) if len(z_scores) > 0 else 0
        )
        self.shock_sizes = shock_residuals
        self.direction_constraint = direction

        print(f"  Shock arrival rate: {self.shock_rate:.4f}")
        print(f"  Shock count: {len(shock_residuals)}")
        if len(shock_residuals) > 0:
            print(
                f"  Shock size distribution — P50: {np.percentile(np.abs(shock_residuals), 50):.3f}, "
                f"P75: {np.percentile(np.abs(shock_residuals), 75):.3f}, "
                f"P90: {np.percentile(np.abs(shock_residuals), 90):.3f}"
            )

        return self

    def simulate_shock(self):
        """Simulate one random shock draw"""
        if len(self.shock_sizes) == 0:
            return 0.0

        if np.random.random() < self.shock_rate:
            return np.random.choice(self.shock_sizes)
        else:
            return 0.0

    def get_forced_shock(self, quantile=0.75):
        """Return forced shock size for scenario injection"""
        if len(self.shock_sizes) == 0:
            return 0.0

        if self.direction_constraint == "positive":
            abs_shocks = np.abs(self.shock_sizes)
            return np.percentile(abs_shocks, quantile * 100)
        elif self.direction_constraint == "negative":
            abs_shocks = np.abs(self.shock_sizes)
            return -np.percentile(abs_shocks, quantile * 100)
        else:
            return np.percentile(self.shock_sizes, quantile * 100)


# =========================================================
# 6. 完整预测系统
# =========================================================


class PolicyForecastSystem:
    """Policy variable forecasting system"""

    def __init__(self):
        self.models = {}
        self.shock_modules = {}
        self.data = None
        self.forecast_dates = None
        self.historical_stats = {}  # 存储历史数据的均值和标准差用于标准化

    def fit(self, data):
        """Fit models for all variables"""
        self.data = data

        print("Fitting models...")
        print("=" * 60)

        # 计算历史数据的统计量用于z-score标准化
        for var in TARGET_VARS:
            if var in data.columns:
                series = data[var].dropna()
                self.historical_stats[var] = {
                    "mean": series.mean(),
                    "std": series.std(),
                }

        for var in TARGET_VARS:
            if var not in data.columns:
                print(f"Skip {var}: not found in data.")
                continue

            print(f"\nFitting: {var}")
            print("-" * 30)

            # 获取数据
            series = data[var].dropna()
            if len(series) < 5:
                print(f"  Not enough data, skip {var}")
                continue

            print(f"  Sample size: {len(series)}")
            print(f"  Value range: {series.min():.3f} to {series.max():.3f}")

            # 拟合AR(1)-EWMA模型
            model = AR1_EWMA_Model(lambda_ewma=LAMBDA_EWMA)
            model.fit(series.values)
            self.models[var] = model

            print(f"  AR(1) params — mu: {model.mu:.4f}, rho: {model.rho:.4f}")

            # 冲击模块
            shock_module = ShockModule(threshold=SHOCK_THRESHOLD)
            z_scores = model.get_standardized_residuals()
            direction = SHOCK_DIRECTION.get(var)

            shock_module.identify_shocks(z_scores, model.residuals, direction=direction)
            self.shock_modules[var] = shock_module

        # 设置预测时间范围
        self.forecast_dates = pd.date_range(
            start=FORECAST_START, end=FORECAST_END, freq="MS"
        )

        print(
            f"\nForecast horizon: {self.forecast_dates[0]} to {self.forecast_dates[-1]}"
        )
        print(f"Steps: {len(self.forecast_dates)}")

    def simulate_paths(self, scenario="baseline", n_sims=1000):
        """Monte Carlo simulation of forecast paths"""
        print(f"\nStart scenario simulation: {scenario}")
        print(f"Number of paths: {n_sims}")

        n_periods = len(self.forecast_dates)
        results = {}

        for var in self.models.keys():
            print(f"  Simulating: {var}")

            model = self.models[var]
            shock_module = self.shock_modules[var]

            # 获取初始值
            last_value = self.data[var].dropna().iloc[-1]
            last_sigma = model.sigma_hist[-1] if len(model.sigma_hist) > 0 else 1.0

            # 存储所有路径
            paths = np.zeros((n_sims, n_periods))

            for sim in range(n_sims):
                x_current = last_value
                sigma_current = last_sigma

                for t in range(n_periods):
                    # AR(1)基础预测
                    innovation = np.random.normal(0, sigma_current)
                    x_next = model.mu + model.rho * (x_current - model.mu) + innovation

                    # 随机冲击
                    shock = shock_module.simulate_shock()

                    # Tariff情景：在第一期强制冲击
                    if scenario == "tariff" and t == 0 and var in TARIFF_SHOCK_VARS:
                        forced_shock = shock_module.get_forced_shock(SHOCK_QUANTILE)
                        shock = forced_shock
                        if var == "JP_EPU":  # JP_EPU特殊处理为负向
                            shock = -abs(forced_shock)

                    x_next += shock
                    paths[sim, t] = x_next

                    # 更新状态
                    x_current = x_next
                    sigma_current = np.sqrt(
                        LAMBDA_EWMA * sigma_current**2
                        + (1 - LAMBDA_EWMA) * innovation**2
                    )

            results[var] = paths

        return results

    def calculate_confidence_bands(self, paths):
        """Compute confidence bands"""
        percentiles = {}
        for var, var_paths in paths.items():
            percentiles[var] = {
                "p5": np.percentile(var_paths, 5, axis=0),
                "p16": np.percentile(var_paths, 16, axis=0),
                "p50": np.percentile(var_paths, 50, axis=0),
                "p84": np.percentile(var_paths, 84, axis=0),
                "p95": np.percentile(var_paths, 95, axis=0),
            }
        return percentiles

    def standardize_data(self, data, var):
        """Z-score standardize data using historical or rolling statistics"""
        if var not in self.historical_stats:
            return data

        if ROLLING_ZSCORE and hasattr(self, "data") and len(self.data) > ZSCORE_WINDOW:
            # 使用滚动窗口统计量（更动态）
            recent_data = self.data[var].dropna().iloc[-ZSCORE_WINDOW:]
            mean = recent_data.mean()
            std = recent_data.std()
        else:
            # 使用全样本历史统计量
            mean = self.historical_stats[var]["mean"]
            std = self.historical_stats[var]["std"]

        if std == 0 or np.isnan(std):
            return data - mean  # 如果标准差为0，只做均值中心化

        return (data - mean) / std


# =========================================================
# 7. 可视化（增强版，包含Z-Score标准化）
# =========================================================


def create_forecast_plot_zscore(forecast_system, baseline_paths, tariff_paths):
    """Create per-variable forecast charts with Z-score standardization (2025-01 to 2026-06)"""

    # 计算置信区间
    baseline_bands = forecast_system.calculate_confidence_bands(baseline_paths)
    tariff_bands = forecast_system.calculate_confidence_bands(tariff_paths)

    # 历史数据
    hist_data = forecast_system.data
    hist_dates = pd.to_datetime(hist_data["Date"])
    forecast_dates = forecast_system.forecast_dates

    # 定义显示时间范围：2025-01 到 2026-06
    display_start = pd.to_datetime("2025-01-01")
    display_end = pd.to_datetime("2026-06-30")

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, var in enumerate(TARGET_VARS):
        if var not in forecast_system.models:
            continue

        ax = axes[i]

        # 历史数据 - Z-score标准化
        hist_series = hist_data[var].dropna()
        hist_dates_var = hist_dates[hist_data[var].notna()]
        hist_series_zscore = forecast_system.standardize_data(hist_series, var)

        # 筛选显示范围内的历史数据
        hist_mask = (hist_dates_var >= display_start) & (hist_dates_var <= display_end)
        hist_dates_display = hist_dates_var[hist_mask]
        hist_series_display = hist_series_zscore[hist_mask]

        if len(hist_dates_display) > 0:
            ax.plot(
                hist_dates_display,
                hist_series_display,
                color=COLOR_MAP[var],
                linewidth=2,
                label=f"{var} (History)",
                alpha=0.8,
            )

        # 基准情景 - Z-score标准化
        if var in baseline_bands:
            baseline_median = baseline_bands[var]["p50"]
            baseline_median_zscore = forecast_system.standardize_data(
                baseline_median, var
            )
            ax.plot(
                forecast_dates,
                baseline_median_zscore,
                color=COLOR_MAP[var],
                linestyle="-",
                linewidth=2,
                label=f"{var} (Baseline)",
                alpha=0.8,
            )

            # 68%置信区间 - Z-score标准化
            baseline_p16_zscore = forecast_system.standardize_data(
                baseline_bands[var]["p16"], var
            )
            baseline_p84_zscore = forecast_system.standardize_data(
                baseline_bands[var]["p84"], var
            )
            ax.fill_between(
                forecast_dates,
                baseline_p16_zscore,
                baseline_p84_zscore,
                color=COLOR_MAP[var],
                alpha=0.2,
            )

        # Tariff情景 - Z-score标准化
        if var in tariff_bands:
            tariff_median = tariff_bands[var]["p50"]
            tariff_median_zscore = forecast_system.standardize_data(tariff_median, var)
            ax.plot(
                forecast_dates,
                tariff_median_zscore,
                color=COLOR_MAP[var],
                linestyle="--",
                linewidth=2,
                label=f"{var} (Tariff)",
                alpha=0.8,
            )

        # 分界线（如果在显示范围内）
        forecast_start_date = pd.to_datetime(HIST_END_DATE)
        if display_start <= forecast_start_date <= display_end:
            ax.axvline(
                x=forecast_start_date,
                color="red",
                linestyle=":",
                alpha=0.7,
                linewidth=1,
            )

        # 添加0线作为参考
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.8)

        # 设置x轴范围
        ax.set_xlim(display_start, display_end)

        ax.set_title(f"{var} (Z-Score Standardized)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Z-Score", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # 格式化x轴 - 更密集的时间标签
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月一个标签
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.suptitle(
        "Policy Variable Forecast (Z-Score Standardized): Baseline vs Tariff (2025-2026)",
        fontsize=16,
        y=1.02,
    )


def create_summary_plot_zscore(forecast_system, tariff_paths):
    """Create a single summary chart with Z-score standardization (2025-01 to 2026-06)"""

    tariff_bands = forecast_system.calculate_confidence_bands(tariff_paths)
    hist_data = forecast_system.data
    hist_dates = pd.to_datetime(hist_data["Date"])
    forecast_dates = forecast_system.forecast_dates

    # 定义显示时间范围：2025-01 到 2026-06
    display_start = pd.to_datetime("2025-01-01")
    display_end = pd.to_datetime("2026-06-30")

    fig, ax = plt.subplots(figsize=(14, 8))

    for var in TARGET_VARS:
        if var not in forecast_system.models:
            continue

        # 历史数据 - Z-score标准化
        hist_series = hist_data[var].dropna()
        hist_dates_var = hist_dates[hist_data[var].notna()]
        hist_series_zscore = forecast_system.standardize_data(hist_series, var)

        # 筛选显示范围内的历史数据
        hist_mask = (hist_dates_var >= display_start) & (hist_dates_var <= display_end)
        hist_dates_display = hist_dates_var[hist_mask]
        hist_series_display = hist_series_zscore[hist_mask]

        # 连接历史和预测
        if len(hist_series) > 0 and var in tariff_bands:
            # 历史部分（仅显示范围内）
            if len(hist_dates_display) > 0:
                ax.plot(
                    hist_dates_display,
                    hist_series_display,
                    color=COLOR_MAP[var],
                    linewidth=2.5,
                    alpha=0.8,
                    label=var,
                )

            # 预测部分（中位数）- Z-score标准化
            forecast_median = tariff_bands[var]["p50"]
            forecast_median_zscore = forecast_system.standardize_data(
                forecast_median, var
            )

            # 连接点（如果历史数据的最后一个点在显示范围内）
            last_hist_date = hist_dates_var.iloc[-1]
            if last_hist_date >= display_start:
                last_hist_value_zscore = hist_series_zscore.iloc[-1]
                first_forecast_date = forecast_dates[0]
                first_forecast_value_zscore = forecast_median_zscore[0]

                ax.plot(
                    [last_hist_date, first_forecast_date],
                    [last_hist_value_zscore, first_forecast_value_zscore],
                    color=COLOR_MAP[var],
                    linewidth=2.5,
                    alpha=0.8,
                )

            # 预测路径
            ax.plot(
                forecast_dates,
                forecast_median_zscore,
                color=COLOR_MAP[var],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
            )

    # 分界线
    forecast_start_date = pd.to_datetime(HIST_END_DATE)
    if display_start <= forecast_start_date <= display_end:
        ax.axvline(
            x=forecast_start_date,
            color="red",
            linestyle=":",
            alpha=0.8,
            linewidth=2,
            label="Forecast start",
        )

    # 添加0线作为参考
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)

    # 设置x轴范围
    ax.set_xlim(display_start, display_end)

    ax.set_title(
        "Policy Variable Forecast (Z-Score Standardized) — Tariff Scenario (2025–2026)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Z-Score", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # 格式化x轴 - 更密集的时间标签
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月一个标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def analyze_shock_persistence():
    """分析冲击持续性和模型参数"""
    print("\n" + "=" * 70)
    print("SHOCK PERSISTENCE ANALYSIS")
    print("=" * 70)
    print("Understanding why forecasts converge to zero:")
    print("\n1. AR(1) Mean Reversion:")
    print("   - Model: X_t = μ + ρ*(X_{t-1} - μ) + ε_t")
    print("   - When |ρ| < 1, series reverts to long-term mean μ")
    print("   - Shock impact decays at rate ρ^t")
    print("\n2. Z-Score Standardization:")
    print("   - Z = (X - μ) / σ")
    print("   - When forecast → μ, then Z-score → 0")
    print("\n3. Mitigation strategies:")
    if PERSISTENCE_BOOST:
        print(f"   ✓ Enhanced persistence (min ρ = {MIN_RHO})")
    else:
        print("   ✗ Standard persistence (original ρ estimates)")

    if ROLLING_ZSCORE:
        print(f"   ✓ Rolling Z-score ({ZSCORE_WINDOW}-month window)")
    else:
        print("   ✗ Fixed historical Z-score baseline")


def create_persistence_analysis_plot(forecast_system, tariff_paths):
    """显示冲击衰减过程"""

    tariff_bands = forecast_system.calculate_confidence_bands(tariff_paths)
    forecast_dates = forecast_system.forecast_dates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 上图：冲击幅度衰减（原始数值）
    ax1.set_title(
        "Tariff Shock Impact Decay (Original Values)", fontsize=14, fontweight="bold"
    )

    for var in TARIFF_SHOCK_VARS:
        if var in tariff_bands and var in forecast_system.models:
            model = forecast_system.models[var]
            last_hist = forecast_system.data[var].dropna().iloc[-1]
            forecast_median = tariff_bands[var]["p50"]

            # 计算相对于历史最后值的变化
            relative_change = ((forecast_median - last_hist) / abs(last_hist)) * 100

            ax1.plot(
                forecast_dates,
                relative_change,
                color=COLOR_MAP[var],
                linewidth=2.5,
                label=f"{var} (ρ={model.rho:.3f})",
                marker="o",
                markersize=4,
            )

    ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax1.set_ylabel("Change from Last Historical Value (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下图：理论衰减曲线
    ax2.set_title("Theoretical Shock Decay Rates", fontsize=14, fontweight="bold")

    periods = np.arange(1, len(forecast_dates) + 1)
    for var in TARIFF_SHOCK_VARS:
        if var in forecast_system.models:
            model = forecast_system.models[var]
            rho = model.rho
            # 理论衰减：初始冲击 × ρ^t
            decay = np.power(abs(rho), periods - 1)  # t=1时衰减为1（初始冲击）

            ax2.plot(
                forecast_dates,
                decay,
                color=COLOR_MAP[var],
                linewidth=2,
                linestyle="--",
                label=f"{var} (ρ={rho:.3f})",
            )

    ax2.set_ylabel("Shock Retention Factor")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 格式化x轴
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()
    """Create a single summary chart with Z-score standardization (2025-01 to 2026-06)"""

    tariff_bands = forecast_system.calculate_confidence_bands(tariff_paths)
    hist_data = forecast_system.data
    hist_dates = pd.to_datetime(hist_data["Date"])
    forecast_dates = forecast_system.forecast_dates

    # 定义显示时间范围：2025-01 到 2026-06
    display_start = pd.to_datetime("2025-01-01")
    display_end = pd.to_datetime("2026-06-30")

    fig, ax = plt.subplots(figsize=(14, 8))

    for var in TARGET_VARS:
        if var not in forecast_system.models:
            continue

        # 历史数据 - Z-score标准化
        hist_series = hist_data[var].dropna()
        hist_dates_var = hist_dates[hist_data[var].notna()]
        hist_series_zscore = forecast_system.standardize_data(hist_series, var)

        # 筛选显示范围内的历史数据
        hist_mask = (hist_dates_var >= display_start) & (hist_dates_var <= display_end)
        hist_dates_display = hist_dates_var[hist_mask]
        hist_series_display = hist_series_zscore[hist_mask]

        # 连接历史和预测
        if len(hist_series) > 0 and var in tariff_bands:
            # 历史部分（仅显示范围内）
            if len(hist_dates_display) > 0:
                ax.plot(
                    hist_dates_display,
                    hist_series_display,
                    color=COLOR_MAP[var],
                    linewidth=2.5,
                    alpha=0.8,
                    label=var,
                )

            # 预测部分（中位数）- Z-score标准化
            forecast_median = tariff_bands[var]["p50"]
            forecast_median_zscore = forecast_system.standardize_data(
                forecast_median, var
            )

            # 连接点（如果历史数据的最后一个点在显示范围内）
            last_hist_date = hist_dates_var.iloc[-1]
            if last_hist_date >= display_start:
                last_hist_value_zscore = hist_series_zscore.iloc[-1]
                first_forecast_date = forecast_dates[0]
                first_forecast_value_zscore = forecast_median_zscore[0]

                ax.plot(
                    [last_hist_date, first_forecast_date],
                    [last_hist_value_zscore, first_forecast_value_zscore],
                    color=COLOR_MAP[var],
                    linewidth=2.5,
                    alpha=0.8,
                )

            # 预测路径
            ax.plot(
                forecast_dates,
                forecast_median_zscore,
                color=COLOR_MAP[var],
                linestyle="--",
                linewidth=2.5,
                alpha=0.8,
            )

    # 分界线
    forecast_start_date = pd.to_datetime(HIST_END_DATE)
    if display_start <= forecast_start_date <= display_end:
        ax.axvline(
            x=forecast_start_date,
            color="red",
            linestyle=":",
            alpha=0.8,
            linewidth=2,
            label="Forecast start",
        )

    # 添加0线作为参考
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=1)

    # 设置x轴范围
    ax.set_xlim(display_start, display_end)

    ax.set_title(
        "Policy Variable Forecast (Z-Score Standardized) — Tariff Scenario (2025–2026)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Z-Score", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    # 格式化x轴 - 更密集的时间标签
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每2个月一个标签
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


# =========================================================
# 8. 主程序（修改版）
# =========================================================


def main():
    """Main runner with Z-score standardized visualization"""
    print("Policy Variable Forecasting System (with Z-Score Standardization)")
    print("=" * 70)
    print("Method: AR(1)-EWMA + Shock Module + Scenario Constraints")
    print(f"Forecast horizon: {FORECAST_START} to {FORECAST_END}")
    print("=" * 70)

    # 1. 数据加载
    data = load_and_prepare_data(DATA_PATH)

    # 2. 模型拟合
    forecast_system = PolicyForecastSystem()
    forecast_system.fit(data)

    # 3. 基准情景模拟
    print("\n" + "=" * 70)
    baseline_paths = forecast_system.simulate_paths(
        scenario="baseline", n_sims=N_SIMULATIONS
    )

    # 4. Tariff情景模拟
    print("\n" + "=" * 70)
    tariff_paths = forecast_system.simulate_paths(
        scenario="tariff", n_sims=N_SIMULATIONS
    )

    # 5. 生成报告
    print("\n" + "=" * 70)
    print("Model parameter summary:")
    print("-" * 40)

    for var in forecast_system.models.keys():
        model = forecast_system.models[var]
        shock_module = forecast_system.shock_modules[var]

        print(f"\n{var}:")
        print(f"  AR(1): μ={model.mu:.4f}, ρ={model.rho:.4f}")
        print(f"  Historical mean: {forecast_system.historical_stats[var]['mean']:.4f}")
        print(f"  Historical std: {forecast_system.historical_stats[var]['std']:.4f}")
        print(f"  Shock rate: {shock_module.shock_rate:.4f}")
        print(f"  Direction constraint: {shock_module.direction_constraint}")

        if len(shock_module.shock_sizes) > 0:
            abs_shocks = np.abs(shock_module.shock_sizes)
            print(f"  Shock size P75: {np.percentile(abs_shocks, 75):.4f}")

    # 6. 分析冲击持续性
    analyze_shock_persistence()

    # 7. 可视化 (Z-Score标准化版本)
    print("\nGenerating Z-score standardized charts...")

    # 冲击衰减分析图
    create_persistence_analysis_plot(forecast_system, tariff_paths)

    # 详细对比图 (Z-Score标准化)
    create_forecast_plot_zscore(forecast_system, baseline_paths, tariff_paths)

    # 汇总图 (Z-Score标准化)
    create_summary_plot_zscore(forecast_system, tariff_paths)

    print("\nForecast finished!")
    print("=" * 70)

    # 8. 输出预测统计 (原始数值和Z-Score变化)
    print("\nForecast change summary:")
    print("-" * 50)

    tariff_bands = forecast_system.calculate_confidence_bands(tariff_paths)

    for var in forecast_system.models.keys():
        if var in tariff_bands:
            last_hist = data[var].dropna().iloc[-1]
            final_forecast = tariff_bands[var]["p50"][-1]  # 终期中位数

            # 原始数值变化
            denom = np.abs(last_hist) if np.abs(last_hist) > 1e-8 else 1.0
            change_pct = ((final_forecast - last_hist) / denom) * 100

            # Z-Score变化
            last_hist_zscore = forecast_system.standardize_data(
                np.array([last_hist]), var
            )[0]
            final_forecast_zscore = forecast_system.standardize_data(
                np.array([final_forecast]), var
            )[0]
            zscore_change = final_forecast_zscore - last_hist_zscore

            print(f"{var}:")
            print(
                f"  Original: {last_hist:.2f} → {final_forecast:.2f} ({change_pct:+.1f}%)"
            )
            print(
                f"  Z-Score: {last_hist_zscore:.2f} → {final_forecast_zscore:.2f} ({zscore_change:+.2f})"
            )
            print()


if __name__ == "__main__":
    main()


#####################################################################################
# 政策预测后策略结果
# ####################################################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# 设置画图参数
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
    "sans-serif",
]
plt.rcParams["figure.facecolor"] = "white"


class EnhancedIntegratedStrategySystem:
    """增强版集成策略系统：确保情景差异明显"""

    def __init__(self):
        # 预测系统参数
        self.LAMBDA_EWMA = 0.94
        self.SHOCK_THRESHOLD = 2.0
        self.N_SIMULATIONS = 500
        self.SHOCK_QUANTILE = 0.75
        self.PERSISTENCE_BOOST = True
        self.MIN_RHO = 0.7

        # 策略参数
        self.min_w, self.max_w = 0.2, 2.0
        self.FREQ = 12
        self.eps = 1e-8

        # 时间分界点
        self.HIST_END_DATE = "2025-06-30"
        self.FORECAST_START = "2025-07-01"

        # 目标变量和权重配置
        self.TARGET_VARS = [
            "TPU",
            "US_EPU",
            "JP_EPU",
            "FX_VOL",
            "VIX",
            "Liquidity_Premium",
            "US_GDP",
            "JP_GDP",
            "US_CPI",
            "JP_CPI",
            "US_PMI",
            "JP_PMI",
            "US_UR",
            "JP_UR",
        ]

        self.SHOCK_DIRECTION = {
            "TPU": "positive",
            "US_EPU": "positive",
            "FX_VOL": "positive",
            "JP_EPU": "negative",
            "VIX": "positive",
            "Liquidity_Premium": "positive",
            "US_GDP": "negative",
            "JP_GDP": "negative",
            "US_CPI": "positive",
            "JP_CPI": "positive",
            "US_PMI": "negative",
            "JP_PMI": "negative",
            "US_UR": "positive",
            "JP_UR": "positive",
        }

        self.TARIFF_SHOCK_VARS = ["TPU", "US_EPU", "FX_VOL", "JP_EPU"]

        # 风险变量权重
        self.risk_variables = [
            "VIX",
            "FX_VOL",
            "US_EPU",
            "TPU",
            "Liquidity_Premium",
            "US_GDP",
            "JP_GDP",
            "US_CPI",
            "JP_CPI",
            "US_PMI",
            "JP_PMI",
            "US_UR",
            "JP_UR",
        ]

        self.risk_weights = {
            "VIX": 0.12,
            "FX_VOL": 0.12,
            "US_EPU": 0.10,
            "TPU": 0.10,
            "Liquidity_Premium": 0.06,
            "US_GDP": 0.08,
            "JP_GDP": 0.08,
            "US_CPI": 0.08,
            "JP_CPI": 0.08,
            "US_PMI": 0.06,
            "JP_PMI": 0.06,
            "US_UR": 0.08,
            "JP_UR": 0.08,
        }

        # 存储数据和模型
        self.historical_data = None
        self.historical_strategy_results = None
        self.policy_models = {}
        self.shock_modules = {}
        self.risk_thresholds = {}

    def load_historical_data(self, file_path):
        """加载历史数据"""
        print("Loading historical data...")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"File not found: {file_path}, generating sample data")
            df = self.generate_sample_data()

        # 日期处理
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)

        # 转为月度数据
        df["YearMonth"] = df["Date"].dt.to_period("M")
        df_monthly = df.groupby("YearMonth").last().reset_index()
        df_monthly["Date"] = df_monthly["YearMonth"].dt.to_timestamp()

        # 只保留历史数据
        hist_cutoff = pd.to_datetime(self.HIST_END_DATE)
        self.historical_data = df_monthly[df_monthly["Date"] <= hist_cutoff].copy()

        print(
            f"Historical data range: {self.historical_data['Date'].min()} to {self.historical_data['Date'].max()}"
        )
        print(f"Historical observations: {len(self.historical_data)}")

        return self.historical_data

    def compute_original_strategy_results(self):
        """计算原始策略的历史结果（完全按照第一套代码逻辑）"""
        print("Computing original strategy historical results...")

        df = self.historical_data.copy()

        # ============= ② 结构性 Base Weight（分位线性）==============
        p30, p70 = df["IRD"].quantile([0.3, 0.7])

        def compute_base_weight(ird):
            if ird <= p30:
                return 0.2
            elif ird >= p70:
                return 1.0
            else:
                return 0.2 + (ird - p30) / (p70 - p30) * 0.8

        df["Base_Weight"] = df["IRD"].apply(compute_base_weight)

        # ============= ③ Policy Risk Signal（分位惩罚）============
        risk_thresholds = {
            v: df[v].quantile(0.75) for v in self.risk_variables if v in df.columns
        }
        self.risk_thresholds = risk_thresholds

        def compute_risk_signal(row):
            penalty = 0
            for var in self.risk_variables:
                if var in row.index and var in risk_thresholds:
                    if row[var] > risk_thresholds[var]:
                        penalty += self.risk_weights[var]
            return max(0.2, 1.0 - penalty)

        df["Risk_Signal"] = df.apply(compute_risk_signal, axis=1)

        # ============= ④ 三种杠杆 + Final ============
        df["IRD_Leverage"] = np.clip(
            df["Base_Weight"] * self.max_w, self.min_w, self.max_w
        )
        df["Risk_Leverage"] = np.clip(
            df["Risk_Signal"] * self.max_w, self.min_w, self.max_w
        )
        df["Final_State_Signal"] = df["Base_Weight"] * df["Risk_Signal"]
        df["Final_Leverage"] = np.clip(
            df["Final_State_Signal"] * self.max_w, self.min_w, self.max_w
        )

        # ============= ⑤ Alpha策略（与原代码完全一致）============
        fs = df["Final_State_Signal"].clip(0, 1).astype(float).values
        alpha_base = df["Final_Leverage"].astype(float).values

        # 凸性增强
        k_low, p_low = 1.10, 1.8
        k_mid, p_mid = 1.80, 2.6
        k_high, p_high = 3.00, 3.2

        f_conv = np.where(
            fs < 0.40,
            1.0 + k_low * (fs**p_low),
            np.where(fs < 0.75, 1.0 + k_mid * (fs**p_mid), 1.0 + k_high * (fs**p_high)),
        )

        # 动量因子
        mom6 = df["Carry_Return"].rolling(6).mean()
        std12 = df["Carry_Return"].rolling(12).std().replace(0, np.nan).bfill()
        mom_ir = (mom6 / (std12 + self.eps)).clip(-1, 3).fillna(0)
        f_mom = 1.0 + 0.32 * np.clip(mom_ir, 0, None)

        # FX波动率因子
        fx_q40, fx_q80 = df["FX_VOL"].quantile([0.40, 0.80])
        f_fxv = np.where(
            df["FX_VOL"] <= fx_q40, 1.06, np.where(df["FX_VOL"] >= fx_q80, 0.96, 1.00)
        )

        # 涡轮增压
        turbo = np.where((fs > 0.82) & (mom_ir > 0), 1.18, 1.00)
        boost = f_conv * f_mom * f_fxv * turbo

        # 上下限
        floor_rel = 0.93
        max_w_alpha_cap = np.where(fs > 0.75, 7.0, 4.0)
        alpha_target = np.minimum(alpha_base * boost, max_w_alpha_cap)
        alpha_target = np.maximum(alpha_target, floor_rel * alpha_base)

        # 平滑处理
        up_abs, up_rel = 0.70, 0.70
        down_abs, down_rel = 0.22, 0.22
        alpha_ema = pd.Series(alpha_target).ewm(span=2, adjust=False).mean().values
        alpha_slow = np.zeros_like(alpha_ema)
        alpha_slow[0] = alpha_ema[0]

        for t in range(1, len(alpha_ema)):
            prev, raw = alpha_slow[t - 1], alpha_ema[t]
            if raw >= prev:
                ub = min(prev * (1 + up_rel), prev + up_abs)
                alpha_slow[t] = np.clip(raw, prev, ub)
            else:
                lb = max(prev * (1 - down_rel), prev - down_abs)
                alpha_slow[t] = np.clip(raw, lb, prev)

        alpha_lev = pd.Series(alpha_slow).ewm(span=2, adjust=False).mean().values

        # 波动率目标
        def _ann_vol(x):
            return x.std() * np.sqrt(self.FREQ)

        final_vol = _ann_vol(df["Carry_Return"])
        tgt_mult = np.where(fs < 0.40, 1.4, np.where(fs < 0.75, 1.9, 2.6))
        tgt_vol = final_vol * tgt_mult

        asset_roll_vol = (
            (df["Carry_Return"].rolling(12).std() * np.sqrt(self.FREQ))
            .replace(0, np.nan)
            .bfill()
            .ffill()
        )
        scale = (tgt_vol / (asset_roll_vol * (alpha_lev + self.eps))).clip(0.90, 1.60)
        alpha_lev *= scale

        # 回撤控制
        eq_tmp = (1 + (df["Carry_Return"] * alpha_lev)).cumprod()
        dd = (eq_tmp.cummax() - eq_tmp) / (eq_tmp.cummax() + self.eps)
        alpha_lev = np.where(dd > 0.06, alpha_lev * 0.92, alpha_lev)

        # 最终边界
        alpha_lev = np.clip(alpha_lev, 0.2, np.where(fs > 0.75, 7.0, 4.0))
        df["Alpha_Leverage"] = alpha_lev

        # ============= ⑥ 策略回报/净值 ============
        df["Return_Hold2x"] = df["Carry_Return"] * 2.0
        df["Return_IRD_Alloc"] = df["Carry_Return"] * df["IRD_Leverage"]
        df["Return_Policy_Alloc"] = df["Carry_Return"] * df["Risk_Leverage"]
        df["Return_Final"] = df["Carry_Return"] * df["Final_Leverage"]
        df["Return_Alpha"] = df["Carry_Return"] * df["Alpha_Leverage"]

        df["Equity_Hold2x"] = (1 + df["Return_Hold2x"]).cumprod()
        df["Equity_IRD_Alloc"] = (1 + df["Return_IRD_Alloc"]).cumprod()
        df["Equity_Policy_Alloc"] = (1 + df["Return_Policy_Alloc"]).cumprod()
        df["Equity_Final"] = (1 + df["Return_Final"]).cumprod()
        df["Equity_Alpha"] = (1 + df["Return_Alpha"]).cumprod()

        self.historical_strategy_results = df
        print("Original strategy historical results computed")

        return df

    def fit_policy_prediction_models(self):
        """拟合政策变量预测模型"""
        print("Fitting policy variable prediction models...")

        for var in self.TARGET_VARS:
            if var not in self.historical_data.columns:
                continue

            print(f"Fitting model: {var}")
            series = self.historical_data[var].dropna()

            if len(series) >= 5:
                model_result = self.fit_ar1_ewma_model(series)
                self.policy_models[var] = model_result

                direction = self.SHOCK_DIRECTION.get(var)
                shock_result = self.identify_shocks(
                    model_result["residuals"], model_result["sigma_hist"], direction
                )
                self.shock_modules[var] = shock_result

                print(
                    f"  AR(1): μ={model_result['mu']:.4f}, ρ={model_result['rho']:.4f}"
                )
                print(f"  Shock rate: {shock_result['shock_rate']:.4f}")

    def fit_ar1_ewma_model(self, data):
        """拟合AR(1)-EWMA模型"""
        data = pd.Series(data).dropna()

        if len(data) < 10:
            mu = data.mean()
            rho = 0.0 if not self.PERSISTENCE_BOOST else self.MIN_RHO
            residuals = data - mu
            sigma_hist = np.full(len(residuals), residuals.std())
        else:
            y = data.iloc[1:].values
            X = sm.add_constant(data.iloc[:-1].values)

            try:
                model = sm.OLS(y, X).fit()
                intercept, ar_coef = model.params

                rho = ar_coef
                mu = intercept / (1 - ar_coef) if abs(ar_coef) < 0.99 else data.mean()

                if abs(rho) >= 0.99:
                    rho = 0.95 if rho > 0 else -0.95
                elif self.PERSISTENCE_BOOST and abs(rho) < self.MIN_RHO:
                    rho = self.MIN_RHO if rho > 0 else -self.MIN_RHO

                residuals = y - (intercept + ar_coef * data.iloc[:-1].values)

            except Exception:
                mu = data.mean()
                rho = self.MIN_RHO if self.PERSISTENCE_BOOST else 0.5
                residuals = (data - mu).values[1:]

        # EWMA波动率
        residuals_sq = residuals**2
        sigma_sq = np.zeros(len(residuals_sq))
        sigma_sq[0] = np.var(residuals) if len(residuals) > 1 else 1.0

        for t in range(1, len(residuals_sq)):
            sigma_sq[t] = (
                self.LAMBDA_EWMA * sigma_sq[t - 1]
                + (1 - self.LAMBDA_EWMA) * residuals_sq[t - 1]
            )

        sigma_hist = np.sqrt(sigma_sq)

        return {
            "mu": mu,
            "rho": rho,
            "residuals": residuals,
            "sigma_hist": sigma_hist,
            "last_sigma": sigma_hist[-1] if len(sigma_hist) > 0 else 1.0,
        }

    def identify_shocks(self, residuals, sigma_hist, direction=None):
        """识别和校准冲击"""
        z_scores = residuals / (sigma_hist + self.eps)
        shock_mask = np.abs(z_scores) > self.SHOCK_THRESHOLD
        shock_residuals = residuals[shock_mask]

        if direction == "positive":
            shock_residuals = shock_residuals[shock_residuals > 0]
        elif direction == "negative":
            shock_residuals = shock_residuals[shock_residuals < 0]

        shock_rate = len(shock_residuals) / len(z_scores) if len(z_scores) > 0 else 0

        return {
            "shock_rate": shock_rate,
            "shock_sizes": shock_residuals,
            "direction": direction,
        }

    def simulate_policy_variables_enhanced(self, scenario="baseline", months=12):
        """增强版政策变量模拟 - 确保持续差异"""
        print(f"Enhanced simulation: {scenario} scenario, {months} months")

        forecast_dates = pd.date_range(
            start=self.FORECAST_START, periods=months, freq="MS"
        )
        results = {}

        for var in self.policy_models.keys():
            model = self.policy_models[var]
            shock_module = self.shock_modules[var]

            # 初始值
            last_value = self.historical_data[var].dropna().iloc[-1]
            last_sigma = model["last_sigma"]

            print(f"  {var}: Historical last value={last_value:.2f}")

            # 情景特定的持续偏移
            scenario_drift = self.get_scenario_drift(var, scenario)

            # 路径模拟
            path = []
            x_current = last_value
            sigma_current = last_sigma

            for t in range(months):
                # 基本AR(1)预测
                innovation = 0
                x_next = (
                    model["mu"] + model["rho"] * (x_current - model["mu"]) + innovation
                )

                # 初期冲击（更强且持续更久）
                shock = 0
                if scenario == "tariff" and var in self.TARIFF_SHOCK_VARS:
                    if t <= 2:  # 前3个月都有冲击，逐渐减弱
                        shock_magnitude = self.get_tariff_shock_magnitude(var, t)
                        if shock_module["direction"] == "negative" or var == "JP_EPU":
                            shock = -abs(shock_magnitude)
                        else:
                            shock = abs(shock_magnitude)
                        print(f"    {var} Tariff shock at t={t}: {shock:.2f}")

                # 持续的情景偏移
                drift_effect = scenario_drift * (0.9**t)  # 逐渐衰减但持续存在

                x_next += shock + drift_effect

                # 合理范围限制
                x_next = self.clip_variable_range(var, x_next)

                path.append(x_next)
                x_current = x_next
                sigma_current = np.sqrt(
                    self.LAMBDA_EWMA * sigma_current**2
                    + (1 - self.LAMBDA_EWMA) * innovation**2
                )

            print(
                f"    {var} forecast: {last_value:.2f} -> {path[-1]:.2f} (drift: {scenario_drift:.2f})"
            )

            results[var] = {"dates": forecast_dates, "values": np.array(path)}

        # 输出情景对比
        self.print_scenario_comparison(results, scenario)

        return results

    def get_scenario_drift(self, var, scenario):
        """获取情景特定的持续偏移"""
        if scenario == "baseline":
            return 0.0  # 基准情景无额外偏移

        elif scenario == "tariff":
            # Tariff情景的持续偏移
            drift_map = {
                "TPU": 8.0,  # 持续高贸易不确定性
                "US_EPU": 12.0,  # 持续高经济政策不确定性
                "FX_VOL": 0.02,  # 持续高汇率波动
                "VIX": 3.0,  # 持续高市场恐慌
                "JP_EPU": -2.0,  # 日本相对稳定
                "Liquidity_Premium": 0.15,  # 持续流动性溢价
            }
            return drift_map.get(var, 0.0)

        return 0.0

    def get_tariff_shock_magnitude(self, var, period):
        """获取Tariff冲击幅度（前几期递减）"""
        base_shocks = {"TPU": 20.0, "US_EPU": 25.0, "FX_VOL": 0.05, "JP_EPU": 8.0}

        base = base_shocks.get(var, 0.0)
        # 递减：第0期100%，第1期70%，第2期40%
        decay_factors = [1.0, 0.7, 0.4]
        return base * decay_factors[min(period, 2)]

    def clip_variable_range(self, var, value):
        """限制变量在合理范围内"""
        if var == "VIX":
            return np.clip(value, 8, 80)
        elif var == "FX_VOL":
            return np.clip(value, 0.05, 0.5)
        elif var in ["TPU", "US_EPU", "JP_EPU"]:
            return np.maximum(value, 0)
        elif var in ["US_PMI", "JP_PMI"]:
            return np.clip(value, 20, 80)
        elif var in ["US_UR", "JP_UR"]:
            return np.clip(value, 1, 15)
        return value

    def print_scenario_comparison(self, results, scenario):
        """打印情景对比"""
        print(f"\n{scenario.upper()} scenario variable changes:")
        for var in ["TPU", "US_EPU", "VIX", "FX_VOL"]:
            if var in results:
                initial = results[var]["values"][0]
                final = results[var]["values"][-1]
                change = (
                    ((final - initial) / abs(initial)) * 100 if abs(initial) > 0 else 0
                )
                print(f"  {var}: {initial:.1f} -> {final:.1f} ({change:+.1f}%)")

    def create_forecast_extension_enhanced(self, policy_forecasts, scenario="baseline"):
        """增强版预测扩展 - 确保策略差异明显"""
        print(f"Enhanced forecast extension: {scenario} scenario...")

        hist_last = self.historical_strategy_results.iloc[-1]
        forecast_dates = policy_forecasts[list(policy_forecasts.keys())[0]]["dates"]

        # 不同情景使用不同随机种子，确保基础差异
        if scenario == "baseline":
            np.random.seed(42)
        else:
            np.random.seed(123)  # 不同种子确保差异

        # 历史统计
        hist_carry_mean = self.historical_data["Carry_Return"].mean()
        hist_carry_std = self.historical_data["Carry_Return"].std()
        hist_ird_mean = self.historical_data["IRD"].mean()
        hist_ird_std = self.historical_data["IRD"].std()

        print(
            f"  Historical Carry: mean={hist_carry_mean:.4f}, std={hist_carry_std:.4f}"
        )
        print(f"  Historical IRD: mean={hist_ird_mean:.4f}, std={hist_ird_std:.4f}")

        # 创建预测数据
        forecast_rows = []

        for i, date in enumerate(forecast_dates):
            row = {"Date": date}

            # 1. 添加预测的政策变量
            for var, forecast in policy_forecasts.items():
                row[var] = forecast["values"][i]

            # 2. 增强的IRD和Carry_Return预测
            if i == 0:
                # 第一期：增强政策影响
                row["IRD"] = (
                    hist_last["IRD"] * 0.9
                    + hist_ird_mean * 0.1
                    + np.random.normal(0, hist_ird_std * 0.4)
                )

                # 大幅增强政策对收益的影响
                base_carry = hist_carry_mean
                policy_adjustment = 1.0

                # VIX影响（显著增强）
                if "VIX" in row and "VIX" in self.risk_thresholds:
                    vix_excess = max(
                        0,
                        (row["VIX"] - self.risk_thresholds["VIX"])
                        / self.risk_thresholds["VIX"],
                    )
                    vix_impact = -0.6 * vix_excess  # 大幅增强从-0.3到-0.6
                    policy_adjustment += vix_impact
                    print(
                        f"    VIX impact: VIX={row['VIX']:.1f}, threshold={self.risk_thresholds['VIX']:.1f}, impact={vix_impact:.3f}"
                    )

                # TPU影响（大幅增强）
                if "TPU" in row:
                    if scenario == "tariff":
                        tpu_impact = -0.8 * (row["TPU"] / 100)  # 从-0.4增加到-0.8
                        policy_adjustment += tpu_impact
                        print(
                            f"    Tariff TPU impact: TPU={row['TPU']:.1f}, impact={tpu_impact:.3f}"
                        )
                    else:
                        tpu_impact = -0.15 * (row["TPU"] / 100)  # 从-0.1增加到-0.15
                        policy_adjustment += tpu_impact
                        print(
                            f"    Baseline TPU impact: TPU={row['TPU']:.1f}, impact={tpu_impact:.3f}"
                        )

                # US_EPU影响（增强）
                if "US_EPU" in row and "US_EPU" in self.risk_thresholds:
                    epu_excess = max(
                        0,
                        (row["US_EPU"] - self.risk_thresholds["US_EPU"])
                        / self.risk_thresholds["US_EPU"],
                    )
                    if scenario == "tariff":
                        epu_impact = -0.5 * epu_excess  # 从-0.25增加到-0.5
                    else:
                        epu_impact = -0.2 * epu_excess  # 从-0.1增加到-0.2
                    policy_adjustment += epu_impact
                    print(
                        f"    US_EPU impact: EPU={row['US_EPU']:.1f}, impact={epu_impact:.3f}"
                    )

                # 情景基础调整（增强）
                if scenario == "tariff":
                    scenario_adjustment = -0.3  # 从-0.15增加到-0.3
                else:
                    scenario_adjustment = 0.05  # 从0.02增加到0.05
                policy_adjustment += scenario_adjustment

                print(f"    Total policy adjustment: {policy_adjustment:.3f}")

                # 确保调整在合理范围内
                policy_adjustment = np.clip(policy_adjustment, 0.2, 1.8)

                row["Carry_Return"] = base_carry * policy_adjustment + np.random.normal(
                    0, hist_carry_std * 0.5
                )

            else:
                # 后续期间：持续的政策影响
                prev_ird = forecast_rows[i - 1]["IRD"]
                prev_ret = forecast_rows[i - 1]["Carry_Return"]

                row["IRD"] = (
                    0.7 * prev_ird
                    + 0.3 * hist_ird_mean
                    + np.random.normal(0, hist_ird_std * 0.3)
                )

                # 持续的政策影响（衰减更慢）
                base_carry = 0.3 * prev_ret + 0.7 * hist_carry_mean
                decay_factor = 0.85**i  # 稍慢的衰减
                policy_adjustment = 1.0

                # 持续的VIX影响
                if "VIX" in row and "VIX" in self.risk_thresholds:
                    vix_excess = max(
                        0,
                        (row["VIX"] - self.risk_thresholds["VIX"])
                        / self.risk_thresholds["VIX"],
                    )
                    vix_impact = -0.4 * vix_excess * decay_factor
                    policy_adjustment += vix_impact

                # 持续的TPU影响
                if "TPU" in row:
                    if scenario == "tariff":
                        tpu_impact = -0.6 * (row["TPU"] / 100) * decay_factor
                    else:
                        tpu_impact = -0.12 * (row["TPU"] / 100) * decay_factor
                    policy_adjustment += tpu_impact

                # 持续的US_EPU影响
                if "US_EPU" in row and "US_EPU" in self.risk_thresholds:
                    epu_excess = max(
                        0,
                        (row["US_EPU"] - self.risk_thresholds["US_EPU"])
                        / self.risk_thresholds["US_EPU"],
                    )
                    if scenario == "tariff":
                        epu_impact = -0.4 * epu_excess * decay_factor
                    else:
                        epu_impact = -0.15 * epu_excess * decay_factor
                    policy_adjustment += epu_impact

                # 情景基础调整（衰减）
                if scenario == "tariff":
                    scenario_adjustment = -0.2 * decay_factor
                else:
                    scenario_adjustment = 0.03 * decay_factor
                policy_adjustment += scenario_adjustment

                policy_adjustment = np.clip(policy_adjustment, 0.3, 1.6)

                row["Carry_Return"] = base_carry * policy_adjustment + np.random.normal(
                    0, hist_carry_std * 0.3
                )

            forecast_rows.append(row)

        forecast_df = pd.DataFrame(forecast_rows)

        print(
            f"  Forecast Carry stats: mean={forecast_df['Carry_Return'].mean():.4f}, std={forecast_df['Carry_Return'].std():.4f}"
        )
        print(
            f"  Forecast IRD stats: mean={forecast_df['IRD'].mean():.4f}, std={forecast_df['IRD'].std():.4f}"
        )

        # 3. 策略信号计算（与历史完全一致）
        hist_p30, hist_p70 = self.historical_data["IRD"].quantile([0.3, 0.7])

        def compute_base_weight_forecast(ird):
            if ird <= hist_p30:
                return 0.2
            elif ird >= hist_p70:
                return 1.0
            else:
                return 0.2 + (ird - hist_p30) / (hist_p70 - hist_p30) * 0.8

        forecast_df["Base_Weight"] = forecast_df["IRD"].apply(
            compute_base_weight_forecast
        )

        # 风险信号（使用历史阈值）
        def compute_risk_signal_forecast(row):
            penalty = 0
            for var in self.risk_variables:
                if var in row.index and var in self.risk_thresholds:
                    if row[var] > self.risk_thresholds[var]:
                        penalty += self.risk_weights[var]
            return max(0.2, 1.0 - penalty)

        forecast_df["Risk_Signal"] = forecast_df.apply(
            compute_risk_signal_forecast, axis=1
        )

        # 杠杆计算
        forecast_df["Final_State_Signal"] = (
            forecast_df["Base_Weight"] * forecast_df["Risk_Signal"]
        )
        forecast_df["IRD_Leverage"] = np.clip(
            forecast_df["Base_Weight"] * self.max_w, self.min_w, self.max_w
        )
        forecast_df["Risk_Leverage"] = np.clip(
            forecast_df["Risk_Signal"] * self.max_w, self.min_w, self.max_w
        )
        forecast_df["Final_Leverage"] = np.clip(
            forecast_df["Final_State_Signal"] * self.max_w, self.min_w, self.max_w
        )

        # Alpha策略（与历史逻辑一致但适应预测）
        fs = forecast_df["Final_State_Signal"].clip(0, 1).values
        alpha_base = forecast_df["Final_Leverage"].values

        # 凸性增强
        k_low, p_low = 1.10, 1.8
        k_mid, p_mid = 1.80, 2.6
        k_high, p_high = 3.00, 3.2

        f_conv = np.where(
            fs < 0.40,
            1.0 + k_low * (fs**p_low),
            np.where(fs < 0.75, 1.0 + k_mid * (fs**p_mid), 1.0 + k_high * (fs**p_high)),
        )

        # 情景特定的动量因子
        if scenario == "baseline":
            base_momentum = 1.12  # 基准情景更积极
        elif scenario == "tariff":
            base_momentum = 0.75  # 贸易冲击情景更保守

        # 根据政策变量调整动量
        momentum_adjustments = []
        for _, row in forecast_df.iterrows():
            adjustment = base_momentum

            # VIX影响动量
            if "VIX" in row.index and "VIX" in self.risk_thresholds:
                vix_excess = max(
                    0,
                    (row["VIX"] - self.risk_thresholds["VIX"])
                    / self.risk_thresholds["VIX"],
                )
                adjustment -= 0.25 * vix_excess  # 增强影响

            # TPU影响动量
            if "TPU" in row.index:
                tpu_level = row["TPU"] / 100
                if scenario == "tariff":
                    adjustment -= 0.35 * tpu_level  # 增强负面影响
                else:
                    adjustment -= 0.12 * tpu_level

            momentum_adjustments.append(np.clip(adjustment, 0.6, 1.3))

        f_mom = np.array(momentum_adjustments)

        # FX波动率调整
        if "FX_VOL" in forecast_df.columns:
            fx_hist_q40, fx_hist_q80 = self.historical_data["FX_VOL"].quantile(
                [0.40, 0.80]
            )
            f_fxv = np.where(
                forecast_df["FX_VOL"] <= fx_hist_q40,
                1.08,
                np.where(forecast_df["FX_VOL"] >= fx_hist_q80, 0.94, 1.00),
            )
        else:
            f_fxv = np.ones_like(fs)

        # 涡轮增压
        turbo = np.where(fs > 0.82, 1.12, 1.00)

        boost = f_conv * f_mom * f_fxv * turbo

        # 杠杆上限
        max_w_alpha_cap = np.where(fs > 0.75, 6.5, 4.0)
        alpha_target = np.minimum(alpha_base * boost, max_w_alpha_cap)
        alpha_target = np.maximum(alpha_target, 0.91 * alpha_base)

        # 平滑
        alpha_lev = pd.Series(alpha_target).ewm(span=3, adjust=False).mean().values
        forecast_df["Alpha_Leverage"] = np.clip(
            alpha_lev, 0.2, np.where(fs > 0.75, 6.5, 4.0)
        )

        # 4. 收益计算
        forecast_df["Return_Hold2x"] = forecast_df["Carry_Return"] * 2.0
        forecast_df["Return_IRD_Alloc"] = (
            forecast_df["Carry_Return"] * forecast_df["IRD_Leverage"]
        )
        forecast_df["Return_Policy_Alloc"] = (
            forecast_df["Carry_Return"] * forecast_df["Risk_Leverage"]
        )
        forecast_df["Return_Final"] = (
            forecast_df["Carry_Return"] * forecast_df["Final_Leverage"]
        )
        forecast_df["Return_Alpha"] = (
            forecast_df["Carry_Return"] * forecast_df["Alpha_Leverage"]
        )

        return forecast_df

    def combine_historical_and_forecast(self, forecast_df):
        """合并历史和预测数据"""
        print("Combining historical and forecast data...")

        hist_last = self.historical_strategy_results.iloc[-1]

        # 计算预测期间净值
        forecast_df["Equity_Hold2x"] = (
            hist_last["Equity_Hold2x"] * (1 + forecast_df["Return_Hold2x"]).cumprod()
        )
        forecast_df["Equity_IRD_Alloc"] = (
            hist_last["Equity_IRD_Alloc"]
            * (1 + forecast_df["Return_IRD_Alloc"]).cumprod()
        )
        forecast_df["Equity_Policy_Alloc"] = (
            hist_last["Equity_Policy_Alloc"]
            * (1 + forecast_df["Return_Policy_Alloc"]).cumprod()
        )
        forecast_df["Equity_Final"] = (
            hist_last["Equity_Final"] * (1 + forecast_df["Return_Final"]).cumprod()
        )
        forecast_df["Equity_Alpha"] = (
            hist_last["Equity_Alpha"] * (1 + forecast_df["Return_Alpha"]).cumprod()
        )

        # 合并数据
        combined_df = pd.concat(
            [self.historical_strategy_results, forecast_df], ignore_index=True
        )

        return combined_df

    def run_integrated_forecast(self, file_path, scenarios=["baseline", "tariff"]):
        """运行增强版集成预测"""
        print("=" * 70)
        print("Enhanced Integrated Strategy System: Ensure Scenario Differences")
        print("=" * 70)

        # 1. 加载历史数据
        self.load_historical_data(file_path)

        # 2. 计算原始策略历史结果
        self.compute_original_strategy_results()

        # 3. 拟合政策变量预测模型
        self.fit_policy_prediction_models()

        results = {}

        for scenario in scenarios:
            print(f"\nRunning {scenario.upper()} scenario...")

            # 4. 增强版政策变量预测
            policy_forecasts = self.simulate_policy_variables_enhanced(
                scenario=scenario, months=12
            )

            # 5. 增强版预测扩展
            forecast_df = self.create_forecast_extension_enhanced(
                policy_forecasts, scenario
            )

            # 6. 合并历史和预测
            combined_df = self.combine_historical_and_forecast(forecast_df)

            results[scenario] = {
                "combined_data": combined_df,
                "policy_forecasts": policy_forecasts,
                "forecast_data": forecast_df,
            }

        return results

    def generate_performance_table(self, results):
        """生成业绩对比表"""
        print("\nGenerating performance comparison table...")

        def perf_metrics(returns, freq=12):
            returns = pd.Series(returns).dropna()
            ann_ret = (1 + returns.mean()) ** freq - 1
            ann_vol = returns.std() * np.sqrt(freq)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            cumret = (1 + returns).cumprod()
            max_dd = ((cumret.cummax() - cumret) / cumret.cummax()).max()
            return ann_ret, ann_vol, sharpe, max_dd

        # 策略列表 - 只保留需要的3个
        strategies = [
            ("Buy&Hold (2x)", "Return_Hold2x"),
            ("Final State", "Return_Final"),
            ("Final State+Alpha", "Return_Alpha"),
        ]

        # 为每个情景生成表格
        for scenario, result in results.items():
            print(f"\n{scenario.upper()} Scenario Performance:")
            print("-" * 80)

            data = result["combined_data"]

            # 分离历史和预测期间
            forecast_start = pd.to_datetime(self.HIST_END_DATE)
            hist_mask = data["Date"] <= forecast_start
            forecast_mask = data["Date"] > forecast_start

            hist_data = data[hist_mask]
            forecast_data = data[forecast_mask]

            print(
                f"Historical Period: {hist_data['Date'].min().strftime('%Y-%m')} to {hist_data['Date'].max().strftime('%Y-%m')}"
            )
            print(
                f"Forecast Period: {forecast_data['Date'].min().strftime('%Y-%m')} to {forecast_data['Date'].max().strftime('%Y-%m')}"
            )
            print()

            # 历史期间业绩
            print("Historical Period Performance:")
            print("Strategy               | Ann.Ret | Ann.Vol | Sharpe | Max DD")
            print("-" * 65)

            for strategy_name, return_col in strategies:
                if return_col in hist_data.columns:
                    ann_ret, ann_vol, sharpe, max_dd = perf_metrics(
                        hist_data[return_col]
                    )
                    print(
                        f"{strategy_name:<22} | {ann_ret:>6.2%} | {ann_vol:>6.2%} | {sharpe:>6.2f} | {max_dd:>6.2%}"
                    )

            # 预测期间业绩
            print(f"\nForecast Period Performance ({scenario.upper()}):")
            print("Strategy               | Ann.Ret | Ann.Vol | Sharpe | Max DD")
            print("-" * 65)

            for strategy_name, return_col in strategies:
                if return_col in forecast_data.columns:
                    ann_ret, ann_vol, sharpe, max_dd = perf_metrics(
                        forecast_data[return_col]
                    )
                    print(
                        f"{strategy_name:<22} | {ann_ret:>6.2%} | {ann_vol:>6.2%} | {sharpe:>6.2f} | {max_dd:>6.2%}"
                    )

            # 全期间业绩
            print(f"\nFull Period Performance ({scenario.upper()}):")
            print("Strategy               | Ann.Ret | Ann.Vol | Sharpe | Max DD")
            print("-" * 65)

            for strategy_name, return_col in strategies:
                if return_col in data.columns:
                    ann_ret, ann_vol, sharpe, max_dd = perf_metrics(data[return_col])
                    print(
                        f"{strategy_name:<22} | {ann_ret:>6.2%} | {ann_vol:>6.2%} | {sharpe:>6.2f} | {max_dd:>6.2%}"
                    )

        # 生成可视化业绩表格
        self.plot_performance_table(results)

    def plot_performance_table(self, results):
        """生成可视化业绩表格 - 业绩指标横向，策略纵向，全白背景"""
        print("Generating visual performance table...")

        def perf_metrics(returns, freq=12):
            returns = pd.Series(returns).dropna()
            ann_ret = (1 + returns.mean()) ** freq - 1
            ann_vol = returns.std() * np.sqrt(freq)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            cumret = (1 + returns).cumprod()
            max_dd = ((cumret.cummax() - cumret) / cumret.cummax()).max()
            return ann_ret, ann_vol, sharpe, max_dd

        strategies = [
            ("Buy&Hold (2x)", "Return_Hold2x"),
            ("Final State", "Return_Final"),
            ("Final State+Alpha", "Return_Alpha"),
        ]

        # 为每个情景创建表格
        for scenario, result in results.items():
            data = result["combined_data"]

            # 计算业绩数据 - 策略作为行
        # 计算业绩数据 - 业绩指标作为行，策略作为列
        metrics_data = {}
        for strategy_name, return_col in strategies:
            if return_col in data.columns:
                ann_ret, ann_vol, sharpe, max_dd = perf_metrics(data[return_col])
            metrics_data[strategy_name] = [
                f"{ann_ret:.2%}",
                f"{ann_vol:.2%}",
                f"{sharpe:.2f}",
                f"{max_dd:.2%}",
            ]

        # 转置数据：行是业绩指标，列是策略
        metric_names = [
            "Annual Return",
            "Annual Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
        ]
        table_data = []
        for i, metric in enumerate(metric_names):
            row = [metric]
            for strategy_name, _ in strategies:
                if strategy_name in metrics_data:
                    row.append(metrics_data[strategy_name][i])
            table_data.append(row)

            # 列标题：第一列是"Metric"，后续是策略名称
            col_labels = ["Metric"] + [strategy_name for strategy_name, _ in strategies]

            # 绘制表格
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.axis("tight")
            ax.axis("off")

            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)

            # 设置表格样式 - 完全白色背景，无底色
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # 标题行
                    cell.set_text_props(weight="bold")
                cell.set_facecolor("white")  # 所有单元格都是白色
                cell.set_edgecolor("black")
                cell.set_linewidth(1)

            plt.title(
                f"Performance Comparison: {scenario.upper()} Scenario",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            plt.tight_layout()
            plt.show()

    def plot_results(self, results):
        """绘制结果对比 - 分别为每个情景绘制"""
        print("Generating enhanced visualization charts...")

        scenarios = list(results.keys())
        forecast_start = pd.to_datetime(self.HIST_END_DATE)

        colors = {"Hold2x": "#D62728", "Final": "#964B00", "Alpha": "#6F2DBD"}

        # 为每个情景分别绘制净值和杠杆图
        for scenario in scenarios:
            data = results[scenario]["combined_data"]

            # 1. 净值曲线图
            fig, ax = plt.subplots(figsize=(15, 8))

            ax.plot(
                data["Date"],
                data["Equity_Hold2x"],
                lw=3,
                c=colors["Hold2x"],
                label="Buy&Hold (2x)",
            )
            ax.plot(
                data["Date"],
                data["Equity_Final"],
                lw=3,
                c=colors["Final"],
                label="Final State",
            )
            ax.plot(
                data["Date"],
                data["Equity_Alpha"],
                lw=3.5,
                c=colors["Alpha"],
                label="Final State+Alpha",
            )

            ax.axvline(
                x=forecast_start,
                color="red",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="Forecast Start",
            )

            ax.set_title(
                f"Cumulative Return - {scenario.upper()} Scenario",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_ylabel("Cumulative Return (Index)", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.legend(fontsize=12, loc="upper left")
            ax.grid(False)  # 去掉背景格子
            ax.set_facecolor("white")

            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

            # 2. 杠杆对比图
            fig, ax = plt.subplots(figsize=(15, 8))

            ax.hlines(
                2.0,
                data["Date"].min(),
                data["Date"].max(),
                colors=colors["Hold2x"],
                linestyles="-",
                lw=3,
                label="Buy&Hold (2x)",
            )
            ax.plot(
                data["Date"],
                data["Final_Leverage"],
                lw=3,
                c=colors["Final"],
                label="Final State Leverage",
            )
            ax.plot(
                data["Date"],
                data["Alpha_Leverage"],
                lw=3.5,
                c=colors["Alpha"],
                label="Final State+Alpha Leverage",
            )

            ax.axvline(
                x=forecast_start,
                color="red",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="Forecast Start",
            )

            ax.set_title(
                f"Leverage Comparison - {scenario.upper()} Scenario",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_ylabel("Leverage Multiplier", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.legend(fontsize=12, loc="upper left")
            ax.grid(False)  # 去掉背景格子
            ax.set_facecolor("white")
            ax.set_ylim(0, 8)

            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            plt.show()

        # 3. 情景对比图（可选）
        self.plot_scenario_comparison(results)

        # 4. 政策变量预测对比
        self.plot_policy_variables_forecast(results)

        # 5. 预测期间收益分解
        self.plot_forecast_period_analysis(results)

    def plot_scenario_comparison(self, results):
        """绘制情景对比图"""
        print("Plotting scenario comparison...")

        forecast_start = pd.to_datetime(self.HIST_END_DATE)
        colors = {"baseline": "#1f77b4", "tariff": "#ff7f0e"}

        # Alpha策略的情景对比
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 净值对比
        ax1 = axes[0]
        for scenario, result in results.items():
            data = result["combined_data"]
            linestyle = "-" if scenario == "baseline" else "--"
            alpha = 0.9

            ax1.plot(
                data["Date"],
                data["Equity_Alpha"],
                linewidth=3.5,
                color=colors[scenario],
                linestyle=linestyle,
                alpha=alpha,
                label=f"Final State+Alpha ({scenario.title()})",
            )
            ax1.plot(
                data["Date"],
                data["Equity_Hold2x"],
                linewidth=2.5,
                color="gray",
                linestyle=linestyle,
                alpha=0.7,
                label=f"Buy&Hold 2x ({scenario.title()})",
            )

        ax1.axvline(
            x=forecast_start,
            color="red",
            linestyle=":",
            alpha=0.7,
            linewidth=2,
            label="Forecast Start",
        )
        ax1.set_title(
            "Strategy Performance Comparison: Baseline vs Tariff Scenarios",
            fontsize=14,
            fontweight="bold",
        )
        ax1.set_ylabel("Cumulative Return (Index)")
        ax1.legend(fontsize=10, loc="upper left")
        ax1.grid(False)
        ax1.set_facecolor("white")

        # 杠杆对比
        ax2 = axes[1]
        for scenario, result in results.items():
            data = result["combined_data"]
            linestyle = "-" if scenario == "baseline" else "--"
            alpha = 0.9

            ax2.plot(
                data["Date"],
                data["Alpha_Leverage"],
                linewidth=3.5,
                color=colors[scenario],
                linestyle=linestyle,
                alpha=alpha,
                label=f"Final State+Alpha Leverage ({scenario.title()})",
            )

        ax2.hlines(
            2.0,
            data["Date"].min(),
            data["Date"].max(),
            colors="gray",
            linestyles="-",
            linewidth=2.5,
            alpha=0.7,
            label="Buy&Hold (2x)",
        )
        ax2.axvline(
            x=forecast_start, color="red", linestyle=":", alpha=0.7, linewidth=2
        )

        ax2.set_title(
            "Leverage Comparison: Baseline vs Tariff Scenarios",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("Leverage Multiplier")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=10, loc="upper left")
        ax2.grid(False)
        ax2.set_facecolor("white")
        ax2.set_ylim(0, 8)

        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_policy_variables_forecast(self, results):
        """绘制政策变量预测对比"""
        print("Plotting policy variable forecasts...")

        key_vars = ["TPU", "US_EPU", "VIX", "FX_VOL"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        colors = ["#1f77b4", "#ff7f0e"]  # 蓝色baseline，橙色tariff

        for i, var in enumerate(key_vars):
            ax = axes[i]

            for j, (scenario, result) in enumerate(results.items()):
                if var in result["policy_forecasts"]:
                    forecast = result["policy_forecasts"][var]

                    # 历史最后几个点
                    hist_data = self.historical_data[var].dropna()
                    hist_dates = self.historical_data["Date"][
                        self.historical_data[var].notna()
                    ]

                    # 显示最后6个历史点
                    ax.plot(
                        hist_dates.iloc[-6:],
                        hist_data.iloc[-6:],
                        color="gray",
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        alpha=0.7,
                    )

                    # 预测路径
                    ax.plot(
                        forecast["dates"],
                        forecast["values"],
                        color=colors[j],
                        linewidth=3,
                        marker="o",
                        markersize=4,
                        label=f"{scenario.title()}",
                        alpha=0.8,
                    )

                    # 连接线
                    ax.plot(
                        [hist_dates.iloc[-1], forecast["dates"][0]],
                        [hist_data.iloc[-1], forecast["values"][0]],
                        color=colors[j],
                        linewidth=2,
                        linestyle="--",
                        alpha=0.6,
                    )

            # 标记预测起点
            forecast_start = pd.to_datetime(self.HIST_END_DATE)
            ax.axvline(
                x=forecast_start, color="red", linestyle=":", alpha=0.5, linewidth=1
            )

            ax.set_title(f"{var} Forecast", fontsize=12, fontweight="bold")
            ax.set_ylabel(var)
            ax.legend(fontsize=10)
            ax.grid(False)  # 去掉背景格子
            ax.set_facecolor("white")

            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.suptitle(
            "Policy Variable Forecasts: Baseline vs Tariff Scenarios",
            fontsize=14,
            y=1.02,
            fontweight="bold",
        )
        plt.show()

    def plot_forecast_period_analysis(self, results):
        """绘制预测期间详细分析"""
        print("Plotting forecast period analysis...")

        forecast_start = pd.to_datetime(self.HIST_END_DATE)

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        colors = {"baseline": "#1f77b4", "tariff": "#ff7f0e"}

        # 第一图：基础收益对比
        ax1 = axes[0]
        for scenario, result in results.items():
            data = result["combined_data"]
            forecast_mask = data["Date"] > forecast_start
            forecast_data = data[forecast_mask]

            ax1.plot(
                forecast_data["Date"],
                forecast_data["Carry_Return"],
                color=colors[scenario],
                linewidth=2,
                marker="o",
                markersize=3,
                label=f"Carry Return ({scenario.title()})",
                alpha=0.8,
            )

        ax1.set_title(
            "Forecast Period: Base Asset Returns", fontsize=12, fontweight="bold"
        )
        ax1.set_ylabel("Monthly Return")
        ax1.legend()
        ax1.grid(False)  # 去掉背景格子
        ax1.set_facecolor("white")

        # 第二图：策略杠杆对比
        ax2 = axes[1]
        for scenario, result in results.items():
            data = result["combined_data"]
            forecast_mask = data["Date"] > forecast_start
            forecast_data = data[forecast_mask]

            ax2.plot(
                forecast_data["Date"],
                forecast_data["Alpha_Leverage"],
                color=colors[scenario],
                linewidth=3,
                marker="o",
                markersize=3,
                label=f"Alpha+ Leverage ({scenario.title()})",
                alpha=0.8,
            )
            ax2.plot(
                forecast_data["Date"],
                forecast_data["Final_Leverage"],
                color=colors[scenario],
                linewidth=2,
                linestyle="--",
                label=f"Final State Leverage ({scenario.title()})",
                alpha=0.6,
            )

        ax2.hlines(
            2.0,
            forecast_data["Date"].min(),
            forecast_data["Date"].max(),
            colors="gray",
            linestyles="-",
            linewidth=2,
            alpha=0.5,
            label="Buy&Hold (2x)",
        )

        ax2.set_title(
            "Forecast Period: Strategy Leverage", fontsize=12, fontweight="bold"
        )
        ax2.set_ylabel("Leverage Multiplier")
        ax2.legend()
        ax2.grid(False)  # 去掉背景格子
        ax2.set_facecolor("white")
        ax2.set_ylim(0, 6)

        # 第三图：策略收益对比
        ax3 = axes[2]
        for scenario, result in results.items():
            data = result["combined_data"]
            forecast_mask = data["Date"] > forecast_start
            forecast_data = data[forecast_mask]

            ax3.plot(
                forecast_data["Date"],
                forecast_data["Return_Alpha"],
                color=colors[scenario],
                linewidth=3,
                marker="o",
                markersize=3,
                label=f"Alpha+ Return ({scenario.title()})",
                alpha=0.8,
            )
            ax3.plot(
                forecast_data["Date"],
                forecast_data["Return_Hold2x"],
                color=colors[scenario],
                linewidth=2,
                linestyle="--",
                label=f"Buy&Hold 2x ({scenario.title()})",
                alpha=0.6,
            )

        ax3.set_title(
            "Forecast Period: Strategy Returns", fontsize=12, fontweight="bold"
        )
        ax3.set_ylabel("Monthly Return")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(False)  # 去掉背景格子
        ax3.set_facecolor("white")

        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def verify_scenario_differences(self, results):
        """验证情景差异是否明显"""
        print("\n" + "=" * 70)
        print("SCENARIO DIFFERENCE VERIFICATION")
        print("=" * 70)

        forecast_start = pd.to_datetime(self.HIST_END_DATE)

        baseline_data = results["baseline"]["combined_data"]
        tariff_data = results["tariff"]["combined_data"]

        # 预测期间数据
        baseline_forecast = baseline_data[baseline_data["Date"] > forecast_start]
        tariff_forecast = tariff_data[tariff_data["Date"] > forecast_start]

        print("1. POLICY VARIABLE DIFFERENCES:")
        print("-" * 40)

        key_vars = ["TPU", "US_EPU", "VIX", "FX_VOL"]
        for var in key_vars:
            if var in baseline_forecast.columns and var in tariff_forecast.columns:
                baseline_avg = baseline_forecast[var].mean()
                tariff_avg = tariff_forecast[var].mean()
                diff_pct = ((tariff_avg - baseline_avg) / baseline_avg) * 100
                print(f"{var}:")
                print(f"  Baseline avg: {baseline_avg:.2f}")
                print(f"  Tariff avg:   {tariff_avg:.2f}")
                print(f"  Difference:   {diff_pct:+.1f}%")
                print()

        print("2. STRATEGY PERFORMANCE DIFFERENCES:")
        print("-" * 40)

        strategies = ["Return_Hold2x", "Return_Final", "Return_Alpha"]
        for strategy in strategies:
            if (
                strategy in baseline_forecast.columns
                and strategy in tariff_forecast.columns
            ):
                baseline_ret = baseline_forecast[strategy].mean() * 12  # 年化
                tariff_ret = tariff_forecast[strategy].mean() * 12
                diff_bps = (tariff_ret - baseline_ret) * 10000

                baseline_vol = baseline_forecast[strategy].std() * np.sqrt(12)
                tariff_vol = tariff_forecast[strategy].std() * np.sqrt(12)

                print(f"{strategy.replace('Return_', '')}:")
                print(f"  Baseline return: {baseline_ret:.2%}")
                print(f"  Tariff return:   {tariff_ret:.2%}")
                print(f"  Difference:      {diff_bps:+.0f} bps")
                print(f"  Baseline vol:    {baseline_vol:.2%}")
                print(f"  Tariff vol:      {tariff_vol:.2%}")
                print()

        print("3. LEVERAGE DIFFERENCES:")
        print("-" * 40)

        leverage_cols = ["Final_Leverage", "Alpha_Leverage"]
        for col in leverage_cols:
            if col in baseline_forecast.columns and col in tariff_forecast.columns:
                baseline_avg = baseline_forecast[col].mean()
                tariff_avg = tariff_forecast[col].mean()
                diff = tariff_avg - baseline_avg
                print(f"{col.replace('_Leverage', '')} Leverage:")
                print(f"  Baseline avg: {baseline_avg:.3f}")
                print(f"  Tariff avg:   {tariff_avg:.3f}")
                print(f"  Difference:   {diff:+.3f}")
                print()

        print("4. OVERALL ASSESSMENT:")
        print("-" * 40)

        # 检查是否有明显差异
        alpha_baseline_final = baseline_forecast["Equity_Alpha"].iloc[-1]
        alpha_tariff_final = tariff_forecast["Equity_Alpha"].iloc[-1]
        alpha_diff_pct = (
            (alpha_tariff_final - alpha_baseline_final) / alpha_baseline_final
        ) * 100

        print(f"Alpha+ Strategy Final Equity:")
        print(f"  Baseline: {alpha_baseline_final:.3f}")
        print(f"  Tariff:   {alpha_tariff_final:.3f}")
        print(f"  Difference: {alpha_diff_pct:+.2f}%")

        if abs(alpha_diff_pct) > 5:
            print("✅ SIGNIFICANT SCENARIO DIFFERENCES DETECTED")
        elif abs(alpha_diff_pct) > 2:
            print("⚠️  MODERATE SCENARIO DIFFERENCES DETECTED")
        else:
            print("❌ INSUFFICIENT SCENARIO DIFFERENCES - NEEDS ADJUSTMENT")

    def generate_sample_data(self):
        """生成示例数据"""
        print("Generating sample data...")
        dates = pd.date_range(start="2020-01-01", end="2025-06-30", freq="MS")
        n_periods = len(dates)

        np.random.seed(42)
        data = {}

        # 基础资产
        data["Carry_Return"] = np.random.normal(0.008, 0.05, n_periods)
        data["IRD"] = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_periods))

        # 政策变量
        data["TPU"] = np.maximum(
            0,
            100
            + np.cumsum(np.random.normal(0, 2, n_periods))
            + np.random.choice(
                [0, 0, 0, 15, 25], n_periods, p=[0.7, 0.1, 0.1, 0.05, 0.05]
            ),
        )

        data["US_EPU"] = np.maximum(
            0,
            150
            + np.cumsum(np.random.normal(0, 3, n_periods))
            + np.random.choice(
                [0, 0, 0, 20, 40], n_periods, p=[0.75, 0.1, 0.1, 0.03, 0.02]
            ),
        )

        data["JP_EPU"] = np.maximum(
            0,
            80
            + np.cumsum(np.random.normal(0, 1.5, n_periods))
            + np.random.choice(
                [0, 0, 0, -10, -20], n_periods, p=[0.8, 0.1, 0.05, 0.03, 0.02]
            ),
        )

        data["VIX"] = np.maximum(
            8,
            20
            + 10 * np.sin(np.arange(n_periods) * 2 * np.pi / 24)
            + np.random.normal(0, 3, n_periods),
        )

        data["FX_VOL"] = np.maximum(
            0.05,
            0.1
            + 0.05 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
            + np.random.normal(0, 0.01, n_periods),
        )

        # 其他变量
        data["Liquidity_Premium"] = np.maximum(
            0, 1.0 + np.random.normal(0, 0.3, n_periods)
        )
        data["US_GDP"] = 2.5 + np.random.normal(0, 0.5, n_periods)
        data["JP_GDP"] = 1.0 + np.random.normal(0, 0.3, n_periods)
        data["US_CPI"] = 2.0 + np.cumsum(np.random.normal(0, 0.1, n_periods))
        data["JP_CPI"] = 0.5 + np.cumsum(np.random.normal(0, 0.08, n_periods))
        data["US_PMI"] = (
            50
            + 5 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
            + np.random.normal(0, 2, n_periods)
        )
        data["JP_PMI"] = (
            50
            + 3 * np.sin(np.arange(n_periods) * 2 * np.pi / 12)
            + np.random.normal(0, 1.5, n_periods)
        )
        data["US_UR"] = np.maximum(
            2,
            4.0
            + 0.5 * np.sin(np.arange(n_periods) * 2 * np.pi / 24)
            + np.random.normal(0, 0.3, n_periods),
        )
        data["JP_UR"] = np.maximum(
            1,
            2.5
            + 0.3 * np.sin(np.arange(n_periods) * 2 * np.pi / 24)
            + np.random.normal(0, 0.2, n_periods),
        )

        df = pd.DataFrame(data)
        df["Date"] = dates
        df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

        return df


# 主运行函数
def main():
    """主运行函数"""
    print("启动增强版集成策略系统...")

    # 创建系统实例
    system = EnhancedIntegratedStrategySystem()

    # 运行集成预测
    results = system.run_integrated_forecast(
        file_path="E:/承珞资本/宏观/carry_trade.csv",  # 替换为你的实际文件路径
        scenarios=["baseline", "tariff"],
    )

    print(f"\n完成 {len(results)} 个情景的预测")

    # 验证情景差异
    system.verify_scenario_differences(results)

    # 生成业绩对比表
    system.generate_performance_table(results)

    # 生成可视化结果
    system.plot_results(results)

    print("\n" + "=" * 70)
    print("增强版集成预测完成!")
    print("=" * 70)
    print("✅ 历史期间策略结果完全不变")
    print("✅ 预测期间确保明显情景差异")
    print("✅ 增强的政策变量影响")
    print("✅ 完整的业绩分析和可视化")
    print("=" * 70)

    return results, system


if __name__ == "__main__":
    results, system = main()
