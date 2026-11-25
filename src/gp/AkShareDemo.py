# -*- coding: utf-8 -*-
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Optional, TypedDict


class EvaluationResult(TypedDict):
    model: Any
    model_name: str
    cv_scores: Dict[str, float]
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: Optional[float]
    confusion: Any
    baseline: float
    n_train: int
    n_test: int
    positive_rate: float


st.set_page_config(
    page_title="股票技术分析",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _apply_page_styles():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { min-width: 280px; max-width: 280px; }
            [data-testid="stSidebar"] > div:first-child { width: 280px; }
            .block-container {
                padding-top: 1.5rem;
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 100%;
            }
            div[data-testid="stMetric"] {
                background: #f7f9fc;
                border: 1px solid #e6eaf0;
                border-radius: 10px;
                padding: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


today = datetime.now().date()

# 股票代码补全市场前缀，供腾讯接口使用
def _with_market_prefix(code):
    code = str(code).strip().lower()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if code.startswith("6"):
        return "sh" + code
    if code.startswith(("0", "3")):
        return "sz" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sh" + code


# 股票代码与名称映射，来自交易所列表，缓存一天
@st.cache_data(ttl=86400, show_spinner="正在获取股票名称...")
def _stock_name_map():
    df = ak.stock_info_a_code_name()
    return dict(zip(df["code"].astype(str), df["name"].astype(str)))


def get_stock_name(code):
    code = str(code).strip().lower()
    code = code[2:] if code.startswith(("sh", "sz", "bj")) else code
    return _stock_name_map().get(code, code)


# 获取股票数据：默认使用腾讯接口，避免东方财富 push2 限流
@st.cache_data(ttl=3600, show_spinner="正在获取行情数据...")
def load_data(code, start, end):
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    df = ak.stock_zh_a_hist_tx(
        symbol=_with_market_prefix(code),
        start_date=start_str,
        end_date=end_str,
        adjust="qfq",
    )
    df = df.rename(columns={
        "date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
        "turnover": "换手率",
    })

    # 转换日期并排序，索引只显示日期不显示 00:00:00
    df['日期'] = pd.to_datetime(df['日期']).dt.date
    df = df.sort_values('日期')
    df.set_index('日期', inplace=True)
    return df


@st.cache_data(ttl=3600, show_spinner="正在获取指数数据...")
def load_market_data(start, end):
    """加载上证与深证成指日收益；失败时返回空表以便优雅降级。"""
    columns = ["上证收益_1日", "深证收益_1日"]
    try:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        frames = []
        for symbol, name in [("sh000001", "上证"), ("sz399001", "深证")]:
            idx = ak.index_zh_daily_tx(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
            )
            if idx.empty:
                return pd.DataFrame(columns=columns)
            frames.append(pd.DataFrame({
                "date": pd.to_datetime(idx["date"]).dt.date,
                f"{name}收盘": pd.to_numeric(idx["close"], errors="coerce"),
            }))
        market = pd.merge(frames[0], frames[1], on="date", how="inner")
        market["上证收益_1日"] = market["上证收盘"].pct_change()
        market["深证收益_1日"] = market["深证收盘"].pct_change()
        return market.set_index("date")[columns].dropna()
    except Exception:
        return pd.DataFrame(columns=columns)


FEATURE_COLUMNS = [
    # 基础量价
    "收益_1日",
    "收益_5日",
    "收益_20日",
    "收盘_5日均线距离",
    "收盘_20日均线距离",
    "RSI",
    "波动率_20日",
    "成交量比_5日",
    "成交额比_5日",
    "换手率",
    # 技术指标
    "MACD_DIF",
    "MACD_DEA",
    "MACD_HIST",
    "KDJ_K",
    "KDJ_D",
    "KDJ_J",
    "BOLL_POS",
    "BOLL_WIDTH",
    "ATR_14",
    # 量价特征
    "振幅",
    "量比_20日",
    "成交额比_20日",
    "换手率变化",
    "量价背离",
    # 趋势动量
    "收益_10日",
    "收益_60日",
    "MA20斜率",
    "MA60斜率",
    "收盘_MA60距离",
    "突破20日新高",
    "跌破20日新低",
    # 市场联动
    "上证收益_1日",
    "深证收益_1日",
    "相对上证_1日",
    "相对深证_1日",
]

MIN_SAMPLES = 80


def build_features(df: pd.DataFrame, market_df: pd.DataFrame = None) -> pd.DataFrame:
    """仅使用当前及历史数据生成特征，避免未来信息泄漏。"""
    out = pd.DataFrame(index=df.index)
    close = df["收盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    volume = df["成交量"].astype(float)
    amount = df["成交额"].astype(float)

    out["收益_1日"] = close.pct_change()
    out["收益_5日"] = close.pct_change(5)
    out["收益_10日"] = close.pct_change(10)
    out["收益_20日"] = close.pct_change(20)
    out["收益_60日"] = close.pct_change(60)

    ma5 = close.rolling(5, min_periods=5).mean()
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    out["收盘_5日均线距离"] = close / ma5 - 1
    out["收盘_20日均线距离"] = close / ma20 - 1
    out["收盘_MA60距离"] = close / ma60 - 1
    out["MA20斜率"] = (ma20 - ma20.shift(5)) / ma20.shift(5).replace(0, np.nan)
    out["MA60斜率"] = (ma60 - ma60.shift(5)) / ma60.shift(5).replace(0, np.nan)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean().replace(0, 1e-6)
    out["RSI"] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    out["MACD_DIF"] = dif
    out["MACD_DEA"] = dea
    out["MACD_HIST"] = (dif - dea) * 2

    low9 = low.rolling(9, min_periods=9).min()
    high9 = high.rolling(9, min_periods=9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    out["KDJ_K"] = k
    out["KDJ_D"] = d
    out["KDJ_J"] = 3 * k - 2 * d

    boll_std = close.rolling(20, min_periods=20).std()
    boll_upper = ma20 + 2 * boll_std
    boll_lower = ma20 - 2 * boll_std
    out["BOLL_POS"] = (close - boll_lower) / (boll_upper - boll_lower).replace(0, np.nan)
    out["BOLL_WIDTH"] = (boll_upper - boll_lower) / ma20.replace(0, np.nan)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = tr.rolling(14, min_periods=14).mean() / close

    out["波动率_20日"] = out["收益_1日"].rolling(20, min_periods=20).std()
    out["振幅"] = (high - low) / prev_close
    out["成交量比_5日"] = volume / volume.rolling(5, min_periods=5).mean().replace(0, np.nan)
    out["量比_20日"] = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)
    out["成交额比_5日"] = amount / amount.rolling(5, min_periods=5).mean().replace(0, np.nan)
    out["成交额比_20日"] = amount / amount.rolling(20, min_periods=20).mean().replace(0, np.nan)
    out["换手率"] = df["换手率"].astype(float)
    out["换手率变化"] = out["换手率"].pct_change()
    vol_change = volume.pct_change()
    out["量价背离"] = (
        ((out["收益_1日"] > 0) & (vol_change < 0))
        | ((out["收益_1日"] < 0) & (vol_change > 0))
    ).astype(float)

    out["突破20日新高"] = close / high.rolling(20, min_periods=20).max() - 1
    out["跌破20日新低"] = low.rolling(20, min_periods=20).min() / close - 1

    if market_df is not None and not market_df.empty:
        aligned = market_df.reindex(df.index).ffill()
        out["上证收益_1日"] = aligned["上证收益_1日"]
        out["深证收益_1日"] = aligned["深证收益_1日"]
        out["相对上证_1日"] = out["收益_1日"] - out["上证收益_1日"]
        out["相对深证_1日"] = out["收益_1日"] - out["深证收益_1日"]

    return out.replace([np.inf, -np.inf], np.nan).dropna()


def make_labels(df: pd.DataFrame, threshold: float = 0.0, horizon: int = 1) -> pd.Series:
    """按未来 horizon 日相对收益生成二分类训练标签，微小波动会剔除。"""
    future_close = df["收盘"].shift(-horizon)
    ret = future_close / df["收盘"] - 1
    labels = pd.Series(index=df.index, dtype="float64")
    labels[ret > threshold] = 1
    labels[ret < -threshold] = 0
    return labels.dropna().astype(int)


def make_multi_horizon_labels(
    df: pd.DataFrame,
    threshold: float = 0.0,
    horizons=(1, 3, 5),
) -> pd.DataFrame:
    """生成展示用多周期标签：1涨、0跌、2横盘、NaN表示未来数据不足。"""
    labels = pd.DataFrame(index=df.index)
    for horizon in horizons:
        future_close = df["收盘"].shift(-horizon)
        ret = future_close / df["收盘"] - 1
        label = pd.Series(index=df.index, dtype="float64")
        label[ret > threshold] = 1
        label[ret < -threshold] = 0
        label[ret.abs() <= threshold] = 2
        labels[f"标签_{horizon}日"] = label
    return labels


def prepare_dataset(df: pd.DataFrame, threshold: float = 0.0, market_df: pd.DataFrame = None):
    features = build_features(df, market_df)
    labels = make_labels(df, threshold, horizon=1)
    index = features.index.intersection(labels.index)
    available_columns = [column for column in FEATURE_COLUMNS if column in features.columns]
    return features.loc[index, available_columns], labels.loc[index]


def _candidate_models(model_option: str):
    models = {
        "随机森林": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
        ),
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "直方图梯度提升": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }
    if model_option != "自动选择":
        return {model_option: models[model_option]}
    return models


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    model_option: str = "自动选择",
) -> Optional[EvaluationResult]:
    """时序切分 + 训练内交叉验证选模，返回测试集多指标结果。"""
    if len(X) < MIN_SAMPLES or y.nunique() < 2:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None

    candidates = _candidate_models(model_option)
    n_splits = min(5, max(2, len(X_train) // 20))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    cv_scores: Dict[str, float] = {}
    for name, model in candidates.items():
        fold_scores = []
        for tr_idx, va_idx in splitter.split(X_train):
            if y_train.iloc[va_idx].nunique() < 2:
                continue
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            pred = model.predict(X_train.iloc[va_idx])
            fold_scores.append(
                f1_score(y_train.iloc[va_idx], pred, zero_division=0)
            )
        cv_scores[name] = float(np.mean(fold_scores)) if fold_scores else 0.0

    best_name = max(cv_scores.keys(), key=lambda name: cv_scores[name])
    best_model = candidates[best_name]
    best_model.fit(X_train, y_train)

    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else None

    return {
        "model": best_model,
        "model_name": best_name,
        "cv_scores": cv_scores,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
        "confusion": confusion_matrix(y_test, pred, labels=[0, 1]),
        "baseline": max(float(y_test.mean()), 1 - float(y_test.mean())),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate": float(y.mean()),
    }


def app():
    _apply_page_styles()
    st.title("股票技术分析")

    st.sidebar.header("数据")
    if st.sidebar.button("刷新行情", use_container_width=True):
        getattr(load_data, "clear")()
        st.sidebar.success("已清除行情缓存，正在重新获取...")
    st.sidebar.caption("行情默认缓存 1 小时，收盘后如需立即更新可点击刷新")

    st.sidebar.header("模型设置")
    lookback_years = st.sidebar.slider("回看年限（年）", 1, 5, 1)
    st.sidebar.caption("调整后自动更新页面上的开始日期")
    threshold_pct = st.sidebar.slider("涨跌标签阈值（%）", 0.0, 2.0, 0.0, 0.1)
    model_option = st.sidebar.selectbox(
        "模型选择",
        ["自动选择", "随机森林", "逻辑回归", "直方图梯度提升"],
    )

    default_start = today - timedelta(days=365 * lookback_years)
    code_col, start_col, end_col = st.columns([2, 1, 1])
    with code_col:
        stock_code = st.text_input("股票代码", "600519")
    with start_col:
        start_date = st.date_input(
            "开始日期",
            default_start,
            key=f"start_date_{lookback_years}",
        )
    with end_col:
        end_date = st.date_input("结束日期", today)

    try:
        if start_date >= end_date:
            st.error("错误：开始日期必须早于结束日期！")
        else:
            stock_name = get_stock_name(stock_code)
            df = load_data(stock_code, start_date, end_date)
            market_df = load_market_data(start_date, end_date)

            if df.empty:
                st.error("数据获取失败，请检查：1.股票代码有效性 2.日期范围包含交易日")
            else:
                label_df = make_multi_horizon_labels(df, threshold_pct / 100)
                display_df = df.tail(10).join(label_df).copy()
                for column in ["标签_1日", "标签_3日", "标签_5日"]:
                    display_df[column] = display_df[column].map(
                        {1: "涨", 0: "跌", 2: "横盘"}
                    ).fillna("—")

                st.subheader(f"{stock_name}（{stock_code}） · 最近 10 个交易日")
                st.dataframe(
                    display_df.style.format({
                        "开盘": "{:.2f}",
                        "收盘": "{:.2f}",
                        "最高": "{:.2f}",
                        "最低": "{:.2f}",
                        "成交量": "{:,.0f}",
                        "成交额": "{:,.0f}",
                        "换手率": "{:.4f}",
                    }),
                    height=340,
                    use_container_width=True,
                )

                X, y = prepare_dataset(df, threshold_pct / 100, market_df)
                result = evaluate_models(X, y, model_option=model_option)

                if result is None:
                    st.warning(
                        "有效样本不足或类别过于单一，暂不进行模型训练。"
                        "请延长回看年限（建议 2 年以上）或降低涨跌标签阈值。"
                    )
                else:
                    latest_features = build_features(df, market_df).iloc[-1:][X.columns]
                    latest_pred = result["model"].predict(latest_features)[0]
                    latest_prob = result["model"].predict_proba(latest_features)[0, 1]

                    st.subheader("模型分析")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("测试集准确率", f"{result['accuracy'] * 100:.1f}%")
                    m2.metric("多数类基线", f"{result['baseline'] * 100:.1f}%")
                    m3.metric("F1", f"{result['f1']:.3f}")
                    m4.metric(
                        "AUC",
                        f"{result['auc']:.3f}" if result["auc"] is not None else "—",
                    )

                    m5, m6, m7, m8 = st.columns(4)
                    m5.metric("精确率", f"{result['precision']:.3f}")
                    m6.metric("召回率", f"{result['recall']:.3f}")
                    m7.metric("最新收盘价", f"{df['收盘'].iloc[-1]:.2f}")
                    m8.metric(
                        "预测明日",
                        "看涨" if latest_pred == 1 else "看跌",
                        delta=f"{latest_prob * 100:.1f}%",
                    )

                    info_col, cm_col = st.columns([2, 1])
                    with info_col:
                        st.caption(
                            f"最佳模型：{result['model_name']}｜"
                            f"交叉验证 F1：{result['cv_scores'][result['model_name']]:.3f}｜"
                            f"训练 {result['n_train']} 条 / 测试 {result['n_test']} 条｜"
                            f"正类占比 {result['positive_rate']:.1%}"
                        )
                        with st.expander("查看候选模型交叉验证结果"):
                            st.dataframe(
                                pd.Series(result["cv_scores"], name="平均F1").round(3),
                                use_container_width=True,
                            )
                    with cm_col:
                        st.write("混淆矩阵")
                        cm = pd.DataFrame(
                            result["confusion"],
                            index=["实际下跌", "实际上涨"],
                            columns=["预测下跌", "预测上涨"],
                        )
                        st.dataframe(cm, use_container_width=True)

                    st.info(
                        "准确率需高于多数类基线才有参考价值；股票涨跌预测噪声很高，"
                        "结果仅作技术分析参考。"
                    )

                st.divider()
                st.subheader("技术指标可视化")
                ma20 = df["收盘"].rolling(20, min_periods=20).mean()
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['开盘'],
                    high=df['最高'],
                    low=df['最低'],
                    close=df['收盘'],
                    increasing_line_color='red',
                    decreasing_line_color='green',
                    name='K线'
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=ma20,
                    line=dict(color='orange', width=2),
                    name='20日均线'
                ))
                fig.update_layout(
                    height=520,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"系统错误：{str(e)}")


# 主程序
if __name__ == "__main__":
    app()