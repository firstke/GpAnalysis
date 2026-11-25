# gp

一个使用 `src` 布局的 Python 项目，包含最小 CLI 示例和基于 Streamlit + AkShare 的股票技术分析示例。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows 使用: .venv\Scripts\activate
pip install -e ".[dev]"
```

运行示例命令：

```bash
gp --name CodeBuddy
# 或
python -m gp --name CodeBuddy
```

## 股票分析示例

启动 Streamlit 应用：

```bash
.venv/bin/streamlit run src/gp/AkShareDemo.py
```

打开 `http://localhost:8501`，在页面顶部输入股票代码（例如 `600519`）并选择日期范围。

侧边栏可调整：

- **刷新行情**：清除行情缓存并立即重新请求，适合收盘后数据刚更新时使用
- **回看年限**：1–5 年，快速设置默认开始日期
- **涨跌标签阈值**：过滤微小涨跌，降低噪声标签；阈值越高，有效样本越少
- **模型选择**：自动选择、随机森林、逻辑回归、直方图梯度提升

模型分析区指标含义：

- **测试集准确率**：按时间顺序切分后的测试集准确率
- **多数类基线**：始终预测多数类的准确率；模型准确率需明显高于该值才有参考价值
- **F1 / 精确率 / 召回率 / AUC**：综合评估分类效果，缓解类别不平衡带来的误判
- **混淆矩阵**：查看预测下跌/上涨与实际结果的对应关系
- **候选模型交叉验证 F1**：训练期内时间序列交叉验证的平均 F1，用于自动选择模型

预测因子覆盖四类：

- **技术指标**：MACD、KDJ、布林带位置与宽度、ATR
- **量价特征**：振幅、量比、成交额比、换手率变化、量价背离
- **趋势动量**：多周期收益、均线斜率、60 日均线距离、突破/跌破形态
- **市场联动**：上证指数、深证成指同期收益及个股相对强弱；指数数据来自腾讯接口，获取失败时自动跳过该类特征

最近 10 日数据表新增 `标签_1日`、`标签_3日`、`标签_5日` 三列，取值为“涨 / 跌 / 横盘 / —”：阈值内波动显示“横盘”，未来数据不足显示“—”。模型训练仍只使用**次日涨跌二分类标签**，3 日和 5 日标签仅用于展示，不改变准确率、F1、AUC 和混淆矩阵口径。由于新增 60 日窗口因子，有效样本不足约 80 条时会跳过训练。

行情数据来自腾讯接口，默认缓存 1 小时以避免短时间重复请求；点击侧边栏“刷新行情”可立即清除缓存重新获取。股票涨跌预测噪声很高，所有结果仅作技术分析参考，不构成投资建议。

## 打包为可执行文件

使用 PyInstaller 打包，启动后直接打开内嵌桌面窗口运行应用，不依赖外部浏览器，页面布局与交互保持不变。

macOS：

```bash
chmod +x build_mac.sh
./build_mac.sh
```

产物在 `dist/StockAnalysis.app` 和 `dist/StockAnalysis/StockAnalysis`。

Windows（需在 Windows 机器上执行）：

```bat
build_windows.bat
```

产物在 `dist\StockAnalysis\StockAnalysis.exe`。

也可通过 GitHub Actions 获取 Windows 版本：

1. 将项目推送到 GitHub 仓库
2. 进入 `Actions` → `Build Windows App` → `Run workflow`
3. 构建完成后下载产物 `StockAnalysis-Windows`

推送 `v*` 标签时也会自动触发 Windows 构建。

说明：

- PyInstaller 不支持跨平台交叉编译，Windows exe 必须在 Windows 上构建，macOS 应用必须在 macOS 上构建
- 应用通过 pywebview 创建本地窗口：macOS 使用系统 WebKit，Windows 使用 Edge WebView2 Runtime（Win10/11 通常已内置，缺失时需安装 Microsoft Edge WebView2 Runtime）
- 默认从 `8501` 起自动选择第一个可用端口，被占用时不会闪退；可用环境变量 `GP_STREAMLIT_PORT` 固定端口，例如 `GP_STREAMLIT_PORT=8502`
- 打包体积约 470 MB，首次启动会联网拉取行情与指数数据

## 运行测试

```bash
pytest
# 或无需安装任何依赖
PYTHONPATH=src python3 -m unittest discover -s tests
```

## 目录结构

```text
GP/
├── .github/workflows/build-windows.yml
├── pyproject.toml
├── requirements-dev.txt
├── README.md
├── desktop_launcher.py
├── build_mac.sh
├── build_windows.bat
├── src/
│   └── gp/
│       ├── __init__.py
│       ├── __main__.py
│       └── AkShareDemo.py
└── tests/
    ├── test_gp.py
    ├── test_akshare_demo.py
    └── test_desktop_launcher.py
```
