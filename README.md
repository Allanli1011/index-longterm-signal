# Index Futures Breadth Strategy

基于市场宽度（Market Breadth）的全球股指期货交易策略系统。

## 策略原理

监控各大股指成份股中处于200日均线之上的比例（市场宽度指标），利用该指标的极端值和动量变化来识别指数的潜在反转点：

- **超卖信号（做多）**: 当宽度比例降至极低水平（如<25%），且开始回升时，说明市场已充分下跌，可能迎来反弹
- **超买信号（做空）**: 当宽度比例升至极高水平（如>75%），且开始回落时，说明市场过热，可能开始调整

### 增强信号

1. **动量确认**: 不仅看宽度的绝对水平，还看其变化速率——宽度从低位开始回升才是真正的买入信号
2. **宽度背离**: 指数创新低但宽度不创新低，暗示卖压减弱，是更强的反转信号
3. **多周期宽度**: 综合50日、100日、200日均线的宽度，形成多层确认
4. **自适应阈值**: 通过回测优化，为每个指数找到最佳参数

## 覆盖市场

| 指数 | 期货代码 | 国家/地区 |
|------|---------|----------|
| S&P 500 | ES | 美国 |
| NASDAQ 100 | NQ | 美国 |
| Dow Jones | YM | 美国 |
| Russell 2000 | RTY | 美国 |
| FTSE 100 | Z | 英国 |
| DAX 40 | FDAX | 德国 |
| CAC 40 | FCE | 法国 |
| Euro Stoxx 50 | FESX | 欧洲 |
| Nikkei 225 | NIY | 日本 |
| Hang Seng | HSI | 香港 |
| S&P/ASX 200 | AP | 澳大利亚 |
| KOSPI 200 | KS200 | 韩国 |
| FTSE China A50 | XIN | 中国 |
| Nifty 50 | NIFTY | 印度 |

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
# 查看所有可用指数
python main.py --list

# 对单个指数运行回测（使用默认参数）
python main.py --index SP500

# 对单个指数进行参数优化
python main.py --index SP500 --optimize

# Walk-Forward 优化（防止过拟合）
python main.py --index SP500 --walk-forward

# 对所有指数进行回测
python main.py --all

# 对所有指数进行快速优化
python main.py --all --optimize --quick

# 指定日期范围和优化指标
python main.py --index DAX --optimize --start 2010-01-01 --scoring sharpe

# 对选定指数进行优化
python main.py --indices SP500 DAX NIKKEI225 --optimize
```

## 项目结构

```
index-longterm-signal/
├── main.py                  # 主入口
├── config/
│   └── indices.py           # 指数配置（期货合约、成份股来源等）
├── data/
│   ├── constituents.py      # 成份股数据获取
│   └── price_data.py        # 价格数据下载与缓存
├── strategy/
│   ├── breadth.py           # 宽度指标计算引擎
│   └── signals.py           # 交易信号生成
├── backtest/
│   ├── engine.py            # 回测引擎
│   └── optimizer.py         # 参数优化（网格搜索 + Walk-Forward）
├── analysis/
│   └── visualization.py     # 可视化与报告生成
└── output/                  # 回测报告输出目录
```

## 信号模式

| 模式 | 说明 |
|------|------|
| `simple` | 纯阈值：宽度低于超卖线做多，高于超买线做空 |
| `momentum_confirmed` | 阈值 + 动量确认：要求宽度开始回升/回落才入场 |
| `divergence` | 阈值 + 背离检测：出现背离时放宽阈值要求 |
| `composite` | 综合评分：加权结合宽度、动量、背离多个维度 |

## 风险管理

- 固定止损（默认5%）
- 移动止损（默认3%）
- 最大持仓天数限制
- 宽度回归中性时自动平仓
