"""
第二步：LightGBM 选股 + A 股规则回测
基线 = 微软 Qlib 官方 benchmark（Alpha158 + LightGBM + TopkDropout），
在此基础上叠加散户行为因子，并对比三种特征集：

  --features retail     只用 16 个散户行为因子（轻量，1~2GB 内存可跑）
  --features alpha158   官方基线（158 个通用量价因子）
  --features both       Alpha158 + 散户因子（推荐，需 8GB+ 内存）

用法示例：
  python run_model.py --pool csi1000 --features retail
  python run_model.py --pool csi1000 --features both --topk 30 --n_drop 3
"""
import argparse
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # 新版 mlflow 兼容

import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord

BENCH = {"csi300": "SH000300", "csi500": "SH000905", "csi1000": "SH000852", "csi800": "SH000906"}

# official: Qlib 官方 LightGBM + Alpha158 benchmark 的超参（examples/benchmarks/LightGBM）
# robust  : 更保守的树（小学习率、少叶子、叶子最少样本多），因子少/噪声大时不容易过拟合
LGB_PRESETS = {
    "official": dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.2, subsample=0.8789,
                     lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210,
                     early_stopping_rounds=50, num_boost_round=1000),
    "robust": dict(loss="mse", colsample_bytree=0.8, learning_rate=0.03, subsample=0.8,
                   subsample_freq=1, lambda_l1=10.0, lambda_l2=500.0, max_depth=6, num_leaves=31,
                   min_data_in_leaf=2000, early_stopping_rounds=100, num_boost_round=2000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--pool", default="csi1000")
    ap.add_argument("--features", choices=["retail", "alpha158", "both"], default="retail")
    ap.add_argument("--train", nargs=2, default=["2015-01-01", "2020-12-31"])
    ap.add_argument("--valid", nargs=2, default=["2021-01-01", "2022-12-31"])
    ap.add_argument("--test", nargs=2, default=["2023-01-01", "2026-09-18"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--n_drop", type=int, default=5)
    ap.add_argument("--account", type=float, default=1_000_000)
    ap.add_argument("--kernels", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lgb", choices=list(LGB_PRESETS), default="robust")
    ap.add_argument("--horizon", type=int, default=1,
                    help="标签=未来N日收益（T+1收盘买入）。1=官方默认；5 更贴合散户因子的周频节奏")
    a = ap.parse_args()

    qlib.init(provider_uri=a.provider, region="cn", kernels=a.kernels)
    bench = BENCH.get(a.pool, "SH000852")

    handler_cls = {"retail": ("RetailAlpha", "retail_factors"),
                   "alpha158": ("Alpha158", "qlib.contrib.data.handler"),
                   "both": ("Alpha158Retail", "retail_factors")}[a.features]
    handler = {
        "class": handler_cls[0], "module_path": handler_cls[1],
        "kwargs": {"instruments": a.pool, "start_time": a.train[0], "end_time": a.test[1],
                   "fit_start_time": a.train[0], "fit_end_time": a.train[1],
                   "label": ([f"Ref($close, -{a.horizon + 1})/Ref($close, -1) - 1"], ["LABEL0"])},
    }
    dataset = init_instance_by_config({
        "class": "DatasetH", "module_path": "qlib.data.dataset",
        "kwargs": {"handler": handler,
                   "segments": {"train": tuple(a.train), "valid": tuple(a.valid), "test": tuple(a.test)}},
    })
    model = init_instance_by_config({
        "class": "LGBModel", "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {**LGB_PRESETS[a.lgb], "num_threads": a.threads},
    })

    port_config = {
        "strategy": {
            "class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": a.topk, "n_drop": a.n_drop},
        },
        "backtest": {
            "start_time": a.test[0], "end_time": a.test[1], "account": a.account,
            "benchmark": bench,
            "exchange_kwargs": {
                "limit_threshold": 0.095,   # 涨跌停买卖不了
                "deal_price": "close",
                "open_cost": 0.0005,        # 买入：佣金+过户
                "close_cost": 0.0010,       # 卖出：佣金+印花税(0.05%)+冲击
                "min_cost": 5,
            },
        },
    }

    tag = f"{a.pool}_{a.features}_{a.lgb}_h{a.horizon}"
    with R.start(experiment_name="retail_alpha", recorder_name=tag):
        model.fit(dataset)
        rec = R.get_recorder()
        SignalRecord(model, dataset, rec).generate()
        SigAnaRecord(rec).generate()
        # 特征重要性：看模型到底靠哪些“散户愚蠢”赚钱
        try:
            imp = model.get_feature_importance(importance_type="gain")
            cols = list(dataset.handler.get_cols("feature"))
            imp.index = [cols[int(str(i).split("_")[-1])] if str(i).startswith("Column_") else i for i in imp.index]
        except Exception:  # noqa
            imp = None
        # 回测前释放训练数据，小内存机器也能跑
        del dataset, model
        import gc; gc.collect()
        PortAnaRecord(rec, port_config, "day").generate()

        ic = rec.load_object("sig_analysis/ic.pkl")
        ric = rec.load_object("sig_analysis/ric.pkl")
        report = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        ana = rec.load_object("portfolio_analysis/port_analysis_1day.pkl")

    os.makedirs("output", exist_ok=True)
    report.to_csv(f"output/report_{tag}.csv")
    ana.to_csv(f"output/analysis_{tag}.csv")

    if imp is not None:
        imp.to_csv(f"output/importance_{tag}.csv", header=["gain"])
        print("\n前 15 重要特征:\n", imp.head(15).round(1).to_string())

    print(f"\n==== {tag} | 测试期 {a.test[0]} ~ {a.test[1]} ====")
    print(f"IC={ic.mean():.4f}  ICIR={ic.mean()/ic.std():.2f}  RankIC={ric.mean():.4f}  RankICIR={ric.mean()/ric.std():.2f}")
    print(ana.round(4).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["axes.unicode_minus"] = False
        strat = (1 + report["return"] - report["cost"]).cumprod()
        bm = (1 + report["bench"]).cumprod()
        exc = (1 + report["return"] - report["cost"] - report["bench"]).cumprod()
        fig, ax = plt.subplots(figsize=(11, 5))
        strat.plot(ax=ax, label="strategy (after cost)")
        bm.plot(ax=ax, label=f"benchmark {bench}")
        exc.plot(ax=ax, label="excess (after cost)", ls="--")
        ax.set_title(f"{tag}  test {a.test[0]}~{a.test[1]}")
        ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(f"output/nav_{tag}.png", dpi=110)
        print(f"净值图: output/nav_{tag}.png")
    except Exception as e:  # noqa
        print("画图失败:", e)


if __name__ == "__main__":
    main()
