"""objects.csv (+ mono_depth.csv, s1_results.csv) → 결과표 · 오차 곡선 · Z* · results/REPORT.md"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import binned_errors, crossover_distance, distance_errors
from src.object_depth import DISP_AGGREGATORS, scale_to_reference

OUT = Path("results")
FIG = OUT / "figures"
METHODS = ("bm", "sgbm", "sgbm_wls")
MAX_Z = 80.0


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def moderate(df: pd.DataFrame) -> pd.DataFrame:
    """KITTI moderate 기준: 박스 높이 ≥ 25 px, 가림 ≤ 1, 잘림 ≤ 0.3, 그리고 0 < Z ≤ 80 m."""
    return df[(df.box_h >= 25) & (df.occluded <= 1) & (df.truncated <= 0.3) & (df.z_gt > 0) & (df.z_gt <= MAX_Z)].copy()


def add_mono_depth(df: pd.DataFrame) -> pd.DataFrame:
    p = OUT / "cache" / "mono_depth.csv"
    if not p.exists():
        return df
    md = pd.read_csv(p)
    gt = md[md.kind == "gt"].rename(columns={"z_mono": "mono_gt"})[["idx", "k", "mono_gt"]]
    det = md[md.kind == "det"].rename(columns={"k": "det", "z_mono": "mono_det"})[["idx", "det", "mono_det"]]
    df = df.merge(gt, on=["idx", "k"], how="left")
    if "det" in df:
        df = df.merge(det, on=["idx", "det"], how="left")
    # 스케일 보정: 같은 영상의 *다른* 객체들의 핀홀 거리로 s를 정한다 (leave-one-out, 자기 자신 복사 방지).
    scaled = np.full(len(df), np.nan)
    for _, g in df.groupby("idx"):
        if len(g) < 2:
            continue
        for i in g.index:
            others = g.drop(index=i)
            _, s = scale_to_reference(others.mono_gt.to_numpy(), others.pin_gt.to_numpy())
            scaled[df.index.get_loc(i)] = g.loc[i, "mono_gt"] * s
    df["mono_gt_scaled"] = scaled
    return df


def stats_table(df: pd.DataFrame, cols: dict[str, str], gt_col: str = "z_gt") -> pd.DataFrame:
    rows = []
    for label, c in cols.items():
        if c in df:
            rows.append({"method": label, **distance_errors(df[c].to_numpy(), df[gt_col].to_numpy())})
    return pd.DataFrame(rows)


def disparity_error_px(df: pd.DataFrame, col: str, fB: pd.Series) -> float:
    """거리 오차를 시차 영역으로 되돌린 |Δd| 중앙값(px) — 스테레오 이론 곡선의 δd."""
    ok = np.isfinite(df[col]) & (df[col] > 0)
    return float(np.median(np.abs(fB[ok] / df.loc[ok, col] - fB[ok] / df.loc[ok, "z_gt"])))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(OUT / "objects.csv")
    df = moderate(raw).reset_index(drop=True)
    df = add_mono_depth(df)
    heights = json.loads((OUT / "class_heights.json").read_text())
    report = ["# 결과 보고서 (자동 생성: `scripts/analyze.py`)\n"]
    report.append(f"- 평가 객체: **{len(df)}**개 / {df.idx.nunique()} 프레임 (KITTI object training 홀수 id, moderate 필터, Z ≤ {MAX_Z:.0f} m)")
    report.append(f"- 클래스별 객체 수: " + ", ".join(f"{c} {n}" for c, n in df.cls.value_counts().items()))
    report.append("- 클래스 평균 높이 (짝수 id에서 추정): " + ", ".join(f"{c} {h:.2f} m" for c, h in heights.items()) + "\n")

    # ---------- S1 ----------
    s1p = OUT / "s1_results.csv"
    if s1p.exists():
        s1 = pd.read_csv(s1p)
        cols = ["method", "block_size", "d1_all", "d1_noc", "epe", "d1_valid", "density", "time_ms"]
        report.append("## S1. 시차 알고리즘 (Stereo 2015, 테스트 100장)\n")
        report.append(md_table(s1[cols]))
        report.append("D1/D1_valid/density 단위 %, EPE px. `d1_all`·`epe`는 KITTI 방식(무효 픽셀을 배경 보간), `d1_valid`는 유효 예측만.\n")

    # ---------- S2 ----------
    s2_rows = []
    for m in METHODS:
        for agg in DISP_AGGREGATORS:
            c = f"st_{m}_{agg}"
            for gtc in ("z_gt", "z_near"):
                s2_rows.append({"stereo": m, "aggregator": agg, "gt": gtc, **distance_errors(df[c].to_numpy(), df[gtc].to_numpy())})
    s2 = pd.DataFrame(s2_rows)
    s2.to_csv(OUT / "s2_results.csv", index=False)
    piv = s2.pivot_table(index=["stereo", "aggregator"], columns="gt", values=["rel_median", "mae"]).reset_index()
    piv.columns = ["stereo", "aggregator"] + [f"{a}({b})" for a, b in piv.columns[2:]]
    piv["fail_rate"] = s2[s2.gt == "z_gt"].set_index(["stereo", "aggregator"]).loc[list(zip(piv.stereo, piv.aggregator)), "fail_rate"].to_numpy()
    report.append("## S2. 객체 시차 추출 방식 (GT 박스)\n")
    report.append(md_table(piv))
    report.append("`z_gt` = 3D 박스 중심 깊이(라벨), `z_near` = 가장 가까운 꼭짓점 깊이(스테레오가 실제로 보는 표면에 가까움). rel_median은 %, MAE는 m.\n")
    best_agg = {m: s2[(s2.stereo == m) & (s2.gt == "z_gt")].sort_values("rel_median").iloc[0].aggregator for m in METHODS}
    report.append("방식별 최적 집계: " + ", ".join(f"{m} → `{a}`" for m, a in best_agg.items()) + "\n")

    st_cols = {f"D3 stereo {m} ({best_agg[m]})": f"st_{m}_{best_agg[m]}" for m in METHODS}
    best_stereo = min(METHODS, key=lambda m: s2[(s2.stereo == m) & (s2.aggregator == best_agg[m]) & (s2.gt == "z_gt")].rel_median.iloc[0])
    st_best_col = f"st_{best_stereo}_{best_agg[best_stereo]}"

    # ---------- D1–D4 ----------
    cols = {
        "D1 pinhole (GT box, class H)": "pin_gt",
        "D1* pinhole (GT box, true H)": "pin_gt_trueH",
        "D2 pinhole (det box)": "pin_det",
        **st_cols,
        f"D3' stereo {best_stereo} (det box)": f"st_det_{best_stereo}",
        "D4 mono depth (raw metric)": "mono_gt",
        "D4 mono depth (pinhole scale)": "mono_gt_scaled",
    }
    cols = {k: v for k, v in cols.items() if v in df}
    d_all = stats_table(df, cols)
    report.append("## D1–D4. 객체 거리 오차 (전체)\n")
    report.append(md_table(d_all))
    report.append("rel은 |Ẑ−Z|/Z (%), bias = 평균(Ẑ−Z) m, fail_rate = 거리를 못 구한 비율 % "
                  "(D2/D3'는 검출 누락 포함, D4 scale은 한 장면에 객체가 1개뿐이면 실패).\n")
    if "pin_det" in df:
        matched = df[np.isfinite(df.pin_det)]
        report.append(f"검출기 매칭(IoU ≥ 0.5) 재현율: **{100 * len(matched) / len(df):.1f}%**. 매칭된 객체만 비교:\n")
        report.append(md_table(stats_table(matched, {"D1 pinhole (GT box)": "pin_gt", "D2 pinhole (det box)": "pin_det",
                                                     f"D3 stereo {best_stereo} (GT box)": st_best_col,
                                                     f"D3' stereo {best_stereo} (det box)": f"st_det_{best_stereo}"})))

    # 클래스별
    per_cls = []
    for c, g in df.groupby("cls"):
        for label, col in (("D1 pinhole", "pin_gt"), (f"D3 stereo {best_stereo}", st_best_col)):
            per_cls.append({"class": c, "method": label, **distance_errors(g[col].to_numpy(), g.z_gt.to_numpy())})
    report.append("### 클래스별\n")
    report.append(md_table(pd.DataFrame(per_cls)[["class", "method", "n", "fail_rate", "mae", "bias", "rel_median"]]))

    # ---------- 구간별 곡선 & Z* ----------
    curve_cols = list(cols.values())
    binned = binned_errors(df, curve_cols, max_m=MAX_Z)
    binned.to_csv(OUT / "binned_errors.csv", index=False)
    binned_near = binned_errors(df, curve_cols, gt_col="z_near", max_m=MAX_Z)

    zs_rows = []
    for label, col in st_cols.items():
        for ref_label, ref in (("D1 GT box", "pin_gt"), ("D2 det box", "pin_det"), ("D4 scaled", "mono_gt_scaled")):
            if ref not in df:
                continue
            for metric in ("rel_median", "mae"):
                zs_rows.append({"stereo": label, "vs": ref_label, "metric": metric,
                                "Z*(z_gt)": crossover_distance(binned, col, ref, metric),
                                "Z*(z_near)": crossover_distance(binned_near, col, ref, metric)})
    zs = pd.DataFrame(zs_rows)
    zs.to_csv(OUT / "crossover.csv", index=False)
    report.append("## 교차 거리 Z*\n")
    report.append(md_table(zs, "{:.1f}"))
    report.append("구간(5 m)별 지표에서 스테레오가 더 나은 상태에서 단안이 더 나아지는 첫 거리 (선형 보간, 구간당 n ≥ 20). – = 교차 없음.\n")

    # 이론 곡선: 스테레오 δZ/Z = Z·δd/(f·B), 핀홀 δZ/Z ≈ δH/H (일정)
    fB_med = float(df.fB.median())
    dd = disparity_error_px(df, st_best_col, df.fB)
    dH = float(np.median(np.abs(df.h_true - df.cls.map(heights)) / df.cls.map(heights)))
    report.append(f"### 이론 대비\n\n- 최적 스테레오({best_stereo})의 시차 오차 중앙값 δd ≈ **{dd:.2f} px** → 이론 상대 오차 Z·δd/(f·B)\n"
                  f"- 높이 가정 오차 중앙값 |δH|/H ≈ **{100 * dH:.1f}%** → 핀홀 상대 오차의 바닥\n"
                  f"- 두 식이 같아지는 이론 교차점 Z* = f·B·(δH/H)/δd ≈ **{fB_med * dH / dd:.1f} m**\n")

    # ---------- 그림 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, ylabel in ((axes[0], "rel_median", "median |ΔZ|/Z (%)"), (axes[1], "mae", "MAE (m)"), (axes[2], "fail_rate", "fail rate (%)")):
        for label, col in cols.items():
            b = binned[(binned.method == col) & (binned.n >= 20)]
            ls = "--" if "det" in label or "true H" in label else "-"
            ax.plot(b.bin_mid, b[metric], ls, marker="o", ms=3, label=label)
        if metric == "rel_median":
            z = np.linspace(2.5, MAX_Z, 100)
            ax.plot(z, 100 * z * dd / fB_med, "k:", label=f"theory stereo δd={dd:.2f}px")
            ax.axhline(100 * dH, color="gray", ls=":", label=f"theory pinhole |δH|/H={100 * dH:.1f}%")
        ax.set_xlabel("GT distance Z (m)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Distance error vs GT distance (5 m bins, KITTI object eval split, moderate)")
    fig.tight_layout()
    fig.savefig(FIG / "error_curves.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, (label, col) in zip(axes, (("D1 pinhole (GT box)", "pin_gt"), (f"D3 stereo {best_stereo}", st_best_col), ("D4 mono (raw)", "mono_gt"))):
        if col in df:
            ax.scatter(df.z_gt, df[col], s=2, alpha=0.3)
        ax.plot([0, MAX_Z], [0, MAX_Z], "k--", lw=1)
        ax.set_title(label)
        ax.set_xlabel("GT Z (m)")
        ax.set_xlim(0, MAX_Z)
        ax.set_ylim(0, MAX_Z * 1.3)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("estimated Z (m)")
    fig.tight_layout()
    fig.savefig(FIG / "scatter.png", dpi=130)
    report.append("## 그림\n\n![error curves](figures/error_curves.png)\n\n![scatter](figures/scatter.png)\n")

    (OUT / "REPORT.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
