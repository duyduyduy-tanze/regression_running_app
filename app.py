# ==============================================================================
# PHÂN TÍCH THÁI ĐỘ ĐỐI VỚI CÔNG NGHỆ NHẬN DIỆN KHUÔN MẶT  -  Streamlit
# Chạy:  streamlit run app.py
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------------------------
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st

st.set_page_config(page_title="Phân tích thái độ đối với công nghệ nhận diện khuôn mặt",
                   page_icon="📊", layout="wide")

# ------------------------------------------------------------------------------
# 2. CẤU HÌNH & LOAD EXCEL DATA
# ------------------------------------------------------------------------------
DEFAULT_EXCEL_PATH = "facial_recognition_SAMPLE_test_data (1).xlsx"   # đặt file cùng thư mục với app.py
SHEET_NAME = 0

# Mapping cột theo VỊ TRÍ (Google Forms lặp tên cột ở 3 scenario block)
COL = {
    "consent":       1,
    "attention":     6,
    "birth_month":   9,
    "privacy_items": [2, 4, 7],
    "blocks": {
        1: {"support": [10, 11, 12], "manip": 13},   # Missing-person identification
        2: {"support": [14, 15, 16], "manip": 17},   # General surveillance / crime prevention
        3: {"support": [18, 19, 20], "manip": 21},   # Protest / political-activity monitoring
    },
}
EXPECTED_HEADERS = {
    1: "agree to take part", 6: "please select 2", 9: "born",
    2: "serious threat to people", 4: "face could be scanned and recorded",
    7: "face data collected by the police will be misused",
    10: "I would support", 11: "should be allowed", 12: "acceptable use",
    14: "I would support", 15: "should be allowed", 16: "acceptable use",
    18: "I would support", 19: "should be allowed", 20: "acceptable use",
    13: "what were the police using", 17: "what were the police using",
    21: "what were the police using",
}
SCENARIO_LABELS = {
    1: "Missing-person identification",
    2: "General surveillance / crime prevention",
    3: "Protest / political-activity monitoring",
}
SCENARIO_LABELS_VI = {
    1: "Tìm người mất tích",
    2: "Giám sát chung / phòng ngừa tội phạm",
    3: "Giám sát biểu tình / hoạt động chính trị",
}
MANIP_CORRECT = {
    1: "Finding missing people",
    2: "Identifying people in a busy area to help prevent crime",
    3: "Identifying people taking part in a protest",
}
MONTH_TO_SCENARIO = {
    "January": 1, "April": 1, "July": 1, "October": 1,
    "February": 2, "May": 2, "August": 2, "November": 2,
    "March": 3, "June": 3, "September": 3, "December": 3,
}


@st.cache_data(show_spinner=False)
def load_excel(source):
    """Đọc file Excel (đường dẫn hoặc file upload) bằng pandas + openpyxl."""
    return pd.read_excel(source, sheet_name=SHEET_NAME, engine="openpyxl").reset_index(drop=True)


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------------------
# 3-4. DATA CLEANING & VARIABLE CONSTRUCTION
# ------------------------------------------------------------------------------
def clean_and_build(raw_data):
    R = {}
    warnings = []
    for idx, text in EXPECTED_HEADERS.items():
        if idx >= raw_data.shape[1] or text.lower() not in str(raw_data.columns[idx]).lower():
            name = raw_data.columns[idx] if idx < raw_data.shape[1] else "không tồn tại"
            warnings.append(f"Cột #{idx} ('{name}') không khớp với nội dung mong đợi '{text}'.")
    R["structure_warnings"] = warnings
    if raw_data.shape[1] < 22:
        R["fatal"] = "File Excel có ít hơn 22 cột nên không khớp cấu trúc khảo sát mong đợi."
        return R

    data = raw_data.copy()
    R["n_initial"] = len(data)

    # Bước 1 - Consent
    consent_col = data.iloc[:, COL["consent"]].astype(str).str.strip()
    data["consent_ok"] = consent_col.str.lower().str.startswith("yes")
    R["n_removed_consent"] = int((~data["consent_ok"]).sum())

    # Bước 2 - Attention check (pass chỉ khi = 2)
    attention_raw = pd.to_numeric(data.iloc[:, COL["attention"]], errors="coerce")
    data["attention_check"] = np.where(attention_raw == 2, "pass", "fail")
    after_consent = data[data["consent_ok"]].copy()
    R["n_removed_attention"] = int((after_consent["attention_check"] == "fail").sum())
    clean_data = after_consent[after_consent["attention_check"] == "pass"].copy().reset_index(drop=True)
    R["n_clean"] = len(clean_data)

    # privacy_risk
    privacy_df = clean_data.iloc[:, COL["privacy_items"]].apply(pd.to_numeric, errors="coerce")
    clean_data["privacy_risk"] = privacy_df.mean(axis=1, skipna=False)

    # scenario: theo block có câu trả lời
    block_matrix = pd.DataFrame({
        b: clean_data.iloc[:, c["support"] + [c["manip"]]].notna().any(axis=1)
        for b, c in COL["blocks"].items()
    })

    def _scn(row):
        answered = [b for b, v in row.items() if v]
        return answered[0] if len(answered) == 1 else np.nan

    clean_data["scenario"] = block_matrix.apply(_scn, axis=1)
    R["n_invalid_block"] = int(clean_data["scenario"].isna().sum())
    month_scn = clean_data.iloc[:, COL["birth_month"]].astype(str).str.strip().map(MONTH_TO_SCENARIO)
    R["n_month_mismatch"] = int(((month_scn != clean_data["scenario"])
                                 & clean_data["scenario"].notna() & month_scn.notna()).sum())

    # support: chỉ dùng câu trả lời của đúng block
    def _support(i):
        s = clean_data.at[i, "scenario"]
        if pd.isna(s):
            return np.nan
        vals = pd.to_numeric(clean_data.iloc[i, COL["blocks"][int(s)]["support"]], errors="coerce")
        return vals.mean() if vals.notna().all() else np.nan

    clean_data["support"] = [_support(i) for i in range(len(clean_data))]

    # manipulation_check: 1 = đúng, 0 = sai
    def _manip(i):
        s = clean_data.at[i, "scenario"]
        if pd.isna(s):
            return np.nan
        ans = str(clean_data.iloc[i, COL["blocks"][int(s)]["manip"]]).strip()
        return 1 if ans == MANIP_CORRECT[int(s)] else 0

    clean_data["manipulation_check"] = [_manip(i) for i in range(len(clean_data))]
    clean_data["scenario_name"] = clean_data["scenario"].map(SCENARIO_LABELS)

    mv = clean_data["manipulation_check"].dropna()
    R["n_manip_pass"] = int((mv == 1).sum())
    R["n_manip_fail"] = int((mv == 0).sum())
    R["manip_rate"] = R["n_manip_pass"] / len(mv) if len(mv) else np.nan

    vars_ = ["privacy_risk", "support", "scenario", "manipulation_check"]
    R["missing_table"] = pd.DataFrame({
        "Biến": vars_,
        "Số giá trị thiếu": [int(clean_data[v].isna().sum()) for v in vars_],
        "Số giá trị hợp lệ": [int(clean_data[v].notna().sum()) for v in vars_],
    })
    R["clean_data"] = clean_data
    return R


# ------------------------------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ------------------------------------------------------------------------------
def run_correlation(df):
    d = df[["privacy_risk", "support"]].dropna()
    n = len(d)
    if n < 3 or d["privacy_risk"].nunique() < 2 or d["support"].nunique() < 2:
        return {"ok": False, "reason": f"Không đủ dữ liệu để tính tương quan (N = {n}, cần ≥ 3 và các biến phải có biến thiên)."}
    r, p = stats.pearsonr(d["privacy_risk"], d["support"])
    matrix = pd.DataFrame([[1.0, r], [r, 1.0]], index=["Privacy Risk", "Support"],
                          columns=["Privacy Risk", "Support"]).round(3)
    matrix.insert(0, "", matrix.index)
    return {"ok": True, "r": r, "p": p, "n": n,
            "direction": "Dương (Positive)" if r > 0 else "Âm (Negative)" if r < 0 else "Không có (r = 0)",
            "matrix": matrix}


# ------------------------------------------------------------------------------
# 6. REGRESSION ANALYSIS
# ------------------------------------------------------------------------------
def run_regression(df):
    d = df[["privacy_risk", "support"]].dropna()
    n = len(d)
    if n < 3 or d["privacy_risk"].nunique() < 2:
        return {"ok": False, "reason": f"Không đủ dữ liệu để chạy hồi quy (N = {n})."}
    X = sm.add_constant(d["privacy_risk"])
    model = sm.OLS(d["support"], X).fit()
    b0, b1 = model.params["const"], model.params["privacy_risk"]
    sign = "+" if b1 >= 0 else "-"
    equation = f"Support = {b0:.3f} {sign} {abs(b1):.3f} × Privacy Risk"
    coef_table = pd.DataFrame({
        "Predictor": ["Intercept", "Privacy Risk"],
        "Coefficient": [f"{b0:.3f}", f"{b1:.3f}"],
        "Std. Error": [f"{model.bse['const']:.3f}", f"{model.bse['privacy_risk']:.3f}"],
        "t-value": [f"{model.tvalues['const']:.3f}", f"{model.tvalues['privacy_risk']:.3f}"],
        "p-value": [fmt_p(model.pvalues["const"]), fmt_p(model.pvalues["privacy_risk"])],
    })
    summary_table = pd.DataFrame({
        "Chỉ số": ["R-squared", "Adjusted R-squared", "N"],
        "Giá trị": [f"{model.rsquared:.3f}", f"{model.rsquared_adj:.3f}", str(int(model.nobs))],
    })
    return {"ok": True, "data": d, "b0": b0, "b1": b1, "equation": equation,
            "r2": model.rsquared, "adj_r2": model.rsquared_adj, "n": n,
            "coef_table": coef_table, "summary_table": summary_table}


# ------------------------------------------------------------------------------
# 7. SCENARIO COMPARISON
# ------------------------------------------------------------------------------
def run_scenario_descriptives(df):
    d = df[["scenario", "support"]].dropna()
    rows = []
    for s in (1, 2, 3):
        g = d.loc[d["scenario"] == s, "support"]
        rows.append({
            "Scenario": f"{s} - {SCENARIO_LABELS[s]}",
            "Mean Support": round(g.mean(), 3) if len(g) else np.nan,
            "Std. Deviation": round(g.std(ddof=1), 3) if len(g) > 1 else np.nan,
            "Std. Error": round(g.std(ddof=1) / np.sqrt(len(g)), 3) if len(g) > 1 else np.nan,
            "N": int(len(g)),
        })
    return pd.DataFrame(rows), d


# ------------------------------------------------------------------------------
# 8. ANOVA
# ------------------------------------------------------------------------------
def run_anova(d):
    groups = {s: d.loc[d["scenario"] == s, "support"].values for s in (1, 2, 3)}
    sizes = {s: len(v) for s, v in groups.items()}
    if any(n < 2 for n in sizes.values()):
        return {"ok": False, "reason": ("Không đủ điều kiện chạy One-way ANOVA: mỗi scenario cần ít nhất 2 quan sát. "
                                        f"Số quan sát hiện có: {sizes}.")}
    if np.var(np.concatenate(list(groups.values()))) == 0:
        return {"ok": False, "reason": "Support Score không có biến thiên nên không thể chạy ANOVA."}
    f, p = stats.f_oneway(*groups.values())
    grand = d["support"].mean()
    ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
    ss_total = ((d["support"] - grand) ** 2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    _, lev_p = stats.levene(*groups.values())
    table = pd.DataFrame({
        "Chỉ số": ["F-statistic", "p-value", "N", "Eta-squared (mức độ ảnh hưởng)", "Levene p-value (đồng nhất phương sai)"],
        "Giá trị": [f"{f:.3f}", fmt_p(p), str(len(d)), f"{eta2:.3f}", fmt_p(lev_p)],
    })
    res = {"ok": True, "F": f, "p": p, "n": len(d), "table": table, "posthoc": None, "posthoc_msg": ""}
    if p < 0.05:
        tk = pairwise_tukeyhsd(d["support"], d["scenario"].astype(int).map(SCENARIO_LABELS), alpha=0.05)
        ph = pd.DataFrame(tk._results_table.data[1:], columns=tk._results_table.data[0])
        ph = ph.rename(columns={"group1": "Nhóm 1", "group2": "Nhóm 2", "meandiff": "Chênh lệch TB",
                                "p-adj": "p (đã hiệu chỉnh)", "lower": "CI dưới", "upper": "CI trên",
                                "reject": "Khác biệt có ý nghĩa"})
        for c in ["Chênh lệch TB", "p (đã hiệu chỉnh)", "CI dưới", "CI trên"]:
            ph[c] = pd.to_numeric(ph[c]).round(3)
        ph["Khác biệt có ý nghĩa"] = ph["Khác biệt có ý nghĩa"].map({True: "Có", False: "Không"})
        res["posthoc"] = ph
        res["posthoc_msg"] = "ANOVA có ý nghĩa thống kê (p < 0.05) nên đã chạy post-hoc Tukey HSD."
    else:
        res["posthoc_msg"] = "ANOVA không có ý nghĩa thống kê (p ≥ 0.05) nên không chạy post-hoc."
    return res


# ------------------------------------------------------------------------------
# 9. VISUALIZATION
# ------------------------------------------------------------------------------
def plot_regression(reg):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if not reg["ok"]:
        ax.text(0.5, 0.5, reg["reason"], ha="center", va="center", wrap=True)
        ax.axis("off")
        return fig
    d = reg["data"]
    ax.scatter(d["privacy_risk"], d["support"], alpha=0.75, color="#2b6cb0", edgecolor="white", s=60,
               label="Respondent")
    xs = np.linspace(d["privacy_risk"].min(), d["privacy_risk"].max(), 100)
    ax.plot(xs, reg["b0"] + reg["b1"] * xs, color="#c53030", lw=2.2, label="Đường hồi quy")
    ax.set_xlabel("Perceived Privacy Risk (điểm trung bình 1–5)")
    ax.set_ylabel("Support for Police Facial Recognition (điểm trung bình 1–5)")
    ax.set_title("Mối quan hệ giữa rủi ro quyền riêng tư và mức độ ủng hộ Facial Recognition", fontsize=11)
    ax.text(0.03, 0.05, f"{reg['equation']}\nR² = {reg['r2']:.3f}", transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999"))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_scenarios(stats_df):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    labels = [SCENARIO_LABELS_VI[s] for s in (1, 2, 3)]
    means = stats_df["Mean Support"].values.astype(float)
    ses = stats_df["Std. Error"].fillna(0).values.astype(float)
    ns = stats_df["N"].values
    bars = ax.bar(labels, means, yerr=ses, capsize=6, color=["#2b6cb0", "#dd6b20", "#805ad5"], alpha=0.85)
    for b, m, n in zip(bars, means, ns):
        if not np.isnan(m):
            ax.text(b.get_x() + b.get_width() / 2, 0.15, f"TB = {m:.2f}\nN = {n}", ha="center",
                    color="white", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("Use Case (tình huống sử dụng)")
    ax.set_ylabel("Mean Support (điểm trung bình 1–5)")
    ax.set_title("So sánh mức độ ủng hộ Facial Recognition giữa ba tình huống sử dụng\n(thanh lỗi: Standard Error)",
                 fontsize=11)
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def show_table(df):
    st.dataframe(df, hide_index=True, use_container_width=True)


# ------------------------------------------------------------------------------
# 10. STREAMLIT INTERFACE
# ------------------------------------------------------------------------------
st.title("Phân tích thái độ đối với công nghệ nhận diện khuôn mặt")
st.write("Ứng dụng phân tích dữ liệu khảo sát về nhận thức rủi ro quyền riêng tư, mức độ ủng hộ việc cảnh sát "
         "sử dụng công nghệ nhận diện khuôn mặt và sự khác biệt giữa các tình huống sử dụng.")
st.info("**Lưu ý:** File dữ liệu hiện tại là dữ liệu mẫu dùng để kiểm tra ứng dụng, không phải kết quả khảo sát thực tế.")

# Dữ liệu đầu vào: ưu tiên file người dùng tải lên, nếu không dùng file mặc định cạnh app.py
with st.sidebar:
    st.header("Dữ liệu đầu vào")
    uploaded = st.file_uploader("Tải file Excel khảo sát (.xlsx)", type=["xlsx"])
    st.caption(f"Nếu không tải lên, ứng dụng dùng file mặc định: {DEFAULT_EXCEL_PATH}")

if uploaded is not None:
    source = uploaded
elif os.path.exists(DEFAULT_EXCEL_PATH):
    source = DEFAULT_EXCEL_PATH
else:
    st.error(f"Không tìm thấy file '{DEFAULT_EXCEL_PATH}'. Hãy đặt file cạnh app.py hoặc tải file lên ở thanh bên.")
    st.stop()

try:
    raw_data = load_excel(source)
except Exception as e:
    st.error(f"Không đọc được file Excel: {e}")
    st.stop()

R = clean_and_build(raw_data)
if R.get("structure_warnings"):
    st.warning("**Cảnh báo cấu trúc file Excel:**\n\n" + "\n".join(f"- {w}" for w in R["structure_warnings"]))
if "fatal" in R:
    st.error(R["fatal"])
    st.stop()

clean_data = R["clean_data"]
correlation_result = run_correlation(clean_data)
regression_result = run_regression(clean_data)
scenario_stats, scenario_data = run_scenario_descriptives(clean_data)
anova_result = run_anova(scenario_data)

# ---- Phần 1
st.header("1. Tổng quan dữ liệu")
rate = R["manip_rate"]
c = st.columns(3)
c[0].metric("Tổng số respondent ban đầu", R["n_initial"])
c[1].metric("Số respondent sau cleaning", R["n_clean"])
c[2].metric("Tỷ lệ manipulation check pass", f"{rate * 100:.1f}%" if not np.isnan(rate) else "N/A")
overview_table = pd.DataFrame({
    "Chỉ số": ["Tổng số respondent ban đầu", "Số respondent bị loại do consent",
               "Số respondent bị loại do attention check", "Số respondent sau cleaning",
               "Số respondent pass manipulation check", "Số respondent fail manipulation check",
               "Tỷ lệ manipulation check pass"],
    "Giá trị": [R["n_initial"], R["n_removed_consent"], R["n_removed_attention"], R["n_clean"],
                R["n_manip_pass"], R["n_manip_fail"], f"{rate * 100:.1f}%" if not np.isnan(rate) else "N/A"],
}).astype({"Giá trị": str})
st.subheader("Tổng quan làm sạch dữ liệu")
show_table(overview_table)
checks_table = pd.DataFrame({
    "Kiểm tra dữ liệu": [
        "Respondent có nhiều hơn 1 scenario block hoặc không có block nào (bị bỏ khỏi phân tích scenario)",
        "Respondent có scenario không khớp với tháng sinh (theo README)",
    ] + [f"Số respondent trong scenario {s}" for s in (1, 2, 3)],
    "Giá trị": [R["n_invalid_block"], R["n_month_mismatch"]] + [int((clean_data["scenario"] == s).sum()) for s in (1, 2, 3)],
}).astype({"Giá trị": str})
st.subheader("Kiểm tra scenario")
show_table(checks_table)
st.subheader("Kiểm tra giá trị thiếu (sau cleaning)")
show_table(R["missing_table"])

# ---- Phần 2
st.header("2. Biến nghiên cứu")
variables_table = pd.DataFrame({
    "Biến": ["privacy_risk", "support", "scenario"],
    "Vai trò": ["Biến độc lập (IV)", "Biến phụ thuộc (DV)", "Biến độc lập phân loại (IV)"],
    "Mô tả": [
        "Nhận thức rủi ro quyền riêng tư: trung bình 3 câu hỏi (mối đe dọa quyền riêng tư; lo ngại bị quét/ghi lại "
        "khuôn mặt mà không biết; nguy cơ dữ liệu khuôn mặt bị lạm dụng). Thang Likert 1–5.",
        "Mức độ ủng hộ cảnh sát dùng nhận diện khuôn mặt: trung bình 3 câu (ủng hộ; nên được phép; là cách dùng chấp nhận được) "
        "của ĐÚNG scenario block mà respondent được phân. Thang Likert 1–5.",
        "Tình huống sử dụng: 1 = Tìm người mất tích; 2 = Giám sát chung / phòng ngừa tội phạm; "
        "3 = Giám sát biểu tình / hoạt động chính trị. Mỗi respondent chỉ thấy một scenario.",
    ],
})
show_table(variables_table)

# ---- Phần 3
st.header("3. Phân tích tương quan")
st.markdown("**H1:** Rủi ro quyền riêng tư cao hơn có liên hệ âm với mức độ ủng hộ.")
if correlation_result["ok"]:
    cr = correlation_result
    st.markdown(f"**Pearson r = {cr['r']:.3f}**  |  **p-value = {fmt_p(cr['p'])}**  |  **N = {cr['n']}**  \n"
                f"Hướng của mối quan hệ: **{cr['direction']}**")
    st.subheader("Correlation matrix")
    show_table(cr["matrix"])
else:
    st.warning(correlation_result["reason"])

# ---- Phần 4
st.header("4. Hồi quy tuyến tính")
if regression_result["ok"]:
    rg = regression_result
    st.markdown(f"**Phương trình hồi quy:** `{rg['equation']}`  \n"
                f"R-squared = **{rg['r2']:.3f}**  |  Adjusted R-squared = **{rg['adj_r2']:.3f}**  |  N = **{rg['n']}**")
    st.subheader("Bảng hệ số hồi quy")
    show_table(rg["coef_table"])
    st.subheader("Tóm tắt mô hình")
    show_table(rg["summary_table"])
else:
    st.warning(regression_result["reason"])
fig_reg = plot_regression(regression_result)
st.pyplot(fig_reg)
plt.close(fig_reg)

# ---- Phần 5
st.header("5. So sánh theo tình huống sử dụng")
st.markdown("**H2:** Mức độ ủng hộ khác nhau giữa các tình huống; dự kiến giám sát biểu tình/chính trị thấp hơn tìm người mất tích.")
st.subheader("Thống kê mô tả theo scenario")
show_table(scenario_stats)
fig_scn = plot_scenarios(scenario_stats)
st.pyplot(fig_scn)
plt.close(fig_scn)
if anova_result["ok"]:
    a = anova_result
    st.markdown(f"**F = {a['F']:.3f}**  |  **p-value = {fmt_p(a['p'])}**  |  **N = {a['n']}**  \n{a['posthoc_msg']}")
    st.subheader("Kết quả One-way ANOVA")
    show_table(a["table"])
    if a["posthoc"] is not None:
        st.subheader("Post-hoc: Tukey HSD")
        show_table(a["posthoc"])
else:
    st.warning(anova_result["reason"])

st.divider()
st.caption("Dữ liệu NIST chỉ là dữ liệu công bố bên ngoài phục vụ phần thảo luận, không nằm trong phân tích khảo sát này.")
