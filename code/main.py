from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="🏫 폐교 예측 분석",
    page_icon="🏫",
    layout="wide",
)

st.title("🏫 폐교 예측 분석")
st.caption("지역별 폐교 기록을 바탕으로 연도별 추세를 확인하고 미래 폐교 수를 예측합니다.")


# --------------------------------------------------
# 데이터 파일 설정 및 불러오기
# --------------------------------------------------
CSV_FILE = Path(__file__).parent / "한국교육학술정보원_학교알리미 폐교정보_20250930.csv"

REQUIRED_COLUMNS = [
    "지역",
    "교육청",
    "학교명",
    "학교급명",
    "폐교일자",
    "주소",
]


@st.cache_data
def load_data():
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {CSV_FILE.name}"
        )

    # 학교알리미 CSV가 cp949로 저장된 경우를 고려
    try:
        raw = pd.read_csv(CSV_FILE, encoding="cp949")
    except UnicodeDecodeError:
        raw = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in raw.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    # 지역별 결측률은 원본 데이터에서 계산
    missing_by_region = {}

    for region, group in raw.groupby("지역", dropna=False):
        total_cells = group[REQUIRED_COLUMNS].size
        missing_cells = group[REQUIRED_COLUMNS].isna().sum().sum()

        missing_rate = (
            missing_cells / total_cells
            if total_cells > 0
            else 0
        )

        region_name = (
            str(region).strip()
            if pd.notna(region)
            else "지역 미상"
        )

        missing_by_region[region_name] = missing_rate

    df = raw.copy()

    # 날짜 처리
    df["폐교일자"] = pd.to_datetime(
        df["폐교일자"],
        errors="coerce"
    )

    # 날짜가 없는 행은 연도별 분석에서 제외
    df = df.dropna(subset=["폐교일자"]).copy()
    df["연도"] = df["폐교일자"].dt.year.astype(int)

    # 지역 이름 앞뒤 공백 제거
    df["지역"] = df["지역"].astype("string").str.strip()

    return df, missing_by_region


try:
    df, missing_by_region = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. CSV 파일명과 열 이름을 확인해 주세요.")
    st.exception(error)
    st.stop()


if df.empty:
    st.error("폐교일자를 확인할 수 있는 데이터가 없습니다.")
    st.stop()


# --------------------------------------------------
# 전체 데이터 요약
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("전체 폐교 기록", f"{len(df):,}건")
col2.metric("지역 수", f"{df['지역'].nunique():,}개")
col3.metric(
    "데이터에 포함된 연도",
    f"{df['연도'].min()}~{df['연도'].max()}년"
)

st.divider()


# --------------------------------------------------
# 지역 선택
# --------------------------------------------------
st.subheader("📍 지역 선택")

regions = sorted(df["지역"].dropna().unique().tolist())

if not regions:
    st.error("선택할 수 있는 지역이 없습니다.")
    st.stop()

selected_region = st.selectbox(
    "분석할 지역을 선택하세요.",
    options=regions,
)

region_df = df[df["지역"] == selected_region].copy()


# --------------------------------------------------
# 결측률 확인
# --------------------------------------------------
missing_rate = missing_by_region.get(selected_region, 0)

st.write(f"선택 지역의 원본 데이터 결측률: **{missing_rate:.1%}**")

if missing_rate >= 0.30:
    st.error(
        f"{selected_region} 지역은 결측률이 30% 이상이므로 "
        "분석 결과를 신뢰하기 어렵습니다."
    )
    st.subheader("결측치")
    st.info("이 지역은 결측률 기준에 따라 폐교 수 예측을 진행하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 연도별 폐교 수 계산
# --------------------------------------------------
yearly_counts = (
    region_df.groupby("연도")
    .size()
    .sort_index()
)

# 데이터 전체에서 가장 이른 연도부터 가장 늦은 연도까지 연도 축 생성
# 해당 연도에 폐교 기록이 없으면 0으로 표시
all_years = range(
    int(df["연도"].min()),
    int(df["연도"].max()) + 1
)

yearly_counts = (
    yearly_counts
    .reindex(all_years, fill_value=0)
    .astype(int)
)

yearly_df = pd.DataFrame({
    "연도": yearly_counts.index,
    "폐교 수": yearly_counts.values,
})


# --------------------------------------------------
# 측정 연도 수 확인
# --------------------------------------------------
# 폐교 기록이 실제로 있는 연도만 측정 연도로 계산
measured_year_count = int((yearly_counts > 0).sum())

st.subheader("📊 데이터 충분성 확인")

st.metric(
    "폐교 기록이 있는 연도 수",
    f"{measured_year_count}년",
)

if measured_year_count < 20:
    st.warning(
        f"데이터가 충분하지 않습니다. "
        f"{selected_region} 지역은 폐교 기록이 있는 연도가 "
        f"{measured_year_count}년입니다. "
        "(예측 기준: 20년 이상)"
    )
else:
    st.success(
        f"{selected_region} 지역은 폐교 기록이 있는 연도가 "
        f"{measured_year_count}년입니다."
    )

st.caption(
    "※ 여기서 ‘폐교 기록이 있는 연도’는 해당 연도에 폐교 기록이 1건 이상 있는 연도입니다. "
    "폐교가 없었던 연도와 자료가 수집되지 않은 연도는 이 데이터만으로 구분할 수 없습니다."
)


# --------------------------------------------------
# 연도별 폐교 수 그래프
# --------------------------------------------------
st.subheader("📈 연도별 폐교 수")

fig = px.bar(
    yearly_df,
    x="연도",
    y="폐교 수",
    title=f"{selected_region} 연도별 폐교 수",
    labels={
        "연도": "연도",
        "폐교 수": "폐교 수",
    },
)

fig.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title="폐교 수",
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 예측 기간 설정
# --------------------------------------------------
st.subheader("🔮 미래 폐교 수 예측")

forecast_years = st.slider(
    "몇 년 뒤까지 예측할까요?",
    min_value=1,
    max_value=20,
    value=5,
)


# --------------------------------------------------
# 예측 수행
# --------------------------------------------------
# 실제 폐교 기록이 있는 연도만 모델 학습에 사용
training_df = yearly_df[yearly_df["폐교 수"] > 0].copy()

if len(training_df) < 2:
    st.warning(
        "예측 모델을 만들려면 폐교 기록이 있는 연도가 최소 2년 필요합니다."
    )
    st.stop()

X = training_df[["연도"]]
y = training_df["폐교 수"]

model = LinearRegression()
model.fit(X, y)

# 모델 성능 지표
train_predictions = model.predict(X)
mae = mean_absolute_error(y, train_predictions)

if len(training_df) >= 2 and y.nunique() > 1:
    r2 = r2_score(y, train_predictions)
else:
    r2 = None


# --------------------------------------------------
# 미래 예측 결과
# --------------------------------------------------
last_year = int(df["연도"].max())

future_years = list(
    range(last_year + 1, last_year + forecast_years + 1)
)

future_df = pd.DataFrame({
    "연도": future_years,
})

future_df["예측 폐교 수"] = model.predict(
    future_df[["연도"]]
)

# 폐교 수는 음수가 될 수 없고, 실제 건수이므로 정수로 표시
future_df["예측 폐교 수"] = (
    future_df["예측 폐교 수"]
    .clip(lower=0)
    .round()
    .astype(int)
)


# --------------------------------------------------
# 모델 지표
# --------------------------------------------------
metric_col1, metric_col2 = st.columns(2)

metric_col1.metric(
    "평균 절대 오차(MAE)",
    f"{mae:.2f}개교",
)

if r2 is not None:
    metric_col2.metric(
        "결정계수(R²)",
        f"{r2:.3f}",
    )
else:
    metric_col2.metric(
        "결정계수(R²)",
        "계산 불가",
    )


# --------------------------------------------------
# 과거 및 미래 예측 그래프
# --------------------------------------------------
historical_plot = yearly_df.rename(
    columns={"폐교 수": "폐교 수"}
).copy()

historical_plot["구분"] = "과거 기록"
future_plot = future_df.rename(
    columns={"예측 폐교 수": "폐교 수"}
).copy()

future_plot["구분"] = "미래 예측"

plot_df = pd.concat(
    [
        historical_plot[["연도", "폐교 수", "구분"]],
        future_plot[["연도", "폐교 수", "구분"]],
    ],
    ignore_index=True,
)

fig_forecast = px.line(
    plot_df,
    x="연도",
    y="폐교 수",
    color="구분",
    markers=True,
    title=f"{selected_region} 폐교 수 예측",
    labels={
        "연도": "연도",
        "폐교 수": "폐교 수",
        "구분": "데이터 구분",
    },
)

fig_forecast.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title="폐교 수",
)

st.plotly_chart(fig_forecast, use_container_width=True)


# --------------------------------------------------
# 예측 결과 표
# --------------------------------------------------
st.subheader("📋 미래 폐교 수 예측 결과")

st.dataframe(
    future_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 과거 데이터 표
# --------------------------------------------------
st.subheader("📚 과거 연도별 폐교 수")

st.dataframe(
    yearly_df,
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 선택 지역의 원본 폐교 기록
# --------------------------------------------------
with st.expander("선택 지역의 폐교 기록 확인"):
    display_df = region_df[
        ["지역", "교육청", "학교명", "학교급명", "폐교일자", "주소"]
    ].sort_values("폐교일자")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 전체 지역별 데이터 결측률
# --------------------------------------------------
with st.expander("전체 지역별 결측률 확인"):
    missing_summary = pd.DataFrame({
        "지역": list(missing_by_region.keys()),
        "결측률": list(missing_by_region.values()),
    })

    missing_summary["결측률"] = missing_summary["결측률"].map(
        lambda value: f"{value:.1%}"
    )

    st.dataframe(
        missing_summary,
        use_container_width=True,
        hide_index=True,
    )


st.caption(
    "데이터 출처: 한국교육학술정보원 학교알리미 폐교정보(2025년 9월 30일 기준). "
    "예측은 과거 폐교 기록에 선형회귀를 적용한 참고용 추정치이며, 실제 폐교 수를 보장하지 않습니다."
)
