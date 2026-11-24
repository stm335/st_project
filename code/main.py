import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from pathlib import Path


# ==========================================
# 1. 앱 기본 설정
# ==========================================
st.set_page_config(
    page_title="폐교 예측 분석",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 폐교 예측 분석")

st.caption(
    "전국 폐교 데이터를 바탕으로 지역별 폐교 추세를 분석하고 "
    "선형회귀 모델로 미래 폐교 학교 수를 예측합니다."
)

# main.py와 같은 폴더에 있는 CSV 파일
DATA_FILE = (
    Path(__file__).parent
    / "한국교육학술정보원_학교알리미 폐교정보_20250930.csv"
)

# 폐교 기록 개수가 이 값보다 적으면 데이터 부족으로 판단
MIN_RECORDS = 5


# ==========================================
# 2. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data(file_path):
    """CSV 파일을 읽고 분석에 필요한 열을 정리합니다."""

    df = pd.read_csv(
        file_path,
        encoding="cp949",
        dtype=str
    )

    # 실제 CSV 열 이름을 앱에서 사용할 이름으로 변경
    df = df.rename(columns={
        "지역": "도",
        "교육청": "세부지역",
        "학교명": "학교 이름",
        "학교급명": "학교급"
    })

    required_columns = [
        "도",
        "세부지역",
        "학교 이름",
        "학교급",
        "폐교일자",
        "주소"
    ]

    # 필요한 열이 모두 있는지 확인
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {missing_columns}"
        )

    # 필요한 열만 사용
    df = df[required_columns].copy()

    # 문자열 앞뒤 공백 제거
    for col in required_columns:
        df[col] = df[col].astype("string").str.strip()

    # 폐교일자 날짜 변환
    # YYYYMMDD 형식과 일반 날짜 형식을 처리
    date_text = (
        df["폐교일자"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    # 우선 YYYYMMDD 형식으로 변환
    parsed_date = pd.to_datetime(
        date_text,
        format="%Y%m%d",
        errors="coerce"
    )

    # YYYY-MM-DD 등 다른 형식은 일반 날짜 변환으로 처리
    remaining_dates = parsed_date.isna()

    parsed_date.loc[remaining_dates] = pd.to_datetime(
        date_text.loc[remaining_dates],
        errors="coerce"
    )

    df["폐교일자"] = parsed_date

    # 날짜가 없거나 변환되지 않은 행 제거
    df = df.dropna(
        subset=["도", "학교 이름", "폐교일자"]
    )

    # 도와 학교 이름이 없는 행 제거
    df = df[
        (df["도"] != "") &
        (df["학교 이름"] != "")
    ].copy()

    # 폐교 연도 추출
    df["폐교연도"] = df["폐교일자"].dt.year.astype(int)

    return df


# ==========================================
# 3. 데이터 자동 불러오기
# ==========================================
if not DATA_FILE.exists():
    st.error("폐교 데이터 파일을 찾을 수 없습니다.")

    st.write(
        "main.py와 CSV 파일이 같은 폴더에 있는지 확인해주세요."
    )

    st.code(
        "main.py\n"
        "한국교육학술정보원_학교알리미 폐교정보_20250930.csv"
    )

    st.stop()


try:
    df = load_data(DATA_FILE)

except Exception as e:
    st.error(
        f"데이터를 불러오는 중 오류가 발생했습니다: {e}"
    )
    st.stop()


if df.empty:
    st.error("분석할 수 있는 폐교 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 전국 폐교 현황
# ==========================================
st.subheader("📊 전국 폐교 현황")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "전체 폐교 학교 수",
        f"{len(df):,}개"
    )

with col2:
    st.metric(
        "분석 대상 지역",
        f"{df['도'].nunique()}개"
    )

with col3:
    st.metric(
        "폐교 기록 기간",
        f"{df['폐교연도'].min()}~{df['폐교연도'].max()}년"
    )


# ==========================================
# 5. 분석 조건 설정
# ==========================================
st.subheader("🔎 분석 조건")

col1, col2 = st.columns(2)

with col1:
    regions = sorted(
        df["도"].dropna().unique().tolist()
    )

    selected_region = st.selectbox(
        "분석할 지역 선택",
        regions
    )

with col2:
    forecast_years = st.slider(
        "미래 예측 기간",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )


# 선택한 지역의 데이터
region_df = df[
    df["도"] == selected_region
].copy()

# 선택한 지역의 폐교 기록 개수
record_count = len(region_df)

# 데이터 부족 여부
is_low_data = record_count < MIN_RECORDS


# ==========================================
# 6. 연도별 폐교 학교 수 계산
# ==========================================

# 전국 데이터에서 가장 이른 연도와 가장 늦은 연도
first_year = int(df["폐교연도"].min())
last_year = int(df["폐교연도"].max())

# 선택한 지역의 연도별 폐교 학교 수
yearly_count = (
    region_df
    .groupby("폐교연도")
    .size()
    .sort_index()
)

# 전국 데이터의 전체 연도 범위 생성
all_years = np.arange(
    first_year,
    last_year + 1
)

# 선택한 지역에 기록이 없는 연도는 0으로 표시
yearly_count = yearly_count.reindex(
    all_years,
    fill_value=0
)

yearly_count.index.name = "연도"


# ==========================================
# 7. 선형회귀 모델 생성
# ==========================================
X = yearly_count.index.to_numpy().reshape(-1, 1)
y = yearly_count.to_numpy()

model = LinearRegression()
model.fit(X, y)

# 과거 추세선
historical_prediction = model.predict(X)

# 미래 예측 연도
future_years = np.arange(
    last_year + 1,
    last_year + forecast_years + 1
)

# 미래 폐교 학교 수 예측
future_prediction = model.predict(
    future_years.reshape(-1, 1)
)

# 폐교 학교 수는 음수가 될 수 없으므로 0 미만은 0으로 처리
future_prediction = np.maximum(
    future_prediction,
    0
)


# ==========================================
# 8. 지역별 통계
# ==========================================
total_closed = len(region_df)

average_closed = yearly_count.mean()

peak_year = int(yearly_count.idxmax())
peak_count = int(yearly_count.max())

future_total = int(
    np.round(future_prediction.sum())
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "누적 폐교 학교 수",
        f"{total_closed:,}개"
    )

with col2:
    st.metric(
        "연평균 폐교 학교 수",
        f"{average_closed:.1f}개"
    )

with col3:
    st.metric(
        "폐교가 가장 많았던 연도",
        f"{peak_year}년"
    )

with col4:
    st.metric(
        "예측 기간 폐교 학교 수",
        f"{future_total:,}개"
    )


# ==========================================
# 9. 폐교 학교 수 그래프
# ==========================================
st.subheader(
    f"📈 {selected_region} 연도별 폐교 학교 수"
)

fig = go.Figure()

# 실제 폐교 학교 수
fig.add_trace(
    go.Scatter(
        x=yearly_count.index,
        y=yearly_count.values,
        mode="lines+markers",
        name="실제 폐교 학교 수",
        line=dict(width=2),
        marker=dict(size=6)
    )
)

# 선형회귀 과거 추세선
fig.add_trace(
    go.Scatter(
        x=yearly_count.index,
        y=np.maximum(historical_prediction, 0),
        mode="lines",
        name="선형회귀 추세선",
        line=dict(
            color="red",
            width=2,
            dash="dash"
        )
    )
)

# 미래 예측
fig.add_trace(
    go.Scatter(
        x=future_years,
        y=future_prediction,
        mode="lines+markers",
        name="미래 예측",
        line=dict(
            color="green",
            width=3,
            dash="dot"
        ),
        marker=dict(size=8)
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="폐교 학교 수",
    hovermode="x unified",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

fig.update_xaxes(dtick=5)
fig.update_yaxes(rangemode="tozero")

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 10. 데이터 부족 안내
# ==========================================
if is_low_data:
    st.warning(
        f"⚠️ {selected_region} 지역의 폐교 기록은 "
        f"{record_count}개로, 예측에 필요한 데이터가 부족합니다. "
        "선형회귀 결과는 참고용으로만 확인해주세요."
    )


# ==========================================
# 11. 선형회귀 모델 분석
# ==========================================
st.subheader("📉 선형회귀 모델 분석")

slope = float(model.coef_[0])
intercept = float(model.intercept_)

# 데이터가 2개 이상일 때만 R² 계산
if len(y) >= 2:
    r_squared = float(model.score(X, y))
else:
    r_squared = None

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "연간 폐교 학교 수 변화",
        f"{slope:+.2f}개"
    )

with col2:
    if r_squared is not None:
        st.metric(
            "결정계수 R²",
            f"{r_squared:.3f}"
        )
    else:
        st.metric(
            "결정계수 R²",
            "계산 불가"
        )

with col3:
    st.metric(
        "예측 마지막 연도",
        f"{last_year + forecast_years}년"
    )

st.markdown(
    f"""
    **회귀식**

    폐교 학교 수 = {slope:.4f} × 연도 + ({intercept:.4f})

    **해석**

    - 회귀계수가 양수이면 과거 폐교 학교 수가 증가하는 추세입니다.
    - 회귀계수가 음수이면 과거 폐교 학교 수가 감소하는 추세입니다.
    - R²는 과거 데이터에 대한 모델의 설명력을 나타냅니다.
    """
)

if is_low_data:
    st.warning(
        "폐교 기록이 적어 예측 결과의 신뢰도가 낮을 수 있습니다. "
        "실제 폐교 학교 수와 차이가 클 수 있으므로 "
        "참고용으로만 활용해주세요."
    )


# ==========================================
# 12. 미래 예측 결과 표
# ==========================================
st.subheader("🔮 미래 폐교 학교 수 예측")

future_table = pd.DataFrame({
    "예측 연도": future_years,
    "예측 폐교 학교 수": np.round(
        future_prediction
    ).astype(int)
})

st.dataframe(
    future_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 13. 연도별 폐교 학교 수 표
# ==========================================
st.subheader("📋 연도별 폐교 학교 수")

history_table = pd.DataFrame({
    "연도": yearly_count.index,
    "실제 폐교 학교 수": yearly_count.values,
    "선형회귀 추세선": np.round(
        np.maximum(historical_prediction, 0),
        2
    )
})

st.dataframe(
    history_table,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 14. 원본 데이터 확인
# ==========================================
with st.expander("원본 폐교 데이터 확인"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 15. 앱 하단 안내
# ==========================================
st.caption(
    "※ 이 앱은 2025년 9월 30일 기준 폐교 데이터로 분석합니다. "
    "미래 예측은 과거 추세를 연장한 통계적 추정치이며, "
    "실제 폐교 학교 수와 다를 수 있습니다."
)
