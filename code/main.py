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

# main.py와 같은 폴더에 CSV 파일이 있어야 합니다.
DATA_FILE = Path(__file__).parent / (
    "한국교육학술정보원_학교알리미 폐교정보_20250930.csv"
)


# ==========================================
# 2. 데이터 자동 불러오기
# ==========================================
@st.cache_data
def load_data(file_path):
    """CSV 파일을 자동으로 읽고 분석에 필요한 열을 정리합니다."""

    df = pd.read_csv(
        file_path,
        encoding="cp949",
        dtype=str
    )

    # 실제 CSV의 열 이름을 앱에서 사용할 이름으로 변경
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

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {missing_columns}"
        )

    df = df[required_columns].copy()

    # 문자열 공백 제거
    for col in required_columns:
        df[col] = df[col].astype("string").str.strip()

    # 폐교일자를 날짜로 변환
    # 이 파일은 연도만 기록된 날짜도 포함합니다.
    df["폐교일자"] = pd.to_datetime(
        df["폐교일자"],
        errors="coerce"
    )

    # 날짜가 없는 행은 분석에서 제외
    df = df.dropna(subset=["폐교일자"])

    # 폐교 연도 추출
    df["폐교연도"] = df["폐교일자"].dt.year.astype(int)

    return df


# ==========================================
# 3. 데이터 로드 및 오류 처리
# ==========================================
if not DATA_FILE.exists():
    st.error(
        "폐교 데이터 파일을 찾을 수 없습니다."
    )

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
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


if df.empty:
    st.error("분석할 수 있는 폐교 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 데이터 요약
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
    regions = sorted(df["도"].dropna().unique().tolist())

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


# ==========================================
# 6. 연도별 폐교 학교 수 계산
# ==========================================
yearly_count = (
    region_df
    .groupby("폐교연도")
    .size()
    .sort_index()
)

# 기록이 없는 연도도 0개로 표시
all_years = np.arange(
    int(yearly_count.index.min()),
    int(yearly_count.index.max()) + 1
)

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

# 미래 예측
last_year = int(yearly_count.index.max())

future_years = np.arange(
    last_year + 1,
    last_year + forecast_years + 1
)

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

# 선형회귀 추세선
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
# 10. 선형회귀 분석 결과
# ==========================================
st.subheader("📉 선형회귀 모델 분석")

slope = float(model.coef_[0])
intercept = float(model.intercept_)
r_squared = float(model.score(X, y))

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "연간 폐교 학교 수 변화",
        f"{slope:+.2f}개"
    )

with col2:
    st.metric(
        "결정계수 R²",
        f"{r_squared:.3f}"
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


# ==========================================
# 11. 미래 예측 결과 표
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
# 12. 연도별 폐교 학교 수 표
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
# 13. 원본 데이터 확인
# ==========================================
with st.expander("원본 폐교 데이터 확인"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "※ 이 앱은 2025년 9월 30일 기준 폐교 데이터로 분석합니다. "
    "미래 예측은 과거 추세를 연장한 통계적 추정치이며, "
    "실제 폐교 학교 수와 다를 수 있습니다."
)
