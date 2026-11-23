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
    "전국 폐교 현황을 분석하고 "
    "지역별 폐교 추세와 미래 폐교 학교 수를 예측합니다."
)

DATA_URL = (
    "https://www.data.go.kr/data/15154585/fileData.do"
)

REQUIRED_COLUMNS = [
    "도",
    "세부지역",
    "학교 이름",
    "학교급",
    "폐교일자",
    "주소"
]


# ==========================================
# 2. 데이터 불러오기
# ==========================================
def read_csv_file(file):
    """한글 인코딩을 고려하여 CSV 파일 읽기"""

    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            file.seek(0)

            return pd.read_csv(
                file,
                encoding=encoding,
                dtype=str
            )

        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 확인해주세요.")


def load_data(file):
    """CSV 또는 엑셀 파일 불러오기"""

    suffix = Path(file.name).suffix.lower()

    if suffix == ".csv":
        df = read_csv_file(file)

    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file, dtype=str)

    else:
        raise ValueError(
            "CSV 또는 엑셀 파일만 사용할 수 있습니다."
        )

    # 열 이름 앞뒤 공백 제거
    df.columns = df.columns.astype(str).str.strip()

    # 첫 번째 열이 불필요한 인덱스인 경우 제거
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    # 필요한 열 확인
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"필요한 열이 없습니다: {', '.join(missing)}"
        )

    # 필요한 열만 사용
    df = df[REQUIRED_COLUMNS].copy()

    # 문자열 공백 제거
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype("string").str.strip()

    # 폐교일자를 날짜 형식으로 변환
    df["폐교일자"] = pd.to_datetime(
        df["폐교일자"],
        errors="coerce"
    )

    # 도와 학교 이름이 없는 행 제거
    df = df.dropna(
        subset=["도", "학교 이름", "폐교일자"]
    )

    df = df[
        (df["도"] != "") &
        (df["학교 이름"] != "")
    ].copy()

    # 폐교 연도 추출
    df["폐교연도"] = df["폐교일자"].dt.year.astype(int)

    return df


# ==========================================
# 3. 데이터 업로드
# ==========================================
st.subheader("📂 폐교 데이터")

st.markdown(
    f"공공데이터포털의 "
    f"[폐교 데이터 다운로드 페이지]({DATA_URL})에서 "
    f"파일을 내려받아 업로드해주세요."
)

uploaded_file = st.file_uploader(
    "폐교 데이터 파일 업로드",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info(
        "폐교 데이터 파일을 업로드하면 "
        "지역별 폐교 추세와 예측 결과를 확인할 수 있습니다."
    )
    st.stop()

try:
    df = load_data(uploaded_file)

except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if df.empty:
    st.error("분석할 수 있는 폐교 데이터가 없습니다.")
    st.stop()


# ==========================================
# 4. 분석 조건 설정
# ==========================================
st.subheader("🔎 분석 조건")

regions = sorted(df["도"].dropna().unique().tolist())

selected_region = st.selectbox(
    "분석할 지역 선택",
    regions
)

min_year = int(df["폐교연도"].min())
max_year = int(df["폐교연도"].max())

forecast_years = st.slider(
    "미래 예측 기간",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="마지막 폐교 기록 연도 이후 몇 년을 예측할지 선택합니다."
)

# 선택한 지역의 데이터
region_df = df[df["도"] == selected_region].copy()

# 연도별 폐교 학교 수
yearly_count = (
    region_df.groupby("폐교연도")
    .size()
    .sort_index()
)

# 데이터가 없는 연도는 폐교 학교 수를 0으로 처리
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
# 5. 선형회귀 모델 생성
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

# 학교 수는 음수가 될 수 없으므로 0 미만의 예측값은 0으로 처리
future_prediction = np.maximum(
    future_prediction,
    0
)

# ==========================================
# 6. 주요 통계
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
# 7. 폐교 학교 수 그래프
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
        marker=dict(size=7)
    )
)

# 과거 선형회귀 추세선
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

# 미래 예측값
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
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    height=500
)

fig.update_xaxes(dtick=1)
fig.update_yaxes(rangemode="tozero")

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 8. 선형회귀 모델 정보
# ==========================================
st.subheader("📊 선형회귀 모델 분석")

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

    **해석:** 연도에 따른 폐교 학교 수의 평균적인 변화 추세를
    직선으로 나타낸 모델입니다.
    """
)

st.caption(
    "R²는 과거 데이터에 대한 모델의 설명력을 나타냅니다. "
    "값이 높더라도 미래 예측이 정확하다는 보장은 없습니다."
)


# ==========================================
# 9. 미래 예측 결과 표
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
# 10. 연도별 폐교 학교 수 표
# ==========================================
st.subheader("📋 연도별 폐교 학교 수 데이터")

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
# 11. 원본 데이터 확인
# ==========================================
with st.expander("원본 폐교 데이터 확인"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "※ 미래 예측은 과거 폐교 기록에 기반한 통계적 추정치입니다. "
    "실제 폐교 학교 수와 다를 수 있습니다."
)
