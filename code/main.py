import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from pathlib import Path

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

DATA_FILE = (
    Path(__file__).parent
    / "한국교육학술정보원_학교알리미 폐교정보_20250930.csv"
)

MIN_RECORDS = 5


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        encoding="cp949",
        dtype=str
    )

    df = df.rename(columns={
        "지역": "도",
        "교육청": "세부지역",
        "학교명": "학교 이름",
        "학교급명": "학교급"
    })

    required_columns = [
        "도", "세부지역", "학교 이름",
        "학교급", "폐교일자", "주소"
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

    for col in required_columns:
        df[col] = df[col].astype("string").str.strip()

    date_text = (
        df["폐교일자"]
        .str.replace(r"\.0$", "", regex=True)
    )

    parsed_date = pd.to_datetime(
        date_text,
        format="%Y%m%d",
        errors="coerce"
    )

    remaining = parsed_date.isna()

    parsed_date.loc[remaining] = pd.to_datetime(
        date_text.loc[remaining],
        errors="coerce"
    )

    df["폐교일자"] = parsed_date

    df = df.dropna(
        subset=["도", "학교 이름", "폐교일자"]
    )

    df = df[
        (df["도"] != "") &
        (df["학교 이름"] != "")
    ].copy()

    df["폐교연도"] = df["폐교일자"].dt.year.astype(int)

    return df


if not DATA_FILE.exists():
    st.error("폐교 데이터 파일을 찾을 수 없습니다.")
    st.stop()

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"데이터 오류: {e}")
    st.stop()

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# 전국 폐교 현황
st.subheader("📊 전국 폐교 현황")

col1, col2, col3 = st.columns(3)

col1.metric("전체 폐교 학교 수", f"{len(df):,}개")
col2.metric("분석 대상 지역", f"{df['도'].nunique()}개")
col3.metric(
    "폐교 기록 기간",
    f"{df['폐교연도'].min()}~{df['폐교연도'].max()}년"
)


# 분석 조건
st.subheader("🔎 분석 조건")

col1, col2 = st.columns(2)

regions = sorted(df["도"].dropna().unique().tolist())

selected_region = col1.selectbox(
    "분석할 지역 선택",
    regions
)

forecast_years = col2.slider(
    "미래 예측 기간",
    1, 20, 5
)

region_df = df[df["도"] == selected_region].copy()

record_count = len(region_df)
is_low_data = record_count < MIN_RECORDS

first_year = int(df["폐교연도"].min())
last_year = int(df["폐교연도"].max())

all_years = np.arange(first_year, last_year + 1)

yearly_count = (
    region_df.groupby("폐교연도")
    .size()
    .reindex(all_years, fill_value=0)
)

yearly_count.index.name = "연도"


# 선형회귀
X = yearly_count.index.to_numpy().reshape(-1, 1)
y = yearly_count.to_numpy()

model = LinearRegression()
model.fit(X, y)

historical_prediction = model.predict(X)

future_years = np.arange(
    last_year + 1,
    last_year + forecast_years + 1
)

future_prediction = np.maximum(
    model.predict(future_years.reshape(-1, 1)),
    0
)


# 지역별 통계
total_closed = len(region_df)
average_closed = yearly_count.mean()
peak_year = int(yearly_count.idxmax())
future_total = int(np.round(future_prediction.sum()))

col1, col2, col3, col4 = st.columns(4)

col1.metric("누적 폐교 학교 수", f"{total_closed:,}개")
col2.metric("연평균 폐교 학교 수", f"{average_closed:.1f}개")
col3.metric("폐교가 가장 많았던 연도", f"{peak_year}년")
col4.metric("예측 기간 폐교 학교 수", f"{future_total:,}개")


# 지역별 그래프
st.subheader(f"📈 {selected_region} 연도별 폐교 학교 수")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly_count.index,
    y=yearly_count.values,
    mode="lines+markers",
    name="실제 폐교 학교 수"
))

fig.add_trace(go.Scatter(
    x=yearly_count.index,
    y=np.maximum(historical_prediction, 0),
    mode="lines",
    name="선형회귀 추세선",
    line=dict(color="red", dash="dash")
))

fig.add_trace(go.Scatter(
    x=future_years,
    y=future_prediction,
    mode="lines+markers",
    name="미래 예측",
    line=dict(color="green", dash="dot")
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="폐교 학교 수",
    hovermode="x unified",
    height=500
)

fig.update_yaxes(rangemode="tozero")

st.plotly_chart(fig, use_container_width=True)
# ==========================================
# 10. 비슷한 폐교 추세 지역끼리 그룹화
# ==========================================

st.subheader("🧩 폐교 추세가 비슷한 지역끼리 묶기")

st.write(
    "K-means 알고리즘을 이용하여 지역별 연도별 폐교 발생 비율이 "
    "비슷한 지역끼리 자동으로 그룹화합니다."
)

region_yearly = (
    df.groupby(["도", "폐교연도"])
    .size()
    .unstack(fill_value=0)
)

region_yearly = region_yearly.reindex(
    columns=all_years,
    fill_value=0
)

region_totals = region_yearly.sum(axis=1)

region_patterns = region_yearly.div(
    region_totals.replace(0, np.nan),
    axis=0
).fillna(0)

unique_patterns = np.unique(
    region_patterns.to_numpy().round(8),
    axis=0
)

if len(unique_patterns) >= 2:

    max_clusters = min(
        8,
        len(unique_patterns),
        len(region_patterns)
    )

    cluster_count = st.slider(
        "군집 개수 선택",
        min_value=2,
        max_value=max_clusters,
        value=min(4, max_clusters)
    )

    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        region_patterns.to_numpy()
    )

    cluster_df = pd.DataFrame({
        "지역": region_patterns.index,
        "군집": labels + 1
    })

    selected_cluster = int(
        cluster_df.loc[
            cluster_df["지역"] == selected_region,
            "군집"
        ].iloc[0]
    )

    similar_regions = cluster_df[
        cluster_df["군집"] == selected_cluster
    ]["지역"].tolist()

    col1, col2 = st.columns(2)

    col1.metric("전체 군집 수", f"{cluster_count}개")

    col2.metric(
        "선택 지역과 같은 군집",
        f"{len(similar_regions)}개 지역"
    )

    st.success(
        f"{selected_region}은 {selected_cluster}번 군집에 속합니다."
    )

    st.markdown("### 📍 비슷한 폐교 추세를 가진 지역")

    st.write(", ".join(similar_regions))

    # 군집별 평균 그래프
    st.markdown("### 📈 군집별 평균 폐교 추세 비교")

    fig_cluster = go.Figure()

    for cluster_num in range(1, cluster_count + 1):

        members = cluster_df[
            cluster_df["군집"] == cluster_num
        ]["지역"].tolist()

        cluster_mean = (
            region_patterns.loc[members].mean(axis=0) * 100
        )

        fig_cluster.add_trace(go.Scatter(
            x=all_years,
            y=cluster_mean.values,
            mode="lines+markers",
            name=f"{cluster_num}번 군집 평균"
        ))

    selected_pattern = (
        region_patterns.loc[selected_region] * 100
    )

    fig_cluster.add_trace(go.Scatter(
        x=all_years,
        y=selected_pattern.values,
        mode="lines",
        name=f"{selected_region} 선택 지역",
        line=dict(color="black", width=4, dash="dash")
    ))

    fig_cluster.update_layout(
        xaxis_title="연도",
        yaxis_title="폐교 발생 비율 (%)",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(
        fig_cluster,
        use_container_width=True
    )

    # 군집별 실제 지역 그래프
    st.markdown("### 🔍 군집별 실제 지역 그래프 비교")

    st.write(
        "각 지역의 실제 폐교 발생 비율을 한 그래프에 표시합니다. "
        "검은색 굵은 점선은 해당 군집의 평균 그래프입니다."
    )

    for cluster_num in range(1, cluster_count + 1):

        members = cluster_df[
            cluster_df["군집"] == cluster_num
        ]["지역"].tolist()

        if not members:
            continue

        with st.expander(
            f"{cluster_num}번 군집 실제 그래프 비교 "
            f"({len(members)}개 지역)",
            expanded=(cluster_num == selected_cluster)
        ):

            cluster_mean = (
                region_patterns.loc[members].mean(axis=0)
            )

            fig_compare = go.Figure()

            # 각 지역의 실제 그래프
            for region in members:

                region_values = region_patterns.loc[region]

                fig_compare.add_trace(go.Scatter(
                    x=all_years,
                    y=region_values.values * 100,
                    mode="lines+markers",
                    name=region,
                    line=dict(width=1.5),
                    marker=dict(size=3),
                    opacity=0.7
                ))

            # 군집 평균
            fig_compare.add_trace(go.Scatter(
                x=all_years,
                y=cluster_mean.values * 100,
                mode="lines",
                name="군집 평균",
                line=dict(
                    color="black",
                    width=5,
                    dash="dash"
                )
            ))

            fig_compare.update_layout(
                title=f"{cluster_num}번 군집 지역별 폐교 추세",
                xaxis_title="연도",
                yaxis_title="폐교 발생 비율 (%)",
                hovermode="x unified",
                height=550
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            # 유사도 계산
            st.markdown("#### 📐 군집 평균과 지역별 유사도")

            similarity_results = []

            for region in members:

                region_values = region_patterns.loc[region]

                if (
                    region_values.std() == 0
                    or cluster_mean.std() == 0
                ):
                    similarity = np.nan
                else:
                    similarity = region_values.corr(
                        cluster_mean
                    )

                similarity_results.append({
                    "지역": region,
                    "군집 평균과의 상관계수": similarity
                })

            similarity_df = pd.DataFrame(
                similarity_results
            )

            similarity_df = similarity_df.sort_values(
                "군집 평균과의 상관계수",
                ascending=False,
                na_position="last"
            )

            valid_similarity = similarity_df[
                "군집 평균과의 상관계수"
            ].dropna()

            if len(valid_similarity) > 0:

                average_similarity = valid_similarity.mean()

                valid_df = similarity_df.dropna(
                    subset=["군집 평균과의 상관계수"]
                )

                most_similar = valid_df.iloc[0]
                least_similar = valid_df.iloc[-1]

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "군집 평균 유사도",
                    f"{average_similarity:.3f}"
                )

                col2.metric(
                    "가장 비슷한 지역",
                    most_similar["지역"]
                )

                col3.metric(
                    "가장 차이가 큰 지역",
                    least_similar["지역"]
                )

                st.info(
                    "상관계수가 1에 가까울수록 군집 평균과 "
                    "비슷한 변화 패턴을 의미합니다. "
                    "0에 가까우면 선형적 유사성이 낮고, "
                    "음수이면 반대 방향의 경향을 보입니다."
                )

            else:

                st.warning(
                    "상관계수를 계산할 수 있는 데이터가 부족합니다."
                )

            similarity_df[
                "군집 평균과의 상관계수"
            ] = similarity_df[
                "군집 평균과의 상관계수"
            ].round(3)

            st.markdown("#### 📋 지역별 유사도 순위")

            st.dataframe(
                similarity_df,
                use_container_width=True,
                hide_index=True
            )

    # 군집별 지역 목록
    st.markdown("### 📋 군집별 지역 목록")

    for cluster_num in range(1, cluster_count + 1):

        members = cluster_df[
            cluster_df["군집"] == cluster_num
        ]["지역"].tolist()

        with st.expander(
            f"{cluster_num}번 군집 ({len(members)}개 지역)"
        ):

            st.write(", ".join(members))

            cluster_mean = (
                region_patterns.loc[members].mean(axis=0) * 100
            )

            cluster_table = pd.DataFrame({
                "연도": all_years,
                "평균 폐교 발생 비율(%)": np.round(
                    cluster_mean.values,
                    2
                )
            })

            st.dataframe(
                cluster_table,
                use_container_width=True,
                hide_index=True
            )

    # 전체 군집 분류 결과
    st.markdown("### 📊 전체 지역 군집 분류 결과")

    cluster_result = cluster_df.sort_values(
        ["군집", "지역"]
    ).reset_index(drop=True)

    st.dataframe(
        cluster_result,
        use_container_width=True,
        hide_index=True
    )

    csv_data = cluster_result.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 군집 분석 결과 CSV 다운로드",
        data=csv_data,
        file_name="폐교_지역별_군집분석.csv",
        mime="text/csv"
    )

else:

    st.warning(
        "지역별 폐교 추세 패턴이 충분히 구분되지 않아 "
        "군집화를 진행할 수 없습니다."
    )
    # ==========================================
# 11. 데이터 부족 안내
# ==========================================

if is_low_data:

    st.warning(
        f"⚠️ {selected_region} 지역의 폐교 기록은 "
        f"{record_count}개로, 예측에 필요한 데이터가 부족합니다. "
        "선형회귀 결과는 참고용으로만 확인해주세요."
    )


# ==========================================
# 12. 선형회귀 모델 분석
# ==========================================

st.subheader("📉 선형회귀 모델 분석")

slope = float(model.coef_[0])

intercept = float(model.intercept_)

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
# 13. 미래 예측 결과 표
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
# 14. 연도별 폐교 학교 수 표
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
# 15. 원본 데이터 확인
# ==========================================

with st.expander("원본 폐교 데이터 확인"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 16. 앱 하단 안내
# ==========================================

st.caption(
    "※ 이 앱은 2025년 9월 30일 기준 폐교 데이터로 분석합니다. "
    "미래 예측은 과거 추세를 연장한 통계적 추정치이며, "
    "실제 폐교 학교 수와 다를 수 있습니다. "
    "군집 분석은 지역별 연도별 폐교 발생 비율을 기준으로 "
    "유사한 추세를 분류한 결과입니다. "
    "군집별 유사도는 상관계수를 이용해 계산합니다."
)
