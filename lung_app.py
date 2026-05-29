import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 한글 폰트 설정 (웹 화면 그래프 글자 깨짐 방지)
plt.rc('font', family='Malgun Gothic')  # Windows 기준
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="폐암 환자 군집 분석", layout="wide")

# 제목 및 설명
st.title("🫁 폐암 환자 데이터 군집 분석 및 새로운 환자 위치 시각화")
st.write("왼쪽 사이드바에서 환자의 정보를 입력하면, 군집 분석 결과 그래프에 환자의 위치가 실시간으로 표시됩니다.")
font_name = "MalgunGothic.ttf" 
import os
if os.path.exists(font_name):
    # 깃허브 서버(리눅스) 환경일 때: 내 폴더의 폰트 파일을 직접 등록
    fm.fontManager.addfont(font_name)
    prop = fm.FontProperties(fname=font_name)
    plt.rc('font', family=prop.get_name())
else:
    # 혹시나 파일이 없을 때를 대비한 기본 설정
    plt.rc('font', family='sans-serif')

# 그래프에서 마이너스(-) 기호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False
# =========================================================================

st.set_page_config(page_title="폐암 환자 군집 분석", layout="wide")
# 2. 데이터 로드 및 모델 학습 (페이지 구동 시 자동 실행)
@st.cache_data
def load_and_train():
    try:
        data = pd.read_csv("lung.csv")
    except FileNotFoundError:
        return None, None, None, None
    
    # 코랩 코드와 동일하게 데이터 컬럼명을 정제합니다.
    data = data.rename(columns={
        'Name': '이름',
        'Surname': '성',
        'Age': '나이',
        'Smokes': '흡연',
        'AreaQ': '지역수준',
        'Alkhol': '음주',
        'Result': '결과',
    })
    
    # 텍스트 데이터(이름, 성) 및 타겟값(결과)을 제외한 순수 분석 특성 데이터 추출
    features = ['나이', '흡연', '지역수준', '음주']
    X = data[features]
    
    # 데이터 스케일링 전처리
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means 모델 생성 및 학습 (군집 수 = 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return data, X, scaler, kmeans

# 글로벌 변수 선언 및 안전하게 데이터 수집
data, X_features, scaler, kmeans = load_and_train()

# 데이터 파일이 없을 때 경고창 띄우기
if data is None:
    st.error("⚠️ 'lung.csv' 데이터 파일이 웹 프로그램 파일(app.py)과 같은 폴더에 없습니다! 파일명을 확인하시고 복사해 주세요.")
    st.stop()

# 3. 사이드바 - 사용자 입력 UI 구성
st.sidebar.header("📊 새로운 환자 정보 입력")

user_inputs = {}
user_inputs['나이'] = st.sidebar.slider("나이", min_value=10, max_value=90, value=40)
user_inputs['흡연'] = st.sidebar.slider("흡연 수준", min_value=0, max_value=40, value=15)
user_inputs['지역수준'] = st.sidebar.slider("지역 수준", min_value=1, max_value=10, value=5)
user_inputs['음주'] = st.sidebar.slider("음주 수준", min_value=0, max_value=10, value=3)

# 시각화 관람용 축 설정
st.subheader("📈 전체 데이터 내 새로운 환자 위치 시각화")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X축 선택 (가로 축)", options=X_features.columns, index=1)  # 기본값: 흡연
with col2:
    y_axis = st.selectbox("Y축 선택 (세로 축)", options=X_features.columns, index=3)  # 기본값: 음주

# 4. 입력 데이터 전처리 및 새 환자 군집 결과 예측
new_patient = pd.DataFrame([user_inputs])
new_patient_scaled = scaler.transform(new_patient)
predicted_cluster = kmeans.predict(new_patient_scaled)[0]

# 🔥 [🔥 이번에 추가된 핵심 로직] 군집별 위험도 및 상세 설명 매칭 딕셔너리
# 코랩의 군집 레이블(0, 1, 2) 순서에 따른 실제 위험도 정의
cluster_details = {
    0: {
        "title": "🚨 고위험군 (군집 0)",
        "description": "평균 나이가 많고 흡연량과 음주 수준이 모두 매우 높은 환자군입니다. 폐암 발생 가능성이 가장 높으므로 즉각적인 정밀 검진과 금연/금주 조치가 강하게 권고됩니다.",
        "type": "error"  # 빨간색 상자
    },
    1: {
        "title": "⚠️ 중위험군 (군집 1)",
        "description": "나이는 보통이거나 높은 편이며, 흡연이나 음주 중 특정 항목의 수치가 높은 환자군입니다. 지속적인 추적 관찰과 생활 습관 개선이 필요합니다.",
        "type": "warning"  # 노란색 상자
    },
    2: {
        "title": "✅ 저위험군 (군집 2)",
        "description": "상대적으로 나이가 적고 흡연량과 음주 수준이 매우 낮은 안전한 환자군입니다. 현재의 건강한 생활 습관을 유지하는 것이 좋습니다.",
        "type": "success"  # 초록색 상자
    }
}

# 결과 정보 및 위험도 해석 출력
st.subheader("🎯 환자 군집 분석 및 위험도 예측")
cluster_info = cluster_details.get(predicted_cluster, {"title": f"군집 {predicted_cluster}", "description": "분석된 군집 정보가 없습니다.", "type": "info"})

# 예측된 군집의 위험도에 따라 알림창 색상을 다르게 보여줍니다.
if cluster_info["type"] == "error":
    st.error(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")
elif cluster_info["type"] == "warning":
    st.warning(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")
else:
    st.success(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")


# 5. 코랩 시각화 로직 적용 (Matplotlib 구현)
fig, ax = plt.subplots(figsize=(10, 5))

# 군집 색상 정의 (0: 빨강, 1: 주황/보라, 2: 초록)
colors = {0: '#ff4b4b', 1: '#ffa500', 2: '#00b050'}

# 기존 데이터셋의 환자 분산 플롯 그리기
for cluster_id in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster_id]
    ax.scatter(
        cluster_data[x_axis], 
        cluster_data[y_axis], 
        color=colors.get(cluster_id, 'gray'),
        label=f'군집 {cluster_id}', 
        alpha=0.6, 
        s=100
    )

# 사용자가 입력한 새로운 환자의 실시간 위치를 커다란 황금색 별 모양(★)으로 강조 표시
ax.scatter(
    new_patient[x_axis].values[0],
    new_patient[y_axis].values[0],
    color='gold',
    marker='*',
    s=450,
    label='새로운 환자 위치',
    edgecolor='black',
    linewidth=1.5,
    zorder=10
)

ax.set_title(f"군집 분포 그래프 ({x_axis} vs {y_axis})", fontsize=14, pad=12)
ax.set_xlabel(x_axis, fontsize=11)
ax.set_ylabel(y_axis, fontsize=11)
ax.legend(loc='best')
ax.grid(True, linestyle='--', alpha=0.3)

# 웹 대시보드 화면에 최종 그래프 출력
st.pyplot(fig)
