import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =========================================================================
# 🔥 [인증서 에러 절대 안 남] 깃허브 업로드 폰트 강제 주입 로직
# =========================================================================
# 현재 파일(lung_app.py) 위치를 기준으로 폰트 경로를 절대 경로로 잡습니다.
current_dir = os.path.dirname(os.path.abspath(__file__))
font_name = "MalgunGothic.ttf"
font_path = os.path.normpath(os.path.join(current_dir, font_name))

# 인터넷 다운로드(urllib)를 절대 쓰지 않고 내 깃허브 파일에서 바로 읽습니다.
if os.path.exists(font_path):
    try:
        # 시스템 캐시 메모리를 거치지 않고 Matplotlib 엔진에 직접 등록합니다.
        font_prop = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
    except Exception as e:
        # 만약의 상황을 대비한 예외 처리 우회
        plt.rc('font', family='sans-serif')
else:
    # 파일이 진짜 없을 때만 사이트에 경고창을 표시합니다.
    st.error(f"❌ 깃허브 폴더 안에 '{font_name}' 파일이 진짜로 없습니다! 파일이 제대로 업로드되었는지 확인해 주세요.")
    plt.rc('font', family='sans-serif')

# 그래프 내 마이너스 기호 깨짐 예방
plt.rcParams['axes.unicode_minus'] = False
# =========================================================================

st.set_page_config(page_title="폐암 환자 군집 분석", layout="wide")

st.title("🫁 폐암 환자 데이터 군집 분석 및 새로운 환자 위치 시각화")
st.write("왼쪽 사이드바에서 환자의 정보를 입력하면, 군집 분석 결과 그래프에 환자의 위치가 실시간으로 표시됩니다.")

# 2. 데이터 로드 및 모델 학습
@st.cache_data
def load_and_train():
    try:
        data = pd.read_csv("lung.csv")
    except FileNotFoundError:
        return None, None, None, None
    
    data = data.rename(columns={
        'Name': '이름',
        'Surname': '성',
        'Age': '나이',
        'Smokes': '흡연',
        'AreaQ': '지역수준',
        'Alkhol': '음주',
        'Result': '결과',
    })
    
    features = ['나이', '흡연', '지역수준', '음주']
    X = data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return data, X, scaler, kmeans

data, X_features, scaler, kmeans = load_and_train()

if data is None:
    st.error("⚠️ 'lung.csv' 파일을 찾을 수 없습니다. GitHub 저장소에 파일이 올바르게 업로드되었는지 확인해 주세요.")
    st.stop()

# 3. 사이드바 UI 구성
st.sidebar.header("📊 새로운 환자 정보 입력")

user_inputs = {}
user_inputs['나이'] = st.sidebar.slider("나이", min_value=10, max_value=90, value=40)
user_inputs['흡연'] = st.sidebar.slider("흡연 수준", min_value=0, max_value=40, value=15)
user_inputs['지역수준'] = st.sidebar.slider("지역 수준", min_value=1, max_value=10, value=5)
user_inputs['음주'] = st.sidebar.slider("음주 수준", min_value=0, max_value=10, value=3)

st.subheader("📈 전체 데이터 내 새로운 환자 위치 시각화")
col1, col2 = st.columns(2)
with col1:
    x_axis = st.selectbox("X축 선택 (가로 축)", options=X_features.columns, index=1)
with col2:
    y_axis = st.selectbox("Y축 선택 (세로 축)", options=X_features.columns, index=3)

# 4. 예측 결과 도출 및 위험도 매칭
new_patient = pd.DataFrame([user_inputs])
new_patient_scaled = scaler.transform(new_patient)
predicted_cluster = kmeans.predict(new_patient_scaled)[0]

cluster_details = {
    0: {
        "title": "🚨 고위험군 (군집 0)",
        "description": "평균 나이가 많고 흡연량과 음주 수준이 모두 매우 높은 환자군입니다. 폐암 발생 가능성이 가장 높으므로 즉각적인 정밀 검진과 금연/금주 조치가 강하게 권고됩니다.",
        "type": "error"
    },
    1: {
        "title": "⚠️ 중위험군 (군집 1)",
        "description": "나이는 보통이거나 높은 편이며, 흡연이나 음주 중 특정 항목의 수치가 높은 환자군입니다. 지속적인 추적 관찰과 생활 습관 개선이 필요합니다.",
        "type": "warning"
    },
    2: {
        "title": "✅ 저위험군 (군집 2)",
        "description": "상대적으로 나이가 적고 흡연량과 음주 수준이 매우 낮은 안전한 환자군입니다. 현재의 건강한 생활 습관을 유지하는 것이 좋습니다.",
        "type": "success"
    }
}

st.subheader("🎯 환자 군집 분석 및 위험도 예측")
cluster_info = cluster_details.get(predicted_cluster)

if cluster_info["type"] == "error":
    st.error(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")
elif cluster_info["type"] == "warning":
    st.warning(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")
else:
    st.success(f"**{cluster_info['title']}** 에 속합니다.\n\n{cluster_info['description']}")

# 5. 분산형 그래프 생성 및 출력
fig, ax = plt.subplots(figsize=(10, 5))
colors = {0: '#ff4b4b', 1: '#ffa500', 2: '#00b050'}

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

ax.set_title(f"군집 분포 결과 ({x_axis} vs {y_axis})", fontsize=14, pad=12)
ax.set_xlabel(x_axis, fontsize=11)
ax.set_ylabel(y_axis, fontsize=11)
ax.legend(loc='best')
ax.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig)
