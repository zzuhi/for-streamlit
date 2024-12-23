
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


# Streamlit 페이지 설정
st.set_page_config(page_title="맥주 추천 시스템", page_icon="🍺", layout="centered")


def recommend_beers(beer_data, diversity_mode, recommended_beer_name):
    # 데이터 정리
    numeric_cols = ["Aroma", "Appearance", "Flavor", "Mouthfeel", "Overall"]
    beer_data[numeric_cols] = beer_data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    beer_data.dropna(subset=numeric_cols, inplace=True)

    # 기준 맥주 인덱스 찾기
    if recommended_beer_name not in beer_data.index:
        raise ValueError(f"Beer name '{recommended_beer_name}' not found in data.")

    reference_beer_index = recommended_beer_name  # 기준 맥주 인덱스

    # 유사성 점수 계산
    user_vector = beer_data.loc[reference_beer_index, numeric_cols].values.reshape(1, -1)
    beer_vectors = beer_data[numeric_cols].values

    # 데이터 정규화
    scaler = MinMaxScaler()
    beer_vectors_scaled = scaler.fit_transform(beer_vectors)
    user_vector_scaled = scaler.transform(user_vector)

    # 가중치 부여
    weight_vector = np.array([2, 1, 2, 1, 4])  # Aroma, Flavor: 2배; Overall: 4배
    weighted_user_vector = user_vector_scaled * weight_vector
    weighted_beer_vectors = beer_vectors_scaled * weight_vector

    # 거리 계산 (유클리드 거리)
    distances = np.linalg.norm(weighted_beer_vectors - weighted_user_vector, axis=1)
    beer_data["Similarity"] = distances

    # 다양성 모드 처리
    if diversity_mode == 1:
        # 다양성 모드: 유사성 점수가 낮은 순서대로 추천하되 선택된 모든 맥주의 카테고리와 다른 맥주 선택
        recommended_beers = []
        recommended_categories = [beer_data.loc[reference_beer_index, "category"]]
        for idx in beer_data.sort_values(by="Similarity", ascending=True).index:
            if idx == reference_beer_index:
                continue
            current_category = beer_data.loc[idx, "category"]
            if current_category not in recommended_categories:
                recommended_beers.append(idx)
                recommended_categories.append(current_category)
            if len(recommended_beers) == 3:  # 3개 추천
                break
        recommendations = beer_data.loc[recommended_beers]
    else:
    # 유사성 모드: 유사도가 낮은 순서대로 추천
      recommendations = beer_data.sort_values(by="Similarity", ascending=True).iloc[1:4]


    return recommendations



# 데이터 및 모델 로드
@st.cache_data
def load_data():
    file_path = "beer_data.csv"
    return pd.read_csv(file_path)

@st.cache_data
def load_model():
    # pkl 파일에서 모델 로드
    with open("./final_model_lgb.pkl", "rb") as file:
        model = pickle.load(file)
    return model

beer_data = load_data()
model = load_model()


# 탭 메뉴 생성
menu = st.radio(
    "단계를 선택하세요:",
    ("설문", "추천 결과"),
    horizontal=True
)

# 세션 상태 초기화
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = None
if 'recommendation_type' not in st.session_state:
    st.session_state['recommendation_type'] = None
if 'recommendation_made' not in st.session_state:
    st.session_state['recommendation_made'] = False

# 설문 페이지
if menu == "설문":
    st.title("🍺 맥주 추천 시스템 - 설문")
    st.markdown("아래 설문을 작성한 후 추천 결과를 확인하세요.")

    # 설문 슬라이더
    def slider_with_caption(question, slider_text, min_caption, max_caption):
        st.subheader(question)
        slider_score = st.slider(
            slider_text,
            min_value=1,
            max_value=10,
            value=5
        )
        slider_layout = st.columns([1, 7, 1])  # Adjust layout proportions
        with slider_layout[0]:
            st.caption(min_caption)
        with slider_layout[2]:
            st.caption(max_caption)
        return slider_score

    q1_score = slider_with_caption(
        "1번 질문",
        "향이 너무 좋으면, ‘이건 대박이야’ 하면서 신나는 편?",
        "1: 향? 그게 뭐 대단해?",
        "10: 향 맡고 바로 ‘대박 맥주’ 외친다!"
    )

    q2_score = slider_with_caption(
        "2번 질문",
        "빛깔 따지는 사람, 혹시 카페라떼랑 맥주도 헷갈리시는 건 아니죠?",
        "1: 맥주를 눈으로 마시나",
        "10: 이 색깔, 내 인생 맥주 맞는 것 같아!"
    )

    q3_score = slider_with_caption(
        "3번 질문",
        "풍 : 풍미에 미 : 미쳤어?",
        "1: 안미침",
        "10: 유서우급 미침"
    )

    # 3번 질문 아래 이미지 추가
    columns = st.columns([5, 1])  # 오른쪽에 이미지 추가
    with columns[1]:
        st.image("sw.jpg", width=400)

    q4_score = slider_with_caption(
        "4번 질문",
        "맥주 한 잔에서 부드러운 목넘김을 느낄 때, 그게 낭만 아니겠어?",
        "1: 낭만 과한데?",
        "10: 목넘김이 곧 예술이다!"
    )

    q5_score = slider_with_caption(
        "5번 질문", 
        "술 한 잔 마시면 바로 황정민 모드? 너는 어때?",
        "1: 그냥 조용히 마시는 편",
        "10: 황정민 모드 ON"
    )

    # 5번 질문 아래 이미지 추가 (가운데 정렬)
    columns = st.columns([1, 6, 1])  # 가운데 정렬을 위해 비율 설정
    with columns[1]:
        st.image("avb.png", width=500)

    # 다양성 선택 (라디오 버튼 추가)
    st.subheader("6번 질문")
    st.markdown("추천 다양성 모드")
    diversity_weight = st.radio(
        "어떤 맥주 조합을 원해?",
        options=[0, 1],  # 0: 유사성 우선, 1: 다양성 고려
        format_func=lambda x: "느낌 비슷한 맥주 네캔!" if x == 0 else "느낌 다른 맥주 네캔!",
        horizontal=True
    )

    # 설문 저장
    if st.button("설문 완료"):
        # 사용자 점수를 DataFrame으로 정리
        user_input = pd.DataFrame([{
            "Aroma": q1_score,
            "Appearance": q2_score,
            "Flavor": q3_score,
            "Mouthfeel": q4_score
        }])

        # `Overall` 값 예측
        overall_prediction = model.predict(user_input)[0]
        user_input["Overall"] = overall_prediction  # 예측된 Overall 값을 추가

        # CSV 데이터에서 필요한 컬럼만 사용
        beer_features = ["Aroma", "Appearance", "Flavor", "Mouthfeel", "Overall"]
        beer_scores = beer_data[beer_features]

        # 정규화
        scaler = MinMaxScaler()
        beer_scores_scaled = scaler.fit_transform(beer_scores)  # 맥주 점수 정규화
        user_vector_scaled = scaler.transform(user_input)       # 사용자 점수 정규화

        # 가중치 부여
        weight_vector = np.array([2, 1, 2, 1, 4])  # Aroma, Flavor: 2배; Overall: 4배
        weighted_user_vector = user_vector_scaled * weight_vector
        weighted_beer_vectors = beer_scores_scaled * weight_vector

        # 유사도 계산 (유클리드 거리)
        distances = np.linalg.norm(weighted_beer_vectors - weighted_user_vector, axis=1)
        most_similar_idx = np.argmin(distances)  # 거리 기준 가장 유사한 맥주 선택
        recommended_beer_name = beer_data.index[most_similar_idx]  # 인덱스에서 맥주 이름 가져오기

        # 결과 저장
        st.session_state['user_input'] = {
            'Aroma': q1_score,
            'Appearance': q2_score,
            'Flavor': q3_score,
            'Mouthfeel': q4_score,
            'Overall': overall_prediction
        }
        st.session_state['diversity_weight'] = diversity_weight
        st.session_state['recommended_beer_name'] = recommended_beer_name
        st.session_state['recommendation_made'] = True
        st.success(f"설문이 완료되었습니다! 맥주 추천을 준비했습니다. 상단 메뉴에서 '추천 결과'를 선택하세요.")



# 추천 결과 페이지
if menu == "추천 결과":
    st.title("🍺 맥주 추천 시스템 - 추천 결과")

    if not st.session_state['recommendation_made']:
        st.warning("먼저 '설문' 탭에서 설문을 완료해주세요.")
    else:
        # 세션 데이터 불러오기
        diversity_mode = st.session_state["diversity_weight"]  # 0: 유사성, 1: 다양성
        recommended_beer_name = st.session_state["recommended_beer_name"]  # 기준 맥주 이름

        # 추천 계산
        try:
            recommended_beers = recommend_beers(beer_data, diversity_mode, recommended_beer_name)
        except ValueError as e:
            st.error(str(e))
            st.stop()

        # 추천된 맥주 이름 가져오기
        final_recommendations = [recommended_beer_name] + recommended_beers.index.tolist()

        # 최종 추천 결과 메시지 생성
        final_recommendations = [beer_data.loc[recommended_beer_name, "맥주이름"]] + [
            beer_data.loc[name, "맥주이름"] for name in recommended_beers.index
        ]
        
        if diversity_mode == 1:
            recommendation_message = (
                f"<div style='text-align:center; color:pink; font-size:24px; font-weight:bold;'>"
                f"버라이어티한 맥주 탐험을 원하는 그대에게</div>"
            )
        else:
            recommendation_message = (
                f"<div style='text-align:center; color:lightgreen; font-size:24px; font-weight:bold;'>"
                f"취향에 꼭 맞는 맥주 힐링을 원하는 그대에게</div>"
            )
        
        # 추천 결과 메시지 출력
        st.markdown("### 🍺 최종 추천 결과 🍺", unsafe_allow_html=True)
        st.markdown(recommendation_message, unsafe_allow_html=True)
        
        # 맥주 목록을 박스로 감싸서 출력
        box_style = """
        <div style='
            border: 2px solid #ccc; 
            border-radius: 10px; 
            padding: 20px; 
            background-color: #f9f9f9;
            text-align: center;
            font-size: 18px;
            font-weight: bold;'>
            {beers}
        </div>
        """
        
        beer_list_html = "<br>".join(final_recommendations)  # 맥주 이름 줄바꿈
        st.markdown(box_style.format(beers=beer_list_html), unsafe_allow_html=True)

        
        # 상세 정보 출력
        st.markdown("### 추천 맥주 상세 정보")
        
        # 기준 맥주 추가
        reference_beer = beer_data.loc[[recommended_beer_name]]  # 기준 맥주 데이터 가져오기
        
        # 기준 맥주와 추천 맥주 병합
        detailed_info = pd.concat([reference_beer, recommended_beers])
        
        # 맥주 이름 열 추가
        detailed_info["맥주 이름"] = detailed_info.index.map(lambda idx: beer_data.loc[idx, "맥주이름"])
        
        # 데이터 출력
        st.dataframe(detailed_info[["맥주 이름", "Aroma", "Appearance", "Flavor", "Mouthfeel", "Overall", "category"]])
        




