import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Fruit Recommendation System",
    layout="centered"
)

@st.cache_data
def load_data():
    df = pd.read_csv("fruit_dataset_100.csv")
    df = df.set_index("fruit_100g")
    df.index.name = None
    return df

df = load_data()

features = df.select_dtypes(include=['number']).fillna(0)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

similarity_matrix = cosine_similarity(features_scaled)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=features.index,
    columns=features.index
)

def recommend_fruits(fruit_name, top_n=3):
    fruit_name = fruit_name.strip().title()

    if fruit_name not in similarity_df.index:
        return None

    similar = similarity_df[fruit_name].sort_values(ascending=False)

    result = similar.iloc[1:top_n+1]
    result = (result * 100).round(2)

    return result

st.title("🍎 Fruit Recommendation System")
st.write("Find fruits with similar nutritional values using cosine similarity.")

# Dropdown
fruit_list = sorted(similarity_df.index.tolist())
selected_fruit = st.selectbox("Select a fruit:", fruit_list)

# Slider
top_n = st.slider("Number of recommendations:", 1, 10, 3)

# Button
if st.button("Recommend"):
    result = recommend_fruits(selected_fruit, top_n)

    if result is None:
        st.error("Fruit not found!")
    else:
        st.subheader(f"Top {top_n} fruits similar to {selected_fruit}")

        # Table
        st.dataframe(result.to_frame(name="Similarity (%)"))

        # Chart
        st.bar_chart(result)

        # Highlight best match
        st.success(f"Most similar fruit: {result.index[0]}")


with st.expander("🔍 Show Raw Data"):
    st.dataframe(df)

st.markdown("---")
st.caption("Built with Streamlit | Content-Based Recommendation System")