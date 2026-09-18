import streamlit as st

st.set_page_config(
    page_title="Nepal Earthquake — Building Damage",
    page_icon="\U0001F3E0",
    layout="wide",
)

pages = [
    st.Page("pages/0_Home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_Predict_Damage.py", title="Predict Damage", icon="🏢"),
    st.Page("pages/2_What_If_Analysis.py", title="What-if Analysis", icon="🔬"),
    st.Page("pages/3_Dataset_Explorer.py", title="Dataset Explorer", icon="📊"),
    st.Page("pages/4_Model_Performance.py", title="Model Performance", icon="📈"),
    st.Page("pages/5_Notebook_Viewer.py", title="Analysis Notebook", icon="📓"),
    st.Page("pages/6_About.py", title="About", icon="ℹ️"),
]

nav = st.navigation(pages)
nav.run()
