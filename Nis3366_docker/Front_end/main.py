import streamlit as st

st.set_page_config(
    page_title="微博爬虫数据分析",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


pg = st.navigation({
    "微博数据爬取": [
        st.Page("./web_pages/Cookie/Cookie.py", title="Cookie", icon=":material/add_circle:"),
        st.Page("./pages/WhatsToday.py", title="今日热点", icon=":material/add_circle:"),
        st.Page("./pages/search.py",title="搜索", icon=":material/add_circle:"),
    ],
    "数据分析": [
        st.Page("./pages/details.py", title="数据分析", icon=":material/add_circle:" ),
        st.Page("./pages/future_analysis.py", title="数据预测", icon=":material/add_circle:"),
    ],
})

pg.run()