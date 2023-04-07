import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data() -> pd.DataFrame:

    return pd.read_csv("Airline Passenger Satisfaction.csv")


@st.cache_data
def load_hist(df, col_name):

    return px.histogram(df, x=col_name)


@st.cache_data
def load_violin(df, col_name):

    return px.violin(df, x="Satisfaction", y=col_name, box = True)


def main():

    st.set_page_config(page_title="Sentiment Analysis", layout="wide")
    st.title("Sentiment Analysis")

    df = load_data()

    tab_hist, tab_corr, tab_violin = st.tabs(['Histogrammes', 'Correlation', 'Violins'])

    with tab_hist:

        nb_col = 2
        cols = st.columns(nb_col)

        for i in range(len(df.columns)):

            c = df.columns[i]

            fig = load_hist(df, c)

            cols[i % nb_col].header(c)
            cols[i % nb_col].plotly_chart(fig)

    with tab_corr:

        st.plotly_chart(px.imshow(df.drop(columns=["id"]).corr(numeric_only=True)))

    with tab_violin:

        c = st.selectbox("Colonne à afficher:", df.select_dtypes(exclude='object').columns)

        st.plotly_chart(load_violin(df, c))


if __name__ == "__main__":
    main()