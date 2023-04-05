import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_data():

    return pd.read_csv("Airline Passenger Satisfaction.csv")


@st.cache_data
def load_hist(df, col_name):

    return px.histogram(df, x=col_name)


def main():

    st.set_page_config("Sentiment Analysis", layout="wide")
    st.title("Sentiment Analysis")

    df = load_data()

    tab_hist, tab2 = st.tabs(['Histogrammes', 'pouet'])

    with tab_hist:

        nb_col = 2
        cols = st.columns(nb_col)

        for i in range(len(df.columns)):

            c = df.columns[i]

            fig = load_hist(df, c)

            cols[i % nb_col].header(c)
            cols[i % nb_col].plotly_chart(fig)

    with tab2:

        st.write("lol")


if __name__ == "__main__":
    main()