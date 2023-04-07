import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def get_dict_files(doss_name, ext):
    import os
    import pickle
    from os.path import dirname, join

    list_dir = os.listdir(join(dirname(dirname(__file__)), doss_name))

    return {i.removesuffix(ext) : pickle.load(open(join(doss_name, i), 'rb')) for i in list_dir if i.endswith(ext)}


def to_dataframe(results):

    d = {
        k.removeprefix("param_model__"): results[k]
        for k in results.keys()
        if k.startswith("param_model__")
        or k == "mean_test_score"
        or k == "rank_test_score"
        }

    return pd.DataFrame(d)


def main():

    st.set_page_config(page_title="Exploration des modèles", layout="wide")
    st.title("Exploration des modèles")

    models = get_dict_files("models", ".mdl")
    confmats = get_dict_files("confusion_matrix", ".cm")

    tabs = {k: v for k, v in zip(models.keys(), st.tabs(list(models.keys())))}

    for model_name in models.keys():

        mdl = models[model_name]
        cm = confmats[model_name]
        tab = tabs[model_name]

        tab.header(model_name)

        tab.subheader("Score des paramètres :")

        df = to_dataframe(mdl.cv_results_)

        pdf = pd.pivot_table(df, values='mean_test_score', index=[df.columns[0]], columns=[df.columns[1]])

        fig_results = px.imshow(pdf, text_auto=True)

        tab.plotly_chart(fig_results)

        tab.subheader("5 meilleurs combo :")
        tab.dataframe(df.sort_values("rank_test_score").set_index("rank_test_score").head(5))

        tab.subheader("Meilleur combo :")

        fig_cm = px.imshow(cm,
                        text_auto = True,
                        labels = {'x':"Prediction", 'y':"True"},
                        x = ['Satisfaction', 'Dissatisfield'],
                        y = ['Satisfaction', 'Dissatisfield']
                    )
        fig_cm.update_xaxes(side = "top")

        tab.write(f"Best score: **{mdl.best_score_}**")
        tab.write(f"Best params: **{mdl.best_params_['model']}**")
        tab.plotly_chart(fig_cm)


        if model_name == "RandomForestClassifier":

            tab.subheader("Feature importance :")
            feature_importances = mdl.best_estimator_.named_steps['model'].feature_importances_
            feature_names = mdl.best_estimator_.named_steps['preparation'].get_feature_names_out()

            feature_importance = pd.Series({feature_names[i] : feature_importances[i] for i in range(len(feature_importances))})

            feature_importance.sort_values(ascending=False, inplace=True)
            tab.dataframe(feature_importance)

if __name__ == '__main__':
    main()