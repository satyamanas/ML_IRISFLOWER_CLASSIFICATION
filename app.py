# Step 1: Load Important Modules
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit as st
import altair as alt
from sklearn.datasets import load_iris
import pickle

# LOAD DATASETS
data = load_iris()
df = pd.DataFrame(data['data'],
                   columns=data['feature_names'])
df['target'] = data['target']
classes = data['target_names']

X = df.iloc[:, :-1]

# MODEL LIST
all_model_name = ['Logistic Regression',
                   'Naive Bayes',
                   'Decision Tree',
                   'Random Forest',
                   'SVM',
                   'KNN']

all_models = []
for i in all_model_name:
    file_name = i + '.pkl'
    with open(f"{file_name}", 'rb') as f:
        model = pickle.load(f)
        all_models.append(model)

# USER INPUT AND PAGE TITLE
st.title("ML Flower Classification Project")
# Image url
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQflZa84TxjhnwvbKHBVm9eG9jhlg7RJB_jR0q3WD4injX8_9W5QD4d-7Bz&s=10"
st.image(url)

# Show Dataframe sample
st.dataframe(df.sample(5))

# LEFT SIDE BAR for USER VALUE INPUT
st.sidebar.title("Select Iris Features")
st.sidebar.image(url)

user_input = []
for i in X:
    min_i = float(X[i].min())
    max_i = float(X[i].max())
    ans = float(st.sidebar.slider(f"Select value of {i}:", min_value=min_i, max_value=max_i))
    user_input.append(ans)

# USER INPUT SHOW
st.markdown("""
<h2> User Input Value</h2>
""", unsafe_allow_html=True)
st.write(user_input)

# Wrap input as a DataFrame with matching column names to avoid
# the "X does not have valid feature names" warning
input_df = pd.DataFrame([user_input], columns=X.columns)

# MODEL PREDICTION
if st.button("Click here to Predict"):
    with st.spinner("Predicting..."):
        time.sleep(2)
        counter = 0
        model_ans = []
        model_prob = []
        for model in all_models:
            ans = model.predict(input_df)[0]
            try:
                prob = model.predict_proba(input_df).max()
            except Exception:
                prob = 1
            model_prob.append(prob)
            class_ans = classes[ans]
            model_ans.append(class_ans)
            # st.write(f"Prediction by: {all_model_name[counter]}===>{class_ans}")
            counter += 1

        st.markdown("""
        <h2> Model Comparison </h2>
        """, unsafe_allow_html=True)

        comp_df = pd.DataFrame({"x": all_model_name,
                                 "y": model_prob,
                                 'Model-Prediction': model_ans})

        chart = (alt.Chart(comp_df).mark_bar().encode(
            x='x',
            y='y',
            tooltip=['x', 'y', 'Model-Prediction']
        ))

        st.altair_chart(chart, use_container_width=True)

        st.markdown("""
        <h2> Final Prediction </h2>
        """, unsafe_allow_html=True)

        datd = pd.Series(model_ans)
        final_ans = datd.mode().values[0]
        st.success(final_ans)

footer = """
<style>
footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: #888888;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
}
</style>
<div class="footer">
    <p>Made with ❤️ using Streamlit | © 2026</p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)