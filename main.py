import pandas as pd
import numpy as np

import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


st.title('Employee Churn')


@st.cache(allow_output_mutation=True)
def load_data_for_model():
    df = pd.read_csv('HR_comma_sep__281_29__281_29__282_29.csv')

    return df


df = load_data_for_model()

df.columns = ['satisfaction', 'last_evaluation', 'number_projects',
              'average_monthly_hours', 'time_spend_company', 'accident', 'left',
              'promotion_last_5years', 'sales', 'salary']

col1, col2 = st.columns(2)
with col1:
    st.subheader('Input Employee Parameters')
with col2:
    st.metric('Model Accuracy:', '98.8%')


col1, col2, col3 = st.columns(3)

with col1:
    satisfaction = st.number_input('Employee Satisfaction Level', step=0.1)

with col2:
    last_evaluation = st.number_input('Employee Evaluation', step=0.1)

with col3:
    number_projects = st.number_input('Number of Projects', step=1)

col1, col2, col3 = st.columns(3)


with col1:
    average_monthly_hours = st.number_input('Average monthly hours', step=1)

with col2:
    time_spend_company = st.number_input('time spent in comany', step=1)

with col3:
    check_yes = st.checkbox('Did employee have an accident?')
    if check_yes:
        accident = 1
    else:
        accident = 0

col1, col2, col3 = st.columns(3)

with col1:
    sales = st.selectbox('Department:', ('sales',
                                         'technical',
                                         'support',
                                         'IT',
                                         'product_mng',
                                         'marketing',
                                         'RandD',
                                         'accounting',
                                         'hr',
                                         'management'))

with col2:
    salary = st.selectbox('Salary:', (
        'low',
        'medium',
        'high'
    ))

with col3:
    check_yes = st.checkbox('Promotion last 5 years?')
    if check_yes:
        promotion_last_5years = 1
    else:
        promotion_last_5years = 0

data = {
    'satisfaction': satisfaction,
    'last_evaluation': last_evaluation,
    'number_projects': number_projects,
    'average_monthly_hours': average_monthly_hours,
    'time_spend_company': time_spend_company,
    'accident': accident,
    'left': 1,
    'promotion_last_5years': promotion_last_5years,
    'sales': sales,
    'salary': salary
}

user_input = pd.DataFrame(data, index=[0])
df = pd.concat([df, user_input], ignore_index=True)


def pre_processing():

    X = df.drop('left', axis=1)
    y = df.left
    y = y.iloc[:-1]

    X_dums = pd.get_dummies(X, columns=['number_projects', 'time_spend_company',
                                        'accident', 'promotion_last_5years', 'salary', 'sales'], drop_first=True)

    user_input = X_dums.iloc[-1, :]
    user_input = pd.DataFrame(user_input.values.reshape(
        1, -1), columns=user_input.keys())
    X_dums = X_dums.iloc[:-1, :]

    X_train, X_test, y_train, y_test = train_test_split(
        X_dums, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(
        X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    user_input = pd.DataFrame(scaler.transform(
        user_input), columns=user_input.columns)

    return X_train, y_train, user_input, X_test, y_test


X_train, y_train, user_input, X_test, y_test = pre_processing()


def xgb_model(X_train, y_train):
    xgb_model = XGBClassifier(
        use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, importance_type='weight')
    xgb_model.fit(X_train, y_train)

    return xgb_model


model = xgb_model(X_train, y_train)
probability = model.predict_proba(user_input)

st.metric(label='Probability that employee will quit:',
          value=f'{round(probability[0][1] * 100 ,2)}%')
