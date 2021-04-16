from sklearn import tree
import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px

df = pd.read_csv('Burnout-rate-After-Cleanse.csv')

#ML - Model
# loading the dataset
df1 = pd.read_csv('Burnout-Rate-Before-modelling.csv')
data = df1[['burnout', 'Employee_burnout']]
# Since the last column is in words, it has to be converted into a suitable scale
data['scale'] = 0
data.loc[data['Employee_burnout'] == 'No burnout', 'scale'] = 0
data.loc[data['Employee_burnout'] == 'Alarming burnout', 'scale'] = 3
data.loc[data['Employee_burnout'] == 'Moderate burnout', 'scale'] = 2
data.loc[data['Employee_burnout'] == 'Total burnout', 'scale'] = 1
# loading the columns in x and y variables
x = data['burnout']
# Reshaping the x columns as a 2D array is expected here
x = x.values.reshape(-1, 1)
y = data['scale']
model = tree.DecisionTreeClassifier()
model.fit(x, y)


st.title("Identifying signs of Employee burnout")
nav = st.sidebar.radio("Navigation", ["Home", "EDA", "Are you burntout?"])

if nav == "Home":
    st.markdown(""" ## Introduction """, True)
    st.text(" ")
    st.markdown(""" ## About Us """, True)
    st.markdown("We are a group of 3, Anagha M, Sanketh Raj Patil, Vishak S from University Visvesvaraya College of Engineering. We are young, enthusiastic and passionate Data Science Engineers. Having been a part of various college activities and volunteering for the same, we know how serious the issue of Burnout and is and its need of address. It was a “Been there, done that” kind of a situation for us and that is exactly what prompted us to take up this problem statement. We want to address the seriousness and unhealthy aspects of being burnt out in our daily life.")
    st.markdown(""" ## Strategy """, True)
    st.markdown(" We started the project by collecting and concatenating the dataset provided to us from SAP labs. Once this was ready, to make the dataset look clean and proficient we had to clean it and generate certain data. We then jumped into EDA, which gave us a brief idea about how the variables are dependent on each other. Once this was looked into and decided, it was time to initiate the next plan. Before we jumped into our code, we had discussions to decide which columns from the dataset is important as not all stress is bad stress. In many situations, stress has proven to be motivating and challenging. Next, we set the bar for the particular columns and then put down a secret number as marks for each column. This number is your burnout number and it tells how much one particular factor contributes to your burnout. Example: If among two employees, one has attended 20 meetings compared to his colleague who has attended only 8 meetings this week, the formers burnout score is bound to be higher. Similarly, each category’s burnout score was added. A standard was set for each burnout score giving us a scale of burnout. Next, we ran various Machine algorithm to tell us which one gave the perfect results. Later a suitable algorithm was finalised and the model was finally deployed on a webapp.")
    st.markdown(""" ## GitHub link """, True)
    st.markdown(" Please click on the below link to visit out GitHub repository, make sure to read the README file before running it. ")
    st.markdown(
        """<a href="https://github.com/sankethrajpatil/SAPHackathon"> GitHub Repo for this website </a> """, unsafe_allow_html=True,)

if nav == "EDA":

    if st.checkbox("Show Table"):
        st.table(df)

    graph = st.selectbox("What kind of Graph ? ", [
                         "Missed Deadlines", "Peer(s) workload", "Time/Duration", "General leave patterns", "Missed deadlines vs General Leaves",
                         "Missed deadlines vs Working hours", "Missed deadlines vs Workload", "Workload vs General Leaves", "Duration vs General Leaves", "Personality Pie Chart",
                         "Personality & Meeting Pie Chart", "Partly – Job satisfaction"])

    if graph == "Missed Deadlines":
        k = df.sort_values(by=['Missed deadlines'], ascending=False)
        fig = px.bar(k, x='Performance', y='Missed deadlines',
                     animation_group='Performance', color='Performance', hover_name='Performance')
        fig.update_layout(title='Performance vs Missed deadlines')
        st.plotly_chart(fig)

    if graph == "Peer(s) workload":
        k = df.sort_values(by=['Peer(s) workload'], ascending=False)
        fig = px.bar(k, x='Performance', y='Peer(s) workload',
                     animation_group='Performance', color='Performance', hover_name='Performance')
        fig.update_layout(title='Performance vs Workload')
        st.plotly_chart(fig)

    if graph == "Time/Duration":
        k = df.sort_values(by=['Time/Duration'], ascending=False)
        fig = px.bar(k, x='Performance', y='Time/Duration',
                     animation_group='Performance', color='Performance', hover_name='Performance')
        fig.update_layout(title='Performance vs Working hours')
        st.plotly_chart(fig)

    if graph == "General leave patterns":
        k = df.sort_values(by=['General leave patterns'], ascending=False)
        fig = px.bar(k, x='Performance', y='General leave patterns',
                     animation_group='Performance', color='Performance', hover_name='Performance')
        fig.update_layout(title='Performance vs General Leaves')
        st.plotly_chart(fig)

    if graph == "Missed deadlines vs General Leaves":
        k = df.sort_values(by=['General leave patterns'], ascending=False)
        fig = px.bar(k, x='Missed deadlines', y='General leave patterns',
                     animation_group='Missed deadlines', color='Missed deadlines', hover_name='Missed deadlines')
        fig.update_layout(title='Missed deadlines vs General Leaves')
        st.plotly_chart(fig)

    if graph == "Missed deadlines vs Working hours":
        k = df.sort_values(by=['Time/Duration'], ascending=False)
        fig = px.bar(k, x='Missed deadlines', y='Time/Duration', animation_group='Missed deadlines',
                     color='Missed deadlines', hover_name='Missed deadlines')
        fig.update_layout(title='Missed deadlines vs Working hours')
        st.plotly_chart(fig)

    if graph == "Missed deadlines vs Workload":
        k = df.sort_values(by=['Peer(s) workload'], ascending=False)
        fig = px.bar(k, x='Missed deadlines', y='Peer(s) workload', animation_group='Missed deadlines',
                     color='Missed deadlines', hover_name='Missed deadlines')
        fig.update_layout(title='Missed deadlines vs Workload')
        st.plotly_chart(fig)

    if graph == "Workload vs General Leaves":
        k = df.sort_values(by=['General leave patterns'], ascending=False)
        fig = px.bar(k, x='Peer(s) workload', y='General leave patterns',
                     animation_group='Peer(s) workload', color='Peer(s) workload', hover_name='Peer(s) workload')
        fig.update_layout(title='Workload vs General Leaves')
        st.plotly_chart(fig)

    if graph == "Duration vs General Leaves":
        k = df.sort_values(by=['General leave patterns'], ascending=False)
        fig = px.bar(k, x='Time/Duration', y='General leave patterns',
                     animation_group='Time/Duration', color='Time/Duration', hover_name='Time/Duration')
        fig.update_layout(title='Duration vs General Leaves')
        st.plotly_chart(fig)

    if graph == "Personality Pie Chart":
        grouped = df.groupby('Personality style').sum().reset_index()
        grouped = grouped.sort_values(
            'Performance', ascending=False).reset_index()
        grouped.drop('index', axis=1, inplace=True)
        grouped = grouped.head(15)
        fig = px.pie(grouped, values="Performance",
                     names="Personality style", template="seaborn")
        fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
        st.plotly_chart(fig)

    if graph == "Personality & Meeting Pie Chart":
        grouped = df.groupby('Personality style').sum().reset_index()
        grouped = grouped.sort_values(
            'Number of meetings participated', ascending=False).reset_index()
        grouped.drop('index', axis=1, inplace=True)
        grouped = grouped.head(15)
        fig = px.pie(grouped, values="Number of meetings participated",
                     names="Personality style", template="seaborn")
        fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
        st.plotly_chart(fig)

    if graph == "Partly – Job satisfaction":
        grouped = df.groupby('Partly – Job satisfaction').sum().reset_index()
        grouped = grouped.sort_values(
            'Performance', ascending=False).reset_index()
        grouped.drop('index', axis=1, inplace=True)
        grouped = grouped.head(15)
        fig = px.pie(grouped, values="Performance",
                     names="Partly – Job satisfaction", template="seaborn")
        fig.update_traces(rotation=90, pull=0.05, textinfo="percent+label")
        st.plotly_chart(fig)

if nav == "Are you burntout?":
    st.header("Check whether you are burntout")
    eid = st.text_input("Employee ID")
    proid = st.text_input("Project ID")
    dur = st.slider("How many days will this project go on for?",
                    min_value=30, max_value=10)
    wld = st.slider("Rate your workload", min_value=0, max_value=10)
    prfrmc = st.slider("Rate your performance", min_value=0, max_value=10)
    msd = st.slider("How many deadlines have you missed?",
                    min_value=0, max_value=15)
    jobsat = st.selectbox("You are satisfied with your job", [
        'Storngly Agree', 'Agree', 'Disagree', 'Strongly Disagree'])
    unplanmeets = st.slider(
        "Number of unplanned meetings", min_value=3, max_value=15)
    meetorg = st.slider("Number of meetings organised",
                        min_value=0, max_value=8)
    meetpart = st.slider("Number of Meetings Participated",
                         min_value=0, max_value=15)
    avgmeet = st.slider("Average meetings", min_value=0, max_value=8)
    genleave = st.slider("Number of Day Offs", min_value=0)
    sickleave = st.slider("Number of Sick Days off", min_value=0)
    officetime = st.slider(
        "On an average how many hours do you spend time on the office laptop?", min_value=8, max_value=16)
    st.text("If you do not know what your personality type is, then please visit: https://www.16personalities.com/free-personality-test")
    personolity = st.selectbox('Your personality would be of what type?', [
                               'ESTJ', 'ENTJ', 'ESFJ', 'ENFJ', 'ISTJ', 'ISFJ', 'INTJ', 'INFJ', 'ESTP', 'ESFP', 'ENTP', 'ENFP', 'ISTP', 'ISFP', 'INTP', 'INFP'])
    burnoutrate = prfrmc + wld + dur + officetime
    burnoutcondition = 'Not burntout at all'

    pred = model.predict([[burnoutrate]])

    if (pred == 0):
        burnoutcondition = 'Not burntout at all'

    if (pred == 1):
        burnoutcondition = 'Burnout Triggered'

    if (pred == 2):
        burnoutcondition = 'Moderate Burnout'

    if (pred == 3):
        burnoutcondition = 'Total Burnout'

    if st.button("Predict"):
        st.text("Your burnout conditions is: ")
        st.text(burnoutcondition)
