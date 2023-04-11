# import basic libraries
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import io
from PIL import Image
import requests
from streamlit_lottie import st_lottie
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import xgboost as xgb

# image = Image.open('download.png')
# col1, col2, col3 = st.columns(3)
#
# with col1:
#     st.write(' ')
#
# with col2:
#     st.image(image)
#
# with col3:
#     st.write(' ')
# # st.image(image)


col1, col2 = st.columns(2)
url = requests.get(
    "https://assets4.lottiefiles.com/packages/lf20_49rdyysj.json")
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in URL")

with col1:
    st_lottie(url_json,
              # change the direction of our animation
              reverse=True,
              # height and width of animation
              height=400,
              width=400,
              # speed of animation
              speed=1,
              # means the animation will run forever like a gif, and not as a still image
              loop=True,
              # quality of elements used in the animation, other values are "low" and "medium"
              quality='high',
              # THis is just to uniquely identify the animation
              key='Home'
              )

with col2:
    st.markdown("<h1 style='text-align:right; color: olive;'>Leetcode Data Analysis"
                "</h1>""<h5 style='text-align: right; color: olive;'>Rohit Khandal -202011064 "
                "</h5>""<h5 style='text-align: right; color: olive;'>Vishnu Swaroop -202011037"
                "</h5>""<h5 style='text-align: right; color: olive;'>Ishant Bisen -202011028 "
                "</h5>""<h5 style='text-align: right; color: olive;'>Vivek Borole -202011018"
                "</h5>""<h5 style='text-align: right; color: olive;'>Gurupal Singh -202011022"
                "</h5>", unsafe_allow_html=True)

# st.markdown("<h4 style='text-align: center; color: olive;'>Rohit Khandal - 202011064 "
#             "</h4>", unsafe_allow_html=True)

st.text("")
st.text("")
st.text("")

st.markdown("***")
st.markdown("<h2 style='text-align: center; color: olive;'>Our first step "
            "</h2>", unsafe_allow_html=True)
st.markdown("***")
col1, col2 = st.columns(2)
url = requests.get(
    "https://assets7.lottiefiles.com/packages/lf20_mxuufmel.json")
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in URL")
with col1:
    st_lottie(url_json,
                  # change the direction of our animation
                  reverse=True,
                  # height and width of animation
                  height=300,
                  width=300,
                  # speed of animation
                  speed=1,
                  # means the animation will run forever like a gif, and not as a still image
                  loop=True,
                  # quality of elements used in the animation, other values are "low" and "medium"
                  quality='high',
                  # THis is just to uniquely identify the animation
                  key='data'
                  )
with col2:
    st.write("Kaggle leetcode username Dataset [link](https://www.kaggle.com/datasets/nidhaypancholi/leetcode-indian-user-ratings)")
    st.write("Our following Steps:")
    st.markdown("- Fetch Leetcode API")
    st.markdown("- [link](https://leetcode.com/graphql/)")
    st.markdown("- Collect Data")

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)
st.markdown("***")

df = pd.read_excel("Dataset.xlsx")
st.markdown("<h4 style='text-align: center; color: olive;'>Data "
            "</h4>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown("***")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.markdown("<h4 style='text-align: left; color: olive;'>Checking for Null values "
            "</h4>", unsafe_allow_html=True)
st.text(s)
st.markdown("***")
st.markdown("<h4 style='text-align: left; color: olive;'>Total Null Values "
            "</h4>", unsafe_allow_html=True)
st.text(df.isnull().sum())
st.markdown("***")
st.markdown("<h4 style='text-align: left; color: olive;'>Data Cleaning "
            "</h4>", unsafe_allow_html=True)
with st.echo():
    df['Badge'] = df['Badge'].fillna("No Badges")
    df = df[df['Rating'].notna()]
    df = df[df['Country'].notna()]
    df.drop(['SolutionCount', 'Name', 'TotalActiveDays'], axis=1, inplace=True)
    df['PostViewCount'] = df['PostViewCount'].fillna(0)
    df['Reputation'] = df['Reputation'].fillna(0)
    df['AttendedContestsCount'] = df['AttendedContestsCount'].fillna(0)
    df['Total'] = df['Total'].fillna(0)
    df['Hard'] = df['Hard'].fillna(0)
    df['Streak'] = df['Streak'].fillna(0)
    df['Easy'] = df['Easy'].fillna(0)
    df['Medium'] = df['Medium'].fillna(0)

st.markdown("***")
st.markdown("<h4 style='text-align: left; color: olive;'>Cleaned Data "
            "</h4>", unsafe_allow_html=True)

st.text(df.isnull().sum())

st.markdown("***")
# st.markdown("<h4 style='text-align: left; color: olive;'>Correlation Matrix "
#             "</h4>", unsafe_allow_html=True)
fig, ax = plt.subplots()
plt.figure(figsize=(20, 20))
corr_df = df.drop(['Username', 'Country', 'ActiveYears', 'Badge'], axis=1).corr(method='pearson')
fig = px.imshow(corr_df, color_continuous_scale='RdBu_r', origin='lower', text_auto=True, aspect="auto"
                , title="Correlation Matrix")
fig.update_layout(
    title=dict(font=dict(family='Times New Roman', size=30)),
    # xaxis_title='',
    # yaxis_title='Names',
    font=dict(family='Arial', size=18),
    height=800,
    width=1000,
    # margin=dict(l=100, r=100, t=100, b=100),
    # plot_bgcolor='#f0f0f0',
    # paper_bgcolor='#f0f0f0',
    # xaxis=dict(showgrid=True, gridwidth=1, gridcolor='white', tickfont=dict(size=14)),
    # yaxis=dict(showgrid=True, gridwidth=1, gridcolor='white', tickfont=dict(size=14))
)
st.write(fig)

st.markdown("***")

# fig, ax = plt.subplots()
sample_df = df.sample(int(0.04 * len(df)))
fig = plt.figure(figsize=(10, 6))
fig = px.scatter(
    sample_df, x='Total', y='Medium', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
    ,title="Total vs Medium Regression")
st.write(fig)
st.markdown("***")
fig = plt.figure(figsize=(10, 6))
fig = px.scatter(
    df, x='PostViewCount', y='Reputation', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
    ,title="PostViewCount vs Reputation Regression")
st.write(fig)

st.markdown("***")
st.markdown("<h2 style='text-align: center; color: olive;'>Distribution PLots "
            "</h2>", unsafe_allow_html=True)
total = [df['Total'].to_numpy()]
easy = [df['Easy'].to_numpy()]
medium = [df["Medium"].to_numpy()]
hard = [df["Hard"].to_numpy()]

group_labels = ["Total"]
fig = ff.create_distplot(total, ["Total"], bin_size=1.1,colors= ['Crimson'])
fig.update_layout(height=600, width=800)
st.write(fig)
fig = ff.create_distplot(easy, ["Easy"], bin_size=1.1,colors= ['olive'])
fig.update_layout(height=600, width=800)
st.write(fig)
fig = ff.create_distplot(medium, ["Medium"], bin_size=1.1,colors= ['azure'])
fig.update_layout(height=600, width=800)
st.write(fig)
fig = ff.create_distplot(hard, ["Hard"], bin_size=1.1,colors= ['darkkhaki'])
fig.update_layout(height=600, width=800)
st.write(fig)


st.markdown("***")
st.markdown("<h2 style='text-align: center; color: olive;'>Number of Badges "
            "</h2>", unsafe_allow_html=True)
host = df['Badge'].value_counts(ascending=False)
fig = px.bar(x=host, y=host.index, orientation='h', color=host.values,
             color_continuous_scale=px.colors.sequential.Viridis)
st.write(fig)




st.markdown("***")
# Data
group = df.groupby("Badge",as_index =False).mean()
class_1 = group["Total"].tolist()
class_2 = group["Easy"].tolist()
class_3 = group["Medium"].tolist()
class_4 = group["Hard"].tolist()

# Categories
categories = ['Guardian', 'Knight', 'No Badges']

# Create the traces
st.markdown("<h2 style='text-align: center; color: olive;'>Categorize number of Question "
            "</h2>", unsafe_allow_html=True)
trace1 = go.Bar(x=categories, y=class_1, name='Total')
trace2 = go.Bar(x=categories, y=class_2, name='Easy')
trace3 = go.Bar(x=categories, y=class_3, name='Medium')
trace4 = go.Bar(x=categories, y=class_4, name='Hard')
# Create the layout
layout = go.Layout(xaxis=dict(title='Category'), yaxis=dict(title='Value'))

# Create the figure
fig = go.Figure(data=[trace1, trace2, trace3,trace4], layout=layout)

# Show the figure
st.write(fig)

st.markdown("***")
st.markdown("<h2 style='text-align: center; color: olive;'>Pair Plot "
            "</h2>", unsafe_allow_html=True)
fig = plt.figure()
new_df = df.drop(["Sno","AttendedContestsCount","PostViewCount","GlobalRanking","ActiveYears","Country","PostViewCount","Reputation","Streak","TotalParticipants"],axis =1)
pairplot_figure = sns.pairplot(new_df, hue="Badge")
# pairplot_figure.fig.set_size_inches(9, 6.5)
st.pyplot(pairplot_figure.fig)
st.markdown("***")

st.markdown("<h2 style='text-align: left; color: olive;'>Predicting Rating "
            "</h2>", unsafe_allow_html=True)

model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

#Caching the model for faster loading
@st.cache
def predict(total, easy, medium, hard, reputation, streak, attended):
    prediction = model.predict(pd.DataFrame([[total,easy,medium,hard,reputation,streak,attended]], columns=['Total', 'Easy', 'Medium', 'Hard', 'Reputation', 'Streak', 'AttendedContestsCount']))
    return prediction


total = st.number_input('Total Questions :', min_value=0, max_value=2617, value=1)
easy = st.number_input('Easy Questions :', min_value=0, max_value=640, value=1)
medium = st.number_input('Medium Questions :', min_value=0, max_value=1392, value=1)
hard = st.number_input('Hard Questions :', min_value=0, max_value=585, value=1)
reputation = st.number_input('Total Reputation :', min_value=0, max_value=26798, value=1)
streak = st.number_input('Streak :', min_value=0, max_value=365, value=1)
attended = st.number_input('Total Contest Attended :', min_value=0, max_value=309, value=1)


if st.button('Predict Price'):
    price = predict(total, easy, medium, hard, reputation, streak, attended)
    st.success(f'The predicted value of the Rating is {price[0]:.2f}')
# time.sleep(20)
# st.balloons()
st.markdown("<h1 style='text-align: center; color: olive;'>!!  Thank You  !! "
            "</h1>", unsafe_allow_html=True)