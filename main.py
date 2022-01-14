import warnings
from datetime import timedelta
import random
import math
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet

cnf = '#393e46'
dth = '#ff2e63'
rec = '#21bf73'
act = '#fe9801'


@st.cache
def load(file):
    return pd.read_csv(file)


st.title('Corona data-analysis based on the future prediction of covid cases')
st.write('')
st.image('virus.jpg')
menu = ['Spread of Covid cases in India over a period',
        'Covid-19 spread in India vs other Countries', 'Age-wise corona spread in India', 'Syptoms caused to covid patients', 'Future prediction of corona cases']
choice = st.sidebar.selectbox('Menu', menu)


data1 = pd.read_csv('Cases_in_India.csv', parse_dates=['Date'])
data = pd.read_csv('states9.csv')
df = pd.read_csv('covid_19_data_cleaned.csv', parse_dates=['Date'])
country_daywise = pd.read_csv('country_daywise.csv', parse_dates=['Date'])
countywise = pd.read_csv('countrywise.csv')
daywise = pd.read_csv('daywise.csv', parse_dates=['Date'])
symptom = pd.read_csv('Symptom.csv')

confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()

if choice == 'Spread of Covid cases in India over a period':
    st.header('Spread of Covid cases in India over a period')
    st.dataframe(data1)
    st.write(" ")
    fig = sns.pairplot(data1)
    st.pyplot(fig)
    st.write(" ")
    fig1 = sns.relplot(x='Total Deceased', y='Total Confirmed',
                       hue='Total Recovered', data=data1)
    st.pyplot(fig1)
    st.write(" ")
    fig2 = sns.catplot(x='Total Confirmed', kind='box', data=data1)
    st.pyplot(fig2)
elif choice == "Covid-19 spread in India vs other Countries":
    st.header("Covid-19 spread in India vs other Countries")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=confirmed['Date'], y=confirmed['Confirmed'],
                             mode='lines+markers', name='Confirmed', line=dict(color="Orange", width=2)))
    fig.add_trace(go.Scatter(x=recovered['Date'], y=recovered['Recovered'],
                             mode='lines+markers', name='Recovered', line=dict(color="Green", width=2)))
    fig.add_trace(go.Scatter(x=deaths['Date'], y=deaths['Deaths'],
                             mode='lines+markers', name='Deaths', line=dict(color="Red", width=2)))
    fig.update_layout(title='Worldwide Covid-19 Cases',
                      xaxis_tickfont_size=14, yaxis=dict(title='Number of Cases'))
    fig.show()
    fig3 = px.choropleth(country_daywise, locations='Country', locationmode='country names', color=np.log(country_daywise['Confirmed']),
                         hover_name='Country', animation_frame=country_daywise['Date'].dt.strftime('%y-%m-%d'),
                         title='Cases over time', color_continuous_scale=px.colors.sequential.Inferno)
    fig3.show()
    top = 15

    fig_c = px.bar(countywise.sort_values('Confirmed').tail(top), x='Confirmed', y='Country',
                   text='Confirmed', orientation='h', color_discrete_sequence=[act])
    fig_d = px.bar(countywise.sort_values('Deaths').tail(top), x='Deaths', y='Country',
                   text='Deaths', orientation='h', color_discrete_sequence=[dth])

    fig_a = px.bar(countywise.sort_values('Active').tail(top), x='Active', y='Country',
                   text='Active', orientation='h', color_discrete_sequence=['#434343'])
    fig_r = px.bar(countywise.sort_values('Recovered').tail(top), x='Recovered', y='Country',
                   text='Recovered', orientation='h', color_discrete_sequence=[rec])

    fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.14,
                        vertical_spacing=0.1,
                        subplot_titles=('Confirmed Cases', 'Deaths Reported'))

    fig.add_trace(fig_c['data'][0], row=1, col=1)
    fig.add_trace(fig_d['data'][0], row=1, col=2)

    fig.add_trace(fig_r['data'][0], row=2, col=1)
    fig.add_trace(fig_a['data'][0], row=2, col=2)

    fig.update_layout(height=3000)
    fig.show()
    fig4 = px.bar(country_daywise, x='Date', y='Confirmed', color='Country', height=600,
                  title='Confirmed', color_discrete_sequence=px.colors.cyclical.mygbm)
    fig4.show()
elif choice == "Age-wise corona spread in India":
    st.header("Age-wise corona spread in India")
    data = pd.read_csv("state_wise.csv")
    data_1 = pd.read_csv("Book3.csv")
    m = data.groupby('State')['Active'].sum().sort_values(ascending=False)
    data.groupby('State')['Active'].sum().sort_values(ascending=False)
    data.groupby('State')['Active'].sum().drop('Total').drop(
        'State Unassigned').sort_values(ascending=False)
    d = data.groupby('State')['Active'].sum().drop('Total').drop(
        'State Unassigned').sort_values(ascending=False)
    d.plot.bar(figsize=(15, 5))
    plt.show(d)
    p = data.groupby('State')['Active'].sum().drop('Total').drop(
        'State Unassigned').sort_values(ascending=False)/403312*100
    p.plot.bar(figsize=(15, 5))
    plt.show()
    sns.scatterplot(x="State", y="Active", data=data)
    sns.scatterplot(data=data_1, x="Age Group", y="No. Cases",
                    hue="No. Deaths", palette="deep")
    l = data_1.drop([11])
    sns.scatterplot(data=l, x="Age Group", y="No. Cases",
                    hue="No. Deaths", size="No. Deaths", sizes=(100, 150))
    age_wise = data_1.groupby('Age Group')['No. Cases'].sum().drop(
        'Total').sort_values(ascending=False)
    age_wise.plot.bar(figsize=(15, 5))
    plt.show()
elif choice == "Syptoms caused to covid patients":
    st.header("Syptoms caused to covid patients")
    st.dataframe(symptom)
    symptom['Percentage'] = symptom['Percentage'].astype(int)
    fig7 = sns.barplot(x=symptom['Symptoms'], y=symptom['Percentage'],
                       data=symptom, palette="muted")
    plt.show(fig7)
elif choice == "Future prediction of corona cases":
    st.header("Future prediction of corona cases")
    st.dataframe(data)
    data.rename(columns={"Date": "ds", "Confirmed": "y"}, inplace=True)
    model = NeuralProphet(growth="linear",changepoints=None,n_changepoints=5,changepoints_range=0.8,trend_reg=0,trend_reg_threshold=False,yearly_seasonality="auto",weekly_seasonality="auto",daily_seasonality="auto",seasonality_mode="additive",seasonality_reg=0,n_forecasts=1,n_lags=0,num_hidden_layers=0,d_hidden=None,ar_sparsity=None,learning_rate=None,epochs=40,loss_func="Huber",normalize="auto",impute_missing=True)
    metrics = model.fit(data, freq="D")
    future = model.make_future_dataframe(data, periods=365, n_historic_predictions=len(data))
    forecast = model.predict(future)
    st.write("SELECT FORECAST PERIOD")
    periods_input = st.number_input('How many days forecast do you want?',
                                    min_value=1, max_value=10000)
    if st.button('Forecast'):
        future = model.make_future_dataframe(data, periods=periods_input)
        forecast = model.predict(future)
        plotting = model.plot(forecast)
        plotting2 = model.plot_components(forecast)
        st.write(plotting)
        st.write(plotting2)