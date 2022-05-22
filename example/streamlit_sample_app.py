# import streamlit as st
# display = ("male", "female")
# options = list(range(len(display)))
# value = st.selectbox("gender", options, format_func=lambda x: display[x])
# st.write(value)

import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache
def load_data(nrows):
    def lowercase(x):
        return str(x).lower()

    webdata = pd.read_csv(DATA_URL, nrows=nrows)
    webdata.rename(lowercase, axis='columns', inplace=True)
    webdata[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return webdata


data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
