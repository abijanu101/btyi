import streamlit as st
from src.ui.views import render_home, render_week

from src.ui.views.week_content.metadata import WK_METADATA
import os

# SIDEBAR NAVIGATION

st.sidebar.title('🫥 this is *btyi*')
st.sidebar.text('better-than-your-internship')

weeks = list(WK_METADATA.keys())
selected_page = st.sidebar.radio('Pages', ['home'] + weeks)

# CONTENT
if 'week_' in selected_page:
    render_week(selected_page)
else:
    render_home()