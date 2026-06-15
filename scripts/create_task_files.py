import os
import pathlib

from src.config.paths import WK_CONTENTS_PATH
from src.ui.views.week_content.metadata import WK_METADATA

for i in range(1,11):
    week = f'week_{i:02d}'
    week_path = WK_CONTENTS_PATH.joinpath(week)
    if not week_path.exists():
        os.mkdir(week_path)
    
    for t in range(1, len(WK_METADATA[week]['tasks']) + 1):
        task = f'task_{t:02d}'
        task_path = week_path.joinpath(task + '.py')
        if not task_path.exists():
            with open(task_path, 'w+') as f:
                f.write(f'''\
import streamlit as st

@st.fragment           
def render():
    st.text("{week}/{task}")

''')
        pass

WK_CONTENTS_PATH
