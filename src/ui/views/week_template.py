import streamlit as st

import os
import importlib

from src.config.paths import WK_CONTENTS_PATH
from src.ui.views.week_content.metadata import WK_METADATA 


def render_week(week:str):
    '''Expects strings like 'week_01', 'week_08', etc.'''

    WEEK_PATH = WK_CONTENTS_PATH.joinpath(week)

    theme = WK_METADATA[week]['theme']
    tasks = WK_METADATA[week]['tasks']
    files = None
    if WEEK_PATH.exists():
        files = {
            int(f[5:-3]) - 1: WEEK_PATH.joinpath(f)
            for f in os.listdir(WEEK_PATH)
            if f[0:5] == 'task_'
        }

    st.markdown(f"<h1>{week[-2:]}: <span style='font-weight:normal'>{theme}</span></h1>", unsafe_allow_html=True)
    
    box = st.container(border=2, gap='xxsmall')
    with box:
        tabs = st.tabs([t['type'] for t in tasks])
        
        for i, t in enumerate(tasks):
            with tabs[i]:
                st.markdown(f"<h2>{t['type']}: <span style='font-weight:normal'>{t['topics']}</span></h2>", unsafe_allow_html=True)
                
                if not files or i not in files.keys():
                    st.text('\n')
                    st.text('\n')
                    st.text('Nothing here yet...', text_alignment='center', width='stretch')
                else:
                    module = importlib.import_module(f'src.ui.views.week_content.{files[i].parent.name}.{files[i].name[:-3]}')
                    if module and hasattr(module, 'render'):
                        st.caption(f"Loaded from '{files[i]}'")
                        module.render()
                    else:
                        st.error(f'{files[i]} does not have a render() function')