import streamlit as st

@st.fragment           
def render():
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('Hello World')

    st.text('''\
I started the day by looking for a YouTube tutorial because I wasn't really in the \
mood to start reading docs. I stumbled across this 1hr35m video which I ended up \
watching till completion as I found it to be really comprehensive and well-structured.\
'''
    )
    st.video('https://youtu.be/o8p7uQCGD0U?si=qqNhl8OK7NdJQUca')

    st.markdown('''\
In it, I learnt about how unlike React, Streamlit doesnt just reconfigure \
specific DOM elements but opts for a complete rerun of your scripts. I later learnt that you can emulate this behavior \
by using the ```@st.fragment``` decorater. Also, there were some other notable decorators for caching too, but they weren't \
very relevant to me right now. 
                
However what was very relevant was how multi-page applications work, which I learnt could be done in a couple of different ways.
1. **The Native Method:** Make a directory called pages/ and let streamlit handle everything
1. **The SPA Method:** Like react, emulate multiple pages by dynamically switching out DOM elements
'''
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('My Goal for Day 1')
    
    st.markdown('''
I realized that for my use-case, it was important to keep UI secondary and set up enough \
infrastructure for future days' UI integration to be a breeze.                
<div style='text-align:center; font-weight:bold'>Reusability was non-negotiable.</div>
<br/>               
I had landed on the idea of a reusable week page following a strict template where only the tab contents \
would need updation in the same way you are seeing right now.
''', unsafe_allow_html=True
    )

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('Imports and ```$env:PYTHONNAME```')
    st.text('Streamlit elements were working just fine, but when I tried to set up the file structure, I kept bumping into the same issue over and over again.')
    st.error(r'''File "E:\Coding\Projects\btyi\src\ui\main.py", line 3, in <module> from src.ui.views import render_home, render_week File "E:\Coding\Projects\btyi\src\ui\views\__init__.py", line 2, in <module> from .week import render_week''')
    
    st.markdown('''\
After a bit of debugging, I eventually learnt that the issue was pretty simple. \
Basically, streamlit's entry point acts as the default root for all imports, so any folder \
outside of the ``src/ui`` folder couldn't be imported.
            
All I had to do to fix it was explicitly declare the project root directory to be the root for all imports. So now, everytime the app is run, you need the following sequence:
'''
    )
    st.code('''\
$env:PYTHONPATH='.'
streamlit run src/ui/main.py
''', language='powershell')

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    st.subheader('The SPA Design')
    st.text('As for the actual building part of it, the architecture is fairly straightforward.')
    st.markdown('''\
1. The **entry point** defines the navigation drawer and selects between the home page and the week template page
1. The **template page** dynamically imports the appropriate content from the file system using *importlib*
1. Important **metadata** used by both the navigation menu and the week template page is kept in a SSoT at ```src.ui.views.week_content.metadata```.
                
So, all I need to do in the future is write my content in the respective ```week_WX/task_YZ.py``` file's ```render()``` function 
''')