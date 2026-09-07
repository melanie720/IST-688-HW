import streamlit as st

page1 = st.Page('HW1.py', title='Homework 1', icon=':material/description:')
page2 = st.Page('HW2.py', title='Homework 2', icon=':material/description:', default=True)

st.set_page_config(page_title='HomeworkApp', page_icon=':material/science:')
pg = st.navigation({'Assignments': [page1, page2]}, position='top')

st.sidebar.markdown("<h3 style='color:#8B0000; '>Summary Options</h3>", unsafe_allow_html=True)

st.sidebar.subheader(':material/translate: Language')
st.session_state['language_select'] = st.sidebar.selectbox(
    'Select language:',
    ('English', 'Spanish', 'French', 'Mandarin'),
    index = None
)

st.sidebar.subheader(':material/summarize: Type')
st.session_state['summary_type_select'] = st.sidebar.selectbox(
    'Select type of summary:',
    ('100 words', '2 connecting paragraphs', '5 bullet points'),
    index = None
)

st.sidebar.divider()

st.sidebar.subheader(':material/computer: Model Selection:')
st.session_state['llm_select'] = st.sidebar.selectbox(
    'Select your LLM:',
    ('OpenAI (default)', 'Gemini')
)

st.session_state['advanced'] = st.sidebar.checkbox('Use Advanced Model')

st.sidebar.write('')

# Setting LLM model based on user selection:
if st.session_state.llm_select == 'OpenAI (default)':
    if not st.session_state.advanced:
        st.sidebar.write("**Using OpenAI's GPT-5.4 Nano**")
        st.sidebar.caption('• Fast, concise, low-cost')
        st.session_state['model'] = "gpt-5.4-nano"
    else:
        st.sidebar.write("**Using OpenAI's GPT-5.6 Terra**")
        st.sidebar.caption('• Detailed, thorough, a little slower')
        st.session_state['model'] = "gpt-5.6-terra"
else:
    if not st.session_state.advanced:
        st.sidebar.write("**Using Google Gemini's 3.8 Flash**")
        st.sidebar.caption('• Fast, efficient, low-cost')
        st.session_state['model'] = "gemini-3.8-flash"
    else:
        st.sidebar.write("**Using Google Gemini's 3.1 Pro**")
        st.sidebar.caption('• Detailed, thorough, a little slower: Best for in-depth analysis')
        st.session_state['model'] = "gemini-3.1-pro-preview"


pg.run()