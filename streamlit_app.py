import streamlit as st

# Page Configurations
st.set_page_config(
    page_title='HomeworkApp', 
    page_icon=':material/science:', 
    layout="centered",
    menu_items={
        'About': "This is my URL summarizer app!"
    }
)

page1 = st.Page('HW1.py', title='Homework 1', icon=':material/description:')
page2 = st.Page('HW2.py', title='Homework 2', icon=':material/description:')
page3 = st.Page('HW3.py', title='Homework 3', icon=':material/description:', default=True)

pg = st.navigation([page1, page2, page3], position='top')

# CSS for Containers
css = '''
    <style>
        .st-key-summary_container, .st-key-llm_container,
        .st-key-options_container, .st-key-model_container {
            background-color: #ffffff;
            border-radius: 12px;
            border: 2px solid #e0e0e0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        .st-key-summary_container .stSelectbox > div > div,
        .st-key-summary_container .stTextInput > div > div,
        .st-key-llm_container .stSelectbox > div > div,
        .st-key-options_container .stSelectbox > div > div,
        .st-key-model_container .stSelectbox > div > div {
            border: 2px solid #e0e0e0 !important;
            border-radius: 8px !important;
        }

        .st-key-url_heading p,
        .st-key-llm_heading p,
        .st-key-options_heading p,
        .st-key-model_heading p {
            font-size: 16px;
            font-weight: 600;
            line-height: 1.4;
            color: var(--text-color);
            margin-bottom: 0;
        }

        .st-key-url_heading [data-testid="stIconMaterial"],
        .st-key-llm_heading [data-testid="stIconMaterial"],
        .st-key-options_heading [data-testid="stIconMaterial"],
        .st-key-model_heading [data-testid="stIconMaterial"] {
            font-size: 18px;
            vertical-align: -3px;
        }

        .st-key-url_heading,
        .st-key-llm_heading,
        .st-key-options_heading,
        .st-key-model_heading {
            margin-bottom: -0.5rem;
        }

        .st-key-submit_status p {
            font-size: 14px;
            font-weight: 600;
            color: #2e9e5b;
            margin-bottom: 0;
        }

        .st-key-submit_status [data-testid="stIconMaterial"] {
            font-size: 16px;
            vertical-align: -3px;
            color: #2e9e5b;
        }

        .st-key-reset_row,
        .st-key-reset_row > div,
        .st-key-reset_row [data-testid="stElementContainer"] {
            display: flex;
            justify-content: flex-end;
            width: 100%;
        }

        .st-key-reset_row {
            margin-top: -0.5rem;
        }

        .st-key-reset_row button {
            font-size: 13px;
            padding: 0.15rem 0.4rem;
            width: auto;
        }
    </style>
'''

# Initialize session states
defaults = {
    "submitted": False,
    "url1_input": "",
    "url2_input": "",
    "url1": None,
    "url2": None,
    "model": "gpt-6-astra",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Green state for submitted fields
if st.session_state.submitted:
    css += '''
        <style>
            .st-key-summary_container .stTextInput > div > div {
                border: 2px solid #2e9e5b !important;
                background-color: #f2fbf5 !important;
            }
        </style>
    '''

st.html(css)

# On-change/callback functions for reset and submit buttons
def reset_fields():
    st.session_state.url1_input = ""
    st.session_state.url2_input = ""
    st.session_state.submitted = False

def mark_submitted():
    if st.session_state.url1_input:
        st.session_state.submitted = True

# Sidebar Title
st.sidebar.header(":material/settings: Summary Options")

# Container 1: Summary Settings
with st.sidebar.container(border=True, height="content", key="summary_container"):
    st.write("")

    with st.container(key="url_heading"):
        st.write(":material/link: Enter up to 2 URLs:")

    st.write("")

    url1 = st.text_input(
        "URL #1",
        key="url1_input",
        placeholder="https://example.com",
        help="The page you want summarized."
    )

    url2 = st.text_input(
        "URL #2 — optional",
        key="url2_input",
        placeholder="https://example.com"
    )

    st.write("")

    st.button("Submit", width="stretch", type="primary", on_click=mark_submitted)

    if st.session_state.submitted:
        with st.container(key="submit_status"):
            st.write(":material/check_circle: Submitted")
        st.session_state.url1 = st.session_state.url1_input or None
        st.session_state.url2 = st.session_state.url2_input or None
    else:
        st.session_state.url1 = None
        st.session_state.url2 = None

    with st.container(key="reset_row"):
        st.button(
            "Reset",
            icon=":material/refresh:",
            type="tertiary",
            help="Clear",
            on_click=reset_fields,
        )

# Container 2: LLM Settings
with st.sidebar.container(border = True, height = "content", key = "llm_container"):
    st.write("")

    with st.container(key="llm_heading"):
        st.write(":material/computer: Select your LLM:")

    st.write("")

    # Model IDs
    model_labels = {
        "gpt-6-astra": "GPT-6 Astra",
        "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    }

    def format_model(model_id):
        return model_labels[model_id]

    # key = "model" binds the selection to st.session_state.model
    st.radio(
        "Select your LLM",
        options=list(model_labels),
        format_func=format_model,
        key="model",
        label_visibility="collapsed",
    )

    st.session_state.model_label = format_model(st.session_state.model)

    # Top container slot for the messages
    info_container = st.container()

    st.write("")

    # Populating the top container with updated values
    with info_container:
        if st.session_state.model == "gpt-6-astra":
                st.markdown(
                    '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using OpenAI\'s GPT-6 Astra</p>',
                    unsafe_allow_html=True,
                )
                st.caption('_• Nuanced, deepest reasoning_')
        else:
                st.markdown(
                    '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using Google Gemini\'s 3.1 Pro</p>',
                    unsafe_allow_html=True,
                )
                st.caption('_• Fast summaries at low cost_')
                st.session_state.model_label = format_model(st.session_state.model)

# Container 3: Output Settings
with st.sidebar.container(border=True, height="content", key="options_container"):
    st.write("") 

    with st.container(key="options_heading"):
        st.write(":material/tune: Output settings:")

    st.write("")

    st.session_state.language_select = st.selectbox(
        ':material/translate: Select language:',
        ('English', 'Spanish', 'French', 'Mandarin'),
        index=None
    )

    st.write("")

    st.session_state.summary_type_select = st.selectbox(
        ':material/summarize: Select type of summary:',
        ('100 words', '2 connecting paragraphs', '5 bullet points'),
        index=None
    )

    st.write("")

# Container 4: Legacy Model Selection
with st.sidebar.container(border=True, height="content", key="model_container"):
    st.write("") 

    with st.container(key="model_heading"):
        st.write(":material/tune: Model selection:")

    st.write("")

    st.session_state.llm_select = st.selectbox(
        ':material/computer: Select your LLM:',
        ('OpenAI (default)', 'Gemini')
    )

    st.session_state.advanced = st.checkbox('Use Advanced Model')

    # Setting LLM model based on user selection:
    if st.session_state.llm_select == 'OpenAI (default)':
        if not st.session_state.advanced:
            st.markdown(
                '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using OpenAI\'s GPT-5.4 Nano</p>',
                unsafe_allow_html=True,
            )
            st.caption('_• Fast, concise, low-cost_')
            st.session_state.legacy_model = "gpt-5.4-nano"
        else:
            st.markdown(
                '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using OpenAI\'s GPT-5.6 Terra</p>',
                unsafe_allow_html=True,
            )
            st.caption('_• Detailed, thorough, a little slower_')
            st.session_state.legacy_model = "gpt-5.6-terra"
    else:
        if not st.session_state.advanced:
            st.markdown(
                '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using Google Gemini\'s 3.8 Flash</p>',
                unsafe_allow_html=True,
            )
            st.caption('_• Fast, efficient, low-cost_')
            st.session_state.legacy_model = "gemini-3.8-flash"
        else:
            st.markdown(
                '<p style="font-size: 14px; font-weight: bold; color: black; margin-bottom: 0px;">Using Google Gemini\'s 3.1 Pro</p>',
                unsafe_allow_html=True,
            )
            st.caption('_• Detailed, thorough: best for in-depth analysis_')
            st.session_state.legacy_model = "gemini-3.1-pro-preview"

    st.write("")

pg.run()