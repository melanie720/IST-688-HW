import requests, streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
from google import genai

st.markdown(
    """
    <div style="
        background-color: #FFFBE6;
        border: 1px solid #FFE58F;
        border-left: 5px solid #FAAD14;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 24px;
        color: #262730;
    ">
        <div style="font-weight: 600; font-size: 1.05rem; color: #8C5300; margin-bottom: 6px;">
            Chatbot Details
        </div>
        <ul style="margin: 0; padding-left: 20px; font-size: 0.9rem; line-height: 1.6;">
            <li>Add 1 or 2 URLs in the sidebar, pick ChatGPT or Gemini, and ask anything.</li>
            <li>Hit <em>'Yes'</em> for more detail when prompted. <strong>Each call keeps the last 6 messages as memory.</strong></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Show title and description.
st.title(":material/web: Mel's Webpage Chatbot")
st.caption("Let's chat about your webpage!")

# Create client based on user's LLM selection - OpenAI or Google Gemini.
if st.session_state.model == 'gpt-6-astra':
    client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
else:
    client = genai.Client(api_key=st.secrets.GEMINI_API_KEY)

# Always overwrite, so switching models swaps the client too.
st.session_state.hw3_client = client

# Checking if API key is valid.
try:
    client.models.list()
except Exception:
    st.error("API key needs to be updated.")
    st.stop()

# Function for reading URLs. Cached so we don't refetch on every rerun.
@st.cache_data(show_spinner=False)
def read_url_content(url):
    if not url:
        return 'No context found. Just answer the question based on your knowledge.'
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException:
        return 'No context found. Just answer the question based on your knowledge.'

# Hiding the "Press Enter to submit" caption in my input field
st.html("""
    <style>
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
""")

# System prompt
system_prompt = {
    "role": "system",
    "content": (
        "Use the content from these URLs to answer the user's questions. Provide clear, concise, and accurate information. "
        f"First URL content: {read_url_content(st.session_state.get('url1', ''))}; Second URL content: {read_url_content(st.session_state.get('url2', ''))}. "
        "Always respond normally first to whatever the user writes, like if it's not an actually question about the provided URLs, then you can nicely ask what they would like to know."
        f"Introduce yourself first when responding. Just state exactly: 'Hi, I'm {st.session_state.model_label}.', then proceed to answer the user's question.'"
    )
}

# More info prompt
more_info_prompt = {
    "role": "system",
    "content": (
        "User would like more information on the topic just explained. Give new details and examples that build on what you "
        "already said. Keep it simple enough for a 10 year old. "
        "Don't ask any questions at the end."
    )
}

buffer = 6

# If "messages" is not already in session state, we initialize it here with a message from assistant.
# Setting other state variables.
if "hw3_messages" not in st.session_state:
    st.session_state.hw3_messages = [system_prompt, {"role": "assistant", "content": "How can I help you?"}]
if "hw3_awaiting_choice" not in st.session_state:
    st.session_state.hw3_awaiting_choice = False
if "hw3_pending" not in st.session_state:
    st.session_state.hw3_pending = None  # None, answer, or more_info

# For each message in st.session_state.messages, we display each message to the user.
chat_box = st.container(border=True, height=300)

with chat_box:
    for message in st.session_state.hw3_messages:
        if message["role"] == "system":
            continue
        chat_message = st.chat_message(message["role"])
        chat_message.write(message["content"])

# Get user input.
# Assign the user's input to prompt.
prompt = st.chat_input("Say 'hey' or ask a question.")

# If user provides a prompt, append it to messages and we are now waiting for a regular answer.
# Clearing awaiting_choice hides the Yes/No buttons if they were showing.
if prompt:
    st.session_state.hw3_messages.append({"role": "user", "content": prompt})
    st.session_state.hw3_awaiting_choice = False
    st.session_state.hw3_pending = "answer"
    st.rerun()

# If we are awaiting a Yes/No choice, show the buttons.
# "ui_only" messages display in the chat but are kept out of the model's context.
if st.session_state.hw3_awaiting_choice:
    with st.container(horizontal=True):
        # If user chooses Yes, append 'Yes' to messages, change awaiting choice back to False and we are now waiting for a "more_info" answer.
        if st.button("Yes", type="primary", key="hw3_yes_button"):
            st.session_state.hw3_messages.append({"role": "user", "content": "Yes", "ui_only": True})
            st.session_state.hw3_awaiting_choice = False
            st.session_state.hw3_pending = "more_info"
            st.rerun()
        # If user chooses No, append 'No' to messages, change awaiting choice back to False and we go back to the initial 'help' message from the bot.
        if st.button("No", type="primary", key="hw3_no_button"):
            st.session_state.hw3_messages.append({"role": "user", "content": "No", "ui_only": True})
            # Go back to asking what the bot can help with.
            st.session_state.hw3_messages.append(
                {"role": "assistant", "content": "What else can I help you with?", "ui_only": True}
            )
            st.session_state.hw3_awaiting_choice = False
            st.rerun()

# Generate a response, either a normal answer or a "more info".
if st.session_state.hw3_pending:

    # System prompt is pinned separately; the buffer applies to the real conversation,
    # skipping any ui_only filler.
    history = [m for m in st.session_state.hw3_messages[1:] if not m.get("ui_only")][-buffer:]

    if st.session_state.model == 'gpt-6-astra':
        messages = [{"role": "system", "content": system_prompt["content"]}]
        messages += [{"role": m["role"], "content": m["content"]} for m in history]

        # On a "more info" answer, add the extra instruction at the end of the request.
        if st.session_state.hw3_pending == "more_info":
            messages.append(more_info_prompt)

        stream = st.session_state.hw3_client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            stream=True,
        )

    else:
        # Interactions API: history goes in as user_input steps, system prompt is its own argument.
        input_steps = [
            {"type": "user_input", "content": [{"type": "text", "text": f"{'User' if m['role'] == 'user' else 'Your previous reply'}: {m['content']}"}]}
            for m in history
        ]

        if st.session_state.hw3_pending == "more_info":
            input_steps.append(
                {"type": "user_input", "content": [{"type": "text", "text": more_info_prompt["content"]}]}
            )

        interaction = st.session_state.hw3_client.interactions.create(
            model=st.session_state.model,
            system_instruction=system_prompt["content"],
            input=input_steps,
            store=False,
        )

        stream = [interaction.output_text]

    with chat_box:
        with st.chat_message("assistant"):
            response = st.write_stream(stream)

    # Store the answer and ask the follow-up question.
    st.session_state.hw3_messages.append({"role": "assistant", "content": response})
    st.session_state.hw3_messages.append(
        {"role": "assistant", "content": "Do you want more info?", "ui_only": True}
    )

    st.session_state.hw3_pending = None
    st.session_state.hw3_awaiting_choice = True
    st.rerun()