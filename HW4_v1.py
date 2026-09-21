__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pymupdf, chromadb, os, re, streamlit as st
from openai import OpenAI, AuthenticationError
from pathlib import Path
from bs4 import BeautifulSoup

# Retrieve API key and create an OpenAI client.
if "hw4_client" not in st.session_state:
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.hw4_client = OpenAI(api_key=openai_api_key)

# Checking if API key is valid.
try:
    st.session_state.hw4_client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, "Homework-04-Data")

def extract_text_from_html(file_path):
    chunks = []

    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Get text with newlines as separators
        text = soup.get_text(separator="\n")
        
        # Replace all newlines (and carriage returns) with a space
        text = re.sub(r'\r?\n', ' ', text)
        
        # Collapse multiple spaces into a single space and strip leading/trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        chunk_size = len(text) // 2

        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        
    return chunks

def add_to_collection(collection, folder_name):
    client = st.session_state.hw4_client
    existing = set(collection.get()['ids'])

    for file in os.listdir(folder_name):
        if not file.lower().endswith('.html') or file in existing:
            continue

        text = extract_text_from_html(os.path.join(folder_name, file))

        for chunk in text:
            response = client.embeddings.create(
                input=chunk,
                model='text-embedding-3-small'
            )

            collection.add(
                documents=[chunk],
                ids=[file + "_" + str(text.index(chunk))],
                embeddings=[response.data[0].embedding],
                metadatas=[{"filename": file}]
            )

if 'HW4_VectorDB' not in st.session_state:
    st.session_state["HW4_VectorDB"] = chromadb.PersistentClient(path = "./ChromaDB_for_Homework")
    collection = st.session_state.HW4_VectorDB.get_or_create_collection('HW4Collection')
    add_to_collection(collection, target_path)
else:
    collection = st.session_state.HW4_VectorDB.get_or_create_collection('HW4Collection')

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
            <li>Ask a question about student orgs. or anything.</li>
            <li>Hit <em>'Yes'</em> for more detail when prompted. <strong>Each call keeps the last 5 messages as memory.</strong></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Show title and description.
st.title(":material/description: Mel's Chatbot using Chunky RAG")
st.caption("Let's chat!")

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
        "Below are excerpts from a document collection. "
        "If the excerpts are relevant to the user's question, start response by " 
        "saying exactly: 'I'm using knowledge from the RAG.' Then, use the context to answer the question." 
        "If the excerpts are not relevant to the user's question, start response by " 
        "saying exactly: 'I'm answering from my general knowledge, not the RAG.'"
    )
}
 
# More info prompt
more_info_prompt = {
    "role": "system",
    "content": (
        "User would like more information on the topic just explained. Give new details and examples that build on what you "
        "already said. Don't ask any questions at the end."
    )
}
 
buffer = 6

# If "messages" is not already in session state, we initialize it here with a message from assistant.
# Setting other state variables.
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [system_prompt, {"role": "assistant", "content": "How can I help you?"}]
if "hw4_awaiting_choice" not in st.session_state:
    st.session_state.hw4_awaiting_choice = False
if "hw4_pending" not in st.session_state:
    st.session_state.hw4_pending = None  # None, answer, or more_info

# For each message in st.session_state.messages, we display each message to the user.
chat_box = st.container(border=True, height=300)

with chat_box:
    for message in st.session_state.hw4_messages:
        if message["role"] == "system":
            continue
        chat_message = st.chat_message(message["role"])
        chat_message.write(message["content"])
 
# Get user input.
# Assign the user's input to prompt.
prompt = st.chat_input(
    "Say 'hey' or ask a question.",
    disabled=st.session_state.hw4_awaiting_choice # Not currently awaiting Yes/No choice, so input field is not disabled.
)

if "hw4_context" not in st.session_state:
    st.session_state.hw4_context = []

# If user provides a prompt, append it to messages and we are now waiting for a regular answer.
if prompt:
    client = st.session_state.hw4_client
    response = client.embeddings.create(
        input=prompt,
        model='text-embedding-3-small'
    )

    # Get the embedding
    query_embedding = response.data[0].embedding

    # Get the text related to this question (this prompt)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10  # The number of closest documents to return
    )

    # Replacing the context each turn.
    st.session_state.hw4_context = [
        {"role": "system", "content": doc}
        for doc in results['documents'][0]
    ]

    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    st.session_state.hw4_pending = "answer"

    st.rerun()

# If we are awaiting a Yes/No choice, show the buttons.
if st.session_state.hw4_awaiting_choice:
    with st.container(horizontal=True):
        # If user chooses Yes, append 'Yes' to messages, change awaiting choice back to False and we are now waiting for a "more_info" answer.
        if st.button("Yes", type="primary", key="hw4_yes_button"):
            st.session_state.hw4_messages.append({"role": "user", "content": "Yes"})
            st.session_state.hw4_awaiting_choice = False
            st.session_state.hw4_pending = "more_info"
            st.rerun()
        # If user chooses No, append 'No' to messages, change awaiting choice back to False and we go back to the initial 'help' message from the bot.
        if st.button("No", type="primary", key="hw4_no_button"):
            st.session_state.hw4_messages.append({"role": "user", "content": "No"})
            # Go back to asking what the bot can help with.
            st.session_state.hw4_messages.append(
                {"role": "assistant", "content": "What else can I help you with?"}
            )
            st.session_state.hw4_awaiting_choice = False
            st.rerun()
 
# Generate a response, either a normal answer or a "more info".
if st.session_state.hw4_pending:

    # Creating the prompt based on buffer and system prompt.
    # System prompt is pinned to the top.
    # The buffer is applied to messages after the system prompt.
    api_messages = (
        st.session_state.hw4_messages[:1] + st.session_state.hw4_messages[1:][-buffer:] + st.session_state.hw4_context
    )
 
    # On a "more info" answer, add the extra instruction at the end of the request.
    if st.session_state.hw4_pending == "more_info":
        api_messages = api_messages + [more_info_prompt]
 
    stream = st.session_state.hw4_client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=api_messages,
        stream=True,
    )

    with chat_box:
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
 
    # Store the answer and ask the follow-up question.
    st.session_state.hw4_messages.append({"role": "assistant", "content": response})
    st.session_state.hw4_messages.append(
        {"role": "assistant", "content": "Do you want more info?"}
    )
 
    st.session_state.hw4_pending = None
    st.session_state.hw4_awaiting_choice = True
    st.rerun()