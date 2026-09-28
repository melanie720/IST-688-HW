# ============================================================================
# HOW THIS APP WORKS
# 1. Setup: connect to OpenAI and load the club HTML pages into a vector database.
# 2. The user asks a question.
# 3. The LLM decides if it needs club info. If so, it asks to run the search tool.
# 4. The app runs the search and sends the results back to the LLM.
# 5. The LLM writes the answer, which is shown and saved to the chat history.
# ============================================================================

# Swap in a newer SQLite, which ChromaDB needs; Must run before importing chromadb.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pymupdf, chromadb, os, re, tiktoken, streamlit as st, json
from openai import OpenAI, AuthenticationError
from pathlib import Path
from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

# Create the OpenAI client once and save it so it survives page reruns.
if "hw4_client" not in st.session_state:
    openai_api_key = st.secrets.OPENAI_API_KEY
    st.session_state.hw4_client = OpenAI(api_key=openai_api_key)

# Make sure the API key works before going any further.
try:
    st.session_state.hw4_client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

# Folder holding the club HTML pages.
script_dir = os.path.dirname(os.path.abspath(__file__))
target_path = os.path.join(script_dir, "Homework-04-Data")

# Tokenizer used to measure text length the same way the embedding model does.
encoding = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# CHUNKING: split each HTML page into two halves
# ---------------------------------------------------------------------------

# ============================================================================
# CHUNKING METHOD: Fixed-size chunking, token-based split each file into 2 documents, cutting at newlines.
# Each HTML page becomes exactly two mini-documents.
#
# Splitting by tokens seemed like the best way to make 2 documents out of 1.
# Tried splitting by character at first, but that broke up words and complete thoughts.
#
# Cutting only at newlines so a sentence, an org name, or a meeting time is never sliced in half.
#
# No third chunk due to leftover tokens. Because the cut lands on a line boundary
# instead of an exact token index, an odd token just stays inside whatever
# line it belongs to and rides along in that half. With no remainder to
# place, exactly two chunks come out every time.
# ============================================================================

def extract_text_from_html(file_path):
    # Read the page and strip out the HTML tags.
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file.read(), "html.parser")

    # Put each page element on its own line and clean up extra spaces.
    text = soup.get_text(separator="\n")
    text = text.replace("\r", "\n")
    text = re.sub(r'[ \t]+', ' ', text)

    # Make a list of the non-empty lines.
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # If the page is one long line, split it into sentences instead.
    if len(lines) < 2:
        lines = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    # Too short to split, so keep it as one chunk.
    if len(lines) < 2:
        return [" ".join(lines)] if lines else []

    # Add up tokens line by line until reaching the halfway point.
    counts = [len(encoding.encode(line)) for line in lines]
    half = sum(counts) / 2
    running = 0
    cut = len(lines) - 1

    for i, count in enumerate(counts):
        running += count
        if running >= half:
            cut = i + 1
            break

    # Make sure the second half always has at least one line.
    cut = min(cut, len(lines) - 1)

    # Return the two halves.
    return [" ".join(lines[:cut]), " ".join(lines[cut:])]


# ---------------------------------------------------------------------------
# VECTOR DATABASE: store the chunks so they can be searched
# ---------------------------------------------------------------------------

def add_to_collection(collection, folder_name):
    client = st.session_state.hw4_client

    # IDs already in the database, so pages aren't added twice.
    existing = set(collection.get()['ids'])

    for file in os.listdir(folder_name):
        # Only look at HTML files.
        if not file.lower().endswith('.html'):
            continue

        # Skip pages that are already stored.
        if any(chunk_id.startswith(file + "_") for chunk_id in existing):
            continue

        chunks = extract_text_from_html(os.path.join(folder_name, file))

        # Turn each chunk into an embedding (a list of numbers) and save it.
        for index, chunk in enumerate(chunks):
            response = client.embeddings.create(
                input=chunk,
                model='text-embedding-3-small'
            )

            collection.add(
                documents=[chunk],
                ids=[file + "_" + str(index)],
                embeddings=[response.data[0].embedding],
                metadatas=[{"filename": file}]
            )

# Open the database once per session and load any new pages.
# The database is saved to disk, so pages stored in earlier runs are kept.
if 'HW4_VectorDB' not in st.session_state:
    st.session_state["HW4_VectorDB"] = chromadb.PersistentClient(path = "./ChromaDB_for_Homework")
    collection = st.session_state.HW4_VectorDB.get_or_create_collection('HW4Collection')
    add_to_collection(collection, target_path)
else:
    collection = st.session_state.HW4_VectorDB.get_or_create_collection('HW4Collection')


# ---------------------------------------------------------------------------
# THE TOOL: search the database for club info
# ---------------------------------------------------------------------------

# The LLM can ask the app to run this function when it needs club info.
def relevant_club_info(query):
    client = st.session_state.hw4_client

    # Turn the search query into an embedding.
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    # The search ranks each half-page on its own, not whole pages.
    # So get extra chunks, to have enough different pages to pick from.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20
    )

    # Keep the first 6 different pages, in order of how well they matched.
    sources = list(dict.fromkeys(m["filename"] for m in results['metadatas'][0]))[:6]

    # Fetch both halves of each chosen page by ID.
    # Short pages only have one half, and missing IDs are just skipped.
    ids = [f"{file}_{i}" for file in sources for i in (0, 1)]
    pages = collection.get(ids=ids)

    # Put the halves back in order, so each page reads first half, then second half.
    text_by_id = dict(zip(pages['ids'], pages['documents']))
    documents = [text_by_id[chunk_id] for chunk_id in ids if chunk_id in text_by_id]

    # The text goes to the LLM. The count and page names are shown to the user.
    return "\n\n".join(documents), len(documents), sources

# Describes the tool to the LLM. The LLM only sees this description, not the function code.
# The query description asks for a complete search phrase, so follow-ups like
# "Where is it held?" get searched with the club's name instead of "it".
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": "Use only when the user's request pertains to student organizations or clubs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A standalone search query about student organizations. "
                            "Resolve references like 'it' or 'that club' using the conversation, "
                            "e.g. 'Where is it held?' after discussing chess club becomes "
                            "'chess club meeting location'."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# PAGE LAYOUT
# ---------------------------------------------------------------------------

# Yellow info box at the top of the page.
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
            <li><strong>Each call keeps the last 5 messages as memory.</strong></li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# Title and caption.
st.title(":material/description: Mel's Chatbot using Tools")
st.caption("Let's chat!")

# Hide the "Press Enter to submit" hint under the input box.
st.html("""
    <style>
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
""")


# ---------------------------------------------------------------------------
# CHAT SETUP
# ---------------------------------------------------------------------------

# Instructions for the LLM. Always sent first, never shown to the user.
system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful chatbot that can provide information on student organizations. "
        "If the user's request depends on student organization information, call the relevant_club_info tool. "
        "You will receive excerpts from a document collection. "
        "If the excerpts are relevant to the user's question, start response by " 
        "saying exactly: 'I'm using knowledge from the RAG.' Then, use the context to answer the question. " 
        "If you did not call the tool, or the excerpts are not relevant to the user's question, start response by " 
        "saying exactly: 'I'm answering from my general knowledge, not the RAG.' "
        "Also, never mention the RAG excerpts to the user. Be professional."
    )
}

# How many recent messages the LLM gets as memory.
buffer = 5

# Start the chat history with the system prompt and a greeting.
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [system_prompt, {"role": "assistant", "content": "How can I help you?"}]

# Flag that says an answer still needs to be written.
if "hw4_pending" not in st.session_state:
    st.session_state.hw4_pending = None

# Shows a collapsed box with what the search did (query, excerpt count, pages).
def show_tool_log(tool_log):
    with st.expander("🔎 Searched student organization info"):
        for entry in tool_log:
            st.write(f"**Query:** *{entry['query']}*")
            if entry["error"]:
                st.write("⚠️ The search failed.")
            else:
                st.write(f"📄 Found {len(entry['sources'])} pages ({entry['count']} excerpts)")
                if entry["sources"]:
                    st.caption("Sources: " + ", ".join(f"`{s}`" for s in entry["sources"]))


# ---------------------------------------------------------------------------
# SHOW THE CHAT
# ---------------------------------------------------------------------------

# Scrollable box that holds the conversation.
chat_box = st.container(border=True, height=300, key="hw4_chat_box")

# Redraw every saved message, since Streamlit rebuilds the page on each rerun.
with chat_box:
    for message in st.session_state.hw4_messages:
        # Don't show the system prompt.
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            # If this answer used the search, show the search box above it.
            if message.get("tool_log"):
                show_tool_log(message["tool_log"])
            st.write(message["content"])


# ---------------------------------------------------------------------------
# GET THE USER'S QUESTION
# ---------------------------------------------------------------------------

prompt = st.chat_input("Say 'hey' or ask a question.")

# Save the question, flag that an answer is needed, and rerun so the question shows right away.
if prompt:
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    st.session_state.hw4_pending = "answer"
    st.rerun()


# ---------------------------------------------------------------------------
# WRITE THE ANSWER
# ---------------------------------------------------------------------------

if st.session_state.hw4_pending:
    history = st.session_state.hw4_messages

    # Messages to send: the system prompt plus the last few messages.
    # Only role and content are sent. The search record is for display only.
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in history[:1] + history[1:][-buffer:]
    ]

    # Keeps track of any searches so they can be shown later.
    tool_log = []

    with chat_box:
        with st.chat_message("assistant"):

            # Loading box that shows each step as it happens.
            with st.status("Deciding whether this needs club info...", expanded=True) as status:

                # FIRST LLM CALL: the LLM either answers or asks to use the search tool.
                response = st.session_state.hw4_client.chat.completions.create(
                    model="gpt-5.4-nano",
                    messages=api_messages,
                    tools=tools,
                )
                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                # The LLM asked to search.
                if tool_calls:
                    # Add the LLM's search request to the messages.
                    # This is only for this answer, not saved to the history.
                    api_messages.append(response_message.to_dict())

                    for tool_call in tool_calls:
                        if tool_call.function.name == "relevant_club_info":
                            # Get the search query the LLM wrote.
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query")

                            status.update(label="Searching student organization info...")
                            st.write(f"🔎 Searching for: *{query}*")

                            # Run the search and show what was found.
                            try:
                                club_info, count, sources = relevant_club_info(query)
                                st.write(f"📄 Found {len(sources)} pages ({count} excerpts)")
                                if sources:
                                    st.caption("Sources: " + ", ".join(f"`{s}`" for s in sources))
                                tool_log.append({"query": query, "count": count, "sources": sources, "error": False})
                            # If the search fails, tell the LLM instead of crashing.
                            except Exception as e:
                                club_info = json.dumps({"error": str(e)})
                                st.write("⚠️ The search failed.")
                                tool_log.append({"query": query, "count": 0, "sources": [], "error": True})

                            # Send the search results back, matched to the LLM's request by ID.
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": club_info,
                            })

                    # SECOND LLM CALL: the LLM reads the search results and writes the answer.
                    # No tools are passed, so it has to answer now.
                    status.update(label="Writing the answer...")
                    stream = st.session_state.hw4_client.chat.completions.create(
                        model="gpt-5.4-nano",
                        messages=api_messages,
                        stream=True,
                    )
                    status.update(label="Searched student organization info", state="complete", expanded=False)

                # The LLM answered without searching.
                else:
                    status.update(label="No search needed", state="complete", expanded=False)

            # Show the answer below the loading box.
            if tool_calls:
                # Type the answer out as it arrives.
                response = st.write_stream(stream)
            else:
                # Show the whole answer at once.
                response = response_message.content
                st.write(response)

    # Save the answer and its search record to the history.
    history.append({"role": "assistant", "content": response, "tool_log": tool_log})

    # Clear the flag and rerun to redraw the chat.
    st.session_state.hw4_pending = None
    st.rerun()