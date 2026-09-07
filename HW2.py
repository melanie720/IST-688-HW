import pymupdf, requests, streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError, NotFoundError
from bs4 import BeautifulSoup
from google import genai

# Show title and description.
st.title("📄 Mel's Webpage Summarizer")
st.caption("Provide a URL, get a summary in return.")

# Create client based on user's LLM selection - OpenAI or Google Gemini.
if st.session_state.llm_select == 'OpenAI (default)':
    client = OpenAI(api_key = st.secrets.OPENAI_API_KEY)
else:
    client = genai.Client(api_key = st.secrets.GEMINI_API_KEY)

# Checking if API key is valid.
try:
    client.models.list()
except AuthenticationError:
    st.error("API key needs to be updated.")
    st.stop()

# Function for reading URLs.
def read_url_content(url): 
    try: 
        response = requests.get(url) 
        response.raise_for_status()  # Raise an exception for HTTP errors 
        soup = BeautifulSoup(response.content, 'html.parser') 
        return soup.get_text() 
    except requests.RequestException as e: 
        print(f"Error reading {url}: {e}") 
        return None 

# Hiding the "Press Enter to submit" caption in my input field
st.html("""
    <style>
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    </style>
""")

# Ask user to provide a URL to summarize.
# Added a button.
with st.form(key="my_form"):
    url = st.text_input("Enter your URL:")
    submitted = st.form_submit_button("Submit", type="primary")

if submitted and not url:
    st.info("Please enter a URL first.")

if submitted and url:
    if st.session_state.llm_select == 'OpenAI (default)':
        messages = [
            {
                "role": "user",
                "content": f"Here's the text from a URL: {read_url_content(url)}. Provide a summary in language {st.session_state.language_select or 'English'} and make sure it is {st.session_state.summary_type_select or '100 words'}."
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "parts": [{"text": f"Here's the text from a URL: {read_url_content(url)}. Provide a summary in language {st.session_state.language_select or 'English'} and make sure it is {st.session_state.summary_type_select or '100 words'}."}]
            }
        ]

    # Needed a spinner:
    with st.spinner(f"Summarizing your webpage with {st.session_state.model}...", show_time=True):
        # Handling model problems separately from key problems.
        try:
            if st.session_state.llm_select == 'OpenAI (default)':
                # Generate an answer using the OpenAI API.
                stream = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=messages,
                    stream=True
                )

                # Stream the response to the app using `st.write_stream`.
                st.write_stream(stream)

            else:
                # Generate content using Google Gemini API.
                stream = client.models.generate_content(
                    model=st.session_state.model,
                    contents=messages
                )

                # Stream the response to the app using `st.write_stream`.
                st.write(stream.text)
                


        # A 404 means the model is retired, renamed, or not on this account.
        except NotFoundError as e:
            st.error(
                f"**{st.session_state.model}** isn't available. "
                "It may have been deprecated, or your account may not have access. "
                "Pick a different model above."
            )
            # Show OpenAI's error message.
            st.caption(f"OpenAI said: {e}")

        # Anything other errors from the API
        except OpenAIError as e:
            st.error(f"The request failed ({type(e).__name__}): {e}")