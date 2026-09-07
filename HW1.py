import pymupdf
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError, NotFoundError

# Show title and description.
st.title("📄 Mel's Document Q&A")
st.caption("Upload a text file and ask a question about it.")

# Ask user for their OpenAI API key via `st.text_input`.
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Model picker options.
MODELS = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4.1": "gpt-4.1",
    "GPT-5.6 Sol": "gpt-5.6-sol",
    "GPT-5 Nano": "gpt-5-nano",
}

if not openai_api_key:
    st.info("Add your OpenAI API key and press Enter.", icon="🔑")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Checking if API key is valid.
    try:
        client.models.list()
    except AuthenticationError:
        st.error("That key didn't work. Check it and try again.")
        st.stop()

    # Creating a function to read PDF files.
    def read_pdf(pdf_file):
        # 1. Read the uploaded file into memory as bytes
        file_bytes = pdf_file.getvalue()

        # Collect each page's text here so it can be joined at the end
        full_text = []

        # 2. Open the PDF from memory using the stream argument
        with pymupdf.open(stream = file_bytes, filetype = "pdf") as doc:
            st.success(f"Successfully loaded: {pdf_file.name}")

            # 3. Iterate through pages and extract text
            for page_num, page in enumerate(doc):
                # Extract text from the current page
                page_text = page.get_text()
                full_text.append(page_text)

        return "\n".join(full_text)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader("Document (.txt or .md)", type=("txt", "pdf"))

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Your question",
        placeholder="Ex. What are the main points in this document?",
        disabled=not uploaded_file,
    )

    # Model picker.
    model_label = st.radio(
        "Model",
        options=list(MODELS),
        index=list(MODELS).index("GPT-5 Nano"),
        horizontal=True,
        help="Which OpenAI model answers your question.",
    )
    model_id = MODELS[model_label]

    # Added a button:
    if st.button("Get answer", type="primary", disabled=not uploaded_file):
        if not question:
            st.warning("Type a question first.")
        else:
            # Process the uploaded file and question.
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'txt':
                document = uploaded_file.getvalue().decode()
            elif file_extension == 'pdf':
                document = read_pdf(uploaded_file)
            else:
                st.error("Unsupported file type.")
                st.stop()

            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Needed a spinner:
            with st.spinner(f"Reading your document with {model_label}...", show_time=True):
                # Handling model problems separately from key problems.
                try:
                    # Generate an answer using the OpenAI API.
                    stream = client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        stream=True,
                    )

                    # Stream the response to the app using `st.write_stream`.
                    st.write_stream(stream)

                # A 404 means the model is retired, renamed, or not on this account.
                except NotFoundError as e:
                    st.error(
                        f"**{model_label}** (`{model_id}`) isn't available. "
                        "It may have been deprecated, or your account may not have access. "
                        "Pick a different model above."
                    )
                    # Show OpenAI's error message.
                    st.caption(f"OpenAI said: {e}")

                # Anything other errors from the API
                except OpenAIError as e:
                    st.error(f"The request failed ({type(e).__name__}): {e}")