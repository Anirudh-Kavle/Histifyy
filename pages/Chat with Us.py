# import streamlit as st
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import json

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# import google.generativeai as genai

# def get_gemini_response(question):
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not question:
#         st.warning('Please enter a non empty question.', icon="⚠")
#         return None  # or return ''
#     genai.configure(api_key=api_key)  # Pass the API key as a keyword argument
    
#     # Initialize the generative model
#     model = genai.GenerativeModel('gemini-pro')

#     # Generate content based on the question
#     response = model.generate_content(question)

#     print(response)

#     # Return the generated response
#     return response.text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details".
#     Context:\n {context}?\n
#     Question: \n{question}\n

#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.3)

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question, conversation_history):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     conversation_history.append({"user_question": user_question, "gemini_response": response["output_text"]})
#     st.write("Reply: ", response["output_text"])
    
#     # Save updated conversation history to session state
#     st.session_state.conversation_history = conversation_history
    
#     return response["output_text"] 

# def get_gemini_response2(question):
#     api_key = os.getenv("GOOGLE_API_KEY")
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel("gemini-pro")
#     chat = model.start_chat(history=[])
#     response=chat.send_message(question,stream=True)
#     return response

# def format_gemini_response(response):
#     response.resolve()  # Ensure response iteration is complete
#     candidates = response.candidates
#     text_parts = [candidate.content.parts[0].text for candidate in candidates]
#     generated_text = '\n'.join(text_parts)
#     return generated_text

# def main():
#     st.set_page_config("Chat PDF")
#     st.header("Any Doubts ?")

#     # Initialize conversation history
#     conversation_history = st.session_state.get("conversation_history", [])

#     question_counter = 0  # Counter to generate unique keys for text inputs
#     button_counter = 0  # Counter to generate unique keys for buttons

#     while True:
#         question_counter += 1
#         user_question = st.text_input(f"Ask a Question from the PDF Files {question_counter}")

#         if user_question:
#             response_from_gemini = user_input(user_question, conversation_history)
#             ask_button_key = f"ask_button_{question_counter}"  # Unique key for ask button
#             if not response_from_gemini:
#                 # Display placeholder text while waiting for response
#                 st.write("Generating response...")
#             elif "Do you want to explore more resources?" in response_from_gemini:
#                 button_counter += 1
#                 if st.button(f"Explore more resources {button_counter}"):
#                     # Get response from Gemini and store it in a variable
#                     gemini_response = get_gemini_response2(user_question + "in detail")
#                     formatted_gemini_response = format_gemini_response(gemini_response)
#                     st.write("Answer: ", formatted_gemini_response)
#                     # Record the exploration in conversation history along with the Gemini response
#                     conversation_history.append({"user_question": user_question, "explore_more_resources": True, "gemini_response": formatted_gemini_response})
#                     continue  # Skip directly to the next iteration without showing "Ask another question" button

#             # Display input bar for the next question
#             continue_next_question = st.button("Ask another question", key=ask_button_key)
#             if continue_next_question:
#                 continue
#         else:
#             # Read text from chapter_data.json
#             with open("pdf_data/chapter_data.json", "r") as json_file:
#                 chapter_data = json.load(json_file)
#                 raw_text = chapter_data["information"]

#             # Process the text
#             text_chunks = get_text_chunks(raw_text)
#             get_vector_store(text_chunks)

#             response_from_gemini = get_gemini_response(user_question)
#             st.write("Reply: ", response_from_gemini)   
#             if response_from_gemini is not None:
#                 if "Do you want to explore more resources?" in response_from_gemini:
#                     button_counter += 1
#                     if st.button(f"Explore more resources {button_counter}"):
#                         # Get response from Gemini and store it in a variable
#                         gemini_response = get_gemini_response2(user_question)
#                         formatted_gemini_response = format_gemini_response(gemini_response)
#                         st.write("Answer: ", formatted_gemini_response)
#                         # Record the exploration in conversation history along with the Gemini response
#                         conversation_history.append({"user_question": user_question, "explore_more_resources": True, "gemini_response": formatted_gemini_response})

#             ask_button_key = f"ask_button_{question_counter}"  # Unique key for ask button
#             if not st.button("Ask another question", key=ask_button_key):
#                 break
    
#     # Save conversation history to session state
#     st.session_state.conversation_history = conversation_history

# if __name__ == "__main__":
#     main()

import time
import os
import joblib
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
import json


new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

def get_summary():
    with open("pdf_data/chapter_data.json", "r") as json_file:
        chapter_data = json.load(json_file)
        raw_text = chapter_data["information"]
        return raw_text


# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except:
    # data/ folder already exists
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = 'New Chat'

st.write('# Any Doubts?')

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(
        f'data/{st.session_state.chat_id}-st_messages'
    )
    st.session_state.gemini_history = joblib.load(
        f'data/{st.session_state.chat_id}-gemini_messages'
    )
    print('old cache')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    print('new_cache made')
st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        st.markdown(message['content'])

summary = get_summary()

predefined_prompt = "Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the question is not related to the provided context, just say 'answer is not available in the context'.\n\nIf the question is not related to the context, always append this line to the end of the response you give, 'Do you want to explore more resources?' The summary is " + summary


# React to user input
if prompt := st.chat_input('Type your doubts here...'):
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )
    ## Send message to AI
    response = st.session_state.chat.send_message(
        prompt + predefined_prompt,
        stream=True,
    )
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        message_placeholder = st.empty()
        full_response = ''
        assistant_response = response
        # Streams in a chunk at a time
        for chunk in response:
            # Simulate stream of chunk
            # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                # Rewrites with a cursor at end
                message_placeholder.write(full_response + '▌')
        # Write full message with placeholder
        message_placeholder.write(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=st.session_state.chat.history[-1].parts[0].text,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(
        st.session_state.messages,
        f'data/{st.session_state.chat_id}-st_messages',
    )
    joblib.dump(
        st.session_state.gemini_history,
        f'data/{st.session_state.chat_id}-gemini_messages',
    )



    

