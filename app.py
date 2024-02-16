import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

from langchain_community.document_loaders import YoutubeLoader # cargador de transcripciones de Youtube
from langchain.text_splitter import RecursiveCharacterTextSplitter # divisor de textos
from langchain_openai import OpenAIEmbeddings # embeddings de OpenAI
from langchain_community.vectorstores import FAISS # almacenamiento de vectores
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def get_youtube_retriever_tool(url, description):
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False
    )

    docs = loader.load()

    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

    embedding=OpenAIEmbeddings()

    persist_directory = 'docs/chroma/' + url.split("?")[-1]

    vector = FAISS.from_documents(
        documents=documents,
        #persist_directory=persist_directory, 
        embedding=embedding)
    
    retriever = vector.as_retriever()

    from langchain.tools.retriever import create_retriever_tool
    retriever_tool = create_retriever_tool(
        retriever,
        "youtube_transcription_retriever",
        description=description,
    )

    return retriever_tool

# Modelo de lenguaje (LLM)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#Prompt
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

st.set_page_config(page_title="Chatbot de YouTube", page_icon="🤖", layout="wide")
# youtube icon
st.image("https://www.youtube.com/img/desktop/yt_1200.png", width=100)
st.title("Chatbot de YouTube")
st.write("Este chatbot es capaz de responder preguntas sobre el contenido de un video de YouTube. Simplemente ingrese la URL de un video de YouTube y una descripción de lo que se trata el video y el chatbot responderá preguntas sobre el contenido del video.") 

# sidebar
st.sidebar.title("Ingresar URL de YouTube y Descripción")
url = st.sidebar.text_input("YouTube URL")
if url == "":
    url = "https://youtu.be/T2M9hSswlIs?si=t3OZA2Xt1wEa9jcM"

description = st.sidebar.text_input("Description")
if description == "":
    description = "Ante cualquier pregunta sobre 'Cómo comenzar un canal de youtube en 2024' debes usar esta herramienta!"

description = "Ante cualquier pregunta sobre '" + description + "' debes usar esta herramienta!"

retriever_tool = get_youtube_retriever_tool(url, description)
#st.experimental_rerun()

tools = [retriever_tool]

# Agente
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("¿En qué te puedo ayudar?")

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # Esto es necesario porque en la mayoría de los escenarios del mundo real, se necesita una identificación de sesión.
    # Realmente no se usa aquí porque estamos usando un ChatMessageHistory simple en memoria.
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Renderizar mensajes actuales de StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Si el usuario ingresa una nueva consulta, generar y mostrar una nueva respuesta
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True )
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])


