import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR

import os

OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):                     # definiendo función para dividir el texto en chunks
    text_splitter = CharacterTextSplitter(     # definiendo el objeto text_splitter   
        separator="\n",                   # separador de los chunks (en este caso, un salto de línea)      
        chunk_size=1000,              # tamaño de los chunks (en este caso, 1000 caracteres)
        chunk_overlap=200,              # solapamiento entre chunks (en este caso, 200 caracteres)
        length_function=len             # función para calcular la longitud de los chunks (en este caso, la función len)
    )
    chunks = text_splitter.split_text(text)   # dividiendo el texto en chunks
    return chunks                   # devolviendo los chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)   # creando el vectorstore a partir de los chunks
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()                        # definiendo el objeto llm
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})                 # definiendo el objeto llm 

    memory = ConversationBufferMemory(                              # definiendo el objeto memory de la librería langchain 
        memory_key='chat_history', return_messages=True)             
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):                                               # definiendo función para manejar la entrada del usuario 
    # Construye un prompt más específico con la perspectiva deseada
    prompt = f"Eres un Ingeniero Eléctrico experto en normativas y estándares eléctricos. Tu tarea es proporcionar respuestas detalladas, precisas y con referencias claras a las fuentes, sin repetir la pregunta en tus respuestas. Tus respuestas deben estar enfocadas solo a los pliegos RIC. Primero responde precisamente lo que se te pregunta y luego agregar un breve comentario. Acota tu respuesta a 80 palabras, salvo que te pidan mas detalles. Al final de la respuesta incluye las citas o referencias con la sección a revisar. La pregunta a responder es: {user_question}?"
    response = st.session_state.conversation({'question': prompt})           # obteniendo la respuesta del modelo de lenguaje
    st.session_state.chat_history = response['chat_history']       # guardando el historial de la conversación

    for i, message in enumerate(st.session_state.chat_history):                 # iterando sobre el historial de la conversación 
        if i % 2 == 0:                                                                      # si el índice es par (mensaje del usuario)
            message.content = message.content.replace(prompt, user_question)               # eliminando el prompt de la respuesta del usuario
            st.write(user_template.replace(                                     # escribiendo el mensaje del usuario en el template user_template 
                "{{MSG}}", message.content), unsafe_allow_html=True)            
        else:                                                               # si el índice es impar (mensaje del bot) 
            st.write(bot_template.replace(                                  # escribiendo el mensaje del bot en el template bot_template
                "{{MSG}}", message.content), unsafe_allow_html=True)        


def main():                                                                         # definiendo función principal
    load_dotenv()                                                                   # cargando variables de entorno desde el archivo .env 
    st.set_page_config(page_title="Asistente IA SEC Chile",            # configurando la página de Streamlit 
                       page_icon=":books:")                             # título de la página y el ícono de la página (un libro) 
    st.write(css, unsafe_allow_html=True)                               # escribiendo el código CSS en la página de Streamlit 

    if "conversation" not in st.session_state:                          # si no existe la variable de sesión conversation 
        st.session_state.conversation = None                            # inicializando la variable de sesión conversation en None 
    if "chat_history" not in st.session_state:                          # si no existe la variable de sesión chat_history 
        st.session_state.chat_history = None                            # inicializando la variable de sesión chat_history en None 

    st.header("Asistente IA para normativa Eléctrica SEC - Chile :books:")                                # escribiendo el título de la página 
    user_question = st.text_input("Haz tu consulta sobre los pliegos técnicos:")       # escribiendo el campo de texto para que el usuario ingrese su pregunta 
    if user_question:                                                           # si el usuario ingresa una pregunta 
        handle_userinput(user_question)                                         # manejando la entrada del usuario 

    with st.sidebar:                                                            # barra lateral de la página 
        st.subheader("RIC SEC")                                          # título de la barra lateral
        pdf_docs = st.file_uploader(                                                                        # campo para subir archivos PDF
            "Selecciona tu normativa y dale click a 'Process'", accept_multiple_files=True)            # texto de ayuda para el campo de subida de archivos PDF
        if st.button("Process"):                                                            # botón para procesar los archivos PDF
            with st.spinner("Procesando"):                                              # spinner para indicar que se está procesando los archivos PDF
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)                                       # obteniendo el texto de los archivos PDF

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)                                     # obteniendo los chunks de texto

                # create vector store
                vectorstore = get_vectorstore(text_chunks)                              # creando el vectorstore

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(                 # creando la cadena de conversación
                    vectorstore)                                                            # pasando el vectorstore como parámetro


if __name__ == '__main__':
    main()
