import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
import streamlit as st

load_dotenv()
groq_api_key = os.getenv("groq_api_key")
model = ChatGroq(model = "llama3.2:1b",groq_api_key = groq_api_key)
embedding = OllamaEmbeddings(model = "llama3.2:1b")
mydb = Chroma(persist_directory="./chroma_db",embedding_function=embedding)
retriever = mydb.as_retriever(search_type = "similarity" ,search_kwargs ={"k":6})


st.title("PDF Reader")
Query = st.chat_input("Ash me anything: ")


system_prompt = (

"You are an assisant for question answaring task."
"Its a pdf reader application."
"Use the following pieces of retrived context to answer the question."
"Be polite while answering."
"\n\n"
"{context}"

)

prompt = ChatPromptTemplate.from_messages(
  [
    ("system",system_prompt),
    ("human","{input}"),
  ]
)


if Query:
    question_answer_chain = create_stuff_documents_chain(model,prompt)
    rag_chain = create_retrieval_chain(retriever , question_answer_chain)
    response = rag_chain.invoke({"input": Query})
    st.write(response['answer'])
