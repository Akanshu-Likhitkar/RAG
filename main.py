from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings



embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",

)

vectorstore = Chroma(
    persist_directory=  "Chroma-Db",
    embedding_function  = embedding_model 
)

retriver = vectorstore.as_retriever(
    search_type = "mmr" , 
    search_kwargs = {
        "k": 4, 
        "fetch_k": 10 ,
        "lambda_mult": 0.5
        }
)


model = ChatOllama(
    model="gemma3:4b",
    temperature=0.7
)


# Prompt Template

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Instructions:
        1. Answer the question using ONLY the provided context in Paragraphs and Bullet Points.
        2. Do not use external knowledge.
        3. Provide clear, accurate, and concise answers.
        4. If the answer is not available in the context,
           respond exactly with:
           "I could not find the answer in the document."
        """
    ),

    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}

        Answer:
        """
    )
])



print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" :context,
        "question": query
    })
    
    response = model.invoke(final_prompt)

    print(f"\n AI: {response.content}")
    