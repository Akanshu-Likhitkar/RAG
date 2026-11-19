# load pdf
# split into chunks
# create the embeddings
# store the embeddings in a vector database


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



data = PyPDFLoader(
    r"A:\Gen-AI\RAG-Project\documents loaders\deeplearning.pdf"
)

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


chunks = splitter.split_documents(docs)


embedd = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",

)

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embedd ,
    persist_directory=  "Chroma-Db"
)

