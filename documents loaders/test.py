# from langchain_community.document_loaders import TextLoader

# data = TextLoader("notes.txt")

# docs = data.load()

# print(docs[0].page_content)



from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Step 1: Load PDF
loader = PyPDFLoader("GRU.pdf")
docs = loader.load()

# Step 2: Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

# Step 3: Split Documents
texts = text_splitter.split_documents(docs)

# Step 4: Print First Chunk
print(texts[0].page_content)