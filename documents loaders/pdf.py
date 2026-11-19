from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"A:\Gen-AI\RAG-Project\documents loaders\GRU.pdf")

docs = data.load()

print((docs[14]))

