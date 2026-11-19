from langchain_community.document_loaders import WebBaseLoader

url = "https://www.samsung.com/in/smartphones/galaxy-z-fold7/?srsltid=AYCX9vJERPnBvAuKbhNM7B7cAmGx0-83D6mG59ekR1PgmylywAPizoDd"

data = WebBaseLoader(url)

docs = data.load()

print(docs)