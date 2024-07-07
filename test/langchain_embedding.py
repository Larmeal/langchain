from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
import langchain

langchain.debug = True

llm = ChatOllama(model="llama3", temperature=0)

embeddings = OllamaEmbeddings(model="llama3")

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("test/src/facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

query = "What is an interesting fact about the planets?"

################################ Create the vectors database #################################

# db = Chroma.from_documents(
#     documents=docs,
#     embedding=embedding,
#     persist_directory="test/src/emb"
# )

##############################################################################################
################################ Query vectors database ######################################

db = Chroma(
    persist_directory="test/src/emb",
    embedding_function=embeddings
)

responses = db.similarity_search(
    query=query
)

for res in responses:
    print(res.page_content)
    print("-" * 50)

print("\n")

##############################################################################################


retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="refine"
)

response = chain.invoke(
    input=query
)

print("-" * 50)
print(response["query"])
print("-" * 50)
print(response["result"])
print("-" * 50)