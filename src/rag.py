from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class URLInput(BaseModel):
    url: str

@app.post("/summarize_webpage/")
async def summarize_webpage(url_input: URLInput):
    try:
        # Execute the code to load webpage, embed content, and summarize
        from langchain_community.document_loaders import WebBaseLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import GPT4AllEmbeddings
        from langchain_community.llms import Ollama
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain import hub
        from langchain.chains import RetrievalQA

        # Load webpage
        loader = WebBaseLoader(url_input.url)
        data = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        all_splits = text_splitter.split_documents(data)

        # Embed and store
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

        # RAG prompt
        QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

        # LLM
        llm = Ollama(model="llama2-uncensored", verbose=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

        # QA chain
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

        # Ask a question to summarize
        question = f"summarize what this blog is trying to say? {url_input.url}"
        result = qa_chain({"query": question})

        return {"summary": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
