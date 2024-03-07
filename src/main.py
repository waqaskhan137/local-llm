from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains import RetrievalQA

app = FastAPI()

# Allow all origins for CORS (you can customize this based on your requirements)
origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define input models for endpoints
class URLInput(BaseModel):
    url: str


class TopicInput(BaseModel):
    topic: str


# Define the summarize_webpage endpoint
@app.post("/summarize_webpage/")
async def summarize_webpage(request_body: dict):
    try:
        # Extract topic and model_name from the request body
        url = request_body.get("url")
        model_name = request_body.get("model_name")
        print(url)

        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=100
        )
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(
            documents=all_splits, embedding=GPT4AllEmbeddings()
        )
        QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
        llm = Ollama(
            model=model_name,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )
        question = f"summarize what this blog is trying to say? {url}"
        result = qa_chain({"query": question})
        return {"summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define the generate_facts endpoint
@app.post("/generate_facts/")
async def generate_facts(request_body: dict):
    try:
        # Extract topic and model_name from the request body
        topic = request_body.get("topic")
        model_name = request_body.get("model_name")

        from langchain_community.llms import Ollama
        from langchain.callbacks.manager import CallbackManager
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain

        llm = Ollama(model=model_name, temperature=0.9)
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="Give me 5 interesting facts about {topic}?",
        )
        chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

        result = chain.invoke(topic)
        return {"facts": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Define your root endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Read the content of your HTML file
    with open("./src/web/index.html", "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content)


@app.get("/local-rag-test", response_class=HTMLResponse)
async def read_local_rag_test():
    # Read the content of your local-rag-test.html file
    with open("./src/web/local-rag-test.html", "r") as file:
        html_content = file.read()

    return HTMLResponse(content=html_content)
