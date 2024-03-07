from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

class TopicInput(BaseModel):
    topic: str

llm = Ollama(model="llama2-uncensored", temperature=0.9)
prompt = PromptTemplate(input_variables=["topic"], template="Give me 5 interesting facts about {topic}?")
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

@app.post("/generate_facts/")
async def generate_facts(topic_input: TopicInput):
    try:
        result = chain.run(topic_input.topic)
        return {"facts": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
