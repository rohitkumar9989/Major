from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.vectorstores.base import BaseRetriever
from langchain_core.messages import trim_messages
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from typing import Any
import pandas as pd
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
session_id="chat1"
config={"configurable": {"session_id":session_id}}
model = ChatGroq(
    model="deepseek-r1-distill-llama-70b"
)
store={}
parser=StrOutputParser()
def get_session_history(session_id:str) -> BaseChatMessageHistory:
  if session_id not in store:
     store[session_id]=ChatMessageHistory()
  return store[session_id]


#weekly checker
store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]=ChatMessageHistory()
    with open ("test.txt") as f:
      data=f.readlines()
    for lines in data:
      store[session_id].add_message(AIMessage(content=lines.strip()))
  return store[session_id]

class Cust_run(Chain):
  @property
  def input_keys(self):
     return ["problem"]
  @property
  def output_keys(self):
     return["solution"]
  def _call(self, inputs:dict)->dict:
     messages=store[session_id].messages

     trimmed=trim_messages(
        messages,
        max_tokens=1000,
        strategy='last',
        token_counter=len
     )
     store[session_id].messages=trimmed
     return {"solution":inputs["problem"]}
  
chat=ChatPromptTemplate.from_messages([
   ("system", "So you have to help the patient based on their queries and on the history you have already been provided with"),
   ("human", "{problem}"),
   ("ai", "Okay patient your summary on the query you provided is:")
])
  
unwrap=RunnableLambda(lambda x: x["solution"])
runnable1=Cust_run()
chain=runnable1 | unwrap | chat | model | parser

runner=RunnableWithMessageHistory(chain, get_session_history=get_session_history)



#Daily adder
data=pd.read_csv("Train_data.csv")
le=LabelEncoder()
data["label"]=le.fit_transform(data["label"])
max_class=max(data["label"])
tok=Tokenizer()
tok.fit_on_texts(data["text"])
def pad_and_tok(data, tok, maxlen):
  sequences=tok.texts_to_sequences([data])
  padded=pad_sequences(sequences, padding="post", maxlen=maxlen)
  return padded

class Model(tf.keras.layers.Layer):
  def __init__ (self, units):
    super(Model, self).__init__()
    self.w1=tf.keras.layers.Dense(units)
    self.w2=tf.keras.layers.Dense(units)
    self.V=tf.keras.layers.Dense(1)
  def __call__(self, state_h, symptoms):
    symptoms=tf.expand_dims(symptoms, 0)
    scores=self.V(tf.nn.tanh(self.w1(state_h)+self.w2(symptoms)))
    atn=tf.nn.softmax(scores, axis=1)
    context=atn * symptoms
    context=tf.reduce_sum(context, axis=-2)
    return context

class Encoder(tf.keras.layers.Layer):
  def __init__(self, input_dim, output):
    super(Encoder, self).__init__()
    self.embed=tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output)
  def __call__(self, text):
    return self.embed(text)
  
class Decoder (tf.keras.Model):
  def __init__(self, units):
    super(Decoder, self).__init__()
    self.model=Model(units)
    self.enc=Encoder(
        input_dim=len(tok.word_index)+1,
        output=128
    )
    for i in range (2):
      setattr(self, f"otp_{i}", tf.keras.layers.Dense(32))
    self.otp=tf.keras.layers.Dense(max_class+1, tf.keras.activations.softmax)
  def __call__(self, state_h, state_c, symptoms):
    symptoms=self.enc(symptoms)
    context=self.model(state_h, symptoms)

    for i in range (2):
      context=getattr(self, f"otp_{i}")(context)
    return self.otp(tf.expand_dims(tf.squeeze(context), axis=0))
class infocarrier(tf.keras.Model):
  def __init__ (self, units):
    super(infocarrier, self).__init__()
    self.lstm=tf.keras.layers.LSTM(units, return_state=True)
  def __call__ (self, symptoms):
    state_h, state_c, _=self.lstm(tf.expand_dims(symptoms, axis=0))
    return state_h, state_c
info=infocarrier(128)
dec=Decoder(128)

info.load_weights("weights/info/info.ckpt")
dec.load_weights("weights/dec/dec.ckpt")

danger=["Typhoid"]
def predict(text):
   global danger
   des=pad_and_tok(text, tok, 100)
   des=tf.convert_to_tensor(des, dtype=tf.float32)
    
   state_h, state_c=info(des)
   preds=tf.squeeze(dec(state_h, state_c, des))

   preds=tf.argmax([preds], axis=1)
   ans=le.inverse_transform([preds])
   print ("The predictions made were: ", ans)
   return True if ans in danger else False, ans

class Custom_chain1(Chain):
    """
    Provide the text to the prompt such that it returns an chain of context and problem
    """
    prompt: BasePromptTemplate
    vector: BaseRetriever

    def __init__ (self, prompt, vector):
        super().__init__(prompt=prompt, vector=vector)
    @property
    def input_keys(self):
        return ["problem"]

    @property 
    def output_keys(self):
        return ["solution"]
    
    def _call(self, inputs:dict) -> dict:
        problem=inputs["problem"]
        sim_search=self.vector.invoke(problem)
        ans=self.prompt.invoke(
            {"problem":problem, "context":sim_search}
        )
        return {"solution": ans}

ret = FAISS.load_local("Abhi_index", embed, allow_dangerous_deserialization=True)
ret=ret.as_retriever()
prompt=ChatPromptTemplate.from_messages([
    ("system", "You are a nurse who creates a summarized description on what the patient "+\
     "is suffering based on the other patient records which are provided, generate within 80 words"),
    ("human", "Hello Nurse, Iam suffering from {problem}"),
    ("ai", "Okay Pateint provide me with the similar patient records"),
    ("human", "Okay here are the records {context}"),
    ("ai", "Doctor here is a summarized report of the patient problem within 80 words: ")
])

c1=Custom_chain1(prompt, ret)
unwrap=RunnableLambda(lambda x: x["solution"])

daily_chain=c1 | unwrap | model | parser

first_chat=True
app=FastAPI(title="Project", version="1.0")

class custom_weekly(BaseModel):
   chat:str
@app.post("/weekly_updates")
async def weekly_ref(prob:custom_weekly):
    problem=prob.chat
    answer=runner.invoke({"problem":problem}, config=config)
    return {"status": answer}


class custom_daily(BaseModel):
    problem:str
@app.post("/daily_updates")
async def daily_rep(prob:custom_daily):
    problem=prob.problem
    _, disease=predict(problem)
    if _ :
        prompt=ChatPromptTemplate.from_messages([
            ("system", "You are a doctor and the patient who is currently "+\
            "with you is suffering from {main} which is considered as a sever problem"+\
            "suggest the user to go to the doctor ASAP"),
            ("human", "Hello doctor, here is my report {report} problem Iam suffering {problem}"),
            ("ai", "I see you have to consult a doctor ASAP because your problem might cause")
        ])

        immediate_chain=prompt | model | StrOutputParser()
        return {
            immediate_chain.invoke({
                "main":disease,
                "report": problem,
                "problem":disease
            })
        }
    else:
        result=daily_chain.invoke({
            "problem":problem
        })
        with open ("test.txt", "a") as f:
            f.write(result + "\n")
        return {
           "Status": True
        }

if __name__=="__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
