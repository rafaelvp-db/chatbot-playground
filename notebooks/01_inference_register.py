# Databricks notebook source
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BlenderbotTokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
from transformers import BlenderbotTokenizerFast 

mname = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizerFast.from_pretrained(mname)
model = BlenderbotForConditionalGeneration.from_pretrained(mname)

def predict(input, history=[]):
  
    history.append(input)
    
    listToStr= ' '.join([str(elem)for elem in history[len(history)-3:]])
    input_ids = tokenizer([(listToStr)], return_tensors="pt",max_length=512,truncation=True)
    next_reply_ids = model.generate(**input_ids,max_length=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0]
    history.append(response)
    response = [(history[i], history[i+1]) for i in range(0, len(history)-1, 2)]  # convert to tuples of list
    return response, history

# COMMAND ----------

response = predict(
  "I'm 40 yo with two kids, which car is best for me?",
  []
)
response[-1]

# COMMAND ----------

# DBTITLE 1,Register PyFunc Model
import os
import mlflow
from blenderbot_wrapper import BlenderbotWrapper

# COMMAND ----------

model_path = "/tmp/blenderbot/model"
tokenizer_path = "/tmp/blenderbot/tokenizer"

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

artifacts = {
  "hf_model_path": model_path,
  "hf_tokenizer_path": tokenizer_path
}

# COMMAND ----------

#Simple test

class MLflowContext:
  def __init__(self, artifacts):
    self.artifacts = artifacts

context = MLflowContext(artifacts)

pyfunc_model = BlenderbotWrapper()
pyfunc_model.load_context(context)

payload = {"question": "I'm 40 yo with two kids, which car is best for me?", "history": []}
pyfunc_model.predict(model_input = payload, context = context)

# COMMAND ----------

mlflow_pyfunc_model_path = "blenderbot"
model_info = None

with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
    artifact_path = mlflow_pyfunc_model_path,
    python_model = BlenderbotWrapper(),
    code_path = ["./blenderbot_wrapper.py"],
    artifacts = artifacts,
    pip_requirements=["transformers==4.21.1", "torch==1.12.1"]
  )

# COMMAND ----------

model_name = "blenderbot"
version_info = mlflow.register_model(model_uri = model_info.model_uri, name = model_name)

# COMMAND ----------


