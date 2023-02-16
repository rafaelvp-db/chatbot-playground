# Databricks notebook source
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from transformers import BlenderbotTokenizer
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig
from transformers import BlenderbotTokenizerFast 
import contextlib

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
  "I'm married with two kids, which car is best for me?",
  []
)
response[-1]

# COMMAND ----------


