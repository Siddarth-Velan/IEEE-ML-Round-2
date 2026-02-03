
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import pickle
from beir.datasets.data_loader import GenericDataLoader
app = FastAPI(title='IEEERound2')
model = SentenceTransformer('/content/drive/MyDrive/finetuned')
index = faiss.read_index('finetuned_idx.index')
with open('data_idxs.pkl' , 'rb') as f:
  data_idxs = pickle.load(f)
with open('texts.pkl', 'rb') as f:
  text2 = pickle.load(f)
class Searching(BaseModel):
  query : str
  k : int = 10
@app.get('/')
def root():
  return {'documents' : len(data_idxs), 'status' : 'running'}
@app.post('/search')
def search(request: Searching):
  if not request.query.strip():
    raise HTTPException(status_code = 400, detail = 'Query is empty')
  if not (0 < request.k < 200) : 
    raise HTTPException(status_code = 400, detail = 'top k has to be between 1 and 199')
  start = time.time()
  q_emb = model.encode([request.query], convert_to_numpy = True)
  faiss.normalize_L2(q_emb)
  s, idx = index.search(q_emb, request.k)
  results = []
  for rank, (i, j) in enumerate(zip(s[0], idx[0], 1)):
    results.append({'rank' : rank, 'id' : j, 'score' : i, 'text' : text2[j]})
  finish = time.time()
  l = finish - start
  return { 'query' : request.query, 'results' : results, 'latency' : l}