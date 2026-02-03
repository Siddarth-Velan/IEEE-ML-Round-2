# Includes colab notebook, trained model, final embeddings and faiss index.
Major issues : couldnt get fastapi to load properly, both through the cors middleware route and from colab cell itself, not familiar enough with cli to implement
             : Hybrid search kept bugging out when used to train the model, so normal search implemented, but hybrid search can be accessed separately through the cells, the search itself is fully functional
             : all metrics are really low as there are qrels for all 8.8M queries, I have a subset of 100K
             : improvement is definitely there for the trained model, would really love to see it on a full dataset of this scale. I originally used the trec-covid dataset and all the data
             : but there were only 360 queries total, not enough to train at all
             : Many of the libraries are new to me
             : Very honest, debugging is almost 60% claude, i couldnt understand the libraries well enough to actually fix all the issues in this small timeframe
Instructions to run:
  The saved files should work, so just import the dataset and mount to drive. Or download and import to colab, and copy the path into the load area. Use the saved presets for embedding and index creation.
  I dont have a full frontend, depending on when you guys see this I may have fastapi actually working, but primarily, the last cell serves as a sort of interface, it gives the rankings and the document of the query
  worst case, you may have to run all the cells, on gpu runtime it takes about 20 minutes, please dont use cpu run time.
  
  
