import spacy
import pandas as pd

nlp = spacy.load("./output_fresh/model-best")
df_Skus = pd.read_csv('source_repo/productos.csv', encoding='utf-8')
df_desc = pd.read_csv('source_repo/productos_dico.csv', encoding='utf-8')

def batch_extract_entities(nlp, texts, batch_size=256):
  ents = []
  for doc in nlp.pipe(texts, batch_size=batch_size):
    d = {}
    for e in doc.ents:
      cleaned = _clean_socket_text(e.text)
      if cleaned:
        d.setdefault(e.label_, []).append(cleaned)
    ents.append(d)
  return pd.Series(ents)

if "entities" not in df_Skus or df_Skus["entities"].isnull().all():
  df_Skus["entities"] = batch_extract_entities(nlp, df_Skus["processed_description"])
  # explicit ensure index alignment
  df_Skus["entities"] = df_Skus["entities"].reindex(df_Skus.index)
  df_Skus.to_csv('productos.csv', index=False)

if "entities" not in df_desc or df_desc["entities"].isnull().all():
  df_desc["entities"] = batch_extract_entities(nlp, df_desc["processed_description"])
  df_desc["entities"] = df_desc["entities"].reindex(df_desc.index)

# replace NaN with empty dicts
df_Skus["entities"] = df_Skus["entities"].apply(lambda x: x if isinstance(x, dict) else {})
df_desc["entities"] = df_desc["entities"].apply(lambda x: x if isinstance(x, dict) else {})

df_desc.to_csv('df_desc.csv', index=False)
df_Skus.to_csv('productos_merged.csv', index=False)
