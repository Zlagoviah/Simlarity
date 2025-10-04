import numpy as np
import pandas as pd
import re
import spacy
import unicodedata

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import fuzz

nltk.download('punkt_tab')
nlp = spacy.load("model-best")

df_skus = pd.read_csv('source_repo/productos.csv', encoding='utf-8')

df_skus_dico = pd.read_csv('source_repo/productos_dico.csv', encoding='utf-8')
df_skus_dico.rename(columns={'category': 'category_old', 'subfamilia': 'category'}, inplace=True)
df_Skus = pd.concat([df_skus, df_skus_dico], ignore_index=True)

df_desc = pd.read_csv('df_descc.csv', encoding='utf-8')

# -------------------------
# Load model
nlp = spacy.load("model-best")  # ensure this path is correct

# Pre-compile frequently used regexes for performance
_UNITS = r'(?:gb|tb|mb|k|rpm|w|hz|mhz|mt)'
_NUM_UNIT_RE = re.compile(rf'(\d+(?:[.,]\d+)?)\s+({_UNITS})\b', flags=re.IGNORECASE)

_REMOVE_PHRASES = [
    "incluye disipador", "no incluye disipador", "ventana de acrilico", "ventana lateral cristal",
    "panel frontal malla", "ventana lateral acrilico", "ventana", "ventana lateral", "ventana lateral vidrio templado",
    "vidrio templado", "ventana vidrio templado", "panel lateral vidrio templado",
    "paneles frontales y laterales de vidrio templado", "panel lateral", "frontal malla", "micro torre", "lateral",
    "cristal templado", "media torre", "mini torre", "alto rendimiento", "alta eficiencia", "numero de",
    "unidad de estado solido"
]

_VENT_PAT = re.compile(r'\b(?:xven|x ven|xvent|3xvent|x vent|vent|x ventiladores)\b', flags=re.IGNORECASE)

# Remove 'cores/threads/hilos/nucleos' with adjacent 1-2 digit numbers (either order)
_CORE_WORDS = r'(?:nucleos|hilos|cores|threads)'
# Por que sí puede haber "8 nucleos"
_CORE_WORDS_ = r'(?:hilos|cores|threads)'

_CORE_AFTER_RE = re.compile(rf'(?<!\w){_CORE_WORDS}\s*\d{{1,2}}(?!\w)', flags=re.IGNORECASE)   # "nucleos 8"
_CORE_BEFORE_RE = re.compile(rf'(?<!\w)\d{{1,2}}\s*{_CORE_WORDS_}(?!\w)', flags=re.IGNORECASE)  # "8 hilos" or "20 threads"

# Remove space in "cl 20" -> "cl20" (supports 1-2 digits; change to \d{{2}} if strictly two digits)
_CL_JOIN_RE = re.compile(r'(?<!\w)cl\s+(\d{1,2})(?!\w)', flags=re.IGNORECASE)

# NEW: remove the word "color" only when it has whitespace before AND after it
_COLOR_SURROUNDED_RE = re.compile(r'(?<=\s)color(?=\s)', flags=re.IGNORECASE)

# Preprocess text (e.g., tokenization, stemming, stop word removal)
def strip_accents(text: str) -> str:
    """
    Turn accented characters into their un-accented ASCII equivalents.
    E.g. 'áéíóúüñ' → 'aeiouun'
    """
    # Normalize into decomposed form (NFKD),
    # then drop all combining marks (accents).
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def preprocess_text(text: str) -> str:
    """Preprocesses Spanish text: lower-case, strip accents, remove tokens."""
    if not isinstance(text, str):
        return ""
    # 1) Lowercase, remove unwanted chars (keep letters, numbers, spaces)
    text = re.sub(r"[^A-Za-z0-9áéíóúüñÁÉÍÓÚÜÑ]+", " ", text.lower())

    # 2) Strip accents → this maps 'í' to 'i', not delete it
    text = strip_accents(text)
    
    # 2.1) Normalize decimal comma to dot (1,5 -> 1.5) to keep consistency
    # text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    
    # 3) Remove specific phrases (use word boundaries)
    for phrase in _REMOVE_PHRASES:
        text = re.sub(r'\b' + re.escape(phrase.lower()) + r'\b', ' ', text)
    
    # 4) Normalize ventilador variants to a single token
    text = _VENT_PAT.sub(' ventiladores', text)
    
     # 5) Remove core/thread/hilos counts (both orders). Do this BEFORE joining number+unit.
    text = _CORE_AFTER_RE.sub(' ', text)
    text = _CORE_BEFORE_RE.sub(' ', text)
    
    # 6) Join "cl 20" -> "cl20" (removes the space safely)
    text = _CL_JOIN_RE.sub(r'cl\1', text)
    
    # 5) Join numbers and units: "16 gb" -> "16gb", "1.5 mhz" -> "1.5mhz"
    # uses compiled regex _NUM_UNIT_RE
    text = _NUM_UNIT_RE.sub(r'\1\2', text)
    
    # 8) NEW: remove 'color' only when surrounded by spaces (preserves color: or color at string ends)
    text = _COLOR_SURROUNDED_RE.sub('', text)
    
    # 6) Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()

    # 7) Tokenize (ensure punkt spanish models installed)
    try:
        tokens = nltk.word_tokenize(text, language='spanish')
    except Exception:
        # fallback simple split if tokenizer is missing
        tokens = text.split()

    # Remove stop words
    spanish_stopwords = set(['technology','simuntaneamente','torre','de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'tambien', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'mi', 'antes', 'algunos', 'qué', 'que', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'el', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'tu', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'ustedes', 'usted', 'mío', 'mía', 'míos', 'mías', 'mio', 'mia', 'mios', 'mias', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estas', 'esta', 'estamos', 'estáis', 'están', 'esté', 'estés', 'estais', 'estan', 'este', 'estes', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'esteis', 'esten', 'estare', 'estaras', 'estara', 'estaremos', 'estaréis', 'estareis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaran', 'estaria', 'estarias', 'estariamos', 'estariais', 'estarian', 'estaba', 'estabas', 'estábamos', 'estabamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvieramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habéis', 'habeis', 'han', 'haya', 'hayas', 'hayamos', 'hayáis', 'hayais', 'hayan', 'habré', 'habrás', 'habrá', 'habre', 'habras', 'habra', 'habremos', 'habréis', 'habrán', 'habría', 'habrías', 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'habreis', 'habran', 'habria', 'habrias', 'habriamos', 'habriais', 'habrian', 'habia', 'habias', 'habiamos', 'habiais', 'habian', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos', 'hubieramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubiesemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas', 'seamos', 'seáis', 'seais', 'sean', 'seré', 'serás', 'será', 'sere', 'seras', 'sera', 'seremos', 'seréis', 'serán', 'sería', 'serías', 'seríamos', 'seríais', 'serían', 'sereis', 'seran', 'seria', 'serias', 'seriamos', 'seriais', 'serian', 'era', 'eras', 'éramos', 'eramos', 'erais', 'eran', 'fui', 'fuiste', 'fue', 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fuéramos', 'fueramos', 'fuerais', 'fueran', 'fuese', 'fueses', 'fuésemos', 'fuesemos', 'fueseis', 'fuesen', 'sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente', 'sentid', 'sentir', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'teneis', 'tienen', 'tenga', 'tengas', 'tengamos', 'tengáis', 'tengais', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendre', 'tendras', 'tendra', 'tendremos', 'tendréis', 'tendrán', 'tendría', 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían', 'tendreis', 'tendran', 'tendria', 'tendrias', 'tendriamos', 'tendriais', 'tendrian', 'tenia', 'tenias', 'teniamos', 'teniais', 'tenian', 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos', 'tuvieramos', 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo', 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened', 'tener']) #set(stopwords.words('spanish'))
    
    tokens = [token for token in tokens if token not in spanish_stopwords]

    # 8) Filtrar tokens vacíos o tokens irrelevantes
    tokens = [t for t in tokens if t.strip()]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

FUZZY_ENTITIES = {
    "BRAND", "SERIES", "COLOR", "FORM_FACTOR", "DISPLAY_TECH", "CERTIFICATION", "LANGUAGE","DDR_GEN","INTERFACE","SIZE","WIRE_LESS","GPU", "CPU", "SWITCH","ILLUMINATION", "CATEGORY", "SERVICE"
}
NUMERIC_PRIORITY = {
    "STORAGE", "CACHE","WATTAGE","ITEMS","CAS_LAT","RAM_SPEED","ROTATION_SPEED","RAM_CAPACITY"
}

EXACT_ENTITIES = {
    "RAM_DESIGN"
}

# Example weights; tune per your domain
CATEGORY_ENTITY_WEIGHTS = {
    "SSD": {"BRAND": .25, "SERIES": .375, "SIZE": .05, "STORAGE": .2, "FORM_FACTOR": .1, "INTERFACE": .025},
    "HDD": {"BRAND": .125, "SERIES": .5, "SIZE": .05, "STORAGE": .225, "INTERFACE": .05, "ROTATION_SPEED": .025,  "COLOR": .025},
    "RAM": {"BRAND": .2, "SERIES": .325, "STORAGE": .15, "DDR_GEN": .025, "RAM_DESIGN": .1, "RAM_SPEED": .075, "CAS_LAT": .075, "COLOR": .05},
    "CPU": {"CPU": .9, "CACHE": .1},
    "GPU": {"BRAND": .2, "SERIES": .225, "GPU": .425, "STORAGE": .15},
    "MOBO": {"BRAND": .35, "SERIES": .65},
    "PSU": {"BRAND": .25, "SERIES": .525, "CERTIFICATION": .1, "WATTAGE": .125},
    "CASE": {"BRAND": .35, "SERIES": .55, "COLOR": .1},
    "MONITOR": {"BRAND": .3, "SERIES": .5, "DISPLAY_TECH": .1, "SIZE": .1},
    "KEYBOARD": {"BRAND": .25, "SERIES": .45, "WIRE_LESS": .1, "SWITCH": .05, "COLOR": .075, "LANGUAGE": .075},
    "MOUSE": {"BRAND": .3, "SERIES": .525, "WIRE_LESS": .075, "COLOR": .1},
    "KITKM": {"BRAND": .3, "SERIES": .475, "WIRE_LESS": .075, "COLOR": .075, "LANGUAGE": .075},
    "AIRC": {"BRAND": .3, "SERIES": .55, "SIZE": .1, "COLOR": .05},
    "WATERC": {"BRAND": .35, "SERIES": .55, "COLOR": .1},
    "FAN": {"BRAND": .3, "SERIES": .45, "SIZE": .05, "ILLUMINATION": .025, "COLOR": .1, "ITEMS": .075},
    "MBRACKET": {"CATEGORY": .075, "BRAND": .35, "SERIES": .575},
    "HEADPHONE": {"BRAND": .3, "SERIES": .575, "WIRE_LESS": .075, "COLOR": .05},
    "LAPTOP": {"BRAND": .15, "SERIES": .4, "CPU": .15, "GPU": .15, "RAM_CAPACITY": .075, "STORAGE": .075},
    # OTHERS
    "MSA": {"BRAND": .2, "SERIES": .7, "CATEGORY": .1},
    "NAS": {"BRAND": .2, "SERIES": .35, "CATEGORY": .05, "STORAGE": .15, "CPU": .2, "COLOR": .05},
    "FILTROS": {"BRAND": .3, "CATEGORY": .1, "SERIES": .45, "SIZE": .15},
    "SERVICE": {"SERVICE": 1},
    "INSTALACION": {"SERIES": 1},
    "MULTI": {"CATEGORY": .1, "BRAND": .4, "SERIES": 5},
    "PRODUCTOS": {"CATEGORY": 1}
}

# Map various category strings to canonical labels used for filtering
CATEGORY_MAP = {
    "INSTALACION": {"Instalacion office", "instalacion so"},
    "PRODUCTOS": {"productos"},
    "SERVICE": {"paqueteria", "instalacion"},
    "CPU":  {"procesador", "procesadores pc", "procesadores servidor", "procesadores amd socket am4", "procesadores intel socket 1200", "procesadores intel socket 1700 12° gen", "procesadores intel socket 1700 13° gen", "procesadores intel socket 1700 14° gen", "procesadores amd socket am5", "procesadores intel socket 1851"},
    "RAM":  {"ram", "rams", "ram laptops", "memorias ram sodimm para laptop ddr4", "memorias ram dimm ddr4", "memorias ram dimm ddr3", "memorias ram sodimm para laptop ddr3", "memorias ram sodimm para laptop ddr5", "memorias ram dimm ddr5"},
    "MOBO": {"motherboard", "tarjetas madre", "tarjeta madre", "t. madre procesador integrado", "t. madre socket am4 (amd)", "t. madre socket 1151 (intel)", "t.madre socket 1200 (intel)", "t. madre socket 1700 (intel", "t. madre socket am5 (amd)", "t. madre socket 1851 (intel)", "t. madre socket 1700 (intel"},
    "HDD":  {"hdd laptop", "hdd", "hdd servidor", "hdd vigilancia", "hdd externo", "hdd 1", "discos duros sata 3.5 pulgadas", "discos duros sata 2.5 pulgadas", "disco duro sas"},
    "SSD":  {"ssd", "unidad estado solido", "ssd unidades de estado solido", "ssd externos"},
    "GPU":  {"tarjetas de video", "tarjeta de video", "tarjeta grafica", "gpu", "tarjeta video", "tarjeta de video  pci", "tarjetas de video pci-exp"},
    "PSU":  {"fuentes de poder", "fuente de poder", "fuente", "fuentes de poder vigilancia"},
    "CASE": {"gabinetes", "gabinete", "gabinetes computadora"},
    "MONITOR":  {"monitores", "monitor", "monitor 2", "monitores led/lcd", "monitores touch"},
    "KEYBOARD": {"teclados", "teclado", "teclados alambricos", "teclados numericos", "teclados para tablets", "teclados inalambricos"},
    "MOUSE":    {"mouses", "raton", "mouse", "mouse opticos/laser"},
    "KITKM":    {"kit teclado y raton", "kit mouse y teclado", "kit m y t inalámbrico", "teclado y raton", "kit de teclado y mouse", "teclado y ratón", "teclado+raton alambrico", "teclado+raton inalambrico"},
    "AIRC": {"disipadores", "disipador", "disipadores y ventiladores para procesador"},
    "WATERC":   {"enfriamiento liquido", "sistemas de enfriamiento líquido"},
    "FAN":  {"ventiladores", "ventilador", "ventiladores para gabinete"},
    "MBRACKET": {"soporte para monitor", "soportes para monitor", "brazo escritorio", "soporte para monitores y televisiones"},
    "HEADPHONE":{"diadema", "audífonos", "audifonos"},
    "LAPTOP":   {"laptop", "laptops"},
    "MSA":   {"almacenamiento msa"},
    "NAS":   {"almacenamiento nas", "hdd nas"},
    "FILTROS":   {"filtros de privacidad"},
    "MULTI": {"wi fi", "limpiadores", "routers", "switches","multifuncionales", "microfonos", "camaras", "lamparas", "webcam", "otro", "proyectores", "tarjetas de red", "adaptador wifi", "ruteadores inalambricos", "iluminación para gabinetes", "otros", "Base pc", "bases para gabinetes", "panel monitor touch", "bases y ventiladores para laptop", "ergonomicos, bases y comodidad", "soporte para microfono", "soporte proyector", "mousepad o tapetes p/raton", "accesorios para mouse", "accesorios para teclado", "pasta termica para procesadores", "toallas antiestaticas", "limpiador antiestatico para pantallas", "kit de limpieza", "locion limpiadora", "aire comprimido", "espuma limpiadora", "limpiador tarjetas electronicas", "grasa y lubricante partes moviles", "alcohol isopropilico", "limpiador de inyectores", "soportes tarjeta de video", "extensiones", "sillas", "bocinas", "escritorios", "muebles", "soporte gpu", "no break"}
    }

DEFAULT_WEIGHTS = {"CATEGORY": .1, "BRAND": .25, "SERIES": .65}  # fallback

# -------------------------
# Helper: batch entity extraction - group entities by label into lists

# Compile once for speed
_socket_patterns = [
    # exact-ish: "socket", "socket am", "socket am3", "socket am4", "socket am5",
    # "socket am4 4", "socket am4 4x"
    r'\b(socket|skt)(?:\s+am(?:\d)?(?:\s*\d[x]?)?)?\b',

    # "socket 1200", "socket 1700" (socket + 3-4 digit number)
    r'\b(socket|skt)\s+\d{3,4}\b',

    # "socket lga 1700 o socket lga 1700 4x"
    r'\b(socket|skt)\s+lga\s*\d{3,4}\b(?:\s*\d[x]?)?\b',

    # "lga 1700" alone
    r'\blga\s*\d{3,4}\b',
]

# join into one regex and make case-insensitive
_SOCKET_RE = re.compile('|'.join(_socket_patterns), flags=re.IGNORECASE)

# helper: clean a single entity text
def _clean_socket_text(s):
    if s is None:
        return None
    s = str(s)
    # remove the socket patterns
    s = _SOCKET_RE.sub('', s)
    # collapse repeated separators or whitespace left over
    s = re.sub(r'[\s\|\-_/]{2,}', ' ', s)    # collapse sequences of separators/spaces
    s = s.strip(' -|_/')                     # trim leading/trailing separators/spaces
    s = s.strip()
    return s if s != '' else None

# revised batch extractor that applies cleaning and skips empty values
def batch_extract_entities(nlp, texts, batch_size=256):
    ents = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        d = {}
        for e in doc.ents:
            cleaned = _clean_socket_text(e.text)
            if cleaned:                       # only keep non-empty cleaned values
                d.setdefault(e.label_, []).append(cleaned)
        ents.append(d)
    # return a Series to align with dataframe indices easily
    return pd.Series(ents)

# -------------------------
# Example numeric extractor (returns list of numbers as ints/floats)
_num_re = re.compile(r"(?<!\d)(\d+(?:[.,]\d+)?)(?:\s*(gb|tb|mb|k|rpm|w|hz|mhz|ventilador(es)?|pieza(s)?|pza(s)?)?)", re.IGNORECASE)
def extract_numeric(text):
    if text is None:
        return []
    text = str(text)
    matches = _num_re.findall(text)
    nums = []
    for match in matches:
        # If regex has multiple groups, match is a tuple
        if isinstance(match, tuple):
            val = match[0]
            unit = match[1] if len(match) > 1 else ""
        else:
            val = match
            unit = ""
        try:
            v = float(val.replace(",", "."))
        except Exception:
            continue
        nums.append((v, unit.lower() if unit else ""))
    return nums

def strip_text(text):
    if text is None:
        return []
    return [str(text).replace(" ", "")]

# -------------------------
# Your config sets (fill / tune these)

# -------------------------
# compare_entities with safer checks
def compare_entities(val_desc, val_sku, entity_label, fuzzy_threshold=80):
    # None/empty check
    if not val_desc or not val_sku:
        return False

    # normalize to strings if lists -> take first or join
    if isinstance(val_desc, list):
        val_desc = " | ".join(val_desc)
    if isinstance(val_sku, list):
        val_sku = " | ".join(val_sku)

    val_desc = str(val_desc).strip()
    val_sku = str(val_sku).strip()

    if entity_label in FUZZY_ENTITIES:
        score = fuzz.ratio(val_desc, val_sku)
        return score >= fuzzy_threshold

    elif entity_label in NUMERIC_PRIORITY:
        nums_a = extract_numeric(val_desc)
        nums_b = extract_numeric(val_sku)
        #print(nums_a, "compared to", nums_b) [(2666.0, 'mhz')] compared to [(3200.0, 'mhz')] [(16.0, '')] compared to [(16.0, '')] [(8.0, 'gb')] compared to [(16.0, 'gb')]
        # simple normalization: compare sets of (value,unit)
        return set(nums_a) == set(nums_b)
    
    elif entity_label in EXACT_ENTITIES:
        exact_a = strip_text(val_desc)
        exact_b = strip_text(val_sku)
        # simple normalization: compare sets of (value,unit)
        return set(exact_a) == set(exact_b)

    else:
        return val_desc.lower() == val_sku.lower()

# -------------------------
# normalized_entity_score unchanged except safer access
def normalized_entity_score(desc_entities, sku_entities, weight_cat, fuzzy_threshold=80):
    # ensure dicts
    desc_entities = desc_entities if isinstance(desc_entities, dict) else {}
    sku_entities = sku_entities if isinstance(sku_entities, dict) else {}

    weights = CATEGORY_ENTITY_WEIGHTS.get(weight_cat, {})
    present = {k: w for k, w in weights.items() if k in desc_entities}
    denom = sum(present.values())
    if denom == 0:
        return 0.0

    score = 0.0
    for label, w in present.items():
        val_desc = desc_entities.get(label)
        val_sku = sku_entities.get(label)
        if compare_entities(val_desc, val_sku, label, fuzzy_threshold):
            score += w
    return score / denom

# -------------------------

# Apply preprocessing to the 'descripcion' column in both DataFrames 
if "processed_description" not in df_Skus:
    # You will define preprocess_text as before (your normalization logic)
    df_Skus["processed_description"] = df_Skus["descripcion"].apply(preprocess_text)

# Ensure dataframes have entities column (explicitly aligned)
if "entities" not in df_Skus or df_Skus["entities"].isnull().all():
    df_Skus["entities"] = batch_extract_entities(nlp, df_Skus["processed_description"])
    # explicit ensure index alignment
    df_Skus["entities"] = df_Skus["entities"].reindex(df_Skus.index)
    df_Skus.to_csv('productos_merged.csv', index=False)

if "processed_description" not in df_desc:
    # You will define preprocess_text as before (your normalization logic)
    df_desc['processed_description'] = df_desc['descripcion'].apply(preprocess_text)

if "entities" not in df_desc or df_desc["entities"].isnull().all():
    df_desc["entities"] = batch_extract_entities(nlp, df_desc["processed_description"])
    df_desc["entities"] = df_desc["entities"].reindex(df_desc.index)

# replace NaN with empty dicts
df_Skus["entities"] = df_Skus["entities"].apply(lambda x: x if isinstance(x, dict) else {})
df_desc["entities"] = df_desc["entities"].apply(lambda x: x if isinstance(x, dict) else {})

# -------------------------
# Build TF-IDF and similarity matrix
vectorizer = TfidfVectorizer()
X_a = vectorizer.fit_transform(df_Skus['processed_description'].fillna(""))
X_b = vectorizer.transform(df_desc['processed_description'].fillna(""))
similarity_matrix = cosine_similarity(X_a, X_b)  # shape (n_skus, n_descs)

# -------------------------
# helper: map df indices to positional indices in the TF-IDF matrices
sku_pos = {idx: pos for pos, idx in enumerate(df_Skus.index)}
desc_pos = {idx: pos for pos, idx in enumerate(df_desc.index)}

# homologate categories (ensure REV_CATEGORY_MAP exists)
REV_CATEGORY_MAP = {}
for target, sources in CATEGORY_MAP.items():
    for src in sources:
        REV_CATEGORY_MAP[src.lower()] = target

def homologate_category(cat):
    return REV_CATEGORY_MAP.get(str(cat).strip().lower(), str(cat).strip())

df_desc["category_homol"] = df_desc["producto"].apply(homologate_category)
df_Skus["category_homol"] = df_Skus["category"].apply(homologate_category)

# -------------------------
# Main matching loop (vectorized access for tfidf_sim)
THRESH = 0.75  # keep consistent with current pipeline

results = []
for i, desc_row in df_desc.iterrows():
    # Skip rows already assigned a SKU (optional if earlier filter not applied)
    if pd.notna(desc_row.get("sku")) and str(desc_row["sku"]).strip() != "":
        continue

    desc_cat = desc_row["category_homol"]
    desc_entities = desc_row.get('entities') or {}

    # 1) Primary candidates: same homologated category (if configured)
    if desc_cat in CATEGORY_ENTITY_WEIGHTS:
        skus_candidates = df_Skus[df_Skus["category_homol"] == desc_cat]
    else:
        skus_candidates = df_Skus  # unknown category → try all

    if skus_candidates.empty:
        continue

    # positional indices for cosine lookup
    candidate_positions = np.fromiter((sku_pos[idx] for idx in skus_candidates.index), dtype=int)
    dpos = desc_pos[i]

    # vectorized TF-IDF sims against candidates
    tfidf_sims = similarity_matrix[candidate_positions, dpos]

    # entity sims per candidate (list-comprehension for speed)
    weight_cats = (
        np.where(
            desc_cat in CATEGORY_ENTITY_WEIGHTS,
            np.array([desc_cat] * len(skus_candidates), dtype=object),
            skus_candidates["category_homol"].to_numpy()
        )
    )
    ent_sims = np.fromiter(
        (normalized_entity_score(desc_entities, (skus_candidates.at[idx, 'entities'] or {}), wc)
         for idx, wc in zip(skus_candidates.index, weight_cats)),
        dtype=float
    )

    final_scores = 0.75 * ent_sims + 0.25 * tfidf_sims

    # 2) Decide on primary category result
    picked = False
    if final_scores.size and final_scores.max() >= THRESH:
        best_pos = int(np.argmax(final_scores))
        best_sku_idx = skus_candidates.index[best_pos]
        best_row = df_Skus.loc[best_sku_idx]
        df_desc.at[i, "sku"] = best_row["sku"]
        df_desc.at[i, "matched_description"] = best_row.get("descripcion", "")
        df_desc.at[i, "similarity"] = float(final_scores[best_pos])
        picked = True

    # 3) Fallback: try all remaining categories if nothing cleared the threshold
    if not picked:
        # exclude the (possibly wrong) desc_cat to avoid repeating the same set
        if desc_cat in CATEGORY_ENTITY_WEIGHTS:
            fallback_cands = df_Skus[df_Skus["category_homol"] != desc_cat]
        else:
            # when desc_cat not configured we already tried all; no fallback benefit
            fallback_cands = pd.DataFrame()

        if not fallback_cands.empty:
            fb_positions = np.fromiter((sku_pos[idx] for idx in fallback_cands.index), dtype=int)
            # reuse same dpos
            tfidf_sims_fb = similarity_matrix[fb_positions, dpos]

            # weight category comes from SKU-side for fallback
            fb_weight_cats = fallback_cands["category_homol"].to_numpy()
            ent_sims_fb = np.fromiter(
                (normalized_entity_score(desc_entities, (fallback_cands.at[idx, 'entities'] or {}), wc)
                 for idx, wc in zip(fallback_cands.index, fb_weight_cats)),
                dtype=float
            )

            final_scores_fb = 0.75 * ent_sims_fb + 0.25 * tfidf_sims_fb

            if final_scores_fb.size and final_scores_fb.max() >= THRESH:
                best_pos = int(np.argmax(final_scores_fb))
                best_sku_idx = fallback_cands.index[best_pos]
                best_row = df_Skus.loc[best_sku_idx]
                df_desc.at[i, "sku"] = best_row["sku"]
                df_desc.at[i, "matched_description"] = best_row.get("descripcion", "")
                df_desc.at[i, "similarity"] = float(final_scores_fb[best_pos])
                
                # Overwrite the (possibly wrong) category_homol in df_desc with the matched SKU category
                df_desc.at[i, "category_homol"] = best_row["category_homol"]
                df_desc.at[i, "category_corrected"] = True
            # else: leave blank (no SKU meets threshold)

df_desc.to_csv('df_desc.csv', index=False)
