# Фролов Павел МТ-202: сравнение четырёх конфигураций A/B/C/D и метрики 
# Критерий 5: конфигурации A-D. 
# Критерий 6: Hit@5 и MRR@5 на 8 открытых запросах.

import numpy as np   # библиотека для работы с числами
import pandas as pd  # для работы с таблицами

from p2_text import make_lem, prep, full_text  
from p3_base import base_search
from p4_tfidf import Engine

# конфигурации: (лемматизация, удалять стоп-слова, максимальная длина n-граммы)
CONF = {
    "A": (False, False, 1),   # без лемм, стоп-слова остаются, униграммы
    "B": (True, False, 1),    # с леммами, стоп-слова остаются, униграммы
    "C": (True, True, 1),     # с леммами, стоп-слова удалены, униграммы
    "D": (True, True, 2),     # с леммами, стоп-слова удалены, униграммы + биграммы
}

# Hit@5, MRR@5 и подробности по каждому размеченному запросу.
def metrics(eng, qs, rel):
    # делаем словарь «какой запрос  "какой документ должен быть найден".
    gold = dict(zip(rel["query_id"], rel["relevant_document_id"]))
    text = dict(zip(qs["query_id"], qs["query"]))
    hit, rr, rows = [], [], []
    for qid, doc in gold.items():
        found = [d for d, _ in eng.search(text[qid])]
        pos = found.index(doc) + 1 if doc in found else 0
        hit.append(1 if pos else 0)
        rr.append(1 / pos if pos else 0)
        rows.append({"query_id": qid, "query": text[qid], "expected": doc,
                     "position": pos, "found_top5": " ".join(found)})
    return float(np.mean(hit)), float(np.mean(rr)), pd.DataFrame(rows)

# метрики для базового поиска
def base_metrics(docs, qs, rel):
    lem = make_lem()
    # предобрабатываем все документы: с лемматизацией и удалением стоп-слов
    dw = [prep(t, lem, True, True) for t in full_text(docs)]
    ids = docs["document_id"].tolist()
    gold = dict(zip(rel["query_id"], rel["relevant_document_id"]))
    text = dict(zip(qs["query_id"], qs["query"]))
    hit, rr = [], []
    for qid, doc in gold.items():
        found = [d for d, _ in base_search(prep(text[qid], lem, True, True), dw, ids)]
        pos = found.index(doc) + 1 if doc in found else 0
        hit.append(1 if pos else 0)
        rr.append(1 / pos if pos else 0)
    return float(np.mean(hit)), float(np.mean(rr))

# прогоняет базовый поиск и A-D, возвращает таблицу метрик, движки и разборы запросов
def compare(docs, qs, rel):
    h, m = base_metrics(docs, qs, rel)
    rows = [{"configuration": "base", "hit_at_5": h, "mrr_at_5": m}]
    engs, errs = {}, {}
    for name, args in CONF.items():
        engs[name] = Engine(docs, *args)
        h, m, det = metrics(engs[name], qs, rel)
        rows.append({"configuration": name, "hit_at_5": h, "mrr_at_5": m})
        errs[name] = det
    return pd.DataFrame(rows).round(4), engs, errs
