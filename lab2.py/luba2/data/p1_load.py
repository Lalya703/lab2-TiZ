# Корпачёва Ульяна МТ-201
# Загружаем корпус и смотрим, что в нём вообще лежит (читаем CSV, смотрим документы и поля)
from pathlib import Path
import pandas as pd

def load(folder):
    # Читаем три файла: документы, запросы и разметку (какие документы подходят к каким запросам)
    # encoding utf-8-sig нужен т.к. в файлах есть BOM (спец. значок в начале), иначе ломаются названия колонок
    # on_bad_lines='skip' - битые строки просто скипаем, иначе ошибка
    f = Path(folder)
    docs = pd.read_csv(f / "documents.csv", encoding="utf-8-sig", on_bad_lines='skip')
    qs = pd.read_csv(f / "queries.csv", encoding="utf-8-sig", on_bad_lines='skip')
    rel = pd.read_csv(f / "dev_relevance.csv", encoding="utf-8-sig", on_bad_lines='skip')
    return docs, qs, rel

def explore(docs, qs, rel):
    # Выводим всякую инфу про данные, чтобы понять, с чем работаем
    # Сколько всего документов, запросов и размеченных пар
    print("Документов:", len(docs), "| запросов:", len(qs), "| размеченных:", len(rel))
    # Какие колонки есть в документах
    print("Поля документов:", list(docs.columns))
    # Какие типы данных у колонок (число, строка и т.д.)
    print("Типы:\n", docs.dtypes.to_string())
    # Считаем пропуски в доках и запросах
    print("Пропуски в документах:", int(docs.isna().sum().sum()),
          "| в запросах:", int(qs.isna().sum().sum()))
    # Проверяем, что ID не повторяются
    print("ID документов уникальны:", docs["document_id"].is_unique,
          "| ID запросов уникальны:", qs["query_id"].is_unique)
    # Ищем документы дубликаты
    print("Дубликатов текста:", int(docs["text"].duplicated().sum()))
    # Ищем пустые тексты или те, где ток пробелы; NaN заменяем на пустую строку, чтоб не было ошибок
    print("Пустых текстов:", int((docs["text"].fillna("").str.strip() == "").sum()))
    # Считаем длину каждого текста в словах и смотрим мин, среднее и макс
    n = docs["text"].str.split().str.len()
    print("Слов в документе: мин", n.min(), "| средн", round(n.mean(), 1), "| макс", n.max())
    # Сколько разделов в корпусе
    print("Разделов:", docs["section"].nunique())
    # Проверяем, что все доки из разметки реально есть в корпусе
    # Берём множество ID из разметки и убираем из доков, что осталось - не надо, лишнее
    miss = set(rel["relevant_document_id"]) - set(docs["document_id"])
    print("Эталонные ID, которых нет в корпусе:", len(miss))