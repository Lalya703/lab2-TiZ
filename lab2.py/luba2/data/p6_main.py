# Богданова Владислава МТ-201: анализ ошибок, сохранение результатов, запуск и воспроизводимость
# Критерий 6 (часть): разбор ошибок поиска. 
# Критерий 7: скрипт запускается целиком.

import argparse                   # обработка аргументов командной строки
from pathlib import Path          # инструмент для работы с путями файлов

import numpy as np                # эффективная работа с массивами чисел (для графика)
import pandas as pd               # работа с таблицами

import matplotlib.pyplot as plt   # для рисования графика

from p1_load import load, explore
from p3_base import base_search
from p5_compare import CONF, compare

# выбирает лучшую конфигурацию по MRR@5, затем по Hit@5.
def pick_best(metrics):
    filtered = metrics[metrics["configuration"].isin(CONF)]
    best = filtered.sort_values(["mrr_at_5", "hit_at_5"], ascending=False).iloc[0]
    return best["configuration"]

# сохранение лучшие 5 результатов поиска для всех запросов
def save_results(queries, engine, out_dir):
    # используем генератор списка для более чистого и быстрого кода
    rows = [
        {
            "query_id": q["query_id"], 
            "rank": rank, 
            "document_id": doc, 
            "score": round(score, 6)
        }
        for _, q in queries.iterrows()
        for rank, (doc, score) in enumerate(engine.search(q["query"]), 1)
    ]
    # превращаем список словарей в таблицу и сохраняем в CSV
    pd.DataFrame(rows).to_csv(out_dir / "search_results.csv", index=False, encoding="utf-8-sig") 

# сохраняем плохие запросы
def save_errors(errors_df, config_name, out_dir):
    bad = errors_df[errors_df["position"] != 1].copy() # оставляем только те запросы, где нужный документ не на первом месте 
    bad.insert(0, "configuration", config_name)
    bad.to_csv(out_dir / "errors.csv", index=False, encoding="utf-8-sig")
    return bad

# рисуем график
def draw(metrics, out_dir):
    plt.switch_backend('Agg')  # неинтерактивный режим (важно для серверов/скриптов)
    
    x = np.arange(len(metrics))
    plt.figure(figsize=(6, 3.5))
    
    plt.bar(x - 0.2, metrics["hit_at_5"], 0.4, label="Hit@5")
    plt.bar(x + 0.2, metrics["mrr_at_5"], 0.4, label="MRR@5")
    
    plt.xticks(x, metrics["configuration"], rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Метрики по 8 открытым запросам")
    plt.tight_layout()
    
    plt.savefig(out_dir / "metrics.png", dpi=150)
    plt.close()  # Освобождаем память

# главное
def main():
    # «помощника», который будет читать аргументы из командной строки
    parser = argparse.ArgumentParser(description="Оркестратор запуска поиска и оценки")
    parser.add_argument("folder", help="Папка с данными варианта")
    parser.add_argument("-o", "--out", default=".", help="Папка для сохранения результатов")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. загрузка и разведочный анализ
    docs, queries, rel = load(args.folder)
    explore(docs, queries, rel)
    
    # 2. базовый поиск (пустой запрос)
    print("Пустой запрос ->", base_search([], [], []))
    
    # 3. сравнение конфигураций
    metrics, engines, errors = compare(docs, queries, rel)
    metrics.to_csv(out_dir / "experiment_metrics.csv", index=False, encoding="utf-8-sig")
    print("\nМетрики:")
    print(metrics.to_string(index=False))
    
    # 4. выбор лучшей конфигурации и сохранение её результатов
    best_config = pick_best(metrics)
    print(f"\nВыбрана конфигурация: {best_config}")
    
    save_results(queries, engines[best_config], out_dir)
    bad_queries = save_errors(errors[best_config], best_config, out_dir)
    
    print("\nОшибки (нужный документ не на 1-м месте):")
    print(bad_queries[["query_id", "expected", "position"]].to_string(index=False))
    
    # 5. визуализация
    draw(metrics, out_dir)
    print(f"\nРезультаты и графики успешно сохранены в папку: {out_dir.resolve()}")

if __name__ == "__main__":
    main()