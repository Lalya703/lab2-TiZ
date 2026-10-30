# Корсеева Доминика МТ-201: TF-IDF и косинусное сходство
# Критерий 4: матрица признаков по корпусу; запрос в то же пространство; сходство и ранжирование.
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# берём препроцессинг из предыдущих пунктов, чтобы не писать заново
from p2_text import make_lem, prep, full_text
from p3_base import TOP


class Engine:
    def __init__(self, docs, use_lem, drop_stop, ng):
        # лемматизатор создаём один раз - на каждый документ это было бы долго
        self.lem = make_lem()
        # запоминаем настройки препроцессинга, чтобы так же обработать запрос
        self.args = (use_lem, drop_stop)
        # id храним отдельно - по ним возвращаем результаты поиска
        self.ids = docs["document_id"].tolist()
        # собираем текст документа и прогоняем через prep, как и запрос потом
        self.texts = [" ".join(prep(t, self.lem, *self.args)) for t in full_text(docs)]
        # токены бьём по пробелам (текст уже нормализован), lowercase=False — иначе
        # векторайзер снова приведёт всё к нижнему регистру, а нам это не нужно
        self.vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, ng))
        # fit делаем только на документах, чтобы idf считался честно по корпусу
        self.mat = self.vec.fit_transform(self.texts)

    def search(self, query, k=TOP):
        """TOP-k документов: [(document_id, косинусное сходство)]."""
        # запрос обрабатываем тем же препроцессингом, что и документы
        q = " ".join(prep(query, self.lem, *self.args))
        # если после чистки ничего не осталось - сразу выходим
        if not q:
            return []
        # transform, а не fit_transform: запрос кидаем в уже готовое пространство,
        # иначе словарь признаков перестроится и сходство будет бессмысленным
        qv = self.vec.transform([q])
        # косинус между запросом и каждым документом, ravel - чтобы массив был плоским
        sc = cosine_similarity(qv, self.mat).ravel()
        # сортируем по убыванию; stable - чтобы при равных score порядок не прыгал
        order = np.argsort(-sc, kind="stable")[:k]
        # возвращаем пары (id, score), score приводим к float
        return [(self.ids[i], float(sc[i])) for i in order]