import multiprocessing
import pickle
import re
import threading
from datetime import datetime
from enum import Enum

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from pymongo import MongoClient

from tfg.utils import guardar_objeto, cargar_objeto, create_folder


class CacheMode(Enum):
    NONE = 0
    ALL = 1
    FILTERED = 2
    ANONYMIZED = 3
    CLEANED = 4
    PROCESSED = 5


class Origin(Enum):
    MONGO = 'MONGO'
    CACHE = 'CACHE'


class DataBase:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="tfg_database", collection="anonymized_posts"):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection
        # Establecer conexión a la base de datos Mongo
        self.client = MongoClient(self.uri)
        # Obtener los documentos de una colección específica
        self.db = self.client[self.db_name]

    def get_dataframe(self, collection="anonymized_posts", use_cache=False):
        inicio = datetime.now()
        if use_cache:
            df = load_cache(CacheMode.ALL)
            if df is not None:
                return df
        self.collection_name = collection
        docs = self.db[self.collection_name].find()
        print("Cargando datos desde la base de datos...")
        # Convertir los resultados en un DataFrame de pandas
        df = pd.DataFrame(list(docs))
        df = df.rename({"contenido": "text", "hilo": "thread", "nombreUsuario": "author"}, axis='columns')
        # df = anonymize_authors(df)
        save_cache(df, CacheMode.ALL)
        fin = datetime.now()
        print(f"Recuperados {len(df)} documentos de la colección {self.collection_name} en {(fin - inicio).total_seconds():.3f} segundos.\n")
        return df

    def insert_one(self, obj, collection=None):
        collection = collection or self.collection_name
        return self.db[collection].insert_one(obj)

    def insert_many(self, objs, collection=None):
        collection = collection or self.collection_name
        return self.db[collection].insert_many(objs.to_dict("records"))

    def get_object(self, id_):
        obj = self.db[self.collection_name].find_one({"_id": id_})
        return obj

    def get_objects(self, collection=None):
        collection = collection or self.collection_name
        objs = self.db[collection]
        return objs

    def anonymize_and_save(self, df: DataFrame, collection="anonymized_dataset"):
        print("Anonymizing authors and documents...")
        adf = anonymize_authors(df, prefix="AUTHOR_")
        print("End of anonymization")
        # Insertar los objetos en la colección
        print(f"Saving anonymized dataset in collection {collection}...")
        many = self.db[collection].insert_many(adf.to_dict("records"))
        print(f"Saved {len(many.inserted_ids)} documents in collection {collection}")
        return many

    def anonymize_and_save_parallel(self, df: DataFrame, collection="anonymized_dataset"):
        print(f"[{threading.current_thread().name}] Anonymizing authors and documents...")
        # Recuperar todos los nombres de los autores
        authors = df["author"].unique()
        print(f"[{threading.current_thread().name}] Generando bloques de datos para el procesamiento paralelo...")
        params = [(df[i:i + 1000], authors, collection) for i in range(0, len(df), 1000)]
        print(f"[{threading.current_thread().name}] Iniciando el procesamiento paralelo de los datos...")
        # Crear un Pool de procesos
        pool = multiprocessing.Pool()
        # Llamar a la función 'mi_algoritmo' con cada parámetro en paralelo
        pool.map(anonymize_specific_authors, params)
        # Cerrar el Pool de procesos
        pool.close()
        pool.join()
        print(f"[{threading.current_thread().name}] Fin - total insertados: {len(df)}")

    def anonymize_main_dataset(self):
        print("Anonymizing main dataset...")
        df = self.get_dataframe()
        print(f"Dataset size: {len(df)}")
        self.anonymize_and_save(df, collection="anonymized_posts")
        print("End of anonymization of main dataset")

    def add_experiment(self, experiment: dict, collection="experiments"):
        experiment["date"] = datetime.now()
        # experiment["_id"] = experiment.pop("id_experiment")
        return self.db[collection].insert_one(experiment)


def _cache_file(cache_mode):
    file = "all_dataset.pkl" if cache_mode == CacheMode.ALL else "dataset.pkl"
    file = "cleaned_dataset.pkl" if cache_mode == CacheMode.CLEANED else file
    file = "filtered_dataset.pkl" if cache_mode == CacheMode.FILTERED else file
    file = "anonymized_dataset.pkl" if cache_mode == CacheMode.ANONYMIZED else file
    file = "processed_dataset.pkl" if cache_mode == CacheMode.PROCESSED else file
    return file


def save_cache(df, cache_mode: CacheMode):
    inicio = datetime.now()
    print("Guardando datos en el disco...")
    file = _cache_file(cache_mode)
    guardar_objeto(df, file, carpeta="tmp")
    fin = datetime.now()
    # imprimir tiempo con dos decimales
    print(f"Tiempo de guardado: {(fin - inicio).total_seconds():.3f} - [{file}]\n")


def load_cache(cache_mode):
    inicio = datetime.now()
    print("Cargando datos desde el disco...")
    file = _cache_file(cache_mode)
    objeto = cargar_objeto(f"tmp/{file}")
    fin = datetime.now()
    print(f"Tiempo de carga: {(fin - inicio).total_seconds():.3f} - [{file}]\n")
    return objeto


def test():
    mdb = DataBase("mongodb://localhost:27017", "test", "test")
    dic = cargar_objeto("spanish_words.dic")
    mdl = cargar_objeto("../modelos/red_neuronal-loss-0.00028787031777482904.mdl")

    objeto = dict(_id="test", dic=dic, mdl=pickle.dumps(mdl))
    # print(mdb.insert_object(objeto))

    read = mdb.get_object("test")
    read["mdl"] = pickle.loads(read["mdl"])
    print(read["dic"])
    # print(dic)


def anonymize_specific_authors(datos: DataFrame, authors: ndarray,
                               cls="author",
                               doc="text",
                               prefix="AUTHOR_"):
    df = datos.copy()
    # authors = authors.tolist()
    print(f"[{threading.current_thread().name}] Prefix: {prefix}")
    print(f"[{threading.current_thread().name}] Anonymize {len(authors)} Authors in {len(df)} documents...")
    # Para cada autor sustituir su nombre por 'author' y un número, y buscar y cambiar su nombre en todos los textos
    for i, auth in enumerate(authors):
        # print(f"[{threading.current_thread().name}] Changing author {auth} to {prefix + str(i)}")
        # definir regex que busque y sustituya todas las coincidencias con el nombre insensible a mayúsculas o minúsculas
        df[doc] = df[doc].apply(lambda x, j=i, a=auth, p=prefix: re.sub(r'(?i)\b' + re.escape(a) + r'\b', p + str(j), x))
        df[cls] = df[cls].apply(lambda x, j=i, a=auth, p=prefix: x.replace(a, p + str(j)))
        if i % 100 == 0:
            print(f"[{threading.current_thread().name}] Authors to anonymize: {len(authors) - i} of {len(authors)}")
    print(f"[{threading.current_thread().name}] End of anonymization")

    return df
    # db = DataBase()
    # # Insertar los objetos en la colección
    # print(f"[{threading.current_thread().name}] Saving anonymized dataset in collection {collection}...")
    # many = db.db[collection].insert_many(df.to_dict("records"))
    # print(f"[{threading.current_thread().name}] Saved {len(many.inserted_ids)} documents in collection {collection}")


def anonymize_authors(datos: DataFrame,
                      cls="author",
                      doc="text",
                      prefix="A",  # "AUTHOR" #
                      seleccionar_para_grafica=False):
    df = datos.copy()
    print(f"Prefix: {prefix}")
    # Recuperar todos los nombres de los autores
    authors = df[cls].unique()
    # Para cada autor sustituir su nombre por 'author' y un número, y buscar y cambiar su nombre en todos los textos
    for i, auth in enumerate(authors):
        print(f"Changing author {auth} to {prefix + str(i)}")
        # definir regex que busque y sustituya todas las coincidencias con el nombre insensible a mayúsculas o minúsculas
        df[doc] = df[doc].apply(lambda x, j=i, a=auth, p=prefix: re.sub(r'(?i)\b' + a + r'\b', p + str(j), x))
        df[cls] = df[cls].apply(lambda x, j=i, a=auth, p=prefix: x.replace(a, p + str(j)) if a == x else x)
        # df[cls] = df[cls].apply(lambda x, j=i, a=auth, p=prefix: x.replace(a, p + str(j)))
        if (i+1) % 25 == 0:
            print(f"Authors to anonymize: {len(authors) - i} of {len(authors)}")
    # Ordenar por nombre de autor
    # df = df.sort_values(by=[cls])

    if seleccionar_para_grafica:
        # Dejar solo 10 archivos del autor 0 y 90 del resto
        author_zero = prefix + "0"
        no_author0: DataFrame = df[df[cls] != author_zero]
        author0: DataFrame = df[df[cls] == author_zero]
        # Seleccionar aleatoriamente 20 documentos del autor Sendero
        no_author0 = no_author0.sample(n=10)
        author0 = author0.sample(n=90)
        # # Concatenar los dos DataFrames
        df = pd.concat([no_author0, author0])
        df = df.sample(frac=1)

    return df


def get_experiment_id():
    return f"EXPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
