# from tfg.datos import DataBase, load_temp_dataset, save_temp_dataset, anonymize_specific_authors
# import multiprocessing
# import threading
#
# from pandas import DataFrame
#
# from tfg.textmining.preprocessing import show_statistics
#
# ###############################################################################
# # Manipulación de los datos
# ###############################################################################
# origen_datos = "mongo"  # "local" o "mongo"
# from tfg.textmining.preprocessing import eliminar_columnas
# from tfg.textmining.preprocessing import select_documents
#
# save_preprocessed_temp = True  # Si True, se guardará el dataset preprocesado en el disco
# # Anonimización de los datos - Si True, se guardará el dataset anonimizado en la base de datos
# anonymize_and_save = True  # Si False, se anonimizarán los datos pero no se guardarán en la base de datos
# collection_name = "anonimizado_util"  # Nombre de la colección en la que se guardarán los datos anonimizados
#
# ###############################################################################
# # Preprocesamiento de los datos
# ###############################################################################
# n_authors = None  # 200,
# min_words = 1  # 30,
# max_words = None  # None,
# min_doc_per_author = 1  # 100,
# max_doc_per_author = None  # 1000,  # Si longest_docs=False se dará error en caso de no existir documentos suficientes para algún autor.
# longest_docs = True  # True: Seleccionar los documentos más largos. False: Seleccionar los documentos aleatoriamente.
# # Estadísticas y gráficas
# show_stats = False
# num_bars = 'auto'  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
#
# ###############################################################################
# # Limpieza de los datos
# ###############################################################################
# stop_words = False
# max_df = 0.8  # Palabras muy frecuentes
# min_df = 1  # Palabras poco frecuentes
# strip_accents = False
# lowercase = False
#
# # True: Tokenizar; False: Eliminar; None: No hacer nada
# tokenize_emoticons = True
# tokenize_numbers = True
# tokenize_simbols = True
# tokenize_punctuation = True
# tokenize_urls = True
#
# ###############################################################################
# # Tokenización de los datos
# ###############################################################################
# ngram_range = (1, 1)
# binarize = False
# balance = ''  # 'all_set' 'sub_set'
# balance_to_target = False
# target = None  # Clase objetivo binarización
#
# ###############################################################################
# # Extracción de características  de los datos
# ###############################################################################
# test_size = 0.3
# sublinear_tf = True  # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf). Default: False.
# smooth_idf = True  # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions. Default: True.
# use_idf = True  # Enable inverse-document-frequency reweighting. If False, idf(t) = 1. Default: True.
#
# # TODO No propagada al Vectorizador
# norm = 'l2'  # Each output row will have unit norm, either: Default: 'l2'
# #    'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied.
# #    'l1': Sum of absolute values of vector elements is 1. See preprocessing.normalize.
# #    None: No normalization.
#
#
# ###############################################################################
# # Clasificación de los datos
# ###############################################################################
# classifiers_list = [
#     # 'BernoulliNB',
#     'RidgeClassifier',
#     # 'LogisticRegression',
#     # 'NearestCentroid',
#     # 'LinearSVC',
#     # 'NuSVC',
#     # 'SVC',
#     # 'RandomForestClassifier',
#     # 'DecisionTreeClassifier',
#     # 'ExtraTreeClassifier',
#     # 'GradientBoostingClassifier',
#     # 'LogisticRegressionCV',
#     # 'RidgeClassifierCV',
#     # 'MLPClassifier',
# ]
#
# if origen_datos == "local":
#     df_all = load_temp_dataset()
# elif origen_datos == "mongo":
#     df_all = DataBase().get_dataframe()
#     save_temp_dataset(df_all)
# else:
#     raise ValueError("El parámetro origen_datos debe ser 'local' o 'mongo'")
# df_all.info()
#
# df_all = eliminar_columnas(df_all, ["_id", "fecha", "numeroEntrada", "cita", "url", "tipoUsuario", "thread"])
# df_all.info()
#
# ###############################################################################
# # Seleccionar los autores más frecuentes
# ###############################################################################
# datos: DataFrame
# datos, metadata = select_documents(
#     df_all,
#     n_authors=n_authors,  # 200,
#     min_words=min_words,  # 30,
#     max_words=max_words,  # None,
#     min_doc_per_author=min_doc_per_author,  # 100,
#     max_doc_per_author=max_doc_per_author,
#     # 1000,  # Si longest_docs=False se dará error en caso de no existir documentos suficientes para algún autor.
#     longest_docs=longest_docs,
#     # True: Seleccionar los documentos más largos. False: Seleccionar los documentos aleatoriamente.
#     show_stats=show_stats,
#     num_bars=num_bars,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
# )
# metadata = dict(selected_documents_statistics=metadata)
#
#
# collection = "anonymized_posts"
# authors = datos["author"].unique()
# print(f"[{threading.current_thread().name}] Generando bloques de datos para el procesamiento...")
# blocks = [datos[i:i + 1000] for i in range(0, len(datos), 1000)]
# print(f"[{threading.current_thread().name}] Iniciando el procesamiento de los datos...")
# # Anonimizar autores
# db = DataBase()
# for i, block in enumerate(blocks):
#     print(f"[{threading.current_thread().name}] BLOCK {i+1} de {len(blocks)}")
#     print(f"[{threading.current_thread().name}] (BLOCK {i+1}) Insertando {len(block)} elementos...")
#     block = anonymize_specific_authors(block, authors)
#     db.insert_many(block, collection="anonymized_all_posts")
#     print(f"[{threading.current_thread().name}] (BLOCK {i+1}) INSERTADOS: {len(block)}")
#
# # db.anonymize_and_save(datos, collection="anonymized_posts")
