import array
from typing import Dict

from tfg.datos import *
from tfg.textmining.preparing import *
from tfg.textmining.preprocessing import *


# pip install --upgrade pip
# pip install pymongo
# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install scikit-learn
# pip install nltk
# pip install requests
# pip install bs4
# pip install unidecode

# # Atribución Automática de Autoría de publicaciones en foros
#
# #### Solución de software para la asignatura de TFG del Grado en Ingeniería Informática de la UNIR
#
# ###### Autor: José Herce Preciado


def experimento(  # ## PARÁMETROS CONFIGURACIÓN EXPERIMENTO
        ###############################################################################
        # Manipulación de los datos
        ###############################################################################
        # "local" o "mongo" - Si "local" se cargará el dataset desde el disco. Si "mongo" se cargará desde la base de datos.
        origen_datos=Origin.MONGO,
        # Si True, se guardará el dataset preprocesado en el disco.
        save_preprocessed_temp=False,
        # Anonimización de los datos - Si True, se guardará el dataset anonimizado en la base de datos
        # Si False, se anonimizarán los datos pero no se guardarán en la base de datos
        anonymize_and_save=False,
        # Nombre de la colección en la que se guardarán los datos anonimizados
        collection="anonymized_posts",  # Nombre de la colección del data set
        ###############################################################################
        # Preprocesamiento de los datos
        ###############################################################################
        n_authors=3,  # 200,
        min_words=50,  # 30,
        max_words=None,  # None,
        min_doc_per_author=None,  # 100,
        # None # 1000 # Si longest_docs=False se dará error en caso de no existir documentos suficientes para algún autor.
        max_doc_per_author=None,
        # True: Seleccionar los documentos más largos. False: Seleccionar los documentos aleatoriamente.
        longest_docs=True,
        # Estadísticas y gráficas
        show_stats=False,
        # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        num_bars='auto',
        ###############################################################################
        # Limpieza de los datos
        ###############################################################################
        my_tech=None,
        stop_words=True,
        max_df=0.8,  # Palabras muy frecuentes
        min_df=1,  # Palabras poco frecuentes
        strip_accents=True,
        lowercase=True,
        # True: Tokenizar; False: Eliminar; None: No hacer nada
        tokenize_emoticons=False,
        tokenize_numbers=False,
        tokenize_simbols=False,
        tokenize_punctuation=False,
        tokenize_urls=False,
        ###############################################################################
        # Tokenización de los datos
        ###############################################################################
        ngram_range=(1, 3),
        binarize=False,
        balance='all_set',  # 'all_set' 'sub_set'
        balance_to_target=False,
        target=None,  # Clase objetivo binarización
        balance_passive=True,
        ###############################################################################
        # Extracción de características de los datos
        ###############################################################################
        test_size=0.3,
        # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf). Default: False.
        sublinear_tf=True,
        # Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once. Prevents zero divisions. Default: True.
        smooth_idf=True,
        # Enable inverse-document-frequency reweighting. If False, idf(t) = 1. Default: True.
        use_idf=True,
        # TODO No propagada al Vectorizador
        norm='l2',  # Each output row will have unit norm, either: Default: 'l2'
        #    'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product when l2 norm has been applied.
        #    'l1': Sum of absolute values of vector elements is 1. See preprocessing.normalize.
        #    None: No normalization.
        ###############################################################################
        # Clasificación de los datos
        ###############################################################################
        description="No description",
        tags=None,
        classifiers_list: array = None,
        use_cache=True,
        verbose=False,
) -> Dict:
    classifiers_list = check_classifiers_list(classifiers_list)
    tags = check_tags(tags)
    if my_tech is not None:
        if my_tech:
            stop_words = False
            strip_accents = False
            lowercase = False
            # True: Tokenizar; False: Eliminar; None: No hacer nada
            tokenize_emoticons = True
            tokenize_numbers = True
            tokenize_simbols = True
            tokenize_punctuation = True
            tokenize_urls = True
        else:
            stop_words = True
            strip_accents = True
            lowercase = True
            # True: Tokenizar; False: Eliminar; None: No hacer nada
            tokenize_emoticons = False
            tokenize_numbers = False
            tokenize_simbols = False
            tokenize_punctuation = False
            tokenize_urls = False

    # Guardar parámetros de configuración en un diccionario
    params = locals().copy()
    params['origen_datos'] = params['origen_datos'].value
    metadata = dict(_id=get_experiment_id())
    metadata.update(params=params)
    metadata["description"] = metadata["params"].pop("description")
    metadata["tags"] = metadata["params"].pop("tags")
    print("\n", description)
    db = DataBase(collection=collection)

    # ## Minería de texto

    # ### Fase de recolección de datos

    if origen_datos == Origin.CACHE:
        df_all = load_cache(CacheMode.ALL)
    elif origen_datos == Origin.MONGO:
        df_all = db.get_dataframe(collection=collection, use_cache=use_cache)
        # save_temp_dataset(df_all)
    else:
        raise ValueError(
            "El parámetro origen_datos debe ser 'local' o 'mongo'")
    df_all.info() if verbose else None

    df_all.count()  # if verbose else None

    # ### Fase de preprocesamiento

    df_all = eliminar_columnas(
        df_all, ["_id", "fecha", "numeroEntrada", "cita", "url", "tipoUsuario", "thread"])
    df_all.info() if verbose else None

    # _ = show_statistics(df_all, title="Estadísticas del dataset completo",
    #                     num_bars='auto',  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
    #                     hist=False)

    ###############################################################################
    # Seleccionar dataset
    ###############################################################################
    datos, metadata = select_documents(
        df_all,
        n_authors=n_authors,  # 200,
        min_words=min_words,  # 30,
        max_words=max_words,  # None,
        min_doc_per_author=min_doc_per_author,  # 100,
        max_doc_per_author=max_doc_per_author,
        # 1000,  # Si longest_docs=False se dará error en caso de no existir documentos suficientes para algún autor.
        longest_docs=longest_docs,
        # True: Seleccionar los documentos más largos. False: Seleccionar los documentos aleatoriamente.
        show_stats=show_stats,
        # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
        num_bars=num_bars,
        metadata=metadata,
    )
    # metadata["selected_documents_statistics"] = md
    # metadata = dict(selected_documents_statistics=metadata)

    _ = show_statistics(datos, title="Estadísticas del dataset seleccionado",
                        # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                        num_bars='auto',
                        hist=False) if verbose else None

    # Anonimizar autores
    if anonymize_and_save:
        db = DataBase()
        anonymized_df_all = db.anonymize_and_save(
            datos, collection="all_anonymized_posts")
        # anonymized_df_all = anonymize_authors(df_all)
        datos = anonymized_df_all
    # elif origen_datos == "mongo":
    datos = anonymize_authors(datos)
    # # Guardar datos preprocesados en el disco
    # if origen_datos == "mongo":
    #     save_temp_dataset(datos)

    author_label = "author"
    doc_author_counts = datos[author_label].value_counts()
    print("Número de autores: {}".format(len(doc_author_counts)))
    word_author_counts = datos.groupby(author_label)['text'].apply(
        lambda x: x.str.split().str.len().sum())
    print("Número de documentos: {}".format(len(datos)))
    word_doc_counts = datos['text'].str.split().str.len()
    print("Número de palabras: {}".format(word_doc_counts.sum()))

    if verbose:
        # Cuantos autores tienen cero palabras
        print("Número de autores con cero palabras: {}".format((word_author_counts == 0).sum()))
        # Cuantos autores tienen cero documentos
        print("Número de autores con cero documentos: {}".format((doc_author_counts == 0).sum()))
        # Cuantos documentos tienen cero palabras
        print("Número de documentos con cero palabras: {}".format((word_doc_counts == 0).sum()))

    print("Número de documentos de cada autor:") if verbose else None
    # Utilizar la función value_counts() para contar la frecuencia de aparición de cada autor
    author_counts = datos['author'].value_counts()
    metadata.update({"total_documents_per_author": author_counts.to_dict()})
    # Imprimir los resultados
    print("\nNúmero de documentos por autor:", author_counts, sep="\n")

    # ### Fase de limpieza de texto

    print("Antes de limpiar el corpus:") if verbose else None
    print(datos['text'][0:12].reset_index(drop=True), "\n") if verbose else None
    print("Limpiando corpus...")
    datos['text'], args = clean_corpus(datos['text'],
                                       tokenize_emoticons=tokenize_emoticons,
                                       tokenize_numbers=tokenize_numbers,
                                       tokenize_simbols=tokenize_simbols,
                                       tokenize_punctuation=tokenize_punctuation,
                                       tokenize_urls=tokenize_urls)

    metadata.update({'args_clean_corpus': args})
    print(datos['text'][0:12].reset_index(drop=True)) if verbose else None

    # ## Vectorizar, TF-IDF y generar conjuntos de test y entrenamiento

    # ##########################################################
    # Vectorizar y generar los conjuntos de entrenamiento y test
    # ##########################################################
    X_train, X_test, y_train, y_test, feature_names, target_names, md = process_dataset(
        datos,
        class_label="author",
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        stop_words=stop_words,
        strip_accents=strip_accents,
        lowercase=lowercase,
        binarize=binarize,
        balance=balance,  # 'all_set', 'sub_set'
        balance_to_target=balance_to_target,
        target=target,
        sublinear_tf=sublinear_tf,
        smooth_idf=smooth_idf,
        test_size=test_size,
        balance_passive=balance_passive,
        verbose=verbose,
    )
    metadata.update({'metadata_process_dataset': md})

    # Imprimir los documentos de cada autor en el conjunto de entrenamiento
    print("[TRAINING] Número de documentos de cada autor:")
    print(y_train.value_counts(), "\n")
    # Imprimir los documentos de cada autor en el conjunto de test
    print("[TEST] Número de documentos de cada autor:")
    print(y_test.value_counts(), "\n")

    # Tamaño del las características
    print("Tamaño del las características: {:,}".format(len(feature_names)))

    from tfg.modelos import classifiers

    classifiers(X_train, y_train, X_test, y_test,
                classifiers_list, metadata, gen_conf_matrix=True)

    return metadata


def check_tags(tags):
    if tags is None:
        return 'experiment'
    return tags


def check_classifiers_list(classifiers_list):
    if classifiers_list is None:
        classifiers_list = [
            'RidgeClassifier',
            'LogisticRegression',
            'BernoulliNB',
            'GaussianNB',
            'NearestCentroid',
            'LinearSVC',
            'ExtraTreeClassifier',
            'NuSVC',
            'SVC',
            'RandomForestClassifier',
            'DecisionTreeClassifier',
            'GradientBoostingClassifier',
            'LogisticRegressionCV',
            'RidgeClassifierCV',
            'MLPClassifier',
        ]
    return classifiers_list


def session():
    results = dict()
    authors = 3
    max_docs = 100
    min_words = 0
    first = True
    ngramas = (1, 1)
    for my_tech in [True, False]:  #
        import winsound
        # Reproducir un pitido corto
        winsound.Beep(440, 200)  # Frecuencia: 440Hz, Duración: 200 milisegundos
        for max_docs in [100, 200, 300]: #, (1, 2), (1, 3), (1, 4)]:  #
            for max_words in range(0, 501, 100):
                result = experimento(
                    description=f"{authors} autores - {min_words} palabras mínimas - "
                                f"{max_words} palabras máximas - "
                                f"{max_docs} documentos máximos por autor - "
                                f"n-gramas hasta {ngramas[1]} - "
                                f"myTech-{my_tech}",
                    origen_datos=Origin.MONGO if first else Origin.CACHE,
                    use_cache=not first,
                    n_authors=authors,
                    min_words=min_words,
                    max_words=max_words,
                    max_doc_per_author=max_docs,
                    longest_docs=False,
                    test_size=0.5,
                    my_tech=my_tech,
                    ngram_range=ngramas,
                    classifiers_list=[
                        'RidgeClassifier',
                        'SVC',
                        # 'LinearSVC',
                        # 'NearestCentroid',
                        'MLPClassifier',
                        # 'RandomForestClassifier',
                    ],
                    tags="maxwords_x_100_200_300_docs",
                    verbose=True
                )
                first = False
                _id = result.get("_id")
                # results.update({f"{_id}": result})
                # results.update({f'{result.get("_id")}': result})
                results[_id] = result
                print(f"Experimento {_id} finalizado")


    return results


if __name__ == "__main__":
    # ejecutar en con diferentes parámetros
    session()
