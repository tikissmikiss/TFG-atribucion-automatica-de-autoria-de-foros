import pandas as pd

from pandas import DataFrame

from tfg.graphics import plot_histogram


def select_documents(df: DataFrame, n_authors=None, min_words=None, max_words=None, max_doc_per_author=None, longest_docs=False,
                     min_doc_per_author=0, random_state=1, show_stats=False, num_bars='auto', metadata=None):
    """Selecciona un subconjunto de elementos según los parámetros especificados.

    Parameters
    ----------

    df : DataFrame
        DataFrame con los datos a filtrar.

    n_authors : int, default=0
        Número de autores a seleccionar. Si es 0, no se filtra por autores.

    min_words : int, default=0
        Número mínimo de palabras que debe tener un documento para ser seleccionado.
        Si menor que 1, o None, no se filtra por número de palabras.

    max_words : int, default=None
        Número máximo de palabras que puede tener un documento para ser seleccionado.
        Si menor que 1, o None, no se filtra por número de palabras.

    longest_docs : bool, default=False
        Al seleccionar documentos de cada autor, se seleccionan los más largos o aleatorios.\n
        - Si es True, se seleccionan los documentos más largos de cada autor.\n
        - Si es False, se seleccionan los documentos aleatoriamente.

    max_doc_per_author : int, default=0
        Número de documentos a seleccionar por autor. Si es 0, no se filtra por número de documentos por autor.\n
        Si menor que 1, o None, se seleccionan todos los documentos de cada autor.\n
        Si longest_docs=False, se dará error en caso de no existir documentos suficientes para algún autor.

    min_doc_per_author : int, default=0
        Número mínimo de documentos que debe tener un autor para ser seleccionado.
        Si menor que 1, o None, no se filtra por número de documentos por autor.

    random_state : int, default=1
        Semilla para el generador de números aleatorios.

    show_stats : bool, default=False
        Mostrar información sobre los datos resultantes del filtrado.

    balance_docs : bool, default=False
        Balancear el número de documentos por autor. Sí es True, se resamplean los autores con menos documentos.

    Returns
    -------
    DataFrame
        DataFrame con los datos seleccionados.
    """
    metadata = metadata or dict()
    filtered_df = df.copy().reset_index(drop=True)
    # Si se especifica el número mínimo de palabras, se filtran los documentos que no lo cumplan
    # if min_words and min_words > 0:
    filtered_df = _filter_min_words(filtered_df, min_words) if min_words and min_words > 0 else _filter_min_words(filtered_df, 1)
    # Si se especifica el número máximo de palabras, se filtran los documentos que no lo cumplan
    if max_words is not None and max_words >= 0:
        filtered_df = _filter_max_words(filtered_df, max_words)
    # Si se especifica el número mínimo de documentos por autor, se filtran los autores que no lo cumplan
    if min_doc_per_author and min_doc_per_author > 0:
        filtered_df = _filter_min_doc_per_author(filtered_df, min_doc_per_author)
    if n_authors and n_authors > 0:
        filtered_df = _select_n_authors(filtered_df, n_authors)
    # sí se especifica el número de documentos por autor
    if max_doc_per_author and max_doc_per_author > 0:
        filtered_df = _select_n_doc_per_author(filtered_df, max_doc_per_author, longest_docs, random_state)
    if show_stats:
        metadata["selected_documents_statistics"] = show_statistics(filtered_df)
    else:
        metadata["selected_documents_statistics"] = statistics(filtered_df, num_bars=num_bars)
    return filtered_df, metadata  # .reset_index(drop=True)


def balance_class(X, class_label='author', n_samples=None, replace=False, random_state=1):
    from sklearn.utils import resample
    # Obtener nombres de todas las clases
    class_labels = X[class_label].unique()
    # Si no se establece n_samples, calcular el mayor número de elementos de una clase
    _n_samples = n_samples if n_samples else X[class_label].value_counts().max()
    # Generar un DataFrame vacío
    balanced = DataFrame(columns=X.columns)
    # Para cada clase
    for label in class_labels:
        # Obtener los registros de la clase
        subset = X[X[class_label] == label]
        # Generar registros aleatorios de la clase hasta igualar el número de registros de la clase mayoritaria
        resampled = resample(subset,
                             replace=replace,  # Muestreo con reemplazo
                             n_samples=_n_samples,  # Igual cantidad que la clase mayoritaria
                             random_state=random_state)  # Semilla aleatoria para reproducibilidad
        # Unir el conjunto de datos equilibrado
        balanced = pd.concat([balanced, resampled], axis=0)
    # Mezclar los datos
    mixed = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return mixed


def balance_class_ok(X, class_label='author', replace=True, random_state=1):
    from sklearn.utils import resample
    # Obtener nombres de todas las clases
    class_labels = X[class_label].unique()
    # Calcular el mayor número de elementos de una clase
    max_class_size = X[class_label].value_counts().max()
    # Generar un DataFrame vacío
    balanced = DataFrame(columns=X.columns)
    # Para cada clase
    for label in class_labels:
        # Obtener los registros de la clase
        subset = X[X[class_label] == label]
        # Generar registros aleatorios de la clase hasta igualar el número de registros de la clase mayoritaria
        resampled = resample(subset,
                             replace=replace,  # Muestreo con reemplazo
                             n_samples=max_class_size,  # Igual cantidad que la clase mayoritaria
                             random_state=random_state)  # Semilla aleatoria para reproducibilidad
        # Unir el conjunto de datos equilibrado
        balanced = pd.concat([balanced, resampled], axis=0)
    # Mezclar los datos
    mixed = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return mixed


def show_statistics(df, author_label='author', title="Información sobre los datos del DataFrame", num_bars='auto', hist=False):
    metadata = statistics(df, author_label=author_label, num_bars=num_bars, hist=hist)
    print('*' * 90, f"{title}:", '*' * 90, sep='\n')
    for key, value in metadata.items():
        print(f" - {key:35}: {value:>10,.0f}" if isinstance(value, int) else f" - {key:35}: {value:>13,.2f}")
        # print(f" - {key:35}: " f"{value:,.2f}" f"{value:,.2f}")
    print()
    return metadata


def statistics(posts_df, author_label='author', num_bars='auto', hist=False):
    doc_author_counts = posts_df[author_label].value_counts()
    if hist:
        plot_histogram(doc_author_counts,
                       title="Distribución del número de documentos por autor",
                       x_label="Número de documentos",
                       y_label="Autores",
                       num_bars=num_bars,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                       font_size=12)
    word_author_counts = posts_df.groupby(author_label)['text'].apply(lambda x: x.str.split().str.len().sum())
    if hist:
        plot_histogram(word_author_counts,
                       title="Distribución del número de palabras por autor",
                       x_label="Número de palabras",
                       y_label="Autores",
                       num_bars=num_bars,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                       font_size=12)
    word_doc_counts = posts_df['text'].str.split().str.len()
    if hist:
        plot_histogram(word_doc_counts,
                       title="Distribución del número de palabras por documento",
                       x_label="Número de palabras",
                       y_label="Documentos",
                       num_bars=num_bars,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                       font_size=12)
    doc_author_counts_std = float(doc_author_counts.std())
    doc_author_counts_mean = float(doc_author_counts.mean())
    cv_doc_author_counts = float(doc_author_counts_std / doc_author_counts_mean * 100)
    word_author_counts_std = float(word_author_counts.std())
    word_author_counts_mean = float(word_author_counts.mean())
    cv_word_author_counts = float(word_author_counts_std / word_author_counts_mean * 100)
    word_doc_counts_std = float(word_doc_counts.std())
    word_doc_counts_mean = float(word_doc_counts.mean())
    cv_word_doc_counts = float(word_doc_counts_std / word_doc_counts_mean * 100)
    return {
        "Número de autores": posts_df[author_label].nunique(),
        "Número de documentos": posts_df.shape[0],
        "Número de palabras": int(posts_df['text'].apply(lambda x: len(x.split())).sum()),
        "Máximo de documentos de un autor": int(doc_author_counts.max()),
        "Mínimo de documentos de un autor": int(doc_author_counts.min()),
        "Media de documentos por autor": doc_author_counts_mean,
        # "Varianza de documentos por autor": doc_author_counts.var(),
        "SD de documentos por autor": doc_author_counts_std,
        "CV de documentos por autor (%)": cv_doc_author_counts,  # Coeficiente de variacion
        "Máximo de palabras de un autor": int(word_author_counts.max()),
        "Mínimo de palabras de un autor": int(word_author_counts.min()),
        "Media de palabras por autor": word_author_counts_mean,
        # "Varianza de palabras por autor": word_author_counts.var(),
        "SD de palabras por autor": word_author_counts_std,
        "CV de palabras por autor (%)": cv_word_author_counts,
        "Máximo de palabras de un documento": int(word_doc_counts.max()),
        "Mínimo de palabras de un documento": int(word_doc_counts.min()),
        "Media de palabras por documento": word_doc_counts_mean,
        # "Varianza de palabras por documento": word_doc_counts.var(),
        "SD de palabras por documento": word_doc_counts_std,
        "CV de palabras por documento (%)": cv_word_doc_counts
    }


def _select_n_doc_per_author_old(filtered_df, n_doc_per_author, longest_docs, random_state):
    # agregar una columna con el número de palabras de cada documento
    filtered_df['num_palabras'] = filtered_df['text'].apply(lambda x: len(x.split()))
    # Si seleccionado seleccionar los n documentos de cada autor con mayor número de palabras
    if longest_docs:
        # ordenar los documentos dentro de cada grupo por número de palabras
        filtered_df = filtered_df.sort_values(['author', 'num_palabras'], ascending=[True, False])
        # seleccionar los primeros n documentos de cada grupo
        filtered_df = filtered_df.groupby('author').head(n_doc_per_author)
    else:
        # seleccionar n documentos aleatorios de cada autor
        filtered_df = filtered_df.groupby('author') \
            .apply(lambda x: x.sample(n_doc_per_author, random_state=random_state, replace=False))
        # seleccionar solo los elementos con el campo "text" no repetido - Necesario si replace=True
        # filtered_df = filtered_df[filtered_df['text'].duplicated() == False]
    return filtered_df


def _select_n_doc_per_author(df: DataFrame, max_doc_per_author, longest_docs, random_state):
    # agregar una columna con el número de palabras de cada documento
    df['num_palabras'] = df['text'].apply(lambda x: len(x.split()))
    # Si seleccionado seleccionar los n documentos de cada autor con mayor número de palabras
    if longest_docs:
        # ordenar los documentos dentro de cada grupo por número de palabras
        df = df.sort_values(['author', 'num_palabras'], ascending=[True, False])
        # seleccionar los primeros n documentos de cada grupo
        df = df.groupby('author').head(max_doc_per_author)
        # mezclar el dataframe
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        # seleccionar hasta n documentos aleatorios de cada autor
        df = df.groupby('author') \
            .apply(lambda x: x.sample(min(len(x), max_doc_per_author), random_state=random_state))
        # resetear los índices
        df = df.reset_index(drop=True)
        # seleccionar solo los elementos con el campo "text" no repetido - Necesario si replace=True
        # filtered_df = filtered_df[filtered_df['text'].duplicated() == False]
    return df


def _select_n_authors(filtered_df, n_authors):
    # agrupamos los datos por autor y contamos el número de registros de cada uno
    author_counts = filtered_df.groupby('author').size().reset_index(name='counts')
    author_counts = author_counts.sort_values(by='counts', ascending=False)
    # seleccionamos los autores con mayor número de registros
    top_authors = author_counts.head(n_authors)['author']
    # Filtramos las filas del dataframe que pertenecen a los autores seleccionados
    filtered_df = filtered_df[filtered_df['author'].isin(top_authors)]
    return filtered_df


def _filter_min_doc_per_author(filtered_df, min_doc_per_author):
    # Contar las veces que aparece cada nombreUsuario en la columna "nombreUsuario".
    author_counts = filtered_df['author'].value_counts()
    # Seleccionar los author con más apariciones.
    n_authos = author_counts[author_counts >= min_doc_per_author]
    # Utilizar una condición para seleccionar todas las filas correspondientes a los author seleccionados.
    filtered_df = filtered_df[filtered_df['author'].isin(n_authos.index)]
    return filtered_df


def _filter_min_words(filtered_df, min_words):
    # filtramos las filas cuyo campo contenido contenga menos de n palabras
    filtered_df = filtered_df[filtered_df['text'].apply(lambda x: len(x.split()) >= min_words)]
    return filtered_df


def _filter_max_words(filtered_df, max_words):
    # filtramos las filas cuyo campo contenido contenga más de n palabras
    filtered_df = filtered_df[filtered_df['text'].apply(lambda x: len(x.split()) <= max_words)]
    return filtered_df

# import matplotlib.pyplot as plt


# @deprecated
# def __get_more_frequent_authors_old(df, n, min_words=0):
#     # Filtramos las filas cuyo campo contenido contenga al menos n palabras
#     df = df[df['contenido'].apply(lambda x: len(x.split()) >= min_words)]
#     # Contar las veces que aparece cada nombreUsuario en la columna "nombreUsuario".
#     user_counts = df['nombreUsuario'].value_counts()
#     # Seleccionar los n nombreUsuario con más apariciones.
#     n_users = user_counts.head(n)
#     # Utilizar una condición para seleccionar todas las filas correspondientes a los nombreUsuario seleccionados.
#     df = df[df['nombreUsuario'].isin(n_users.index)]
#     # Renumerar los índices y devolver resultado
#     return df.reset_index(drop=True)
#
#
# @deprecated
# def __get_more_frequent_authors_alternative(df, n_authors, min_words=0):
#     filtered_df = df.copy()
#     # filtramos las filas cuyo campo contenido contenga menos de n palabras
#     filtered_df = filtered_df[filtered_df['contenido'].apply(lambda x: len(x.split()) >= min_words)]
#     # agrupamos los datos por autor y contamos el número de registros de cada uno
#     author_counts = filtered_df['nombreUsuario'].value_counts()
#     # seleccionamos los autores con mayor numero de registros
#     top_authors = author_counts.head(n_authors)
#     # Filtramos las filas del dataframe que pertenecen a los autores seleccionados
#     filtered_df = filtered_df[filtered_df['nombreUsuario'].isin(top_authors.index.values)]
#     return filtered_df


def eliminar_columnas(df, columnas_eliminar):
    # Comparar las columnas a eliminar con las columnas del dataframe
    columnas_eliminar = [col for col in columnas_eliminar if col in df.columns]
    print("Eliminando columnas: ", columnas_eliminar)
    return df.drop(columnas_eliminar, axis='columns')
