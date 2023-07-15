from matplotlib import pyplot as plt

from tfg.datos import load_temp_dataset, save_temp_dataset
from tfg.graphics import plot_histogram, plot_pie
from tfg.textmining.preprocessing import show_statistics, select_documents, eliminar_columnas

print("INICIO")

# df_all = DataBase().get_dataframe()

df_all = load_temp_dataset()

df_all = eliminar_columnas(df_all, ["_id", "fecha", "numeroEntrada", "cita", "url", "tipoUsuario", "thread"])
df_all.info()

# set_temp_dataset(df_all)


def test_statistics():
    _ = show_statistics(df_all, title="Estadísticas del dataset completo",
                        num_bars='auto',     # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
                        hist=False)


def test_select_documents():
    ###############################################################################
    # Seleccionar los autores más frecuentes
    ###############################################################################
    datos, metadata = select_documents(
        df_all,
        n_authors=5,  # 200,
        min_words=None,  # 30,
        max_words=400,  # None,
        min_doc_per_author=None,  # 100,
        max_doc_per_author=100,  # 1000,  # Si longest_docs=False se dará error en caso de no existir documentos suficientes para algún autor.
        longest_docs=True,    # True: Seleccionar los documentos más largos. False: Seleccionar los documentos aleatoriamente.
        show_stats=False,
        num_bars='auto',     # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.
    )
    metadata = dict(selected_documents_statistics=metadata)

    save_temp_dataset(datos)


def test_histogramas():
    datos = load_temp_dataset()
    author_label = "author"
    doc_author_counts = datos[author_label].value_counts()
    word_author_counts = datos.groupby(author_label)['text'].apply(lambda x: x.str.split().str.len().sum())
    word_doc_counts = datos['text'].str.split().str.len()
    num_bars = 'scott'  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.

    plot_histogram(doc_author_counts,
                   # range=(1, 50),
                   title="Distribución de autores según cantidad de documentos",
                   x_label="Número de documentos",
                   y_label="Número de autores",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)
    plot_histogram(word_author_counts,
                   # range=(1, 600),
                   title="Distribución de autores según número de palabras",
                   x_label="Número de palabras",
                   y_label="Número de autores",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)
    plot_histogram(word_doc_counts,
                   # range=(1, 200),
                   title="Distribución de documentos según número de palabras",
                   x_label="Número de palabras",
                   y_label="Número de documentos",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)

    plot_histogram(datos['text'].apply(lambda x: len(x.split())).reset_index(drop=True),
                   title="Palabras por documento - máximo 382 palabras",
                   x_label="Número de palabras",
                   y_label="Documentos",
                   num_bars=50,  # 'auto',
                   font_size=12
                   )

    plot_histogram(datos['text'].apply(lambda x: len(x.split())).reset_index(drop=True),
                   title="Palabras por documento - máximo 382 palabras",
                   x_label="Número de palabras",
                   y_label="Documentos",
                   num_bars=50,  # 'auto',
                   font_size=12
                   )


def test_histogramas1():
    datos = load_temp_dataset()
    author_label = "author"
    doc_author_counts = datos[author_label].value_counts()
    plot_histogram(doc_author_counts,
                   # range=(1, 50),
                   title="Distribución de autores según cantidad de documentos",
                   x_label="Número de documentos",
                   y_label="Número de autores",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)


def test_histogramas2():
    datos = load_temp_dataset()
    author_label = "author"
    word_author_counts = datos.groupby(author_label)['text'].apply(lambda x: x.str.split().str.len().sum())
    plot_histogram(word_author_counts,
                   # range=(1, 600),
                   title="Distribución de autores según número de palabras",
                   x_label="Número de palabras",
                   y_label="Número de autores",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)


def test_histogramas3():
    datos = load_temp_dataset()
    word_doc_counts = datos['text'].str.split().str.len()
    plot_histogram(word_doc_counts,
                   # range=(1, 200),
                   title="Distribución de documentos según número de palabras",
                   x_label="Número de palabras",
                   y_label="Número de documentos",
                   num_bars=50,  # 'auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', or 'sqrt'.  'all'
                   font_size=12)


def test_histogramas4():
    datos = load_temp_dataset()
    plot_histogram(datos['text'].apply(lambda x: len(x.split())).reset_index(drop=True),
                   title="Palabras por documento - máximo 382 palabras",
                   x_label="Número de palabras",
                   y_label="Documentos",
                   num_bars=50,  # 'auto',
                   font_size=12
                   )


def test_plot_pie():
    datos = load_temp_dataset()
    lengths = datos["text"].apply(lambda x: len(x.split()))
    plot_pie(datos, lengths)
    plt.show()

