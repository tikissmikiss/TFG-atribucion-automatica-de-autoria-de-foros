import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from attribution.tfg.utils import create_folder
from tfg.utils import get_full_path, create_folder


def _plot_histogram(df, title=None, num_bars=25):
    fig, ax = plt.subplots(figsize=(16, 8))  # Ajustar el tamaño de la figura

    counts, bins, patches = ax.hist(df, bins=num_bars, align='left', color='#440154', edgecolor='black', linewidth=0.5)

    bin_range = max(df) - min(df)
    # Agrupar las barras en rangos proporcionales si hay más de num_bars barras
    if bin_range > num_bars:
        counts_grouped, bins_grouped = [], []
        bin_width = bin_range / num_bars
        for i in range(num_bars - 1):
            bin_start = min(df) + i * bin_width
            bin_end = min(df) + (i + 1) * bin_width
            count_grouped = sum(counts[(bins[:-1] >= bin_start) & (bins[:-1] < bin_end)])
            counts_grouped.append(count_grouped)
            bins_grouped.append(bin_start)

        # Agrupar las barras restantes
        count_grouped = sum(counts[(bins[:-1] >= bin_end) & (bins[:-1] <= max(df))])
        counts_grouped.append(count_grouped)
        bins_grouped.append(bin_end)

        patches.remove()  # Eliminar las barras anteriores
        ax.bar(bins_grouped, counts_grouped, width=bin_width, color='#440154', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Número de palabras', color='grey', fontsize=10)  # Tamaño de los números
    ax.set_ylabel('Frecuencia', color='grey', fontsize=10)  # Tamaño de los números
    ax.set_title(title, color='grey', fontsize=12)  # Tamaño del título
    ax.grid(axis='y', alpha=0.5)

    # Configuración del fondo en gris oscuro
    ax.set_facecolor('#333333')

    rotation = 25 if num_bars > 30 else 0  # Ajuste del ángulo de rotación
    for rect in patches:
        height = rect.get_height()
        ha='left' if num_bars > 30 else 'center'  # Ajuste de la alineación horizontal
        ax.text(rect.get_x() + rect.get_width() / 2, height, f'{int(height)}', ha=ha, va='bottom',
                color='#FDE724', rotation=rotation, fontsize=10)  # Tamaño de los números

    # Configuración de los ejes en blanco
    ax.spines['bottom'].set_color('grey')
    ax.spines['left'].set_color('grey')
    ax.spines['top'].set_color('grey')
    ax.spines['right'].set_color('grey')
    ax.tick_params(axis='both', colors='grey')

    # Ajustar las marcas del eje x
    ax.set_xticks([bin - bin_width/2 for bin in bins_grouped] + [bins_grouped[-1] + bin_width/2])
    ax.set_xticklabels([f'{int(bin)}' for bin in bins_grouped] + [f'{int(bins_grouped[-1] + bin_width)}'], rotation=rotation, ha='right')

    plt.tight_layout()
    plt.savefig('histogram.png')
    plt.show()


def plot_histogram_(df, title=None):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.hist(df, bins=np.arange(df.min(), df.max()+1), align='left', color='#440154', edgecolor='black', linewidth=0.5, width=0.8)
    ax.set_xticks(np.arange(df.min(), df.max()+1))
    ax.set_xlabel('Número de palabras', color='white')  # Establecer el color del texto en blanco
    ax.set_ylabel('Frecuencia', color='white')  # Establecer el color del texto en blanco
    ax.set_title(title, color='white')  # Establecer el color del texto en blanco
    ax.grid(axis='y', alpha=0.5)

    # Configuración del fondo en gris oscuro
    ax.set_facecolor('#333333')

    # Añadir los valores dentro de cada barra en el tono '#FDE724' con un tamaño de fuente más grande
    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height/2, f'{int(height)}', ha='center', va='center', color='#FDE724', fontsize=12)

    # Configuración de los ejes en blanco
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='both', colors='white')

    plt.tight_layout()
    plt.show()


def plot_author_frequencies(dataframe, column_name, bin_size=200, max_value=50):
    # Contar las ocurrencias de cada autor
    author_counts = dataframe['author'].value_counts()

    # Obtener los nombres de los autores y las frecuencias
    authors = author_counts.index
    frequencies = author_counts.values

    # Calcular los límites del rango
    min_value = max(0, min(frequencies))
    max_value = max(frequencies) if max_value is None else max_value
    bin_size = 1
    bin_range = range(min_value, max_value + bin_size, bin_size)

    # Establecer el tamaño de la figura
    plt.figure(figsize=(8, 3.7))  # Ajusta el ancho y alto según tus preferencias

    # Trazar el histograma con barras para rangos de 200
    plt.hist(frequencies, bins=bin_range, edgecolor='black')

    # plt.tight_layout()
    # Personalizar el gráfico
    plt.title("Frecuencias de Autores")
    plt.xlabel("Frecuencia")
    plt.ylabel("Cantidad de Autores")
    _save_plot('Frecuencias de Autores', '/resultados/graphics', format='svg', dpi=1200)

    # Mostrar el gráfico
    plt.show()


def plot_words_frequencies(dataframe, column_name, bin_size=200, max_value=150):
    word_counts = dataframe[column_name].apply(lambda x: len(str(x).split()))
    frequencies = word_counts.values
    # Calcular los límites del rango
    min_value = max(0, min(frequencies))
    max_value = max(frequencies) if max_value is None else max_value
    bin_size = 3
    bin_range = range(min_value, max_value + bin_size, bin_size)

    # Establecer el tamaño de la figura
    plt.figure(figsize=(8, 3.85))  # Ajusta el ancho y alto según tus preferencias

    # Trazar el histograma con barras para rangos de 200
    plt.hist(frequencies, bins=bin_range, edgecolor='black')

    # plt.tight_layout()
    # Personalizar el gráfico
    plt.title("Frecuencias documentos según cantidad palabras")
    plt.xlabel("Frecuencia en rango de palabras")
    plt.ylabel("Cantidad de documentos")
    _save_plot('Frecuencias de palabras', '/resultados/graphics', format='svg', dpi=1200)

    # Mostrar el gráfico
    plt.show()


def plot_histogram(df, title=None, num_bars=None, fig_size=(12, 6), font_size=12, x_label=None, y_label=None,
                   rotation=60, no_labels=False, range=None, folder="resultados/graphics", show=True):
    if num_bars in ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt']:
        num_bars = num_bars
    elif num_bars == 'all':
        num_bars = df.max() - df.min() + 1
    elif not num_bars or num_bars <= 0:
        num_bars = int(df.max() - df.min() + 1)
    # elif num_bars and num_bars > 50:
    #     # num_bars = 50
    #     no_labels = True

    if num_bars and not isinstance(num_bars, str) and range:
        num_bars = int(range[1] - range[0])

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.grid(axis='y', alpha=0.2)
    counts, bins, patches = ax.hist(df, align='left', color='#440154', edgecolor='black', linewidth=0.5, bins=num_bars,
                                    range=range)  # , log=True)

    ax.set_xlabel(x_label, color='#BFC0C2', fontsize=int(font_size * 1.3))
    ax.set_ylabel(y_label, color='#BFC0C2', fontsize=int(font_size * 1.3))
    ax.set_title(f'{title}', color='#BFC0C2', fontsize=int(font_size * 1.6))
    # Configuración del fondo en gris oscuro
    ax.set_facecolor('#777777')
    fig.set_facecolor('#282C34')

    num_bars = bins.shape[0] - 1
    if num_bars > 50:
        no_labels = True

    if not no_labels:
        bin_width = bins[1] - bins[0]
        ax.set_xticks([bin - bin_width / 2 for i, bin in enumerate(bins) if int(bins[i - 1]) != int(bin)])
        rotation_ = rotation if num_bars and not isinstance(num_bars,
                                                            str) and num_bars > 20 else None  # Ajuste del ángulo de rotación
        ax.set_xticklabels([f'{int(bin)}' for i, bin in enumerate(bins) if int(bins[i - 1]) != int(bin)],
                           rotation=rotation_, ha='center', fontsize=font_size)
        # Añadir los valores dentro de cada barra
        for rect in ax.patches:
            height = rect.get_height()
            ha = 'left' if num_bars and not isinstance(num_bars,
                                                       str) and num_bars > 20 else 'center'  # Ajuste de la alineación horizontal
            vp = height if num_bars and not isinstance(num_bars,
                                                       str) and num_bars > 20 else height / 2  # Ajuste de la posición vertical
            ax.text(rect.get_x() + rect.get_width() / 2, vp, f'{int(height) if height > 0 else ""}', ha=ha, va='bottom',
                    color='#FDE724', rotation=rotation_, fontsize=font_size)

    # Configuración de los ejes en gris
    ax.spines['bottom'].set_color('#BFC0C2')
    ax.spines['left'].set_color('#BFC0C2')
    ax.spines['top'].set_color('#BFC0C2')
    ax.spines['right'].set_color('#BFC0C2')
    ax.tick_params(axis='both', colors='#BFC0C2')

    plt.tight_layout()
    create_folder(folder)
    # guardar svg
    plt.savefig(f'{folder}/{title}.svg', format='svg', dpi=1200)
    if show:
        plt.show()
    plt.close()


def plot_pie2(dataframe):
    # Contar las ocurrencias de cada autor
    author_counts = dataframe['author'].value_counts()

    # Obtener los nombres de los autores y las frecuencias
    authors = author_counts.index
    frequencies = author_counts.values

    # Trazar el gráfico de pastel
    plt.pie(frequencies, labels=authors, autopct='%1.1f%%')

    # Personalizar el gráfico
    plt.title("Frecuencias de Autores")

    # Mostrar el gráfico
    plt.show()


def plot_pie(df, lengths, style_label='default', folder="resultados/graphics", show=True):
    # make figure and assign axis objects
    fig, ax1 = plt.subplots(figsize=(6, 4))
    plt.style.use(style_label)
    fig.subplots_adjust(wspace=0)

    overall_ratios = [
        df[lengths <= 15].shape[0] / len(df),
        df[(lengths > 15) & (lengths <= 22)].shape[0] / len(df),
        df[(lengths > 22) & (lengths <= 30)].shape[0] / len(df),
        df[(lengths > 30) & (lengths <= 45)].shape[0] / len(df),
        df[(lengths > 45) & (lengths <= 99)].shape[0] / len(df),
        df[(lengths > 99) & (lengths <= 328)].shape[0] / len(df),
        df[lengths > 328].shape[0] / len(df), ]
    labels = ['0-15', '16-22', '22-30', '31-45',
              '46-99', '100-328', '>329']
    explode = [0.01, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03]
    pctdistance = 0.6
    labeldistance = 1.1
    # ([0.1]+([0.05]*len(overall_ratios-1)))
    wedges, texts, autotexts = ax1.pie(overall_ratios,
                                       autopct='%1.1f%%',
                                       pctdistance=pctdistance,
                                       labeldistance=labeldistance,
                                       labels=labels,
                                       explode=explode,
                                       # shadow=True,
                                       )

    plt.setp(autotexts, size=10, weight="bold")

    title = "Documentos por número de palabras"
    ax1.set_title(title, fontsize=12, fontweight='bold')

    _save_plot(title, f'{folder}', format='svg', dpi=1200)

    plt.tight_layout()
    create_folder(folder)
    # guardar svg
    plt.savefig(f'{folder}/{title}.svg', format='svg', dpi=1200)
    if show:
        plt.show()
    plt.close()

    # plt.savefig(f'./resultados/graphics/pie.svg', format='svg', dpi=1200)
    # plt.show()


def generar_matriz_confusion(clf, y_test, prediction, target_names, folder, show=True, save=True):
    create_folder(folder)
    scale = (len(target_names)/15+1/6)
    # _, ax = plt.subplots(figsize=(20 * scale / 2.4, 20 * scale / 2.4))
    _, ax = plt.subplots(figsize=(15*scale*0.55, 15*scale*0.55))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax)
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(f"{clf.__class__.__name__}\non test set")
    plt.tight_layout()
    if save:
        _save_plot('absolut_confusion_matrix', folder, format='svg', dpi=10)
        # plt.savefig(f'{folder}/absolut_confusion_matrix.svg', format='svg', dpi=1200)
    else:
        print("ATENCION NO SE GUARDA LA MATRIZ DE CONFUSION")
    if show:
        plt.show()
    plt.close()
    #
    _, ax = plt.subplots(figsize=(15*scale*0.65, 15*scale*0.65))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax, normalize='pred')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(f"Conf. Matrix for {clf.__class__.__name__}\non TEST documents")
    plt.tight_layout()
    if save:
        _save_plot('normalized_confusion_matrix', folder, format='svg', dpi=10)
        # plt.savefig(f'{folder}/normalized_confusion_matrix.svg', format='svg', dpi=1200)
    else:
        print("ATENCION NO SE GUARDA LA MATRIZ DE CONFUSION")
    if show:
        plt.show()
    plt.close()


def plot_graphs_ngram__min_word__my_tech(data):
    """
    Gráfica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech' y 'ngram_range'
    Una gráfica por cada rango de ngramas
    """
    num_models = len(data[0]['models'])
    if num_models > 6:
        mid_idx = num_models // 2
        data1 = [{**d, 'models': d['models'][:mid_idx]} for d in data]
        data2 = [{**d, 'models': d['models'][mid_idx:]} for d in data]
        plot_graphs_ngram__min_word__my_tech(data1)
        plot_graphs_ngram__min_word__my_tech(data2)
    else:
        authors = data[0]['params']['n_authors']
        docs = data[0]['selected_documents_statistics']['Número de documentos']
        # Agrupa los datos por 'ngram_range'
        data_by_ngram = {}
        for d in data:
            ngram = tuple(d['params']['ngram_range'])
            if ngram not in data_by_ngram:
                data_by_ngram[ngram] = [d]
            else:
                data_by_ngram[ngram].append(d)

        # Por cada valor de 'ngram_range', separa los datos en dos listas dependiendo del valor de 'my_tech'
        data_by_ngram_and_tech = {ngram: {'true': [], 'false': []} for ngram in data_by_ngram.keys()}
        for ngram, data in data_by_ngram.items():
            data_by_ngram_and_tech[ngram]['true'] = sorted(
                [d for d in data if d['params']['my_tech']], key=lambda x: x['params']['min_words'])
            data_by_ngram_and_tech[ngram]['false'] = sorted(
                [d for d in data if not d['params']['my_tech']], key=lambda x: x['params']['min_words'])

        # Define el conjunto de colores para cada modelo (suponiendo que hay no más de 10 modelos)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

        # Inicializa la figura
        scale = 0.8
        _ = plt.figure(figsize=(scale * 10, scale * 6 * len(data_by_ngram_and_tech)))

        # Por cada valor de 'ngram_range', grafica los datos en su propio subplot
        lines_true = {}
        lines_false = {}
        for i, (ngram, _data) in enumerate(data_by_ngram_and_tech.items()):
            plt.subplot(len(data_by_ngram_and_tech), 1, i + 1)
            for d in ['true', 'false']:

                for m, model in enumerate(_data['true'][0]['models']):
                    name = model.get("name")
                    name = name.replace('RidgeClassifier', 'RC').replace('SVC', 'SVM') \
                        .replace('MLPClassifier', 'MLP').replace('RandomForestClassifier', 'RFC')

                    x = [d['params']['min_words'] for d in _data[d]]
                    y = [d['models'][m]['accuracy'] for d in _data[d]]
                    plt.plot(x, y, marker='.', color=colors[m], linestyle='-' if d == 'true' else '--',
                             label=f"ad-hoc ({name})" if d == 'true' else f"generic ({name})")


        #         lines_true.update({m: {'x': [d['params']['min_words'] for d in _data['true']],
        #                                'y': [d['models'][m]['accuracy'] for d in _data['true']],
        #                                'n': name, 'c': colors[m]}})
        #         lines_false.update({'x': [d['params']['min_words'] for d in _data['false']],
        #                             'y': [d['models'][m]['accuracy'] for d in _data['false']],
        #                             'n': name, 'c': colors[m]})
        #
        # for e in lines_true.items():
        #     x, y, name, c = e[1]['x'], e[1]['y'], e[1]['n'], e[1]['c']
        #     # Grafica para 'my_tech' = True
        #     plt.plot(x, y, marker='.', color=colors[c], linestyle='-',
        #              label=f"ad-hoc ({name})", )
        #
        # for e in lines_false:
        #     x, y, name, c = e
        #     # Grafica para 'my_tech' = False
        #     plt.plot(x, y, marker='.', color=colors[c], linestyle='--',
        #              label=f"genérica ({name})", )

        # Configura el subplot
        plt.title(f'Resultados experimentos [authors: {authors}, docs: {docs}]')
        plt.xlabel('Min words')
        plt.ylabel('Accuracy')
        plt.legend(
            title='Limpieza (Algoritmo)')  # fancybox=True, shadow=True, bbox_to_anchor=(1.00, 1), loc='upper left'

        plot_and_show('graphs_ngram_my_tech', format='svg', dpi=1200, tight_layout=True)


def _save_plot(file, folder, format='svg', dpi=1200):
    folder = folder.removeprefix('./').removeprefix('/').removeprefix('\\').removesuffix('./').removesuffix(
        '/').removesuffix('\\')
    path = get_full_path(f"{file}.{format}", folder)
    # Crear la carpeta si no existe
    create_folder(folder)
    now = ''
    # Comprobar si existe el fichero
    if os.path.isfile(path):
        # Sí existe, añadir YYMMDDHHMMSS delante del nombre
        now = datetime.now().strftime("%y%m%d%H%M%S") + '_'
    plt.savefig(f'{folder}/{now}{file}.{format}', format=format, dpi=dpi)


def plot_and_show(filename, folder='./resultados', format='svg', dpi=1200, show=True, close=True, tight_layout=True):
    # Muestra el gráfico
    if tight_layout:
        plt.tight_layout()
    if format == 'all':
        _save_plot(filename, f'{folder}/graphics', format='svg', dpi=dpi)
        _save_plot(filename, f'{folder}/graphics', format='png', dpi=dpi)
    else:
        _save_plot(filename, f'{folder}/graphics', format=format, dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close()


def plot_all_together(data):
    """
    Gráfica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech'
    Todas las gráficas en la misma figura
    """
    # Primero, separa los datos en dos listas diferentes dependiendo del valor de 'my_tech'
    data_true = [d for d in data if d['params']['my_tech'] == True]
    data_false = [d for d in data if d['params']['my_tech'] == False]

    # Ordena los datos por 'min_words' para asegurar que las líneas de los gráficos sean coherentes
    data_true = sorted(data_true, key=lambda x: x['params']['min_words'])
    data_false = sorted(data_false, key=lambda x: x['params']['min_words'])

    # Inicializa la figura y los ejes
    _, axs = plt.subplots(figsize=(10, 10))

    # Para cada modelo, grafica los datos en el gráfico para 'my_tech' = True
    for i, model in enumerate(data_true[0]['models']):
        x = [d['params']['min_words'] for d in data_true]
        y = [d['models'][i]['accuracy'] for d in data_true]
        axs.plot(x, y, label=f"{model.get('name')} (my_tech=True)")

    # Repite para 'my_tech' = False
    for i, model in enumerate(data_true[0]['models']):
        x = [d['params']['min_words'] for d in data_false]
        y = [d['models'][i]['accuracy'] for d in data_false]
        axs.plot(x, y, linestyle='--', label=f"{model.get('name')} (my_tech=False)")

    # Configura el gráfico
    axs.set_title('Model Accuracy')
    axs.set_xlabel('min_words')
    axs.set_ylabel('accuracy')
    axs.legend()

    plot_and_show('plot_all_together', format='svg', dpi=1200)  # Muestra el gráfico  # plt.show()
