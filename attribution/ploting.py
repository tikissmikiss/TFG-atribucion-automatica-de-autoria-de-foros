import os
from datetime import datetime

import matplotlib.pyplot as plt

from tfg.datos import DataBase
from tfg.utils import create_folder, get_full_path

print_all = False


def poc():
    db = DataBase()
    collection = db.get_objects(collection="experiments")
    filter = {"tags": "poc"}
    experiments = [e for e in collection.find(filter)]
    mdls = collection.find({"tags": "poc"}).next().get('models')

    for experiment in experiments:
        min_word = experiment['params']['min_words']
        models = experiment['models']
        lines = []
        y = []
        for model in models:
            x = [x.get('params').get('min_words') for x in experiments]
            accuracy = model['accuracy']
            name = model['name']
            x.append(min_word)
            y.append(accuracy)
        lines.append((x, y, name))
    # plot lines

    models: dict = experiments[0]["models"]
    for document in experiments:
        print(document)
    pass


def grafica_accuracy(datos):
    """
    Grafica la precisión de los modelos en función de 'min_words' con puntos
    todas las graficas juntos en la misma figura
    """
    plt.figure(figsize=(10, 6))

    # Para cada elemento en los datos
    for elemento in datos:
        modelos = elemento['models']
        min_words = elemento['params']['min_words']

        # Para cada modelo
        for modelo in modelos:
            accuracy = modelo['accuracy']

            plt.plot(min_words, accuracy, marker='-o-', label=modelo.get('name'))

    plt.xlabel('min_words')
    plt.ylabel('accuracy')
    plt.title('Accuracy por modelo en función de min_words')
    plt.legend(loc='lower right')

    plot_and_show('grafica_accuracy', format='svg', dpi=1200)  # plt.show()


# Grafica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech'
def plot_pares_my_tech(data):
    """
    Grafica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech'
    Una gráfica por cada modelo todas en la misma figura
    Parameters
    ----------
    data

    Returns
    -------

    """
    # Primero, separa los datos en dos listas diferentes dependiendo del valor de 'my_tech'
    data_true = [d for d in data if d['params']['my_tech'] == True]
    data_false = [d for d in data if d['params']['my_tech'] == False]

    # Ordena los datos por 'min_words' para asegurar que las líneas de los gráficos sean coherentes
    data_true = sorted(data_true, key=lambda x: x['params']['min_words'])
    data_false = sorted(data_false, key=lambda x: x['params']['min_words'])

    # Número de modelos
    num_models = len(data_true[0]['models'])

    # Inicializa la figura y los ejes
    _, axs = plt.subplots(num_models, figsize=(10, num_models * 5))

    # Para cada modelo, grafica los datos en su propio subplot
    for i, model in enumerate(data_true[0]['models']):
        # Grafica para 'my_tech' = True
        x = [d['params']['min_words'] for d in data_true]
        y = [d['models'][i]['accuracy'] for d in data_true]
        axs[i].plot(x, y, label=f"Text tokenized")

        # Grafica para 'my_tech' = False
        x = [d['params']['min_words'] for d in data_false]
        y = [d['models'][i]['accuracy'] for d in data_false]
        axs[i].plot(x, y, linestyle='--', label=f"Text cleaned")

        # Configura el subplot
        axs[i].set_title(f'Model: {model.get("name")}')
        axs[i].set_xlabel('min_words')
        axs[i].set_ylabel('accuracy')
        axs[i].legend()

    plot_and_show('grafica_precision_min_words_my_tech', format='svg',
                  dpi=1200)  # # Muestra el gráfico  # plt.tight_layout()  # plt.savefig(f'Resultados/graphics/grafica_precision_min_words_my_tech.svg', format='svg', dpi=1200)  # plt.savefig(f'Resultados/graphics/grafica_precision_min_words_my_tech.png', format='png', dpi=1200)  # plt.show()


# def grafica_ngrams_1(data):
#     unique_n_grams = set(tuple(d['params']['ngram_range']) for d in data)
#     # unique_n_grams = set(d['params']['ngram_range'] for d in data)  # Obtiene todos los valores únicos de ngram_range
#
#     # Separa los datos en una estructura de datos anidada, donde el primer nivel son los valores de ngram_range y el segundo nivel son los valores de my_tech
#     data_dict = {
#         n: {
#             my_tech: sorted([d for d in data if d['params']['my_tech'] == my_tech and d['params']['ngram_range'] == n],
#                             key=lambda x: x['params']['min_words'])
#             for my_tech in [True, False]
#         }
#         for n in unique_n_grams
#     }
#
#     num_models = len(data[0]['models'])
#
#     # Para cada modelo, grafica los datos en su propia figura
#     for i, model in enumerate(data[0]['models']):
#         fig, axs = plt.subplots(len(unique_n_grams), figsize=(10, 10))
#
#         for ax, (n, my_tech_dict) in zip(axs, data_dict.items()):
#             for my_tech, data in my_tech_dict.items():
#                 x = [d['params']['min_words'] for d in data]
#                 y = [d['models'][i]['accuracy'] for d in data]
#                 linestyle = '-' if my_tech else '--'
#                 ax.plot(x, y, linestyle=linestyle, label=f"my_tech={my_tech}, ngram_range={n}")
#                 ax.set_title(f'Model: {model.get("name")}, ngram_range: {n}')
#                 ax.set_xlabel('min_words')
#                 ax.set_ylabel('accuracy')
#                 ax.legend()
#
#         plot_and_show('grafica_precision_min_words_my_tech_ngrams', format='svg', dpi=1200)
#
#         # plt.tight_layout()
#         # plt.savefig(f'Resultados/graphics/grafica_precision_min_words_my_tech_ngrams.svg', format='svg', dpi=1200)
#         # # plt.savefig(f'Resultados/graphics/grafica_precision_min_words_my_tech.png', format='png', dpi=1200)
#         # plt.show()
#         # plt.close()


def plot_graphs_ngram_my_tech(data):
    """
    Gráfica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech' y 'ngram_range'
    Una gráfica por cada rango de ngramas
    """
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
        data_by_ngram_and_tech[ngram]['true'] = sorted([d for d in data if d['params']['my_tech']],
            key=lambda x: x['params']['min_words'])
        data_by_ngram_and_tech[ngram]['false'] = sorted([d for d in data if not d['params']['my_tech']],
            key=lambda x: x['params']['min_words'])
    # Para cada modelo, grafica los datos en su propia figura
    for m, model in enumerate(data[0]['models']):
        # Inicializa la figura
        _ = plt.figure(figsize=(10, 5 * len(data_by_ngram_and_tech)))
        name = model.get("name")
        # Por cada valor de 'ngram_range', grafica los datos en su propio subplot
        for i, (ngram, data) in enumerate(data_by_ngram_and_tech.items()):
            # Grafica para 'my_tech' = True
            x = [d['params']['min_words'] for d in data['true']]
            y = [d['models'][m]['accuracy'] for d in data['true']]
            plt.subplot(len(data_by_ngram_and_tech), 1, i + 1)
            plt.plot(x[:-1], y[:-1], label=f"my_tech=True")
            # Grafica para 'my_tech' = False
            x = [d['params']['min_words'] for d in data['false']]
            y = [d['models'][m]['accuracy'] for d in data['false']]
            plt.plot(x[:-1], y[:-1], linestyle='--', label=f"my_tech=False")
            # Configura el subplot
            plt.title(f'Model: {name}, Ngram Range: {ngram}')
            plt.xlabel('min words')
            plt.ylabel('Accuracy')
            plt.legend()

        plot_and_show('graphs_ngram_my_tech', format='svg', dpi=1200)


def plot_graphs_ngram_my_tech_2(data):
    """
    Gráfica la precisión de los modelos en función de 'min_words' para cada valor de 'my_tech' y 'ngram_range'
    Una gráfica por cada rango de ngramas
    """
    num_models = len(data[0]['models'])
    if num_models > 6:
        mid_idx = num_models // 2
        data1 = [{**d, 'models': d['models'][:mid_idx]} for d in data]
        data2 = [{**d, 'models': d['models'][mid_idx:]} for d in data]
        plot_graphs_ngram_my_tech_2(data1)
        plot_graphs_ngram_my_tech_2(data2)
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
        for i, (ngram, _data) in enumerate(data_by_ngram_and_tech.items()):
            plt.subplot(len(data_by_ngram_and_tech), 1, i + 1)
            for m, model in enumerate(_data['true'][0]['models']):
                name = model.get("name")
                name = name.replace('RidgeClassifier', 'RC').replace('SVC', 'SVM')\
                    .replace('MLPClassifier', 'MLP').replace('RandomForestClassifier', 'RFC')

                # Grafica para 'my_tech' = True
                x = [d['params']['min_words'] for d in _data['true']]
                y = [d['models'][m]['accuracy'] for d in _data['true']]
                plt.plot(x, y, marker='.',  color=colors[m], linestyle='-',
                         label=f"Propuesta ({name})",)
                # Grafica para 'my_tech' = False
                x = [d['params']['min_words'] for d in _data['false']]
                y = [d['models'][m]['accuracy'] for d in _data['false']]
                plt.plot(x, y, marker='.',  color=colors[m], linestyle='--',
                         label=f"Estándar ({name})",)

            # Configura el subplot
            plt.title(f'Resultados experimentos [authors: {authors}, docs: {docs}]')
            plt.xlabel('Min words')
            plt.ylabel('Accuracy')
            plt.legend(title='Limpieza (Algoritmo)')  # fancybox=True, shadow=True, bbox_to_anchor=(1.00, 1), loc='upper left'

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


def plot_and_show(filename, folder='./Resultados', format='svg', dpi=1200, show=True, close=True, tight_layout=True):
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


if __name__ == "__main__":
    db = DataBase()
    collection = db.get_objects(collection="experiments")
    find = {"tags": "tfg_ngrams_1000_git"}
    data_list = [e for e in collection.find(find)]
    print_all = False
    print("Iniciando...")

    plot_graphs_ngram_my_tech_2(data_list)
    if print_all:
        plot_graphs_ngram_my_tech(data_list)
        # poc()
        grafica_accuracy(data_list)
        plot_pares_my_tech(data_list)
        plot_all_together(data_list)

    print("FIN")
