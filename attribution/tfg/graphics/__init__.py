import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from tfg.utils import create_folder


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


def plot_histogram(df, title=None, num_bars=None, fig_size=(12, 6), font_size=12, x_label=None, y_label=None,
                   rotation=60, no_labels=False, range=None, folder="Resultados/graphics", show=True):
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


def plot_pie(df, lengths, style_label='default', folder="Resultados/graphics", show=True):
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

    plt.tight_layout()
    create_folder(folder)
    # guardar svg
    plt.savefig(f'{folder}/{title}.svg', format='svg', dpi=1200)
    if show:
        plt.show()
    plt.close()

    # plt.savefig(f'./Resultados/graphics/pie.svg', format='svg', dpi=1200)
    # plt.show()


def generar_matriz_confusion(clf, y_test, prediction, target_names, folder, show=False):
    create_folder(folder)
    scale = len(target_names)/15+1/6
    # _, ax = plt.subplots(figsize=(20 * scale / 2.4, 20 * scale / 2.4))
    _, ax = plt.subplots(figsize=(20*scale, 12*scale))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax)
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(f"{clf.__class__.__name__}\non test set")
    print("ATENCION NO SE GUARDA LA MATRIZ DE CONFUSION")
    # plt.savefig(f'{folder}/absolut_confusion_matrix.svg', format='svg', dpi=1200)
    if show:
        plt.show()
    plt.close()
    #
    _, ax = plt.subplots(figsize=(20*scale, 12*scale))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, ax=ax, normalize='pred')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    _ = ax.set_title(f"Conf. Matrix for {clf.__class__.__name__}\non TEST documents")
    # guardar svg
    print("ATENCION NO SE GUARDA LA MATRIZ DE CONFUSION")
    # plt.savefig(f'{folder}/normalized_confusion_matrix.svg', format='svg', dpi=1200)
    if show:
        plt.show()
    plt.close()
