import io
import os
import pickle
import random
import re
from time import sleep, time

import nltk
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from unidecode import unidecode

from tfg.textmining.preprocessing import (balance_class, show_statistics,
                                          statistics)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
# nltk.download('words')  # Descargar el corpus de palabras en inglés
# nltk.download('cess_esp') # Descargar el corpus de palabras en inglés


# REGEX Definidos a nivel global para que no se compilen cada vez que se llama a una función
url_regex = re.compile(
    r'(?:(?:https?|ftps?)://)?'  # Optional protocol
    r'(?P<domain>(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or IP
    r'(?::\d+)?'  # optional port
    r'(?:/[\w\-./]*)?'  # optional path
    r'(?:\?\S*)?'  # optional query string
    r'(?:#\S*)?',  # optional fragment identifier
    re.IGNORECASE
)
# Busca números enteros y decimales con separador decimal punto o coma
number_regex = re.compile(r'(?P<pref>^|\W)(?P<int>\d+)(?:[,.](?P<dec>\d+))?')
# Busca repeticiones de signos de puntuación
repeated_admiration_regex = re.compile(r'\s*(?P<all>(?P<sig>[¿¡!?.,:;\'])+)\s*', flags=re.IGNORECASE)
# Emoticonos
emoticon_regex = re.compile(r':(?P<emoticon>\w+):')
# Busca repeticiones de caracteres
repeated_chars_regex = re.compile(r'(?P<chars>(?P<char>[A-Za-z])\2+)(?P<suffix>\S*)', flags=re.IGNORECASE)
# Busca repeticiones más de 3 caracteres. Para caracteres dobles del castellano como ll, rr, ee, cc, etc.
repeated_chars_regex_3 = re.compile(r'(?P<chars>(?P<char>[A-Za-z])\2\2+)(?P<suffix>\S*)', flags=re.IGNORECASE)



#
# repeated_chars_regex = re.compile(r'(?P<chars>(?P<prefix>\w)(?!(?P<no>\w))(?P<char>\w)\2+)(?P<suffix>\S*)', flags=re.IGNORECASE)
# repeated_chars_regex_3 = re.compile(r'(?P<chars>(?P<char>\w)\2\2+)(?P<suffix>\S*)', flags=re.IGNORECASE)
#
# repeated_chars_regex = re.compile(r'(?P<prefix>\w)(?!(?P<no>\w))(?P<chars>\w\2+)(?P<suffix>\S*)', flags=re.IGNORECASE)



ruta_fichero_spanish_words = "spanish_words.dic"


def clean_corpus(corpus, verbose=False, tokenize_numbers=True, tokenize_urls=True, tokenize_emoticons=True,
                 tokenize_punctuation=True, tokenize_simbols=True, lower_case=False, remove_accents=False,
                 remove_stop_words=False):
    df = list(corpus.copy())
    for i in range(len(df)):
        df[i], args = _clean_text_(
            df[i], verbose=verbose, tokenize_numbers=tokenize_numbers,
            tokenize_urls=tokenize_urls, tokenize_emoticons=tokenize_emoticons,
            tokenize_punctuation=tokenize_punctuation, tokenize_simbols=tokenize_simbols,
            lower_case=lower_case, remove_accents=remove_accents,
            remove_stop_words=remove_stop_words)
    return df, args


def _clean_text_(text, verbose=False, tokenize_numbers=True, tokenize_urls=True, tokenize_emoticons=True,
                 tokenize_punctuation=True, tokenize_simbols=True, lower_case=False, remove_accents=False,
                 remove_stop_words=False):
    # Tokenizar o eliminar URL's
    if tokenize_urls is not None:
        text = _tokenize_urls(text, verbose=verbose) if tokenize_urls else _remove_urls(text)
    # Tokenizar Emoticonos
    if tokenize_emoticons is not None:
        text = _tokenize_emoticons(text, verbose=verbose) if tokenize_emoticons else _remove_emoticons(text)
    # Tokenizar signos de puntuación
    if tokenize_punctuation is not None:
        text = _tokenize_punctuation(text) if tokenize_punctuation else text
    # Tokenizar signos de puntuación
    if tokenize_simbols is not None:
        text = _tokenize_simbols(text) if tokenize_simbols else _remove_simbols(text)
    # Eliminar acentos
    text = _remove_accents(text) if remove_accents else text  # Ya lo hace el vectorizador de TF-IDF (TfidfVectorizer)
    # Tokenizar o eliminar números
    if tokenize_numbers is not None:
        text = _tokenize_numbers(text) if tokenize_numbers else _remove_numbers(text)
    # Eliminar stop words
    text = _remove_stop_words(text) if remove_stop_words else text  # Lo hace el vectorizador TF-IDF (TfidfVectorizer)
    # Convertir a minúsculas
    text = text.lower() if lower_case else text  # Ya lo hace el vectorizador de TF-IDF (TfidfVectorizer)
    return text, dict(verbose=verbose, tokenize_numbers=tokenize_numbers, tokenize_urls=tokenize_urls,
                      tokenize_emoticons=tokenize_emoticons, tokenize_punctuation=tokenize_punctuation,
                      tokenize_simbols=tokenize_simbols, lower_case=lower_case, remove_accents=remove_accents,
                      remove_stop_words=remove_stop_words)


# def __clean_corpus_parallel(corpus, verbose=False, tokenize_numbers=True,
#                             tokenize_urls=True, tokenize_emoticons=True,
#                             tokenize_punctuation=True, tokenize_simbols=True):
#     with ThreadPoolExecutor(max_workers=6) as executor:
#         results = executor.map(_clean_text, [text for text in corpus], [verbose] * len(corpus),
#                                [tokenize_numbers] * len(corpus), [tokenize_urls] * len(corpus),
#                                [tokenize_emoticons] * len(corpus), [tokenize_punctuation] * len(corpus),
#                                [tokenize_simbols] * len(corpus))
#     return [result for result in results]


def __clean_corpus_serial(corpus, verbose=False, tokenize_numbers=True,
                          tokenize_urls=True, tokenize_emoticons=True,
                          tokenize_punctuation=True, tokenize_simbols=True):
    for i in range(len(corpus)):
        corpus[i] = _clean_text_(corpus[i], verbose=verbose, tokenize_numbers=tokenize_numbers,
                                 tokenize_urls=tokenize_urls, tokenize_emoticons=tokenize_emoticons,
                                 tokenize_punctuation=tokenize_punctuation, tokenize_simbols=tokenize_simbols)
    return corpus


def _remove_whitespaces(text):
    # Eliminar espacios en blanco del texto
    filtered_text = re.sub(r'\s+', ' ', text)
    return filtered_text


class EraseUrlException(Exception):
    pass


def _remove_urls(text, verbose=False):
    filtered_text = re.sub(url_regex, '', text)
    if verbose and filtered_text != text:
        print("· URLS eliminadas en el texto:", text, filtered_text, sep='\n')
    return filtered_text


def _tokenize_urls(text, verbose=False):
    # filtered_text = re.sub(url_regex, 'URLTOKEN', text)
    filtered_text = re.sub(url_regex,
                           lambda match: 'URLTOKEN_' + match.group('domain').replace('.', '_').upper(),
                           text)
    if verbose and filtered_text != text:
        print("· URLS tokenizada en el texto:", text, filtered_text, sep='\n')
    return filtered_text


def _tokenize_numbers(text):
    # Sustituir números por token
    filtered_text = re.sub(
        number_regex,
        lambda match:
        (match.group('pref') if match.group(1) else '') + ' NUMTOKEN_' +
        match.group('int') + (('_' + match.group(3) + ' ') if match.group('dec') else ' '),
        text)
    return filtered_text


def _remove_emoticons(text, verbose=False):
    filtered_text = re.sub(emoticon_regex, '', text)
    if verbose and filtered_text != text:
        print("· Emoticonos eliminados en el texto:", text, filtered_text, sep='\n')
    return filtered_text


def _tokenize_emoticons(text, verbose=False):
    """ Sustituir emoticonos por token """
    filtered_text = re.sub(
        emoticon_regex,
        lambda match: 'TK_EMOTICON_' + match.group('emoticon').upper(),
        text)
    if verbose and filtered_text != text:
        print("· Emoticonos tokenizados en el texto:", text, filtered_text, sep='\n')
    return filtered_text


def _load_spanish_words(path):
    if os.path.isfile(path):
        print("El fichero", path, "existe.")
        # Cargar la lista desde el archivo
        with open(path, "rb") as file:
            dic = pickle.load(file)
    else:
        print("El fichero", path, "no existe.")
        dic = []
    return dic


def _tokenize_punctuation(text):
    filtered_text = text
    while True:
        aux = filtered_text
        filtered_text = repeated_chars_regex.sub(lambda m: _tokenize_repeat_chars(m), filtered_text)
        filtered_text = repeated_chars_regex_3.sub(lambda m: _tokenize_repeat_chars(m, 2), filtered_text)
        if aux == filtered_text:
            break
    filtered_text = repeated_admiration_regex.sub(lambda m: _tokenize_repeat_sing(m), filtered_text)
    return filtered_text


def _tokenize_repeat_sing(match):
    if match.group('sig') == '¡':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'OPEN_EXCLAMATION '
    elif match.group('sig') == '¿':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'OPEN_INTERROGATION '
    elif match.group('sig') == '!':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'CLOSE_EXCLAMATION '
    elif match.group('sig') == '?':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'CLOSE_INTERROGATION '
    elif match.group('sig') == ',':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'COMMA '
    elif match.group('sig') == ':':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'COLON '
    elif match.group('sig') == ';':
        return (' TK_REPEATED_' if len(match.group('all')) > 1 else ' TK_') + 'SEMICOLON '
    elif match.group('sig') == '.':
        if len(match.group('all')) == 1:
            return ' TK_POINT '
        elif len(match.group('all')) == 2:
            return ' TK_REPEATED_POINT '
        elif len(match.group('all')) == 3:
            return ' TK_SUSPENSION_POINTS '
        elif len(match.group('all')) > 3:
            return ' TK_REPEATED_SUSPENSION_POINTS '


# Caracteres repetidos consecutivamente del castellano:
# - muy habituales: ll, rr, cc
# - poco habituales: ee, oo, zz - ej. microondas, pizza, leer, hollywood
def _tokenize_repeat_chars(match, rep=1):
    """"""
    if match.group('char') == '_':
        return match.group('char') + match.group('suffix') + ' TK_REPEATED_CHAR'
    if match.group('chars').lower() in {'ll', 'rr', 'cc'}:
        return match.group('chars') + match.group('suffix')
    elif match.group('chars').lower() in {'tt', 'ee', 'oo', 'zz'}:
        return match.group('char') + match.group('suffix') + ' TK_REPEATED_CHAR_' + match.group('char').upper()
    else:
        return match.group('char') * rep + match.group('suffix') + ' TK_REPEATED_CHAR_' + match.group('char').upper()


def _remove_stop_words(text):
    # Tokenizar el texto
    tokens = word_tokenize(text, language='spanish')
    # Stop words en español
    stop_words = set(stopwords.words('spanish'))
    # Eliminar stop words
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    # Unir los tokens en un texto
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def _remove_simbols(text):
    # Eliminar signos de puntuación del texto
    filtered_text = re.sub(r'[^\w\s]+', '', text)
    return filtered_text


def _tokenize_simbols(text):
    # Tokenizar signos de puntuación del texto
    filtered_text = text
    filtered_text = filtered_text.replace('@', ' TOKEN_AT ')
    filtered_text = filtered_text.replace('#', ' TK_HASH ')
    filtered_text = filtered_text.replace('/', ' TK_SLASH ')
    filtered_text = filtered_text.replace('\\', ' TK_BACK_SLASH ')
    filtered_text = filtered_text.replace('&', ' TK_AMPERSAND ')
    filtered_text = filtered_text.replace('%', ' TK_PERCENT ')
    filtered_text = filtered_text.replace('+', ' TK_PLUS ')
    filtered_text = filtered_text.replace('$', ' TK_DOLLAR ')
    filtered_text = filtered_text.replace('-', ' TK_MINUS ')
    filtered_text = filtered_text.replace('€', ' TK_EURO ')
    filtered_text = filtered_text.replace('=', ' TK_EQUAL ')
    filtered_text = filtered_text.replace('£', ' TK_POUND ')
    filtered_text = filtered_text.replace('*', ' TK_ASTERISK ')
    filtered_text = filtered_text.replace('º', ' TK_ORDINAL_O ')
    filtered_text = filtered_text.replace('ª', ' TK_ORDINAL_A ')
    filtered_text = filtered_text.replace('(', ' TK_OPEN_PARENTHESIS ')
    filtered_text = filtered_text.replace(')', ' TK_CLOSE_PARENTHESIS ')
    filtered_text = filtered_text.replace('[', ' TK_OPEN_BRACKET ')
    filtered_text = filtered_text.replace(']', ' TK_CLOSE_BRACKET ')
    filtered_text = filtered_text.replace('{', ' TK_OPEN_BRACE ')
    filtered_text = filtered_text.replace('}', ' TK_CLOSE_BRACE ')
    filtered_text = filtered_text.replace('<', ' TK_OPEN_ANGLE_BRACKET ')
    filtered_text = filtered_text.replace('>', ' TK_CLOSE_ANGLE_BRACKET ')
    filtered_text = filtered_text.replace('«', ' TK_OPEN_DOUBLE_ANGLE_BRACKET ')
    filtered_text = filtered_text.replace('»', ' TK_CLOSE_DOUBLE_ANGLE_BRACKET ')
    filtered_text = filtered_text.replace('\'', ' TK_SINGLE_QUOTE ')
    filtered_text = filtered_text.replace('\"', ' TK_DOUBLE_QUOTE ')
    filtered_text = filtered_text.replace('`', ' TK_BACK_QUOTE1 ')
    filtered_text = filtered_text.replace('´', ' TK_BACK_QUOTE2 ')
    filtered_text = filtered_text.replace('’', ' TK_SINGLE_QUOTE1 ')
    filtered_text = filtered_text.replace('‘', ' TK_SINGLE_QUOTE2 ')
    filtered_text = filtered_text.replace('“', ' TK_DOUBLE_QUOTE1 ')
    filtered_text = filtered_text.replace('”', ' TK_DOUBLE_QUOTE2 ')
    filtered_text = filtered_text.replace('—', ' TK_DASH1 ')
    filtered_text = filtered_text.replace('–', ' TK_DASH2 ')
    filtered_text = filtered_text.replace('•', ' TK_BULLET1 ')
    filtered_text = filtered_text.replace('·', ' TK_BULLET2 ')
    filtered_text = filtered_text.replace('§', ' TK_SECTION ')
    filtered_text = filtered_text.replace('†', ' TK_DAGGER ')
    filtered_text = filtered_text.replace('‡', ' TK_DOUBLE_DAGGER ')
    filtered_text = filtered_text.replace('…', ' TK_ELLIPSIS ')
    filtered_text = filtered_text.replace('‰', ' TK_PER_MIL ')
    filtered_text = filtered_text.replace('¶', ' TK_PILCROW ')
    filtered_text = filtered_text.replace('©', ' TK_COPYRIGHT ')
    filtered_text = filtered_text.replace('®', ' TK_REGISTERED ')
    filtered_text = filtered_text.replace('™', ' TK_TRADEMARK ')
    filtered_text = filtered_text.replace('°', ' TK_DEGREE ')
    filtered_text = filtered_text.replace('¢', ' TK_CENT ')
    filtered_text = filtered_text.replace('¥', ' TK_YEN ')
    filtered_text = filtered_text.replace('¤', ' TK_CURRENCY ')
    filtered_text = filtered_text.replace('ƒ', ' TK_FLORIN ')
    filtered_text = filtered_text.replace('¬', ' TK_NEGATION ')
    filtered_text = filtered_text.replace('ˆ', ' TK_CIRCUMFLEX ')
    filtered_text = filtered_text.replace('˜', ' TK_TILDE ')
    filtered_text = filtered_text.replace('˚', ' TK_RING ')
    filtered_text = re.sub(r'[^\w\s]', ' TK_SIMBOL ', filtered_text)
    return filtered_text


def _remove_numbers(text):
    # Eliminar números del texto
    filtered_text = re.sub(r'\d+|\d+\.\d+', '', text)
    return filtered_text


def _remove_accents(text):
    # Sustituir caracteres acentuados por caracteres sin acentuar
    text_without_accents = unidecode(text)
    return text_without_accents


"""
Clase TfidfVectorizer:
  @ref https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  · input: especifica el tipo de entrada que se utilizará. Puede ser 'content' (texto), 'file' (archivo) o 'filename' (nombre de archivo).
  · encoding: especifica la codificación de caracteres del corpus de texto.
  · decode_error: especifica cómo manejar los errores de decodificación de caracteres.
  · strip_accents: especifica si se deben eliminar los acentos del texto.
  · lowercase: especifica si se deben convertir todos los caracteres del texto a minúsculas.
  · preprocessor: especifica una función que se utilizará para preprocesar el texto antes de tokenizarlo.
  · tokenizer: especifica una función que se utilizará para tokenizar el texto.
  · stop_words: especifica las palabras de parada que se deben eliminar del texto.
  · token_pattern: especifica el patrón de tokenización que se utilizará para dividir el texto en términos o tokens.
  · ngram_range: especifica el rango de n-gramas que se deben generar a partir del texto.
  · analyzer: especifica el análisis que se utilizará para generar las características.
  · max_df: especifica el máximo número de documentos en los que un término debe aparecer para ser considerado como una característica.
  · min_df: especifica el mínimo número de documentos en los que un término debe aparecer para ser considerado como una característica.
  · max_features: especifica el número máximo de características que se deben generar.
  · vocabulary: especifica el vocabulario que se utilizará para generar las características.
  · binary: especifica si se deben generar características binarias o contar la frecuencia de términos.
  · dtype: especifica el tipo de datos que se utilizará para almacenar la matriz de características.
  · norm: especifica la normalización que se debe aplicar a las características.

  · sublinear_tf:
    La opción sublinear_tf en el TfidfVectorizer de scikit-learn se utiliza para aplicar una 
    transformación sublineal a las frecuencias de los términos en el corpus de texto. La 
    transformación sublineal se utiliza para suavizar la importancia de los términos que aparecen 
    con alta frecuencia en el corpus, ya que estos términos pueden tener un gran impacto en el 
    cálculo del peso TF-IDF.

    La transformación sublineal que se utiliza en sublinear_tf es la función logarítmica. En lugar 
    de utilizar la frecuencia absoluta de un término en el corpus, se utiliza el logaritmo de la 
    frecuencia. Esto ayuda a reducir la importancia de los términos que aparecen con alta frecuencia 
    en el corpus, ya que el logaritmo de un número grande es menor que el número en sí.

    Por defecto, sublinear_tf=False, lo que significa que no se aplica la transformación sublineal 
    a las frecuencias de los términos y se utiliza la frecuencia absoluta. Al establecer 
    sublinear_tf=True, se habilita la transformación sublineal.

  """


def tf_idf(corpus, ngram_range=(1, 1), max_df=1.0):
    # Vectorizar el corpus
    vectorizer = TfidfVectorizer(
        stop_words=stopwords.words('spanish'),
        token_pattern=r'\b\w\w+\b',
        # Ignorar los términos que tienen una frecuencia de documentos estrictamente mayor al umbral. Si es un entero
        # se refiere al conteo absoluto del término. (1.0 o 1 = no ignorar).
        max_df=max_df,
        # Al construir el vocabulario ignora los términos que tienen una frecuencia de documentos estrictamente
        # inferior al umbral dado. Este valor también se llama corte en la literatura. Si es un entero se refiere al
        # conteo absoluto del término. (0.0 o 0 = no ignorar).
        min_df=10,
        sublinear_tf=True,
        strip_accents='unicode',
        lowercase=True,
        ngram_range=ngram_range
    )
    tf_idf_matrix = vectorizer.fit_transform(corpus)

    df = pd.DataFrame(tf_idf_matrix.toarray(),
                      columns=vectorizer.get_feature_names())
    return df


def _size_mb(docs):
    return sum(len(c.encode("utf-8")) + sum(len(s.encode("utf-8")) for s in docs[c]) for c in docs[['text']]) / 1e6


# def _balance(X, y, replace=True, random_state=1):
#     balanced = X.copy()
#     # Añadir una columna con la clase de cada registro
#     class_label = '_class_'
#     balanced[class_label] = y
#     balanced = balance_class(balanced, class_label=class_label, random_state=random_state, replace=replace)
#     y_balanced = balanced[class_label]
#     x_balanced = balanced.drop([class_label], axis=1)
#     return x_balanced, y_balanced


def process_dataset(dataset, class_label, corpus_field='text', verbose=False, lowercase=False, smooth_idf=False,
                    sublinear_tf=False, ngram_range=(1, 1), binarize=False, target=None,
                    balance=None, balance_to_target=False, strip_accents=None, max_df=1.0, min_df=1,
                    stop_words=stopwords.words('spanish'), test_size=0.2, random_state=0, balance_passive=True):
    """
    Parameters
    ----------
    class_label :
        Nombre de la columna que contiene la clase.
    dataset :
        Dataset a procesar
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an extra document
        was seen containing every term in the collection exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    corpus_field:
        Nombre de la columna que contiene el texto.
    verbose:
        Imprimir información adicional.
    lowercase:
        Convertir el texto a minúsculas.
    ngram_range:
        Rango de n-gramas que se deben generar a partir del texto.
    strip_accents:
        Eliminar caracteres acentuados.
    max_df:
        When building the vocabulary ignore terms that have a document frequency strictly higher than the given
        threshold (corpus-specific stop words). If float in range [0.0, 1.0], the parameter represents a proportion
        of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
    min_df:
        When building the vocabulary ignore terms that have a document frequency strictly lower than the given
        threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter
        represents a proportion of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.
    stop_words:
        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
        Only applies if analyzer == 'word'.
        If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically
        detect and filter stop words based on intra corpus document frequency of terms.
    test_size:
        Proporción de los datos que se utilizarán para la prueba.
    binarize:
        Si la clase es binaria. Si es True, se binariza, se cambia el valor de la clase a 'other' para todas las clases
        que no coincidan con `target`, o al más frecuente si `target` es None.
    balance: str {'all_set', 'sub_set'} or None, default=None
        Balancear el número de elementos de cada clase. Si es 'all_set', se balancea el dataset antes de dividirlo
        en los conjuntos de entrenamiento y prueba. Si es 'sub_set', si es 'sub_set', se balancean los conjuntos de
        entrenamiento y prueba por separado.
    balance_to_target: bool, default=False
        Establece si se balancea el dataset respecto el tamaño de la clase objetivo.
        Solo aplica si balance=True, o si binarize=True.
        Cuando balance=True, se utiliza para seleccionar a que valor se balancea el dataset. Si es True, se
        balancean las clases al tamaño de target. Si es False, se balancean las clases al tamaño de la clase
        más grande.
        Cuando binarize=True, se utiliza para seleccionar a que valor se balancea el dataset. Si es True, se
        balancea al tamaño de la clase que no se ha cambiado a 'other'.
    target: str or None, default=None
        Selecciona un valor de clase como true class. Solo aplica si binary_class=True, o si balance_passive=True.
        Cuando binary_class=True, se utiliza para seleccionar el valor de la clase que se utilizará para
        binarizar la clase. Si es None, se selecciona el autor más frecuente.
        Cuando balance_passive=True, se utiliza para seleccionar el valor de la clase que se utilizará para
        balancear el dataset. Si es None, se selecciona el autor más frecuente.
    random_state: int, default=0
        Semilla para la generación de números aleatorios.
    """
    data = pd.DataFrame(dataset.copy())

    # DOCUMENTACION BINARIZER SCIKIT LEARN
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html#sklearn.preprocessing.LabelBinarizer

    # # Recuperar todos los nombres de los autores
    # authors = data["author"].unique()
    # # Para cada autor sustituir su nombre por 'author' y un número, y buscar y cambiar su nombre en todos los textos
    # datos3 = data.copy()
    # for i, author in enumerate(authors):
    #     # definir un regex que busque y sustituya todas las coincidencias con el nombre independientemente de mayúsculas y minuscules
    #     datos3["text"] = datos3["text"].apply(lambda x: re.sub(r'(?i)\b' + author + r'\b', "A" + str(i+1), x))
    #     # datos3["text"] = datos3["text"].apply(lambda x: x.replace(author, "AUTHOR" + str(i)))
    #     datos3["author"] = datos3["author"].apply(lambda x: x.replace(author, "A" + str(i+1)))
    # data = datos3.copy()

    data_statistics = dict({'in_data': show_statistics(data, title='In Data Stats') if verbose else statistics(data)})
    # Si solo 2 clases
    if binarize:
        target = _binarize_class(class_label, data, target)
    # Balancear el número de elementos de cada clase
    # Si balance_to_target es True, se balancean las clases al tamaño de target
    if balance == 'all_set':
        data = balance_class(data, class_label=class_label, random_state=random_state,
                             n_samples=_n_samples(data, balance_passive, balance_to_target, class_label, target))
        in_data_balanced_statistics = show_statistics(
            data, title='Info dataset balanced', author_label=class_label) if verbose else statistics(data)
        data_statistics.update({'in_data_balanced': in_data_balanced_statistics})
    target_names = np.unique(data[class_label])
    y = data[class_label]
    x = data.drop(class_label, axis=1)
    # Separar los datos en los conjuntos de entrenamiento y prueba
    x_train_data, x_test_data, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    #
    # Balancear el número de elementos de cada clase solo en el conjunto de entrenamiento
    if balance == 'sub_set':
        x_train_data, y_train = _balance_subset(
            x_train_data, y_train, class_label, target, balance_to_target, data_statistics, random_state, verbose,
            title='Info training data balanced - No balanced training data use to fit vectorizer',
            dict_key='training_data_balanced', balance_passive=balance_passive)
        x_test_data, y_test = _balance_subset(
            x_test_data, y_test, class_label, target, balance_to_target, data_statistics, random_state, verbose,
            title='Info test data balanced', dict_key='test_data_balanced', balance_passive=balance_passive)

    #
    # para generar graficas para la memoria
    # traint = pd.concat([x_train_data, y_train], axis='columns')
    # # Dejar solo 20 archivos para el autor Sendero
    # no_author0 = traint[traint["author"] != "A1"]
    # author0 = traint[traint["author"] == "A1"]
    # # Seleccionar aleatoriamente 20 documentos del autor Sendero
    # # no_author0 = no_author0.sample(n=10)
    # # author0 = author0.sample(n=90)
    # # Concatenar los dos DataFrames
    # datos4 = pd.concat([no_author0.sample(n=1000), author0.sample(n=1000)])
    # traint = datos4.sample(frac=1)
    # x_train_data, y_train = traint.drop(class_label, axis='columns'), traint[class_label]
    #
    # test = pd.concat([x_train_data, y_train], axis='columns')
    # # test = pd.concat([x_test_data, y_test], axis='columns')
    # # Dejar solo 20 archivos para el autor Sendero
    # no_author0 = test[test["author"] != "A1"]
    # author0 = test[test["author"] == "A1"]
    # # Seleccionar aleatoriamente 20 documentos del autor Sendero
    # # no_author0 = no_author0.sample(n=10)
    # # author0 = author0.sample(n=90)
    # # Concatenar los dos DataFrames
    # datos4 = pd.concat([no_author0.sample(n=100), author0.sample(n=100)])
    # test = datos4.sample(frac=1)
    # x_test_data, y_test = test.drop(class_label, axis='columns'), test[class_label]

    t0 = time()
    vectorizer = TfidfVectorizer(
        smooth_idf=smooth_idf,
        ngram_range=ngram_range,
        token_pattern=r'\b\w+\b',
        sublinear_tf=sublinear_tf,  # True, False
        max_df=max_df,
        min_df=min_df,
        strip_accents=None if not strip_accents else 'unicode',  # 'ascii', 'unicode', None
        lowercase=lowercase,
        stop_words=None if not stop_words else stopwords.words('spanish'),
        use_idf=True,
    )
    # TODO IMPORTANTE: hacer fit() en vez de fit_transform(), y antes de balancear las clases, para que calcular las
    #  frecuencias en documentos (df) sin los posibles registros duplicados en el caso de que se balanceen las
    #  clases solo en el conjunto de entrenamiento
    # vectorizer.fit(x_train_data[corpus_field])
    duration_fit = time() - t0
    #
    # # Balancear el número de elementos de cada clase solo en el conjunto de entrenamiento
    # if balance == 'no_sub_set':
    #     # n_samples = data[data[class_name] == target].shape[0] if balance_to_target and target else None
    #     to_balance = pd.concat([x_train_data, y_train], axis='columns')
    #     balanced_train_data = balance_class(to_balance, class_label=class_label, n_samples=n_samples, random_state=random_state)
    #     train_data_balanced_stats = show_statistics(
    #         balanced_train_data, title='Only training data balanced and do not use to fit vectorizer',
    #         author_label=class_label) if verbose else statistics(x_train_data)
    #     data_statistics.update({'training_data_balanced': train_data_balanced_stats})
    #     x_train_data = balanced_train_data.drop(class_label, axis='columns')
    #     y_train = balanced_train_data[class_label]
    #
    # Se calcula TF-IDF usando las frecuencias de documentos (df) calculadas en el conjunto de entrenamiento sin tener
    # en cuenta los posibles registros duplicados en el caso de que se balanceen las clases solo en el conjunto de
    # entrenamiento
    t0 = time()
    x_train = vectorizer.fit_transform(x_train_data[corpus_field])
    # x_train = vectorizer.transform(x_train_data[corpus_field])
    duration_train = time() - t0
    # Extracción de características de los datos de prueba utilizando el mismo vectorizador
    # usando los valores de los parámetros calculados en el conjunto de entrenamiento
    t0 = time()
    x_test = vectorizer.transform(x_test_data[corpus_field])
    duration_test = time() - t0
    #
    feature_names = vectorizer.get_feature_names_out()
    data_train_size_mb = _size_mb(x_train_data)
    data_test_size_mb = _size_mb(x_test_data)
    metadata = {
        # 'feature_names': feature_names.tolist(),
        'target_names': target_names.tolist(),  # np.unique(y)
        'data_train_size_mb': data_train_size_mb,
        'data_test_size_mb': data_test_size_mb,
        'duration_fit': duration_fit,
        'duration_train': duration_train,
        'duration_test': duration_test,
        'n_train': len(x_train_data),
        'n_test': len(x_test_data),
        'n_classes': len(target_names),
        'n_samples': len(x_train_data) + len(x_test_data),
        'n_features': len(feature_names),
        'n_train_documents_by_class': y_train.value_counts().to_dict(),
        'n_test_documents_by_class': y_test.value_counts().to_dict(),
        'n_gramas': ngram_range,
        'stop_words': stop_words,
        'lowercase': lowercase,
        'strip_accents': strip_accents,
        'smooth_idf': smooth_idf,
        'sublinear_tf': sublinear_tf,
        'max_df': max_df,
        'min_df': min_df,
        'test_size': test_size,
        'balance': balance,
        'binarize': binarize,
        'balance_to_target': balance_to_target,
        'target': target,
        'corpus_field': corpus_field,
        'class_name': class_label,
        'data_stats': data_statistics,
        # 'TfidfVectorizer_params': vectorizer.get_params(deep=True),
    }
    # compute size of loaded data
    salida = _string_data_processed(
        data_test_size_mb, data_train_size_mb, duration_test, duration_train,
        feature_names, target_names, x_test_data, x_train_data)
    metadata.update({'info': salida})
    if verbose:
        print(salida)
    # # Convertir la matriz de CSR en un DataFrame
    # X_train = pd.DataFrame.sparse.from_spmatrix(X_train)
    # X_test = pd.DataFrame.sparse.from_spmatrix(X_test)
    # for field in more_fields:
    #     X_train[field] = X_train_data[field].values
    #     X_test[field] = X_test_data[field].values
    return x_train, x_test, y_train, y_test, feature_names, target_names, metadata


def _balance_subset(X, y, class_label, target, balance_to_target, data_statistics, random_state,
                    verbose, title, dict_key, balance_passive):
    to_balance = pd.concat([X, y], axis='columns')
    balanced_train_data = balance_class(to_balance, class_label=class_label, random_state=random_state,
                                        n_samples=_n_samples(to_balance, balance_to_target=balance_to_target, class_label=class_label, target=target, balance_passive=balance_passive))
    train_data_balanced_stats = show_statistics(
        balanced_train_data, title=title, author_label=class_label) if verbose else statistics(X)
    data_statistics.update({dict_key: train_data_balanced_stats})
    return balanced_train_data.drop(class_label, axis='columns'), balanced_train_data[class_label]


def _n_samples(data, balance_passive, balance_to_target, class_label, target):
    if balance_passive:
        # devolver el numero de documentos del autor con menos documentos
        return data[data[class_label] == data[class_label].value_counts().idxmin()].shape[0]
    return data[data[class_label] == target].shape[0] if balance_to_target and target in data[
        class_label].unique() else None


def _string_data_processed(data_test_size_mb, data_train_size_mb, duration_test, duration_train, feature_names,
                           target_names, x_test_data, x_train_data):
    ancho = 90
    head = "Info data set processed "
    tab_head = " " * ((ancho - len(head) - 2) // 2)
    output = io.StringIO()
    print("*" * ancho, "*" + tab_head + head + tab_head + "*", "*" * ancho, sep="\n", file=output)
    print(" " * 2 + f"{len(target_names)} Classes: {target_names}", sep="\n", file=output)
    # " " * 2 + f"{x_train.shape[1]} Features", sep="\n", file=output)
    print("-" * ancho, " " * 2 + "Training set:", sep="\n", file=output)
    print(" " * 4 + f"vectorize training done in {duration_train:.3f} seg. "
                    f"at {data_train_size_mb / duration_train:.3f} MB/s", file=output)
    print(" " * 4 + f"{len(x_train_data)} documents - {data_train_size_mb:.2f}MB (training set)", file=output)
    print("-" * ancho, " " * 2 + "Test set:", sep="\n", file=output)
    print(" " * 4 + f"vectorize testing done in {duration_test:.3f} seg. "
                    f"at {data_test_size_mb / duration_test:.3f} MB/s", file=output)
    print(" " * 4 + f"{len(x_test_data)} documents - {data_test_size_mb:.2f} MB (test set)", file=output)
    print("-" * ancho, " " * 2 + f"Features: {feature_names.shape[0]}", sep="\n", file=output)
    print("*" * ancho, file=output)
    salida = output.getvalue()
    return salida


def _binarize_class(class_name: str, dataset: DataFrame, binary_target: str = None) -> str:
    if binary_target is not None and binary_target in dataset[class_name].unique():
        true_class = binary_target
    else:
        if binary_target is not None:
            print(f'El autor {binary_target} no está en el dataset\n')
        # agrupamos los datos por autor y contamos el número de registros de cada uno
        author_counts = dataset.groupby(class_name).size().reset_index(name='counts')
        author_counts = author_counts.sort_values(by='counts', ascending=False)
        # seleccionamos el autor con mayor número de registros
        true_class = author_counts.iloc[0][class_name]
    # sustituimos los demás autores por 'other'
    dataset[class_name] = dataset[class_name].apply(lambda x: x if x == true_class else 'other')
    return true_class


def __load_dataset(X, y, verbose=False, lowercase=False, smooth_idf=False, sublinear_tf=False, ngram_range=(1, 1),
                   strip_accents=None, max_df=1.0, min_df=1, stop_words=stopwords.words('spanish'), test_size=0.2):
    """
    smooth_idf : bool, default=True Smooth idf weights by adding one to document frequencies, as if an extra document
    was seen containing every term in the collection exactly once. Prevents zero divisions.

    sublinear_tf : bool, default=False
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    """
    target_names = np.unique(y)
    #
    # Separar los datos en los conjuntos de entrenamiento y prueba
    X_train_data, X_test_data, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)
    #
    # Extraer características de los datos de entrenamiento usando un vectorizador TfidfVectorizer
    t0 = time()
    vectorizer = TfidfVectorizer(
        smooth_idf=smooth_idf,
        ngram_range=ngram_range,
        token_pattern=r'\b\w+\b',
        sublinear_tf=sublinear_tf,  # True, False
        max_df=max_df,
        min_df=min_df,
        strip_accents=strip_accents,  # 'ascii', 'unicode', None
        lowercase=lowercase,
        stop_words=stop_words
    )
    # Se calcula TF-IDF con el conjunto de entrenamiento y se transforma
    # el conjunto de entrenamiento en una matriz TF-IDF
    X_train = vectorizer.fit_transform(X_train_data)
    duration_train = time() - t0
    #
    # Extracción de características de los datos de prueba utilizando el mismo vectorizador
    # usando los valores de los parámetros calculados en el conjunto de entrenamiento
    t0 = time()
    X_test = vectorizer.transform(X_test_data)
    duration_test = time() - t0
    #
    feature_names = vectorizer.get_feature_names_out()
    #
    if verbose:
        # compute size of loaded data
        data_train_size_mb = _size_mb(X_train_data)
        data_test_size_mb = _size_mb(X_test_data)
        print(
            f"{len(X_train_data)} documents - "
            f"{data_train_size_mb:.2f}MB (training set)"
        )
        print(f"{len(X_test_data)} documents - {data_test_size_mb:.2f} MB (test set)")
        print(f"{len(target_names)} categories")
        print(
            f'vectorize training done in {duration_train:.3f} s '
            f'at {data_train_size_mb / duration_train:.3f} MB/s'
        )
        print(f"n_samples: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
        print(
            f"vectorize testing done in {duration_test:.3f} s "
            f"at {data_test_size_mb / duration_test:.3f} MB/s"
        )
        print(f"n_samples: {X_test.shape[0]}, n_features: {X_test.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names, target_names


def __generar_diccionario(ruta_fichero="spanish_words.dic"):
    """
    Genera un diccionario de palabras en español a partir de la página web https://www.palabrasaleatorias.com/
    """
    url = 'https://www.palabrasaleatorias.com/?fs=1&fs2=0&Submit=Nueva+palabra'
    if os.path.isfile(ruta_fichero):
        print("El fichero existe.")
        # Cargar la lista desde el archivo
        with open(ruta_fichero, "rb") as file:
            dic = pickle.load(file)
    else:
        print("El fichero no existe.")
        dic = []
    print("Tamaño:", len(dic))
    for i in range(10000):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        palabra = soup.select_one('body center center div').text.split()[0].lower()
        dic.append(palabra)
        print("Tamaños:", len(dic), "y", len(set(dic)), "\tAñadido:", palabra)
        sleep(random.randint(1, 100) / 100.0)
    # Guardar las palabras en un archivo serializado
    with open(ruta_fichero, "wb") as file:
        # Serializar la lista en el archivo
        pickle.dump(list(set(dic)), file)
    print(dic)


# ##############################################################
# Tests
# ##############################################################

def _stop_words_test():
    text = "Este es un ejemplo de cómo eliminar las stop words del español de un corpus de texto"
    text2 = _remove_stop_words(text)
    print(text)
    print(text2)


# def remove_urls(text):
#     filtered_text = re.sub(url_regex, 'URLTOKEN', text)
#     if filtered_text != text:
#         print("URLS eliminadas en el texto: ", text, "\n")
#     return filtered_text


# print(re.sub(url_regex, 'URLTOKEN', "https://www.ejemplo3.com/path/to/page.html"))

def _remove_urls_test2():
    urls = [
        "https://www.ejemplo1.com",
        "http://www.ejemplo2.com",
        "https://www.ejemplo3.com/path/to/page.html",
        "http://www.ejemplo4.com/path/to/page.html",
        "https://www.ejemplo5.com/?q=query",
        "http://www.ejemplo6.com/?q=query",
        "https://www.ejemplo7.com/path/to/page.html?q=query",
        "http://www.ejemplo8.com/path/to/page.html?q=query",
        "https://www.ejemplo9.com/path/to/page.html#anchor",
        "http://www.ejemplo10.com/path/to/page.html#anchor",
        "https://www.ejemplo11.com/?q=query#anchor",
        "http://www.ejemplo12.com/?q=query#anchor",
        "https://www.ejemplo13.com/path/to/page.html?q=query#anchor",
        "http://www.ejemplo14.com/path/to/page.html?q=query#anchor",
        "ftp://ftp.ejemplo15.com",
        "ftps://ftps.ejemplo16.com",
        "ftp://ftp.ejemplo17.com/path/to/file.txt",
        "ftps://ftps.ejemplo18.com/path/to/file.txt",
        "www.ejemplo19.com",
        "www.ejemplo20.com/path/to/page.html",
        "www.ejemplo21.com/?q=query",
        "www.ejemplo22.com/path/to/page.html?q=query",
        "www.ejemplo23.com/path/to/page.html#anchor",
        "www.ejemplo24.com/?q=query#anchor",
        "www.ejemplo25.com/path/to/page.html?q=query#anchor",
        "https://ejemplo26.com",
        "http://ejemplo27.com",
        "https://ejemplo28.com/path/to/page.html",
        "http://ejemplo29.com/path/to/page.html",
        "https://ejemplo30.com/?q=query",
        "http://ejemplo31.com/?q=query",
        "https://ejemplo32.com/path/to/page.html?q=query",
        "http://ejemplo33.com/path/to/page.html?q=query",
        "https://ejemplo34.com/path/to/page.html#anchor",
        "http://ejemplo35.com/path/to/page.html#anchor",
        "https://ejemplo36.com/?q=query#anchor",
        "http://ejemplo37.com/?q=query#anchor",
        "https://ejemplo38.com/path/to/page.html?q=query#anchor",
        "http://ejemplo39.com/path/to/page.html?q=query#anchor",
        "ftp://ftp.ejemplo40.com",
        "ftps://ftps.ejemplo41.com",
        "ftp://ftp.ejemplo42.com/path/to/file.txt",
        "ftps://ftps.ejemplo43.com/path/to/file.txt",
        "ejemplo44.com",
        "ejemplo45.com/path/to/page.html",
        "ejemplo46.com/?q=query",
        "ejemplo47.com/path/to/page.html?q=query",
        "ejemplo48.com/path/to/page.html#anchor",
        "ejemplo49.com/?q=query#anchor",
        "ejemplo50.com/path/to/page.html?q=query#anchor"
    ]
    urls_copy = urls.copy()
    for i in range(len(urls)):
        urls2 = _tokenize_urls(urls_copy[i], verbose=True)
        if not re.fullmatch(r'URLTOKEN_\w+', urls2):
            raise EraseUrlException(
                "Error al eliminar la URL: ", urls[i], urls2)


def _remove_urls_test():
    # Texto con URLs
    text = "Este es un ejemplo de cómo eliminar las URLs de un texto, como  www.forodetalles.com, " \
           "https://www.example.com o http://example.com"
    text2 = _tokenize_urls(text, verbose=True)


def _separate_numbers_test():
    # Texto con números
    text = "Este es un ejemplo de cómo separar los números del texto, como nombre1234, 1234, 12.34, 12,34, 12-34, " \
           "12/34, 12:34, 12;34, 12_34, 12*34, 12<34, 12>34, 12(34, 12)34, 12[34, 12]34, 12{34, 12}34, 12!34, 12?34, " \
           "12¿34, 12¡34, 12=34, 12+34, 12-34, 12%34, 12&34, 12$34, 12#34, 12@34, 12|34, 12€34, 12£34, 12¥34, 12¢34, " \
           "12₡34, 12₢34, 12₣34, 12₤34, 12₥34, 12₦34, 12₧34, 12₨34, 12₩34, 12₪34, 12₫34, 12₭34, 12₮34, 12₯34, 12₰34, " \
           "12₱34, 12₲34, 12₳34, 12₴34, 12₵34, 12₶34, 12₷34, 12₸34, 12₹34, 12₺34, 12₻34, 12₼34, 12₽34, 12₾34, 12₿34"
    text2 = _tokenize_numbers(text)
    print(text)
    print(text2)


def _tokenize_emotes_test():
    text = "yo estoy sentada si srta. kiss por favor acuda al aparato :cuñao: ....a ese no , al otro oye mándame un " \
           "mail a donde te he puesto porfa plis, es urgente"
    text2 = _tokenize_emoticons(text, verbose=True)
    print(text)
    print(text2)


def _remove_punctuation_test():
    # Texto con signos de puntuación
    text = "Este es un ejemplo de cómo eliminar los signos de puntuación del texto, como . , : ; - _ ! ? <>*"
    text2 = _remove_simbols(text)
    print(text)
    print(text2)


def _tokenize_punctuation_test():
    # Texto con signos de puntuación
    text = "Este es un ejemplo de cómo eliminar los signos de puntuación del texto, como . , : ; - _ ! ? <>*"
    text2 = _tokenize_simbols(text)
    print(text)
    print(text2)


def _remove_accents_test():
    # Texto con caracteres acentuados
    text = "Este es un ejemplo de cómo sustituir los caracteres acentuados por caracteres sin acentuar, como á é í ó " \
           "ú â ê î ô û ä ë ï ö ü à è ì ò ù ã ñ õ ç"
    text2 = _remove_accents(text)
    print(text)
    print(text2)


def _tokenize_punctuation_test():
    texts = [
        "Esteee microondas... pizza leer hollywoood! microoondas!! pizzza! leeer!!! hollywoood!!! ......",
        "¿Esteee textooo tiiene muchhoss caaaracterees repeetidoss cooooonsssseccccuuuuuutivaaaaaamenteeee......?",
        "¿Esteee textooo? ¡tiiene! muchhoss???? caaaracterees!!!! repeetidoss!!!? consecutivamenteeee......",
        "¿Este texto? ¡tiene! muchos???? caracteres!!!! repetidos!!!? consecutivamente......",
    ]
    for t in texts:
        print(t)
        print(_tokenize_punctuation(t))


def run_tests():
    _tokenize_punctuation_test()
    _tokenize_emotes_test()
    _separate_numbers_test()
    _stop_words_test()
    _remove_urls_test()
    _remove_urls_test2()
    _remove_punctuation_test()
    _remove_accents_test()

# run_tests()
