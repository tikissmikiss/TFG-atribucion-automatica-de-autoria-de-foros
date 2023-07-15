import json
import pickle
import sys
from datetime import datetime
from time import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from tfg.datos import DataBase
from tfg.graphics import generar_matriz_confusion
from tfg.utils import create_folder


class ClassifierSelector:
    """Clase para seleccionar el clasificador a utilizar.

    Parameters
    ----------

    classifier_name : str
        Nombre del clasificador a utilizar. Debe ser uno de los siguientes:

        - 'RandomForestClassifier'
        - 'GradientBoostingClassifier'
        - 'LogisticRegression'
        - 'LogisticRegressionCV'
        - 'RidgeClassifier'
        - 'RidgeClassifierCV'
        - 'BernoulliNB'
        - 'NearestCentroid'
        - 'LinearSVC'
        - 'NuSVC'
        - 'SVC'
        - 'DecisionTreeClassifier'
        - 'ExtraTreeClassifier'
        - 'MLPClassifier'

    random_state : int, default=1
        Semilla para el generador de números aleatorios.

    verbose : bool, default=False
        Mostrar información sobre el clasificador seleccionado.
    """

    _classifiers = [
        'BernoulliNB',
        'GaussianNB',
        'RidgeClassifier',
        'LogisticRegression',
        'NearestCentroid',
        'LinearSVC',
        'NuSVC',
        'SVC',
        'RandomForestClassifier',
        'DecisionTreeClassifier',
        'ExtraTreeClassifier',
        'GradientBoostingClassifier',
        'LogisticRegressionCV',
        'RidgeClassifierCV',
        'MLPClassifier',
    ]

    def __init__(self, classifier_name: str, random_state=1, verbose=False):
        """Inicializa el selector de clasificador.

        Parameters
        ----------

        classifier_name : str
            Nombre del clasificador a utilizar. Debe ser uno de los siguientes:

            - 'RandomForestClassifier'
            - 'GradientBoostingClassifier'
            - 'LogisticRegression'
            - 'LogisticRegressionCV'
            - 'RidgeClassifier'
            - 'RidgeClassifierCV'
            - 'BernoulliNB'
            - 'NearestCentroid'
            - 'LinearSVC'
            - 'NuSVC'
            - 'SVC'
            - 'DecisionTreeClassifier'
            - 'ExtraTreeClassifier'
            - 'MLPClassifier'

        random_state : int, default=1
            Semilla para el generador de números aleatorios.

        verbose : bool, default=False
            Mostrar información sobre el clasificador seleccionado.
        """
        self.classifier_name = classifier_name
        self.random_state = random_state
        self.verbose = verbose

    def __int__(self):
        self.classifier_name = ''
        self.random_state = None
        self.verbose = None

    def select(self, **kwargs):
        """Selecciona el clasificador a utilizar.

        Parameters
        ----------
        kwargs : dict
            Parámetros para el clasificador.

        Returns
        -------
        Classifier
            Clasificador seleccionado.
        """
        if self.classifier_name == 'RandomForestClassifier':
            classifier = RandomForestClassifier(random_state=self.random_state, **kwargs)
        elif self.classifier_name == 'GradientBoostingClassifier':
            classifier = GradientBoostingClassifier(random_state=self.random_state, **kwargs)
        elif self.classifier_name == 'LogisticRegression':
            classifier = LogisticRegression(random_state=self.random_state, max_iter=1000, **kwargs)
        elif self.classifier_name == 'LogisticRegressionCV':
            classifier = LogisticRegressionCV(random_state=self.random_state, max_iter=1000, **kwargs)
        elif self.classifier_name == 'RidgeClassifier':
            classifier = RidgeClassifier(
                random_state=self.random_state,
                max_iter=1000,
                alpha=0.3,  # 1.0    # menor más ajuste, mayor riesgo de overfitting
                tol=1e-7,  # 1e-4
                solver="auto",  # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
                class_weight=None,  # weigth,  # inv_weigth,  #  'balanced'  #
                **kwargs)
        elif self.classifier_name == 'RidgeClassifierCV':
            classifier = RidgeClassifierCV(
                alphas=[1e-3, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                cv=None,
                **kwargs)
        elif self.classifier_name == 'BernoulliNB':
            classifier = BernoulliNB(**kwargs)
        elif self.classifier_name == 'GaussianNB':
            classifier = GaussianNB(**kwargs)
        elif self.classifier_name == 'NearestCentroid':
            classifier = NearestCentroid(**kwargs)
        elif self.classifier_name == 'LinearSVC':
            classifier = LinearSVC(random_state=self.random_state, max_iter=10000, **kwargs)
        elif self.classifier_name == 'NuSVC':
            classifier = NuSVC(random_state=self.random_state, **kwargs)
        elif self.classifier_name == 'SVC':
            classifier = SVC(random_state=self.random_state, **kwargs)
        elif self.classifier_name == 'DecisionTreeClassifier':
            classifier = DecisionTreeClassifier(random_state=self.random_state, min_impurity_decrease=0.003, **kwargs)
        elif self.classifier_name == 'ExtraTreeClassifier':
            classifier = ExtraTreeClassifier(random_state=self.random_state, **kwargs)
        elif self.classifier_name == 'MLPClassifier':
            # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
            classifier = MLPClassifier(
                random_state=self.random_state,
                max_iter=1000,
                hidden_layer_sizes=(10, 10),  # (100, 100, 100)
                activation="relu",  # 'identity', 'logistic', 'tanh', 'relu'
                solver="adam",  # 'lbfgs', 'sgd', 'adam'
                alpha=1e-5,  # 1e-4 - por defecto:0.0001
                learning_rate="adaptive",  # 'constant', 'invscaling', 'adaptive'
                learning_rate_init=1.5e-3,  # 1e-3
                shuffle=True,
                tol=1e-4,  # 1e-4
                verbose=True, **kwargs)
        else:
            raise ValueError(f"Clasificador '{self.classifier_name}' no encontrado.")
        return classifier

    class IteratorClassifiers:
        def __init__(self, classifiers=None):
            # columnas_eliminar = [col for col in columnas_eliminar if col in df.columns]
            self.classifiers = classifiers or ClassifierSelector._classifiers
            self.classifiers = [clf for clf in self.classifiers if clf in ClassifierSelector._classifiers]
            self.datos = list(self.classifiers)
            self.indice_actual = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.indice_actual >= len(self.datos):
                raise StopIteration
            resultado = self.datos[self.indice_actual]
            self.indice_actual += 1
            return ClassifierSelector(resultado).select()


def classify(clf, x_test, x_train, y_test, y_train, metadata: dict, gen_conf_matrix=True):
    print("#"*90)
    print(clf.__class__.__name__)
    print("#"*90)
    # metadata = metadata or dict()
    md = dict()
    # Entrenar el modelo (train)
    t0 = time()
    clf.fit(x_train, y_train)
    duration = time() - t0
    # Evaluar el modelo (test)
    y_pred = clf.predict(x_test)
    cr = classification_report(y_test, y_pred, output_dict=True)
    md.update({'classification_report': cr})
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(f"Training done in {duration:.3f} seg.", '\n')
    id_experiment = metadata.get('_id')
    _save_experiment(clf, accuracy, duration, x_train, y_train, x_test, y_test, id_experiment, md, gen_conf_matrix)
    metadata.get("models", list()).append(dict(name=clf.__class__.__name__, accuracy=accuracy, duration=duration, **md))
    # metadata.get("models")[clf.__class__.__name__] = dict(name=clf.__class__.__name__, accuracy=accuracy, duration=duration, **md)
    return accuracy


def probar_clasificadores(X_train, y_train, X_test, y_test):
    weigth = dict(y_train.value_counts(normalize=True))
    inv_weigth = {k: 1 / v for k, v in weigth.items()}
    clfs = [
        LogisticRegression(),
        LogisticRegression(
            penalty='l2',  # 'l1', 'elasticnet', 'none'
            tol=1e-7,  # 1e-4
            C=1.0,  # 1.0
            fit_intercept=True,
            class_weight=None,  # weigth,  # inv_weigth,  #  'balanced'  #
            random_state=1,
            solver="saga",  # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
            max_iter=1000,
            multi_class="auto",  # 'auto', 'ovr', 'multinomial'
            verbose=0,
            warm_start=False,
            n_jobs=-1,
            l1_ratio=None,
        ),
        BernoulliNB(
            alpha=1.0,  # 1.0
            binarize=0.0,  # 0.0
            fit_prior=True,
            class_prior=None,  # weigth,  # inv_weigth,  #  'balanced'  #
        ),
        DecisionTreeClassifier(
            max_depth=None,  # 3,  #
            min_samples_split=2,  # 2, #
            min_samples_leaf=1,  # 1, #
            max_leaf_nodes=None,  # None, #
            min_impurity_decrease=1e-4,  # 1e-7, #
            class_weight=None,  # None 'balanced', #
        ),
        ExtraTreeClassifier(),
        LinearSVC(),
        LinearSVC(multi_class='crammer_singer', max_iter=1000),
        LogisticRegression(multi_class='multinomial', max_iter=1000),
        LogisticRegressionCV(multi_class='multinomial', max_iter=1000),
        MLPClassifier(
            hidden_layer_sizes=(20, 20, 20),  # (100, 100, 100)
            activation="relu",  # 'identity', 'logistic', 'tanh', 'relu'
            # solver="adam",  # 'lbfgs', 'sgd', 'adam'
            # alpha=1e-4,  # 1e-4
            learning_rate="adaptive",  # 'constant', 'invscaling', 'adaptive'
            learning_rate_init=1e-3,  # 1e-3
            shuffle=True,
            random_state=1,
            tol=1e-4,  # 1e-4
            verbose=False),
        NearestCentroid(),
        RandomForestClassifier(),
        NuSVC(),
        SVC(),
        GradientBoostingClassifier(),
        RidgeClassifierCV(
            alphas=[1e-3, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            cv=None),
        RidgeClassifier(
            alpha=0.3,  # 1.0    # menor más ajuste, mayor riesgo de overfitting
            tol=1e-7,  # 1e-4
            solver="auto",  # 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
            random_state=1,
            class_weight=None,  # weigth,  # inv_weigth,  #  'balanced'  #
        )]
    accuracy_biggest = 0
    for i, clf in enumerate(clfs):
        print("##################################################")
        print(clf.__class__.__name__)
        print("##################################################")
        # Entrenar el modelo (train)
        t0 = time()
        clf.fit(X_train, y_train)
        duration = time() - t0
        # Evaluar el modelo (test)
        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        accuracy_biggest = accuracy if accuracy > accuracy_biggest else accuracy_biggest
        print("Actual Accuracy :", accuracy)
        print("Biggest Accuracy:", accuracy_biggest)
        print(f"Training done in {duration:.3f} seg.", '\n')


def all_classifiers(x_train, y_train, x_test, y_test, metadata=None, gen_conf_matrix=True):
    best_accuracy = 0.0
    best_clf = 'unknown'
    metadata = metadata or dict()
    for clf in ClassifierSelector.IteratorClassifiers():
        try:
            accuracy = classify(clf, x_test, x_train, y_test, y_train, metadata, gen_conf_matrix=gen_conf_matrix)
            best_clf = clf.__class__.__name__ if accuracy > best_accuracy else best_clf
            best_accuracy = accuracy if accuracy > best_accuracy else best_accuracy
        except:
            print("Detalles de la excepción:", sys.exc_info())
        finally:
            print(f"\n>>> Best Accuracy: ({best_clf}) {best_accuracy:.4} <<<\n")


def classifiers(x_train, y_train, x_test, y_test, classifiers_list, metadata=None, gen_conf_matrix=True):
    best_accuracy = 0.0
    best_clf = 'unknown'
    metadata = metadata or dict()
    metadata["models"] = list()
    for clf in ClassifierSelector.IteratorClassifiers(classifiers_list):
        try:
            accuracy = classify(clf, x_test, x_train, y_test, y_train, metadata,
                                gen_conf_matrix=gen_conf_matrix)
            best_clf = clf.__class__.__name__ if accuracy > best_accuracy else best_clf
            best_accuracy = accuracy if accuracy > best_accuracy else best_accuracy
        except:
            ex = sys.exc_info()
            print("Tipo de excepción:", ex[0])
            print("Detalles de la excepción:", ex[1])
            print("Traceback:", ex[2])
            print()
            # raise (ex[0], ex[1], ex[2])
        finally:
            print(f"\n>>> Best Accuracy: ({best_clf}) {best_accuracy:.4} <<<\n")
    db = DataBase()
    db.add_experiment(metadata)


class Experiment:
    __slots__ = [
        'clf',
        'accuracy',
        'duration',
        'params',
        'x_train',
        'y_train',
        'x_test',
        'y_test',
    ]

    def __init__(self, clf, accuracy, duration, params, x_train, y_train, x_test, y_test):
        self.clf = clf
        self.accuracy = accuracy
        self.duration = duration
        self.params = params
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def save(self, path):
        with open(f"{path}", 'wb') as f:
            pickle.dump(self, f)

    def todict(self):
        return {'experiment data - clf': self.clf.__class__.__name__,
                'experiment data - accuracy': self.accuracy,
                'experiment data - duration': self.duration,
                'experiment data - params': self.params,
                # 'experiment data - x_train': pd.DataFrame.sparse.from_spmatrix(self.x_train).to_json(),
                # 'experiment data - y_train': self.y_train.to_dict(),
                # 'experiment data - x_test': pd.DataFrame.sparse.from_spmatrix(self.x_test).to_json(),
                # 'experiment data - y_test': self.y_test.to_dict()
                }

    @staticmethod
    def load(clf_name):
        with open(f"{clf_name}.pickle", 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        return f"{self.clf.__class__.__name__}:\n" \
               f"  Accuracy: {self.accuracy}\n" \
               f"  Duration: {self.duration}\n" \
               f"  Params: {self.params}\n" \
               f"  X_train: {self.x_train}\n" \
               f"  Y_train: {self.y_train}\n" \
               f"  X_test: {self.x_test}\n" \
               f"  Y_test: {self.y_test}\n"


def _save_experiment(clf, accuracy, duration, x_train, y_train, x_test, y_test,
                     id_experiment, metadata: dict = None, gen_conf_matrix=True):
    metadata = metadata or dict()
    experiment = Experiment(
        clf,
        accuracy,
        duration,
        clf.get_params(),
        x_train,
        y_train,
        x_test,
        y_test)
    target_names = clf.classes_
    # name = f"{len(target_names)} authors - {len(y_train)+len(y_test)} documents"
    # folder = f"Resultados/{datetime.now().strftime('%Y-%m-%d')}/{name}/{datetime.now().strftime('%H%M%S')}" \
    #          f" - {clf.__class__.__name__} - accuracy {accuracy:.3f}" \
    #          f" - duration {duration:.3f}" \
    #          f" - balanced {metadata.get('metadata_process_dataset').get('balance')}"\
    #          f" - binarize {metadata.get('metadata_process_dataset').get('binarize')}"
    folder_name = f"{len(target_names)}-authors_{len(y_train) + len(y_test)}-documents"
    folder = f"Resultados/{datetime.now().strftime('%Y-%m-%d')}/{folder_name}/{id_experiment}/" \
             f"{clf.__class__.__name__}"
    create_folder(f"{folder}")
    experiment.save(f"{folder}/{clf.__class__.__name__}.xpt")
    metadata.update(experiment.todict())
    with open(f"{folder}/metadata.txt", "w", encoding="utf-8") as archivo:
        print(metadata, file=archivo)
    with open(f"{folder}/metadata.json", "w", encoding="utf-8") as archivo:
        json.dump(metadata, archivo)
    prediction = clf.predict(x_test)
    print(f'accuracy: {accuracy_score(y_test, prediction):.4}')
    print(f'balanced accuracy: {balanced_accuracy_score(y_test, prediction):.4}')
    print(f'precision (macro)   : {precision_score(y_test, prediction, average="macro", zero_division=0):.4}')
    print(f'precision (micro)   : {precision_score(y_test, prediction, average="micro", zero_division=0):.4}')
    print(f'precision (weighted): {precision_score(y_test, prediction, average="weighted", zero_division=0):.4}')
    print(f'recall (macro)   : {recall_score(y_test, prediction, average="macro", zero_division=0):.4}')
    print(f'recall (micro)   : {recall_score(y_test, prediction, average="micro", zero_division=0):.4}')
    print(f'recall (weighted): {recall_score(y_test, prediction, average="weighted", zero_division=0):.4}')
    print(f'f1 (macro)   : {f1_score(y_test, prediction, average="macro", zero_division=0):.4}')
    print(f'f1 (micro)   : {f1_score(y_test, prediction, average="micro", zero_division=0):.4}')
    print(f'f1 (weighted): {f1_score(y_test, prediction, average="weighted", zero_division=0):.4}')
    # print("confusion matrix:\n", confusion_matrix(y_test, prediction))
    if gen_conf_matrix:
        generar_matriz_confusion(clf, y_test, prediction, target_names, folder)
    return experiment
