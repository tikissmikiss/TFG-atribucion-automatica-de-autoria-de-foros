import os
import pickle


def cargar_objeto(ruta_fichero, objeto=None):
    if not os.path.isfile(ruta_fichero):
        print("El fichero no existe.")
        return None
    # Cargar desde el archivo
    with open(ruta_fichero, "rb") as file:
        objeto = pickle.load(file)
    return objeto


def guardar_objeto(objeto, nombre_archivo, carpeta=""):
    create_folder(carpeta)
    name_file = get_full_path(nombre_archivo, carpeta)
    print("Guardar fichero:", name_file)
    with open(name_file, "wb") as file:
        # Guardar el objeto en un archivo serializado
        pickle.dump(objeto, file)
    print("Objeto guardado en el archivo", nombre_archivo)


def get_full_path(nombre_archivo, carpeta):
    name_file = nombre_archivo if len(carpeta) == 0 else "./" + carpeta + "/" + nombre_archivo
    return name_file


def create_folder(carpeta):
    if len(carpeta) > 0 and not os.path.exists(carpeta):
        carpeta = os.path.abspath(carpeta)
        os.makedirs(carpeta)
        print(f"Creada carpeta {carpeta}\n")
    return carpeta
