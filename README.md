
# TFG - Atribución Automática de Autoría de foros

Este es el repositorio de código correspondiente al Trabajo de Fin de Grado de Ingeniería Informática.


![Escudo](https://img.shields.io/badge/status-in%20Development-red) ![Escudo](https://img.shields.io/github/languages/count/tikissmikiss/TFG-atribucion-automatica-de-autoria-de-foros) ![Escudo](https://img.shields.io/github/languages/top/tikissmikiss/TFG-atribucion-automatica-de-autoria-de-foros) <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">![Escudo](https://img.shields.io/badge/license-in%20CC%20BY--NC--SA%204.0-yellow)</a>


## Herramienta para la atribución automática de autoría

Dentro de la carpeta [attribution](./attribution) se encuentra el código fuente de la herramienta utilizada para llevar a cabo la investigación. En el siguiente enlace se puede ver el resultado de una ejecución con una muestra de 10,000 publicaciones de 10 autores diferentes.

[Ver notebook](./Atribucion%20Autoria%20Foros.ipynb)

## Web Scraper

Dentro de la carpeta [scraper](./scraper) se encuentra la implementación del web scraper utilizado para la construcción del dataset.

## Dataset

El dataset recopilado con el scraper y utilizado en este proyecto de investigación consta de 905,129 publicaciones de foros de discusión, con al menos 100 publicaciones de cada uno de los 1,362 autores. Las publicaciones están disponibles para su descarga en formato JSON y BSON.

Cada instancia tiene la siguiente estructura:

```json
{
    "author": "autor del post",
    "text": "texto del post"
}
```

Disponible en formato [JSON](https://mega.nz/file/SY5HkDIa#q8njIJ-5ptDLFbDLJ0YRwvVLZ3p5LigvGGxe2CD4ook) y [BSON](https://mega.nz/folder/mJxlXLjS#lcTOFd35EK5rnnYFIPxiXg) ([.bson](https://mega.nz/file/GdpHQQgA#jcI0JpkRntCF4RQAfEuk_XG_IeNUGQ4P_xp-7ZlTTrk), [metadata](https://mega.nz/file/7EY2DQiZ#8E3Q584E1tm-loaY5rrr_XWDeM5P0DhzEjTTLwyZYG8)).

### Ejemplo de clasificación para 100 autores

[...]