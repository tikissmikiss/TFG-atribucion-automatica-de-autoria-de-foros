# TFG-atribucion-automatica-de-autoria-de-foros
Código relativo al trabajo de fin de estudios del grado de ingeniería informática

## Web Scraper
[...]

## Herramienta para Atribución Automática de Autoría
[Ver notebook](./Atribucion%20Autoria%20Foros.ipynb)

### Ejemplo de clasificación para 100 autores
[...]

## dataset
Contiene el dataset utilizado para el entrenamiento y evaluación de los modelos. Se trata de un dataset de 905129 posts de foros de discusión, con al menos 100 posts de cada uno de los 1362 autores. Los posts se encuentran en formato JSON, con la siguiente estructura:

```
{
    "author": "autor del post",
    "text": "texto del post",
}
```
Disponible en [JSON](https://mega.nz/file/SY5HkDIa#q8njIJ-5ptDLFbDLJ0YRwvVLZ3p5LigvGGxe2CD4ook) y [BSON](https://mega.nz/folder/mJxlXLjS#lcTOFd35EK5rnnYFIPxiXg) ([.bson](https://mega.nz/file/GdpHQQgA#jcI0JpkRntCF4RQAfEuk_XG_IeNUGQ4P_xp-7ZlTTrk), [metadata](https://mega.nz/file/7EY2DQiZ#8E3Q584E1tm-loaY5rrr_XWDeM5P0DhzEjTTLwyZYG8))
