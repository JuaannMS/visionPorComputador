import cv2
import numpy as np
import os

# Ruta de la carpeta que contiene las imágenes
carpeta_imagenes = 'cajas'

# Crear una carpeta para guardar las imágenes resultantes
carpeta_resultados = 'resultados'
if not os.path.exists(carpeta_resultados):
    os.makedirs(carpeta_resultados)

# Obtener la lista de archivos en la carpeta
archivos_imagenes = [f for f in os.listdir(carpeta_imagenes) if os.path.isfile(os.path.join(carpeta_imagenes, f))]

for archivo_imagen in archivos_imagenes:
    # Ruta completa de la imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo_imagen)

    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print(f"Error: No se pudo cargar la imagen {archivo_imagen}.")
        continue

    #Desde esta linea comienza el tratamiento de la imagen

    # Convertir la imagen a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro bilateral para preservar los bordes
    desenfoque = cv2.bilateralFilter(gris, 9, 75, 75)

    # Detectar bordes con el filtro Canny
    bordes = cv2.Canny(desenfoque, 30, 150)

    # Aplicar dilatación y erosión para cerrar huecos en los bordes
    kernel = np.ones((3, 3), np.uint8)
    bordes_dilatados = cv2.dilate(bordes, kernel, iterations=1)
    bordes_morfo = cv2.morphologyEx(bordes_dilatados, cv2.MORPH_CLOSE, kernel)

    # Encontrar contornos en la imagen
    contornos, jerarquia = cv2.findContours(bordes_morfo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (cuadrado o figura de cuatro lados más grande)
    contorno_grande = max(contornos, key=cv2.contourArea)

    # Dibujar el contorno más grande en verde en la imagen original
    cv2.drawContours(imagen, [contorno_grande], 0, (0, 255, 0), 2)

    # Guardar la imagen resultante con el contorno más grande resaltado
    nombre_resultado = f'resultado_{archivo_imagen}.jpg'
    ruta_resultado = os.path.join(carpeta_resultados, nombre_resultado)
    cv2.imwrite(ruta_resultado, imagen)
    print(f"Imagen guardada como '{ruta_resultado}'")

# Cerrar todas las ventanas abiertas
cv2.destroyAllWindows()