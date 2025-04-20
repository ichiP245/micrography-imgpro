import cv2
import skimage as ski
import numpy as np

from common import getContours, filterContours, getFirstElementOfContour, applyFlooding

def getMeFlashes(base_img,
                 cont_mult=2.5):
    # Eliminación de ruido y mejora de contraste
    test_1 = cv2.GaussianBlur(base_img, (7, 7), 0)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10, 10))
    test_2 = cv2.GaussianBlur(clahe.apply(test_1), (11, 11), 0)

    # Aplicar umbralización multiotsu para segmentar test_2
    list_ts_test_2 = ski.filters.threshold_multiotsu(test_2, classes=5)

    # Segementación de test_2 para separar la mayor cantidad de flashes
    flashes = test_2.copy()
    flashes[test_2 <= np.mean([list_ts_test_2[-1],255])] = 0

    # Obtener contornos de la imagen binaria
    contours_flashes, contours_img = getContours(flashes)

    # Filtrar contornos por longitud
    contours_filtered, contours_filtered_img = filterContours(flashes, contours_flashes, mult=cont_mult, mode='UP')

    # Obtener coordenadas de los contornos filtrados
    coordinates_flashes = getFirstElementOfContour(contours_filtered)

    # Aplicar flooding para obtener la máscara de los flashes
    mask_flashes, list_masks_flashes = applyFlooding(flashes, coordinates_flashes)

    return mask_flashes