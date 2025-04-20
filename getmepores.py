import cv2
import skimage as ski
import numpy as np

from common import getContours, getFirstElementOfContour, applyFlooding, applyWatershed

def getMetPores(base_img,
                first_kernel_size=(5,5),
                second_kernel_size=(3,3)):
    
    # Aplicar umbralización multiotsu para segmentar la imagen
    list_ts_base_img = ski.filters.threshold_multiotsu(base_img, classes=5)

    # Aplicar umbralización a la imagen original
    # Se utiliza el primer umbral para obtener la máscara de las burbujas
    test_1 = base_img.copy()
    test_1[base_img <= list_ts_base_img[0]] = 255
    test_1[base_img > list_ts_base_img[0]] = 0
    test_1 = np.uint8(test_1)

    # Mejora de la máscara de las burbujas
    test_2 = cv2.morphologyEx(test_1, cv2.MORPH_OPEN, np.ones(first_kernel_size,np.uint8),iterations=1)

    # Aplicar una operación de apertura más para eliminar el ruido
    test_3 = cv2.morphologyEx(test_2, cv2.MORPH_OPEN, np.ones(second_kernel_size,np.uint8),iterations=1)

    # Obtener contornos de la imagen binaria
    contours_bubbles, contours_img = getContours(test_3)

    coordinates_bubbles = getFirstElementOfContour(contours_bubbles)
    mask_bubbles, list_masks_bubbles = applyFlooding(test_3, coordinates_bubbles)

    # Obtener regiones indefinidas
    undefined_region_mask = test_1.copy()
    undefined_region_mask[mask_bubbles == 255] = 0

    return mask_bubbles, undefined_region_mask
