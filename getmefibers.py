import cv2
import skimage as ski
import numpy as np

from common import getBlackHatMask, getContours, filterContours, getFirstElementOfContour, applyFlooding, applyWatershed

def getMeFibers(base_img,
                bh_ks=(7,7),
                bhm_iter=4,
                bhm_mult=60,
                cont_mult=2.5,
                ws_ths_factor=0.025,
                ws_gl_vecinity=15,###################### VV nuevo
                otsu_classes=5,
                otsu_range=(2, None)):
    
    # Eliminación de ruido y mejora de contraste
    test_1 = cv2.GaussianBlur(base_img, (7, 7), 0)
    
    # Mejorar contraste
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(10, 10))
    test_2 = cv2.GaussianBlur(clahe.apply(test_1), (11, 11), 0)

    # Aplicar Black Hat a la imagen seleccionada
    black_hat_img = getBlackHatMask(test_2, kernel_size=bh_ks)

    # Obtener máscara binaria mejorada de la imagen Black Hat
    kernel_bh = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.uint8)

    black_hat_mask = cv2.morphologyEx(black_hat_img, cv2.MORPH_CLOSE, kernel_bh,iterations=bhm_iter)

    # Aplicar la máscara a la imagen original
    test_3 = np.int16(test_2) - bhm_mult*np.int16(black_hat_mask)
    test_3[test_3 < 0] = 0
    test_3 = np.uint8(test_3)

    # Aplicar umbralización multiotsu para segmentar test_3
    list_ts_test3 = ski.filters.threshold_multiotsu(test_3, classes=otsu_classes)

    ## Dynamic slicing // older
    #start, end = otsu_range
    #selected_thresholds = list_ts_test3[start:end]
    #
    ## Segementación de test_3 para separar la mayor cantidad de fibras    
    #thresh_1 = np.zeros(np.shape(test_3), dtype=np.uint8)
    #thresh_1[test_3 > np.mean(selected_thresholds)] = 255

    # ------ NEWER attempt ---------- 
    # Segment into discrete classes
    regions = np.digitize(test_3, bins=list_ts_test3)

    start, end = otsu_range
    if end is None:
        end = otsu_classes - 1
    if start is None:
        start = 0
    if start > end:
        start, end = end, start
    start = int(np.clip(start, 0, otsu_classes - 1))
    end   = int(np.clip(end,   0, otsu_classes - 1))

    thresh_1 = ((regions >= start) & (regions <= end)).astype(np.uint8) * 255
    # --------- ######## -----------
    
    test_4 = test_3.copy()
    test_4[~(thresh_1 == 255)] = 0
    test_4[test_4 > 0] = 255

    # Obtener contornos de la imagen binaria
    contours, contours_img = getContours(test_4)

    # Filtrar contornos por longitud
    contours_filtered, contours_filtered_img = filterContours(test_4, contours, mult=cont_mult)

    # Aplicar flooding para obtener la máscara de las fibras
    coordinates = getFirstElementOfContour(contours_filtered)
    mask_flood, list_masks = applyFlooding(test_4, coordinates)

    # Se mejora la máscara de las fibras con una operación morfológica de cierre
    kernel_mfl = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.uint8)
    
    test_5 = cv2.morphologyEx(mask_flood, cv2.MORPH_CLOSE, kernel_mfl,iterations=2)

    # Aplicar watershed para obtener la segmentación final
    regions_result, binary_mask = applyWatershed(test_2, test_5, threshold_factor=ws_ths_factor, gl_vecinity=ws_gl_vecinity)

    return binary_mask, contours_filtered_img, list_masks

### ------ Version lolo ----- VVVV

# esto es un helper
def _filterContoursByContMultRange(base_img, contours, cont_mult_range=(0.0, None)):
    contours_lengths = [len(c) for c in contours]

    dim = np.shape(base_img)
    empty_img = cv2.drawContours(np.zeros((dim[0], dim[1], 3)), [], -1, (0, 255, 0), 1)
    empty_img = np.uint8(empty_img)

    if len(contours_lengths) == 0:
        return [], empty_img

    cl_mean = np.mean(contours_lengths)
    cl_std = np.std(contours_lengths)

    min_mult, max_mult = cont_mult_range
    lower = -np.inf if min_mult is None else cl_mean + float(min_mult) * cl_std
    upper = np.inf if max_mult is None else cl_mean + float(max_mult) * cl_std

    if lower > upper:
        lower, upper = upper, lower

    contours_filtered = [c for c in contours if lower <= len(c) <= upper]
    contours_filtered_img = cv2.drawContours(
        np.zeros((dim[0], dim[1], 3)),
        contours_filtered,
        -1,
        (0, 255, 0),
        1,
    )
    contours_filtered_img = np.uint8(contours_filtered_img)

    return contours_filtered, contours_filtered_img

# este es
def getMeFibersGammaOtsuWatershed(base_img,
                                 gamma=1.0,
                                 gain=1.0,
                                 cont_mult_range=(0.0, None),
                                 ws_ths_factor=0.025,
                                 ws_gl_vecinity=15,
                                 otsu_classes=5,
                                 otsu_range=(2, None)):
    gamma_img = ski.exposure.adjust_gamma(base_img, gamma=gamma, gain=gain)
    test_1 = np.clip(gamma_img, 0, 255).astype(np.uint8)

    list_ts_test_1 = ski.filters.threshold_multiotsu(test_1, classes=otsu_classes)
    regions = np.digitize(test_1, bins=list_ts_test_1)

    start, end = otsu_range
    if end is None:
        end = otsu_classes - 1
    if start is None:
        start = 0
    if start > end:
        start, end = end, start
    start = int(np.clip(start, 0, otsu_classes - 1))
    end = int(np.clip(end, 0, otsu_classes - 1))

    thresh_1 = ((regions >= start) & (regions <= end)).astype(np.uint8) * 255

    contours, _ = getContours(thresh_1)
    contours_filtered, contours_filtered_img = _filterContoursByContMultRange(
        thresh_1,
        contours,
        cont_mult_range=cont_mult_range,
    )

    coordinates = getFirstElementOfContour(contours_filtered)
    mask_flood, list_masks = applyFlooding(thresh_1, coordinates)

    regions_result, binary_mask = applyWatershed(
        test_1,
        mask_flood,
        threshold_factor=ws_ths_factor,
        gl_vecinity=ws_gl_vecinity,
    )

    return binary_mask, contours_filtered_img, list_masks
