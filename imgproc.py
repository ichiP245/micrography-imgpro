import cv2
from skimage.segmentation import flood
import numpy as np

def getDegradedFiberWithBubbles(base_imgch,first_bound):
    img_degraded_fiber = base_imgch.copy()

    img_degraded_fiber[base_imgch < first_bound] = 255
    img_degraded_fiber[base_imgch >= first_bound] = 0

    return img_degraded_fiber

def getNotDegradedFiber(base_imgch,second_bound):
    img_not_degraded_fiber = base_imgch.copy()

    img_not_degraded_fiber[base_imgch < second_bound] = 0
    img_not_degraded_fiber[base_imgch >= second_bound] = 255

    return img_not_degraded_fiber

def applyClosing(base_img,kernel_size=3):
    kernel_cl = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    closing = cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel_cl)

    return closing

def applyOpening(base_img,kernel_size=3):
    kernel_op = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,kernel_size))
    opening = cv2.morphologyEx(base_img, cv2.MORPH_OPEN, kernel_op)

    return opening

def getBubblesMasks(base_img,list_bubbles_coordinates,tolerance=1):
    list_masks = []

    for coordinate in list_bubbles_coordinates:
        mask = flood(base_img, coordinate, tolerance=tolerance)
        list_masks.append(mask)

    return list_masks

def applyBubbleMasks(base_img,list_bubble_masks,glvalue=255):
    img_with_bubbles = base_img.copy()

    for mask in list_bubble_masks:
        img_with_bubbles[mask] = glvalue

    return img_with_bubbles    

def mergeBinaryPictures(first_img,second_img):
    merged_img = first_img + second_img
    merged_img[merged_img > 255] = 255
    merged_img[merged_img < 0] = 0

    return np.uint8(merged_img)

def makeLabelMask(base_img):
    label_mask = np.zeros(np.shape(base_img))
    label_mask[790:869,1425:1552] = 255

    return label_mask

def applyLabelMask(base_img):
    img_unlabeled = base_img.copy()
    label_mask = makeLabelMask(base_img)
    img_unlabeled[label_mask == 255] = 0

    return img_unlabeled

def invertBinaryPicture(base_img):
    inverted_img = 255 - np.int16(base_img)
    inverted_img[inverted_img < 0] = 0

    return np.uint8(inverted_img)

def getFinalSegmentation(base_img,list_img_no_label):
    reconstruction = np.zeros(np.shape(base_img))

    reconstruction[list_img_no_label[3] == 255] = 150 #resina
    reconstruction[list_img_no_label[2] == 255] = 30 #burbujas
    reconstruction[list_img_no_label[0] == 255] = 250 #fibra buena
    reconstruction[list_img_no_label[1] == 255] = 70 #fibra mala
    reconstruction[list_img_no_label[4] == 255] = 0 #etiqueta

    return np.uint8(reconstruction)

def getMaterialProportions(list_segmentations,label=True):
    pixels_notready = [np.count_nonzero(img) for img in list_segmentations]

    if label:
        label_pixels = pixels_notready[-1]
    else:
        label_pixels = 0
    
    pixels_ready = [pixels_notready[0]-pixels_notready[1],pixels_notready[1],pixels_notready[2],pixels_notready[3]]

    total = np.shape(list_segmentations[0])[0]*np.shape(list_segmentations[0])[1] - label_pixels

    proportions = [100*quantity/total for quantity in pixels_ready]

    return proportions

def getImgAnalysis(base_img,gl_boundaries,list_bubbles_coordinates):
    
    # Capa verde
    #base_img = base_img[:,:,1]

    # Fronteras de niveles de grises
    first_bound = gl_boundaries[0]
    second_bound = gl_boundaries[1]

    # Fibra degradada + Burbujas + Etiqueta (si la hay)
    img_test_1 = getDegradedFiberWithBubbles(base_img,first_bound)

    # Fibra no degrada + Etiqueta (si la hay)
    img_test_2 = getNotDegradedFiber(base_img,second_bound)

    # Fibra degradada + Burbujas + Etiqueta (si la hay) + MorfMat 1
    img_test_3 = applyOpening(applyClosing(img_test_1))

    # Fibra degradada + Sin burburjas + Etiqueta (si la hay) + MorfMat 1
    img_test_4 = applyBubbleMasks(img_test_3,getBubblesMasks(img_test_3,list_bubbles_coordinates),glvalue=0)

    # Fibras completas + Etiqueta
    img_step_5 = mergeBinaryPictures(applyClosing(img_test_4,5),applyClosing(img_test_2,5))

    # Fibras completas + Etiqueta + MorfMat 2
    img_test_6 = applyClosing(img_step_5,kernel_size=5)

    # Resina + Etiqueta
    img_test_7 = invertBinaryPicture(img_test_6)

    # Burbujas sin etiqueta
    img_test_8 = applyBubbleMasks(np.zeros(np.shape(base_img)),getBubblesMasks(img_test_3,list_bubbles_coordinates),glvalue=255)


    list_img_with_labels = [img_test_6,img_test_4,img_test_8,img_test_7]
    list_img_no_label = [applyLabelMask(img) for img in list_img_with_labels]
    list_img_no_label.append(makeLabelMask(base_img))

    img_test_9 = getFinalSegmentation(base_img,list_img_no_label)
    proportions = getMaterialProportions(list_img_no_label)

    dictionary = dict()
    elements = ["Fnd", "Fde", "Bur", "Res"]

    for proportion, element in zip(proportions,elements):
        dictionary[element] = proportion
    
    return dictionary, img_test_9, list_img_no_label

