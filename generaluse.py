from os import listdir
from cv2 import imread, cvtColor, COLOR_BGR2RGB
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def getImgNamesAndPaths(imgs_folder_name):

    # Lista de nombres de imágenes
    list_img_names = listdir(imgs_folder_name)
    
    # Lista de path de cada imagen
    list_paths_to_img = []
    for img_name in list_img_names:
        img_path = imgs_folder_name + '/' + img_name
        list_paths_to_img.append(img_path)

    # Creación de diccionario - nombre_img:path_img
    dictionary = dict()
    for i in range(len(list_img_names)):
        dictionary[list_img_names[i]] = list_paths_to_img[i]

    return dictionary

def getImagesFromPathfile(list_pathfile, mode=COLOR_BGR2RGB):
    list_imgs = []
    for i in range(len(list_pathfile)):
        img = cvtColor( imread(list_pathfile[i]) , mode)
        list_imgs.append(img)

    return list_imgs

def plotFigures(list_names, list_figs):
    nrows = len(list_names) // 3 + 1
    ncols = 3

    fig = plt.figure(figsize=(21, int(5*nrows)))

    for i in range(len(list_names)):

        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(list_figs[i])
        ax.set_title(list_names[i])

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.grid(which='both',linestyle='--')

    plt.show()

def plotHist(list_names, list_figs):
    nrows = len(list_names) // 3 + 1
    ncols = 3

    fig = plt.figure(figsize=(21, int(5*nrows)))

    for i in range(len(list_names)):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.plot(list_figs[i])
        ax.set_title(list_names[i])

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax.grid(which='both',axis='x')

    plt.show()