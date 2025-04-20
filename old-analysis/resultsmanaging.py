import matplotlib.pyplot as plt
import os
from imgproc import getImgAnalysis
import pandas as pd

def figureMaker(img_base,img_name,segmentation,results):
    proportions = [*results.values()]

    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(2, 1, 1)
    plt.imshow(img_base)
    plt.title(f"Original - {img_name}")
    plt.axis("off")

    ax = fig.add_subplot(2, 1, 2)
    plt.imshow(segmentation,cmap="viridis")
    plt.title(f"Fnd: {proportions[0]:.2f} % - Fde: {proportions[1]:.2f} % - Bur: {proportions[2]:.2f} % - Res: {proportions[3]:.2f} %")
    plt.axis("off")

    # Agregar la barra de color con etiquetas personalizadas
    cbar = plt.colorbar(orientation="vertical",shrink=0.5)
    cbar.set_ticks([30,70,150,250])
    cbar.set_ticklabels(["Burbujas", "Fibra degradada", "Resina", "Fibra no degradada"])

def plotResults(img_base,img_name,segmentation,results):

    figureMaker(img_base,img_name,segmentation,results)

    plt.show()

    plt.close()

def savePlotResults(img_base,img_name,segmentation,results,savings_folder):
    figureMaker(img_base,img_name,segmentation,results)

    os.makedirs(savings_folder, exist_ok=True)
    fname = os.path.join(savings_folder, f'seg-{img_name}')

    plt.savefig(fname=fname)

    plt.close()

def showResults(list_imgs,list_filenames,list_graylevel_boundaries,list_bubbles_coordinates):

    list_results = []

    for base_img, img_name, gl_boundaries, bubble_coords in zip(list_imgs,
                                                                list_filenames,
                                                                list_graylevel_boundaries,
                                                                list_bubbles_coordinates):
        
        results, segmentation, _ = getImgAnalysis(base_img,
                                                gl_boundaries,
                                                bubble_coords)
    
        list_results.append(results)
        plotResults(base_img,img_name,segmentation,results)

    return list_results

def saveResults(list_imgs,list_filenames,list_graylevel_boundaries,list_bubbles_coordinates,savings_folder,dataframe):

    folder = "procimgs/"+savings_folder

    for base_img, img_name, gl_boundaries, bubble_coords in zip(list_imgs,
                                                                list_filenames,
                                                                list_graylevel_boundaries,
                                                                list_bubbles_coordinates):
        
        results, segmentation, _ = getImgAnalysis(base_img,
                                                gl_boundaries,
                                                bubble_coords)

        savePlotResults(base_img,img_name,segmentation,results,folder)

    pd.DataFrame(dataframe).to_excel(f"{folder}/{savings_folder}.xlsx", index=False)