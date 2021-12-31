import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys

from gui.mainWindow_ui import Ui_MainWindow
from gui.noiseAdd_ui import Ui_noiseAddDialog
from gui.noiseRemove_ui import Ui_noiseRemoveDialog
from gui.filters_ui import Ui_filterDialog
from backend.histogram import *
from backend.noise import *
from backend.filters import *
from backend.fourier import *

from PIL import Image

# Global variables
original_Image = processed_Image = None
original_H_Image = processed_H_Image = None
original_F_Image = processed_F_Image = None


app = QApplication(sys.argv)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)


		
MainWindow.show()

def form_histogramImage(hist):
    img = np.full([100, 256], 255, dtype="uint8")
    for i in range(hist.shape[0]):
#         print(int(hist[i][0]))
        maxindex = int((hist[i][0]/hist.max())*100)
#         print(maxindex)
        img[maxindex:, i] = 0
    return img

def loadImage():
    global original_Image, processed_Image
    img_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
    original_Image = cv2.imread(img_path)
    

    # cv2.imshow("image", original_Image)
    # cv2.waitKey(0)
    original_Image = cv2.cvtColor(original_Image, cv2.COLOR_BGR2RGB)
    image = QImage(original_Image, original_Image.shape[1],original_Image.shape[0],original_Image.strides[0],QImage.Format_RGB888)
    ui.originalImage_label.setPixmap(QtGui.QPixmap.fromImage(image))

    original_Image = cv2.cvtColor(original_Image, cv2.COLOR_RGB2GRAY)
    processed_Image = equalizeHistogram(original_Image)
    # cv2.imshow("image", processed_Image)
    # cv2.waitKey(0)
    image = processed_Image #cv2.applyColorMap(processed_Image, cv2.COLORMAP_AUTUMN)
    image = QImage(image, image.shape[1],image.shape[0],image.strides[0],QImage.Format_Grayscale8)
    ui.processedImage_label.setPixmap(QtGui.QPixmap.fromImage(image))
    
    updateImages()
		

def updateImages():
    global original_Image, processed_Image
    original_H = calculateHistogram(original_Image)
    original_H_Image = form_histogramImage(original_H)
    processed_H = calculateHistogram(processed_Image)
    processed_H_Image = form_histogramImage(processed_H)

    original_F_Image = shifted_dft(original_Image)
    processed_F_Image = shifted_dft(processed_Image)

    if(ui.radioButton_histogram.isChecked):
        # cv2.imshow("image", original_H_Image)
        # cv2.waitKey(0)
        image = QImage(original_H_Image, original_H_Image.shape[1],original_H_Image.shape[0],original_H_Image.strides[0],QImage.Format_Grayscale8)
        ui.original_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))

        # cv2.imshow("image", processed_H_Image)
        # cv2.waitKey(0)
        image = QImage(processed_H_Image, processed_H_Image.shape[1],processed_H_Image.shape[0],processed_H_Image.strides[0],QImage.Format_Grayscale8)
        ui.processed_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))
    elif ui.radioButton_fourier.isChecked:
        image = QImage(original_F_Image, original_F_Image.shape[1],original_F_Image.shape[0],original_F_Image.strides[0],QImage.Format_Grayscale8)
        ui.original_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))

        image = QImage(processed_F_Image, processed_F_Image.shape[1],processed_F_Image.shape[0],processed_F_Image.strides[0],QImage.Format_Grayscale8)
        ui.processed_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))
    else:
        image = QImage(original_F_Image, original_H_Image.shape[1],original_H_Image.shape[0],original_H_Image.strides[0],QImage.Format_Grayscale8)
        ui.original_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))

        image = QImage(processed_F_Image, processed_H_Image.shape[1],processed_H_Image.shape[0],processed_H_Image.strides[0],QImage.Format_Grayscale8)
        ui.processed_HF_label.setPixmap(QtGui.QPixmap.fromImage(image))

    

ui.loadImage_Button.clicked.connect(loadImage)
ui.reset_Button.clicked.connect(updateImages)

# cv2.destroyAllWindows()
# if original_Image != None:
#     updateImages()
sys.exit(app.exec_())
