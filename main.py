import os
import sys

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from gui.mainWindow_ui import Ui_MainWindow
from gui.filters_ui import Ui_filterDialog
from gui.noiseAdd_ui import Ui_noiseAddDialog
from gui.noiseRemove_ui import Ui_noiseRemoveDialog

from backend.histogram import *
from backend.fourier import *
from backend.filters import *
from backend.noise import *

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

#######################################################################

offset = 0
def mousePressEvent(event):
    global offset
    offset = event.pos()
def mouseMoveEvent(event):
    global offset
    try:
        x = event.globalX()
        y = event.globalY()
        x_w = offset.x()
        y_w = offset.y()
        if activeDialog is None:
            MainWindow.move(x - x_w, y - y_w)
        else:
            activeDialog.move(x - x_w, y - y_w)
    except:
        pass
#######################################################################

app = QApplication(sys.argv)
style = open('gui\Eclippy.qss').read()
app.setStyleSheet(style)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.setWindowFlags(QtCore.Qt.FramelessWindowHint)
MainWindow.mousePressEvent = mousePressEvent
MainWindow.mouseMoveEvent = mouseMoveEvent
MainWindow.show()

#######################################################################
"""
    Global variables
"""

original_Image = processed_Image = None
underprocessing_Image = None
undo, redo = [], []

#######################################################################
"""
    Helper Functions
"""


def apply_cmap_to_image(img, cmap):
    plt.imsave('temp.jpg', img, cmap=cmap)
    new_image = plt.imread('temp.jpg')
    os.remove('temp.jpg')
    return new_image


def getHistogramImage(img):
    img_gray = getGrayImage(img)
    # make a Figure and attach it to a canvas.
    fig = Figure(figsize=(5, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting here
    ax = fig.add_subplot(111)
    ax.hist(img_gray.ravel(), 256, [0, 256])

    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    X = np.asarray(buf)
    X = cv.cvtColor(X, cv.COLOR_RGBA2BGR)
    return X


def getPixmap(img=None):
    if img is None:
        return QPixmap()

    cv.imwrite('tempo.jpg', img)
    image_Pixmap = QPixmap('tempo.jpg')
    os.remove('tempo.jpg')

    return image_Pixmap


def getGrayImage(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


#######################################################################
"""
    Main Functions
"""


def loadImage(init = False):
    global original_Image, processed_Image

    if init is False:
        img_path = QFileDialog.getOpenFileName(filter="Image (*.jpg *.png *.jpeg *.tif *.jfif)")[0]
        if img_path != '':
            original_Image = processed_Image = cv.imread(img_path)
    else:
        original_Image = processed_Image = cv.imread('images\placeholder.jpg')
    update()

def reset():
    global original_Image, processed_Image
    processed_Image = original_Image
    update()


def update():
    updateImages()
    updateHF()


def updateImages():
    ui.originalImage_label.setPixmap(getPixmap(original_Image))
    ui.processedImage_label.setPixmap(getPixmap(processed_Image))


def updateHF():
    if original_Image is None:
        ui.original_HF_label.setPixmap(getPixmap())
        ui.processed_HF_label.setPixmap(getPixmap())
        return

    if (ui.histogram_radioButton.isChecked()):
        original_HF_pixmap = getPixmap(getHistogramImage(original_Image))
        processed_HF_pixmap = getPixmap(getHistogramImage(processed_Image))
    else:
        dftimg = dft_magnitude(shifted_dft(getGrayImage(original_Image)))
        dftimg_gray = apply_cmap_to_image(dftimg, cmap='gray')
        original_HF_pixmap = getPixmap(dftimg_gray)

        dftimg = dft_magnitude(shifted_dft(getGrayImage(processed_Image)))
        dftimg_gray = apply_cmap_to_image(dftimg, cmap='gray')
        processed_HF_pixmap = getPixmap(dftimg_gray)

    ui.original_HF_label.setPixmap(original_HF_pixmap)
    ui.processed_HF_label.setPixmap(processed_HF_pixmap)


def equalizeHist():
    global processed_Image
    processed_Image = equalizeHistogram(getGrayImage(processed_Image))
    update()


def convertToGray():
    global processed_Image
    processed_Image = getGrayImage(processed_Image)
    update()


def saveImage():
    filename = QFileDialog.getSaveFileName(caption="Save Processed Image", filter='JPEG (*.jpg)')

    if filename == ('', ''):
        return

    img = processed_Image.copy()
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imsave(filename[0], img)
    else:
        plt.imsave(filename[0], img, cmap='gray')

####################################################################################
"""
    Mask Filter Functions
"""
number_of_points=0
point1 = point2 = []
factor_h = factor_w = 0
mf_fourierImage=None

def mf_reset_points():
    global activeDialog, processed_Image, underprocessing_Image
    global number_of_points, point1, point2
    number_of_points=0
    activeDialog.ui.label_point1MaskFilter.setText( "point_1: ?")
    activeDialog.ui.label_point2MaskFilter.setText( "point_2: ?")
    activeDialog.ui.apply_button_5.setEnabled(False)
    activeDialog.ui.pushButton_preview.setEnabled(False)
    underprocessing_Image = processed_Image
    mf_set_Fourier_Image(underprocessing_Image)
    

def mf_tabsChange():
    global activeDialog, processed_Image, underprocessing_Image, number_of_points

    if activeDialog.ui.tabs.currentIndex() == 5: # mask filter
        number_of_points=0
        mf_set_Fourier_Image(processed_Image)
        activeDialog.ui.image_label.mousePressEvent = lambda e: mf_mousePressed(e)
        activeDialog.ui.pushButton_reset.clicked.connect(mf_reset_points)
        activeDialog.ui.pushButton_preview.clicked.connect(mf_preview_result)
    else:
        underprocessing_Image = processed_Image
        refresh_dialog()

def mf_mousePressed(event):
    global number_of_points, point1, point2, factor_h, factor_w
    global original_Image
    
    hi, wi = original_Image.shape[:2]
    hl, wl = 340, 550
    factor_w = wi/wl
    factor_h = hi/hl

    y, x = (int(factor_w * event.pos().x()), int(factor_h * event.pos().y()))

    if number_of_points == 0:
        point1 = x, y
        text = "point_1: " + str(point1)
        activeDialog.ui.label_point1MaskFilter.setText(text)
        number_of_points += 1
    elif number_of_points == 1:
        point2 = (x, y)
        text = "point_2: " + str(point2)
        activeDialog.ui.label_point2MaskFilter.setText(text)
        activeDialog.ui.apply_button_5.setEnabled(True)
        activeDialog.ui.pushButton_preview.setEnabled(True)
        number_of_points += 1
        mf_processImage()
        mf_set_Fourier_Image(underprocessing_Image)

def mf_set_Fourier_Image(image):
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    dftimg = shifted_dft(getGrayImage(image))
    mf_fourierImage = dftimg
    dftimg_gray = apply_cmap_to_image(dft_magnitude(dftimg), cmap='gray')
    processed_HF_pixmap = getPixmap(dftimg_gray)
    activeDialog.ui.image_label.setPixmap(processed_HF_pixmap)

def mf_processImage():
    global factor_h, factor_w
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    if number_of_points == 2:
        underprocessing_Image = maskFilter(mf_fourierImage, point1, point2, filter_size=10)
        underprocessing_Image = inverse_shifted_dft(underprocessing_Image)
def mf_preview_result():
    global number_of_points
    global activeDialog, processed_Image, underprocessing_Image, mf_fourierImage
    mf_processImage()

    cv.imshow("After Mask Filter", np.uint8(underprocessing_Image))
    mf_set_Fourier_Image(underprocessing_Image)
    number_of_points = 0
    

####################################################################################
"""
    Dialogs Functions
"""


def applyChanges():
    global activeDialog, processed_Image, underprocessing_Image
    undo.append(processed_Image.copy())
    ui.undo_Button.setEnabled(True)
    processed_Image = underprocessing_Image.copy()
    update()
    activeDialog.close()

def undoChanges():
    global processed_Image, underprocessing_Image
    redo.append(processed_Image.copy())
    ui.redo_Button.setEnabled(True)
    processed_Image = undo.pop()
    if len(undo) == 0:
        ui.undo_Button.setEnabled(False)
    update()

def redoChanges():
    global processed_Image, underprocessing_Image
    undo.append(processed_Image.copy())
    ui.undo_Button.setEnabled(True)
    processed_Image = redo.pop()
    if len(redo) == 0:
        ui.redo_Button.setEnabled(False)
    update()


def updateFilter():
    global activeDialog, processed_Image, underprocessing_Image
    slider1 = activeDialog.ui.slider1.value()
    slider2 = activeDialog.ui.slider2.value()
    dx = activeDialog.ui.dx_checkBox.isChecked()
    dy = activeDialog.ui.dy_checkBox.isChecked()

    if slider1 & 1 ^ 1:
        slider1 += 1
        activeDialog.ui.slider1.setValue(slider1)
    if slider2 & 1 ^ 1:
        slider2 += 1
        activeDialog.ui.slider2.setValue(slider2)

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Sobel Filter
        if dx or dy:
            underprocessing_Image = sobel_filter(processed_Image, slider1, dx, dy)
    elif idx == 1:  # Laplace Filter
        underprocessing_Image = laplacian_filter(processed_Image, slider2)

    refresh_dialog()


def updateNoiseAdd():
    global activeDialog, processed_Image, underprocessing_Image

    slider1 = activeDialog.ui.slider1.value() / 100.0
    slider2 = activeDialog.ui.slider2.value() / 100.0

    slider3 = activeDialog.ui.slider3.value() / 100.0
    slider4 = activeDialog.ui.slider4.value() / 100.0

    slider5 = activeDialog.ui.slider5.value()
    slider6 = activeDialog.ui.slider6.value()
    slider7 = activeDialog.ui.slider7.value()
    slider8 = activeDialog.ui.slider8.value()


    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))
    activeDialog.ui.slider5_counter.setText(str(slider5))
    activeDialog.ui.slider6_counter.setText(str(slider6))
    activeDialog.ui.slider7_counter.setText(str(slider7))
    activeDialog.ui.slider8_counter.setText(str(slider8))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Salt and Pepper Noise
        underprocessing_Image = add_salt_and_pepper_noise(processed_Image, slider1, slider2)
    elif idx == 1:  # Gaussian Noise
        underprocessing_Image = gaussianNoise(processed_Image, slider3, slider4)
    elif idx == 2:  # Periodic Noise
        underprocessing_Image = add_periodic_noise(processed_Image, slider5, slider6, slider7, slider8)

    refresh_dialog()


def updateNoiseRemove():
    global activeDialog, processed_Image, underprocessing_Image

    slider1 = activeDialog.ui.slider1.value()
    slider2 = activeDialog.ui.slider2.value()

    slider3 = activeDialog.ui.slider3.value()

    slider4 = activeDialog.ui.slider4.value()
    slider5 = activeDialog.ui.slider5.value() 
    slider6 = activeDialog.ui.slider6.value() 

    slider7 = activeDialog.ui.slider7.value()

    slider8 = activeDialog.ui.slider8.value()

    if slider3 & 1 ^ 1:
        slider3 += 1
        activeDialog.ui.slider3.setValue(slider3)
    if slider4 & 1 ^ 1:
        slider4 += 1
        activeDialog.ui.slider4.setValue(slider4)


    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))
    activeDialog.ui.slider5_counter.setText(str(slider5))
    activeDialog.ui.slider6_counter.setText(str(slider6))
    activeDialog.ui.slider7_counter.setText(str(slider7))
    activeDialog.ui.slider8_counter.setText(str(slider8))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Average Filter
        if slider1 > 0 and slider2 > 0:
            underprocessing_Image = averaging_filter(processed_Image, slider1, slider2)
    elif idx == 1:  # Median Filter
        underprocessing_Image = median_filter(processed_Image, slider3)
    elif idx == 2:  # Gaussian Blur
        underprocessing_Image = gaussianFilter(processed_Image, slider4, slider5, slider6)
    elif idx == 3: # Notch Filter
        underprocessing_Image = notch_filter(getGrayImage(processed_Image), slider7)
    elif idx == 4: # Band Filter
        underprocessing_Image = band_filter(getGrayImage(processed_Image), slider8)
    elif idx == 5: # Mask Filter
        mf_reset_points()
        pass

    refresh_dialog()



dialogs = [Ui_filterDialog, Ui_noiseAddDialog, Ui_noiseRemoveDialog]
updateFunctions = [updateFilter, updateNoiseAdd, updateNoiseRemove]

activeDialog = None


def setup_dialog(dialogUI):
    global activeDialog
    activeDialog = QtWidgets.QDialog(None, QtCore.Qt.WindowCloseButtonHint)
    activeDialog.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    activeDialog.mousePressEvent = mousePressEvent
    activeDialog.mouseMoveEvent = mouseMoveEvent
    activeDialog.ui = dialogUI()
    activeDialog.ui.setupUi(activeDialog)


def refresh_dialog():
    global activeDialog, underprocessing_Image
    activeDialog.ui.image_label.setPixmap(getPixmap(underprocessing_Image))
    activeDialog.ui.image_label.setScaledContents(True)


def show_dialog(idx):
    global activeDialog, underprocessing_Image, processed_Image
    setup_dialog(dialogs[idx])
    underprocessing_Image = processed_Image.copy()
    refresh_dialog()

    components = activeDialog.ui.getComponents()
    for c in components:
        if type(c) is QtWidgets.QTabWidget:
            activeDialog.ui.tabs.currentChanged.connect(mf_tabsChange)
        if type(c) is QtWidgets.QPushButton:
            c.clicked.connect(applyChanges)
        if type(c) is QtWidgets.QSlider:
            c.valueChanged['int'].connect(updateFunctions[idx])
        if type(c) is QtWidgets.QCheckBox:
            c.clicked.connect(updateFunctions[idx])

    activeDialog.ui.close_button.clicked.connect(close_dialog)
    activeDialog.ui.minimize_button.clicked.connect(activeDialog.showMinimized)
    activeDialog.exec_()

def close_dialog():
    global activeDialog
    activeDialog.close()
    activeDialog = None


#######################################################################

loadImage(True)

ui.close_button.clicked.connect(app.exit)
ui.minimize_button.clicked.connect(MainWindow.showMinimized)

ui.loadImage_Button.clicked.connect(loadImage)
ui.reset_Button.clicked.connect(reset)
ui.undo_Button.clicked.connect(undoChanges)
ui.redo_Button.clicked.connect(redoChanges)
ui.saveImage_Button.clicked.connect(saveImage)

ui.histogram_radioButton.clicked.connect(updateHF)
ui.fourier_radioButton.clicked.connect(updateHF)

ui.cvttograyscale_Button.clicked.connect(convertToGray)
ui.equalizeHistogram_Button.clicked.connect(equalizeHist)
ui.filtering_Button.clicked.connect(lambda: show_dialog(0))
ui.addNoise_Button.clicked.connect(lambda: show_dialog(1))
ui.removeNoise_Button.clicked.connect(lambda: show_dialog(2))

#######################################################################

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

sys.excepthook = except_hook

sys.exit(app.exec_())
