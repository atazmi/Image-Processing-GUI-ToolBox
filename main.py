import os
import sys

from PyQt5 import QtWidgets
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

app = QApplication(sys.argv)
MainWindow = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

#######################################################################
"""
    Global variables
"""

original_Image = processed_Image = None
underprocessing_Image = None

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
    img_gray = convertToGray(img)
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

    height, width = img.shape[0:2]
    bytesPerLine = img.strides[0]

    if len(img.shape) == 3:
        img = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
    else:
        img = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)

    image_Pixmap = QPixmap()
    image_Pixmap.convertFromImage(img)

    return image_Pixmap


def convertToGray(img):
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


#######################################################################
"""
    Main Functions
"""


def loadImage():
    global original_Image, processed_Image
    # TODO: Specify allowed image extensions
    img_path = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
    if img_path != '':
        original_Image = processed_Image = cv.imread(img_path)
        update()


def reset():
    global original_Image, processed_Image
    original_Image = processed_Image = None
    update()


def update():
    updateImages()
    updateHF()


def updateImages():
    ui.originalImage_label.setPixmap(getPixmap(original_Image))
    ui.processedImage_label.setPixmap(getPixmap(processed_Image))
    # TODO: Fix Scaling Problem
    ui.originalImage_label.setScaledContents(True)
    ui.processedImage_label.setScaledContents(True)


def updateHF():
    if original_Image is None:
        ui.original_HF_label.setPixmap(getPixmap())
        ui.processed_HF_label.setPixmap(getPixmap())
        return

    if (ui.histogram_radioButton.isChecked()):
        original_HF_pixmap = getPixmap(getHistogramImage(original_Image))
        processed_HF_pixmap = getPixmap(getHistogramImage(processed_Image))
    else:
        dftimg = dft_magnitude(shifted_dft(convertToGray(original_Image)))
        dftimg_gray = apply_cmap_to_image(dftimg, cmap='gray')
        original_HF_pixmap = getPixmap(dftimg_gray)

        dftimg = dft_magnitude(shifted_dft(convertToGray(processed_Image)))
        dftimg_gray = apply_cmap_to_image(dftimg, cmap='gray')
        processed_HF_pixmap = getPixmap(dftimg_gray)

    ui.original_HF_label.setPixmap(original_HF_pixmap)
    ui.processed_HF_label.setPixmap(processed_HF_pixmap)
    # TODO: Fix Scaling Problem
    ui.original_HF_label.setScaledContents(True)
    ui.processed_HF_label.setScaledContents(True)


def equalizeHist():
    global processed_Image
    processed_Image = equalizeHistogram(convertToGray(processed_Image))
    update()


def saveImage():
    filename = QFileDialog.getSaveFileName(caption="Save Processed Image", filter='JPEG (*.jpg)')

    img = processed_Image.copy()
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imsave(filename[0], img)
    else:
        plt.imsave(filename[0], img, cmap='gray')


####################################################################################
"""
    Dialogs Functions
"""


def applyChanges():
    global activeDialog, processed_Image, underprocessing_Image
    processed_Image = underprocessing_Image.copy()
    update()
    activeDialog.close()


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

    slider3 = activeDialog.ui.slider3.value()
    slider4 = activeDialog.ui.slider4.value()

    activeDialog.ui.slider1_counter.setText(str(slider1))
    activeDialog.ui.slider2_counter.setText(str(slider2))
    activeDialog.ui.slider3_counter.setText(str(slider3))
    activeDialog.ui.slider4_counter.setText(str(slider4))

    idx = activeDialog.ui.tabs.currentIndex()
    if idx == 0:  # Salt and Pepper
        underprocessing_Image = add_salt_and_pepper_noise(processed_Image, slider1, slider2)
    elif idx == 1:  # Gaussian
        underprocessing_Image = gaussianNoise(processed_Image, slider3, slider4)
    elif idx == 2:  # Periodic
        pass

    refresh_dialog()


def updateNoiseRemove():
    global activeDialog, processed_Image, underprocessing_Image
    # Code goes here
    refresh_dialog()


dialogs = [Ui_filterDialog, Ui_noiseAddDialog, Ui_noiseRemoveDialog]
updateFunctions = [updateFilter, updateNoiseAdd, updateNoiseRemove]

activeDialog = None


def setup_dialog(dialogUI):
    global activeDialog
    activeDialog = QtWidgets.QDialog()
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
            # TODO: Reset image at tabs navigation
            pass
        if type(c) is QtWidgets.QPushButton:
            c.clicked.connect(applyChanges)
        if type(c) is QtWidgets.QSlider:
            c.valueChanged['int'].connect(updateFunctions[idx])
        if type(c) is QtWidgets.QCheckBox:
            c.clicked.connect(updateFunctions[idx])

    activeDialog.exec_()


#######################################################################

# TODO: Disable all buttons when images are None or place a default image with program logo


ui.loadImage_Button.clicked.connect(loadImage)
ui.reset_Button.clicked.connect(reset)
ui.saveImage_Button.clicked.connect(saveImage)

ui.histogram_radioButton.clicked.connect(updateHF)
ui.fourier_radioButton.clicked.connect(updateHF)

ui.equalizeHistogram_Button.clicked.connect(equalizeHist)
ui.filtering_Button.clicked.connect(lambda: show_dialog(0))
ui.addNoise_Button.clicked.connect(lambda: show_dialog(1))
ui.removeNoise_Button.clicked.connect(lambda: show_dialog(2))


#######################################################################

def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


sys.excepthook = except_hook

sys.exit(app.exec_())
