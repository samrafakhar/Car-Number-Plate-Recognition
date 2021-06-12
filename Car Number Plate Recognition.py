import ImageProcessing as IP
from PIL import Image as im
import numpy as np
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget
from PyQt5.QtCore import QSize

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V','W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lRate = 0.5
targetError = 0.0035
hiddenLayerPerceptron = 100
w = 18
h = 16
result=[]

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

class printDetectedPlate(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(640, 480))
        self.setWindowTitle("Detected Lisence Plate")

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        gridLayout = QGridLayout(self)
        centralWidget.setLayout(gridLayout)

        title = QLabel(listToString(result), self)
        title.setAlignment(QtCore.Qt.AlignCenter)
        gridLayout.addWidget(title, 0, 0)

def feedForward(normalized, ItoH, HtoO, HBias, OBias):
    wSum = 0
    [count, h, w] = ItoH.shape
    hiddenOutput = []
    hiddenInput = []

    for hiddenLayer in range(0, count):
        for i in range(h):
            for j in range(w):
                inputWeight = ItoH[hiddenLayer, i, j] * normalized[i, j]
                wSum = wSum + inputWeight

        wSum = wSum + HBias[hiddenLayer]
        hiddenOutput.append(logistic(wSum))
        hiddenInput.append(wSum)
        wSum = 0

    hiddenWeight = hiddenOutput * HtoO
    netInput = np.sum(hiddenWeight, axis=1)
    perceptronOutput = []

    for outputNeuron in range(36):
        input = netInput[outputNeuron] + OBias[outputNeuron]
        perceptronOutput.append(logistic(input))
    return perceptronOutput, hiddenOutput

def backPropagate(ItoH, HtoO,normalized, outputError, perceptronOutput, hiddenOutput, momentum):
    oldHtoO = np.array(HtoO[:, :])
    [count, h, w] = ItoH.shape

    for o in range(36):
        for hl in range(count):
            adjustment = (lRate * outputError[o] * perceptronOutput[o] * (1 - perceptronOutput[o]) * hiddenOutput[hl])
            HtoO[o, hl] = (momentum * HtoO[o, hl]) - adjustment
    for hl in range(count):
        deltaMSE = 0
        for o in range(36):
            deltaErrorOPerceptron = outputError[o] * perceptronOutput[o] * (1 - perceptronOutput[o]) * oldHtoO[o, hl]
            deltaMSE = deltaMSE + deltaErrorOPerceptron
        for i in range(h):
            for j in range(w):
                deltaMSEItoToH = deltaMSE * hiddenOutput[hl] * (1 - hiddenOutput[hl]) * normalized[i, j]
                ItoH[hl, i, j] = (momentum * ItoH[hl, i, j]) - (lRate * deltaMSEItoToH)
    return ItoH, HtoO

def training(ItoH, HtoO, HBias, OBias, numberOfTrainingSamples, momentum, maxError):
    I = 1
    MSE = 1

    while MSE > maxError:
        for i in range(36):
            targetOutput = np.zeros(36)
            targetOutput[i] = 1

            for n in range(0, numberOfTrainingSamples):
                test = 'trainData/%s%d.png' % (letters[i], n)
                BW = IP.convertToBW(im.open(test))
                normalized = IP.crop(BW, w, h)
                perceptronOutput, hiddenOutput = feedForward(normalized, ItoH, HtoO, HBias, OBias)
                outputError = OutputError(perceptronOutput, targetOutput)
                ItoH, HtoO = backPropagate(ItoH, HtoO, normalized, outputError, perceptronOutput, hiddenOutput, momentum)

        MSE = 0
        for x in range(36):
            sq = 0.5 * outputError[x] ** 2
            MSE = MSE + sq

        print('Iteration ' + str(I))
        print('Current Error = %f' % MSE)
        I = I + 1

    return (ItoH, HtoO, HBias, OBias)

def testing(img, ItoH, HtoO, HBias, OBias):
    BW = IP.convertToBW(img)
    [lines, lineCount] = IP.cropLines(BW)

    for l in range(lineCount):
        [chars, charCount] = IP.cropCharacters(lines[l])
        for c in range(charCount):
            inputNormalized = IP.crop(chars[c], w, h)
            output = recognizeCharacter(inputNormalized, ItoH, HtoO, HBias, OBias)
            result.append(output)
            result.append("  ")
            print(output)

def logistic(summation):
    num = 1 / (1 + np.exp(-summation))
    return num

def recognizeCharacter(inputNormalized, ItoH, HtoO, HBias, OBias):
    perceptronOutput, hiddenOutput = feedForward(inputNormalized, ItoH, HtoO, HBias, OBias)
    char = np.argmax(perceptronOutput)
    return letters[char]

def OutputError(perceptronOutput, targetOutput):
    Error = []
    for o in range(36):
        outputError = perceptronOutput[o] - targetOutput[o]
        Error.append(outputError)
    return Error

def main():
    filename='input1.png'
    IP.ExtractNumberplate(filename)

    img = im.open('cropped.jpg')
    ItoH = np.random.random(size=(hiddenLayerPerceptron, h, w)) - 0.5
    HtoO = np.random.random(size=(36, hiddenLayerPerceptron)) - 0.5
    HBias = np.random.random(hiddenLayerPerceptron) - 0.5
    OBias = np.random.random(36) - 0.5
    
    print("Training...")
    ItoH, HtoO, HBias, OBias = training(ItoH, HtoO, HBias, OBias, 4, 1, targetError)
    print("Testing...")
    testing(img, ItoH, HtoO, HBias, OBias)

if __name__ == "__main__":
    main()
    app = QtWidgets.QApplication(sys.argv)
    mainWin = printDetectedPlate()
    mainWin.show()
    sys.exit( app.exec_() )