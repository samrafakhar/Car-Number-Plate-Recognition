import cv2
import numpy as np
from PIL import Image as im

def ExtractNumberplate(name):
    #parts of numberplate cropping taken from https: // github.com / AjayAndData / Licence - plate - detection - and -recognition - --using - openCV - only / blob / master / Car % 20Number % 20Plate % 20Detection.py
    image = cv2.imread(name)
    cv2.imshow("Orignal Image ", image)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Black and white ", gray)
    cv2.waitKey(0)
    edged = cv2.Canny(gray, 170, 200)
    cv2.imshow("Edged ", edged)
    cv2.waitKey(0)
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img1 = gray.copy()
    cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = gray[y:y + h, x:x + w]
            cv2.imwrite('cropped' + '.jpg', new_img)
            break
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    Cropped_img_loc = 'cropped.jpg'
    cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))
    cv2.waitKey(0)

def convertToBW(imageIn):
    BW = imageIn.convert('1')
    BW = np.array(BW)*1
    return (BW^1)

def crop(string, w, h):
    [r , c] = string.shape
    vSum = np.sum(string,axis=0)
    l = False
    for i in range(0,c):
        if vSum[i] > 0 and l == False:
            l = True
            left = i
        elif vSum[i] > 0 and l == True:
            right = i

    hSum= np.sum(string,axis=1)
    t = False
    for i in range(0,r):
        if hSum[i] > 0 and t == False:
            t = True
            top = i
        elif hSum[i] > 0 and t == True:
            bottom = i

    vBW= string[:,(range(left,right+1))]
    imgBW = vBW[(range(top,bottom+1)),:]

    imgBW= im.fromarray(imgBW)
    normalized = imgBW.resize((w,h),im.HAMMING)
    result = np.array(normalized)
    return result

#copied from github https://github.com/GeorgeSaman/Optical-Character-Recognition-BackPropagation
def cropLines(blackAndWhite):
    [numberRowPixels , _] = blackAndWhite.shape
    h_firstBlackPixelDetected = False
    firstBlackPixelRow = 0
    lastBlackPixelRow = 0
    topOfLines = []
    bottomOfLines = []

    for i in range (numberRowPixels):
        sumOfAllPixelsInRow_i = sum(blackAndWhite[i,:])
        if sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == False:
            h_firstBlackPixelDetected = True
            firstBlackPixelRow = i
            lastBlackPixelRow = i

        elif sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == True:
            lastBlackPixelRow = i

        elif sumOfAllPixelsInRow_i < 1 and h_firstBlackPixelDetected == True:
            h_firstBlackPixelDetected = False
            topOfLines.append(firstBlackPixelRow)
            bottomOfLines.append(lastBlackPixelRow)                          # Save LastBlackPixels in a list

    numberOfLines = len(topOfLines)
    croppedLinesList = []
    for i in range(numberOfLines):                                         # Make a list containing croppedLines
        croppedLine = blackAndWhite[(range(topOfLines[i],bottomOfLines[i])),:]
        croppedLinesList.append(croppedLine)

    return croppedLinesList,numberOfLines

#copied from github https://github.com/GeorgeSaman/Optical-Character-Recognition-BackPropagation
def cropCharacters(croppedLine):
    [_ , numberOfLineColumnPixels] = croppedLine.shape
    l = False
    firstBlackPixelColumn = 0
    lastBlackPixelColumn = 0
    leftOfCharacters = []
    rightOfCharacters = []

    for i in range (numberOfLineColumnPixels):
        sumOfAllPixelsInColoumn_i = sum(croppedLine[:,i])
        if sumOfAllPixelsInColoumn_i >= 1 and l == False:
            l = True
            firstBlackPixelColumn = i
            lastBlackPixelColumn  = i

        elif sumOfAllPixelsInColoumn_i >= 1 and l == True:
            lastBlackPixelColumn  = i

        elif sumOfAllPixelsInColoumn_i < 1 and l == True:
            l = False
            leftOfCharacters.append(firstBlackPixelColumn)
            rightOfCharacters.append(lastBlackPixelColumn)

    numberOfCharacters = len(leftOfCharacters)
    croppedCharactersList = []
    for i in range(numberOfCharacters):
        croppedCharacter = croppedLine[:,(range(leftOfCharacters[i],rightOfCharacters[i]))]
        croppedCharactersList.append(croppedCharacter)

    return croppedCharactersList,numberOfCharacters