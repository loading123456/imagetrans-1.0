import concurrent.futures
from curses import noecho
from pickle import APPENDS
from traceback import print_tb
from typing import List
from xmlrpc.client import Boolean
import cv2
import numpy as np
import pytesseract
import googletrans 
import time
from pytesseract import Output
import re
from PIL import Image, ImageFont, ImageDraw
import math
import os



class Box:
    def __init__(self, rect, bgColor ) -> None:
        self.bgColor = bgColor
        self.stPoint = [rect[0], rect[1]]
        self.enPoint = [rect[0] + rect[2], rect[1] + rect[3]]
        self.cePoint = [rect[0] + rect[2] / 2, rect[1] + rect[3] / 2]
        self.w = rect[2]
        self.h = rect[3]
        self.group ='dm'

class Word(Box):
    def __init__(self, rect, bgColor, text, confident ) -> None:
        self.text = text
        self.conf = confident
        super().__init__(rect, bgColor)

    def getRect(self) -> List:
        return self.stPoint + [self.w, self.h]

class Group(Box):
    def __init__(self, box:Box) -> None:
        rect = box.stPoint + [box.w, box.h]
        super().__init__(rect, box.bgColor)
        self.words = []

    def getLines(self):
        self.sortWords()
        lines = []

        for word in self.words:
            inserted = False
            for line in lines:
                if line.insertWord(word):
                    inserted = True
                    break
            if not inserted:
                lines.append(Line(word))
        self.updateLine(lines)
        return lines

    def sortWords(self):
        self.words = sorted(self.words, key=lambda k: (k.stPoint[0]))
    def updateLine(self, lines):
        for line in lines:
            line.isStart = (abs(line.stPoint[0] - self.stPoint[0]) <= line.h)
            line.isEnd = (abs(line.enPoint[0] - self.enPoint[0]) <= line.h)
            line.enPoint[1] += line.h * 0.2
            line.h += int(line.h * 0.3)

class Line(Box):
    def __init__(self, word:Word) -> None:
        super().__init__(word.getRect(), word.bgColor)
        self.text = word.text
        self.words = [word]
        self.X_NOUN = 1
        self.Y_NOUN = 0.5
        self.H_NOUN = 0.5
        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN

    def insertWord(self, word:Word) -> Boolean:
        r = self.__isValid(word) 
        if r == 1:
            self.__update(word)
            return True
        return False

    def __isValid(self, word:Word) -> int:
        if (abs(self.cePoint[1] - word.cePoint[1]) <= self.h * self.Y_NOUN
                or abs(self.cePoint[1] - word.cePoint[1]) <= word.h * self.Y_NOUN
        ):
            if word.stPoint[0] - self.enPoint[0] <= self.epX:
                return 1
            return 0
        return -1

    def __update(self, word:Word) -> None:
        if self.stPoint[0] > word.stPoint[0]:
            self.stPoint[0] = word.stPoint[0]
        if self.stPoint[1] > word.stPoint[1]:
            self.stPoint[1] = word.stPoint[1]

        if self.enPoint[0] < word.enPoint[0]:
            self.enPoint[0] = word.enPoint[0]
        if self.enPoint[1] < word.enPoint[1]:
            self.enPoint[1] = word.enPoint[1]
        
        self.cePoint[0] = (self.stPoint[0] + self.enPoint[0])/2
        self.cePoint[1] = (self.stPoint[1] + self.enPoint[1])/2
        self.w = self.enPoint[0] - self.stPoint[0]
        self.h = self.enPoint[1] - self.stPoint[1]

        self.epX = self.h * self.X_NOUN
        self.epY = self.h * self.Y_NOUN
        self.epH = self.h * self.H_NOUN
        self.text += ' ' + word.text


def translate(imagePath, savePath):
    # print("===========",imagePath,'===================')
    st = time.time()
    img = cv2.imread(imagePath)

    imgData = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT)
    cImgData = getCImgData(img, savePath)
    words = getWords(imgData, cImgData, img)
    groups = getGroups(img)

    insertWordTGroup(words, groups)
    lines = getLines(groups)
    lines = sorted(lines, key=lambda k: (k.cePoint[1]))

    for line in lines:
        if line.text.isupper():
            line.text = line.text.lower()
    
    # paragraphs = getParagraphs(lines)
    # for paragraph in paragraphs:
    #     print(paragraph.getText())
    #     print("================================================||")
    # return 0

    stt = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futureDraw = {executor.submit(transText,  id, lines[id].text): id for id in range(len(lines))}
        for future in concurrent.futures.as_completed(futureDraw):
            id, text = future.result()
            lines[id].text = text
    print("Translate: ", time.time() - stt)
    outputImg = Image.open(imagePath).convert("RGB")

    for line in lines:
        draw(outputImg, line)
    outputImg.save(savePath)



    # print("Excuse time: ", time.time() - st)


def getCImgData(img, savePath) ->  List:
    cImg = getContrastImg(img)
    cv2.imwrite('cImg.jpg', cImg)
    cImg = cv2.imread('cImg.jpg')

    cImgData = pytesseract.image_to_data(cImg, lang='eng', output_type=Output.DICT)
    os.remove('cImg.jpg')
    return cImgData

def getWords(imgData, cImgData, img):
    imgDataL = len(imgData['text'])
    nWords = []
    for i in range(imgDataL):
        x, y = imgData['left'][i], imgData['top'][i]
        w, h = imgData['width'][i], imgData['height'][i]
        text = imgData['text'][i]
        conf = int(imgData['conf'][i])
        if (int(imgData['conf'][i]) >= 50
            and not imgData['text'][i].isspace()
            and w < img.shape[1]
            and h < img.shape[0]
        ):
            nWords.append(Word([x, y, w, h], 0, text, conf))


    cImgDataL = len(cImgData['text'])
    cWords = []
    for i in range(cImgDataL):
        x, y = cImgData['left'][i], cImgData['top'][i]
        w, h = cImgData['width'][i], cImgData['height'][i]
        text = cImgData['text'][i]
        conf = int(cImgData['conf'][i])
        if (int(cImgData['conf'][i]) >= 50
            and not cImgData['text'][i].isspace()
            and w < img.shape[1]
            and h < img.shape[0]
        ):
            cWords.append(Word([x, y, w, h], 1, text, conf))
    return filterWords(nWords, cWords, img)


def filterWords(wordsA, wordsB, img):
    words = wordsA[:] + wordsB[:]
    wordsSize = len(words)
    for i in range(wordsSize):
        for j in range(i+1, wordsSize):
            if (words[i] != None 
                and words[j] != None
            ):
                if (words[i].stPoint == words[j].stPoint
                    or words[i].enPoint == words[j].enPoint
                    or words[i].cePoint == words[j].cePoint
                    or ((
                        abs(words[i].cePoint[1] - words[j].cePoint[1]) <= words[i].h * 0.5
                        or abs(words[i].cePoint[1] - words[j].cePoint[1]) <= words[j].h * 0.5
                )
                        and (abs(words[i].cePoint[0] - words[j].cePoint[0]) <= words[j].w * 0.5 
                            or abs(words[i].cePoint[0] - words[j].cePoint[0]) <= words[j].w * 0.5 
                        )
                    )
                ):
                        if getBgColor(img[words[i].stPoint[1]:words[i].enPoint[1],
                                          words[i].stPoint[0]:words[i].enPoint[0]  ]) != 0:
                            words[j] = None
                        else:
                            words[i] = None
    words =  [ elem for elem in words if elem != None]
    return words


def getBgColor(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, im_bw = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    count = 0
    for c in im_bw:
        for r in c:
            if r==0:
                count += 1
    if count / (im_bw.shape[0] * im_bw.shape[1]) < 0.5:
        return 1
    return 0


def getContrastImg(img):
    return 255 - img    


def getGroups(img):
    contours, cContours = getContours(img)
    lines = []
     
    for ctn in contours:
        x, y, w, h = cv2.boundingRect(ctn)
        if( w < img.shape[1]
            and h <img.shape[0]
        ):  
            lines.append(Box([x, y, w, h], 0))
            
    lines = sorted(lines, key=lambda k: (k.stPoint[0]))

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if (abs(lines[i].cePoint[1] - lines[j].cePoint[1]) <= lines[i].h*0.2
                and abs(lines[i].enPoint[0] - lines[j].stPoint[0]) <= lines[i].h*0.2
            ):
                lines[j].stPoint[0] = lines[i].stPoint[0]

                if lines[i].stPoint[1] < lines[j].stPoint[1]:
                    lines[j].stPoint[1] = lines[i].stPoint[1]

                if lines[i].enPoint[1] > lines[j].enPoint[1]:
                    lines[j].enPoint[1] = lines[i].enPoint[1]
                
                lines[j].cePoint[0] = (lines[j].enPoint[0] + lines[j].stPoint[0])/2
                lines[j].cePoint[1] = (lines[j].enPoint[1] + lines[j].stPoint[1])/2
                lines[j].w = int(lines[j].enPoint[0]) - int(lines[j].stPoint[0])
                lines[j].h = int(lines[j].enPoint[1]) - int(lines[j].stPoint[1])

                lines[i] = None
                break

    result = [ Group(box) for box in lines[:] if box != None]
    # ----------------------------------------------------------
    lines = []
    for ctn in cContours:
        x, y, w, h = cv2.boundingRect(ctn)
        if( w < img.shape[1]
            and h <img.shape[0]
        ):  
            lines.append(Box([x, y, w, h], 1))

    lines = sorted(lines, key=lambda k: (k.stPoint[0]))
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            if (abs(lines[i].cePoint[1] - lines[j].cePoint[1]) <= lines[i].h*0.2
                and abs(lines[i].enPoint[0] - lines[j].stPoint[0]) <= lines[i].h*0.2
            ):
                lines[j].stPoint[0] = lines[i].stPoint[0]
                if lines[i].stPoint[1] < lines[j].stPoint[1]:
                    lines[j].stPoint[1] = lines[i].stPoint[1]

                if lines[i].enPoint[1] > lines[j].enPoint[1]:
                    lines[j].enPoint[1] = lines[i].enPoint[1]


                lines[j].cePoint[0] = (lines[j].enPoint[0] + lines[j].stPoint[0])/2
                lines[j].cePoint[1] = (lines[j].enPoint[1] + lines[j].stPoint[1])/2

                lines[j].w = lines[j].enPoint[0] - lines[j].stPoint[0]
                lines[j].h = lines[j].enPoint[1] - lines[j].stPoint[1]

                lines[i] = None
                break
    result += [ Group(box) for box in lines[:] if box != None]
    result = sorted(result, key=lambda k: (k.w*k.h))

    return result


def getContours(img):
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bImg = cv2.threshold(gImg, 0, 255, cv2.THRESH_OTSU 
                                            | cv2.THRESH_BINARY_INV)[1]
    cBImage = np.where(bImg == 0 , 255, 0).astype('ubyte')
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 3))
    dilation = cv2.dilate(bImg, rect_kernel, iterations = 1)
    cDilation = cv2.dilate(cBImage, rect_kernel, iterations = 1)

    contours = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    cContours = cv2.findContours(cDilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)[0]
    return contours, cContours

def insertWordTGroup(words, groups):
    for word in words:
        for group in groups:
            if( abs(word.cePoint[0] - group.cePoint[0]) <= group.w/2 
                and abs(word.cePoint[1] - group.cePoint[1]) <= group.h/2
            ):
                group.words.append(word)
                break 
    lines = getLines(groups)

def getLines(groups):
    lines = []
    for group in groups:
        if group.words != []:
            lines += group.getLines()
    return lines



def transText(id, text) -> None:
    tText = (googletrans.Translator()
                .translate(text, dest='vi').text)

    tText = re.sub('\\s+', ' ', tText).strip()
    return id, tText

def draw(img, line):
    x, y = line.stPoint
    w, h = line.w, line.h
    bg = Image.new(mode="RGBA", size=(w, h), color=(235, 150, 235))
    img.paste(bg, (x, y))   

    font = ImageFont.truetype(r'font/Arimo-VariableFont_wght.ttf', h)
    _, _, textWidth, textHeight = font.getbbox(line.text)
    if textWidth < w:
        textWidth = w
    textBox = Image.new(mode="RGBA", size=(textWidth, textHeight ), color=(0, 0, 0, 0))
    d = ImageDraw.Draw(textBox)
    d.text((0, 0), line.text, font=font, fill=(0, 0, 0))
    textBox.thumbnail((w, 1000  ), Image.Resampling.LANCZOS)
    textBox = textBox.crop((0, 0, w, h))

    img.paste(textBox, (x, y), textBox.convert("RGBA"))



