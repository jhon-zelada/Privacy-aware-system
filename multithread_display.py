from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
import cv2
import time
from threading import Thread
import sys
import numpy as np
import os


from display_thermal import pithermalcam

path = '.\data'
fileName = 'outdoor_test.txt'

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)
        self.mw  = MainWindow
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("images/H.png"))
        self.label.setObjectName("label")

        # adding another label for second video
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("images/H.png"))
        self.label_4.setObjectName("label")

        self.horizontalLayout.addWidget(self.label)
        self.horizontalLayout.addWidget(self.label_4)


        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        #self.label_2 = QtWidgets.QLabel(self.centralwidget)
        #self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        #self.label_2.setObjectName("label_2")
        #self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1)

        self.horizontalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)

        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        spacerItem = QtWidgets.QSpacerItem(313, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        #self.statusbar = QtWidgets.QStatusBar(MainWindow)
        #self.statusbar.setObjectName("statusbar")
        #MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)


        self.th = {}

        self.pause = False 
        self.pushButton.clicked.connect(self.pause_video) 
        self.pushButton_2.clicked.connect(self.run_threads)

        

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        

        self.started = False
        self.started2 = False

        self.thermcam = pithermalcam(os.path.join(path,fileName))  # Instantiate class

    def pause_video(self):
        if self.pause:
            self.pause = False 
            self.pushButton.setText('Pause')
        else:
            self.pause = True 
            self.pushButton.setText('Continue')

    def play_videos(self,notePath):
        print(notePath)
        if notePath == 'pushButton_2':
            self.loadThermal()
        if notePath == 'pushButton_3':
            self.loadImage2()
	
    def run_threads(self):
        self.th['pushButton_2'] = Thread(target = self.play_videos, args = ('pushButton_2',)) 
        self.th['pushButton_2'].start()
        self.th['pushButton_3'] = Thread(target = self.play_videos, args = ('pushButton_3',)) 
        self.th['pushButton_3'].start()
        
    def loadThermal(self):
        """ This function will display a thermal image that has been taken by,
            the thermal array 
        """
        if self.started:
            self.started=False
            self.pushButton_2.setText('Start')	
        else:
            self.started=True
            self.pushButton_2.setText('Stop')
    
        
        for i,frame in enumerate(self.thermcam.frames):
            self.thermcam.mlx = frame
            self.thermcam.update_image_frame()
            image = self.thermcam._image
            self.update(image,self.label,i)
            
            time.sleep(0.5)
            key = cv2.waitKey(1) & 0xFF

            while (self.pause):
                self.update(image,self.label,i)

            if self.started==False:
                break
                
    
    def calibration(self,frame):
        self.bg = np.mean(frame[0:5], axis= 0)

    def loadImage2(self):
        """ This function will display a thermal image that has been taken by,
            the thermal array 
        """

        if self.started2:
            self.started2=False
            self.pushButton_2.setText('Start')	
        else:
            self.started2=True
            self.pushButton_2.setText('Stop')
        
        
        thermcam2 = pithermalcam(os.path.join(path,fileName))

        #self.calibration(thermcam2.frames)

        thermcam2.background_extraction()

        for i,frame in enumerate(thermcam2.foreground):
            #f_diff = np.abs(frame - self.bg)
            thermcam2.mlx =frame 
            thermcam2._raw_image=thermcam2._temps_to_rescaled_uints(thermcam2.mlx,0,8)
            #thermcam2._pull_raw_image()
            image = np.resize(thermcam2._raw_image,(24,32))
            thermcam2._raw_image = cv2.GaussianBlur(image,(3,3),0.1)
            
            #thermcam2._process_raw_image()
            image =cv2.resize(thermcam2._raw_image, (640,480), interpolation=cv2.INTER_AREA)
            image = cv2.flip(image, 0)

            self.setGray(image,self.label_4,i)
            #self.update(image,self.label_4,i)
            time.sleep(0.5)

            while (self.pause):
                self.setGray(image,self.label_4,i)
                #self.update(image,self.label_4,i)

            if self.started2==False:
                break



    def setPhoto(self,image, label):
        """ This function will take image input and resize it 
            only for display purpose and convert it to QImage
            to set at the label.
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(image))

    def setGray(self,image,label,frame):
        
        gray_image = image#cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img =cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((11,11),np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        text  =  'Frame: '+str(frame)
        img= cv2.putText(img, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (255,255,255), 2, cv2.LINE_AA) 
        
        time.sleep(0.3)
        image = QImage(img, 640, 480, QImage.Format_Grayscale8)
        label.setPixmap(QtGui.QPixmap.fromImage(image))



    def update(self,image,label,frame):
        """ This function will show text on the photos
        """
        img = image
        # Here we add display text to the image
        text  =  'Frame: '+str(frame)
        img= cv2.putText(img, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (255,255,255), 2, cv2.LINE_AA) 
        
        text = str(time.strftime("%H:%M %p"))
        img= cv2.putText(img, text, (image.shape[1]-180,30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (255,255,255), 2, cv2.LINE_AA) 
        
        time.sleep(0.3)
        self.setPhoto(img,label)
    

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Display of sensors"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        #self.pushButton_3.setText(_translate("MainWindow", "Start2"))
        self.pushButton.setText(_translate("MainWindow", "Pause"))
    

# Subscribe to PyShine Youtube channel for more detail! 

# WEBSITE: www.pyshine.com


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())