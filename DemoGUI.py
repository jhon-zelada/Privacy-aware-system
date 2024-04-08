import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFrame, QVBoxLayout, QPushButton, QLabel,
                             QGroupBox,QGridLayout,QWidget)
from display_thermal import pithermalcam
from PyQt5.QtGui import QPixmap, QImage
import cv2
from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import constants as const 

class ImageDisplayWidget(QWidget):
    def __init__(self,label):
        super().__init__()
        self.label1 = label
        self.current_image_index = 0
        self.paused = False
        self.image_label = QLabel()
        self.timestamp = 0
        self.presence = False
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.thermal = pithermalcam(const.P_EXPERIMENT_THERMAL)
    
        
    def update_image(self):
        if self.paused:
            return 
        if self.current_image_index < len(self.thermal.frames):
            self.thermal.mlx = self.thermal.frames[self.current_image_index]
            self.timestamp = self.thermal.timestamps[self.current_image_index]
            self.thermal.update_image_frame()
            image = self.thermal._image
            contours = self.find_contours(self.current_image_index)
            for cntr in contours:
                if cv2.contourArea(cntr)>3:
                    self.presence = True
                    self.label1.setStyleSheet("background-color: green; border-radius: 25px;")
                    x,y,w,h = cv2.boundingRect(cntr)
                    cv2.rectangle(image, (20*x, 20*y), (20*x+20*w, 20*y+20*h), (0, 0, 255), 2)
                    #print("x,y,w,h:",x,y,w,h)
            if self.presence:
                self.presence= False
            else:
                self.label1.setStyleSheet("background-color: red; border-radius: 25px;")

            text  =  'Frame: '+str(self.current_image_index)
            image= cv2.putText(image, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (255,255,255), 2, cv2.LINE_AA)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))
            
            self.current_image_index += 1
        else:
            return
            
    def find_contours(self,index):
        
        reescaled=self.thermal._temps_to_rescaled_uints(self.thermal.foreground[index],0,8)
        image = np.resize(reescaled,(24,32))
        filtered_img = cv2.GaussianBlur(image,(3,3),0.1)
        _, img =cv2.threshold(filtered_img, 50, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.flip(img, 0)
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        return contours


class ImageDisplayThread(QThread):
    update_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        

    def run(self):
        while True:
            self.update_signal.emit()
            self.msleep(400)  # Wait for 1 second
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the UI components
        self.setWindowTitle("Presence detection")
        self.running = False
        self.display_image = False
        self.init_ui()

    def init_ui(self):
        

        # Set the central widget
        self.central_widget = QFrame(self)
        
        self.setCentralWidget(self.central_widget)

        # Create three circular QFrames (labels)
        self.setGeometry(60, 60, 700, 400)
        self.label1 = QFrame(self)
        self.box1 = QGroupBox("Thermal")
        self.group_layout = QVBoxLayout(self.box1)
        self.group_layout.addWidget(self.label1)
        
        self.label2 = QFrame(self)
        self.box2 = QGroupBox("mmWave")
        self.group_layout2 = QVBoxLayout(self.box2)
        self.group_layout2.addWidget(self.label2)

        self.label3 = QFrame(self)
        self.box3 = QGroupBox("Thermal+ mmWave")
        self.group_layout3 = QVBoxLayout(self.box3)
        self.group_layout3.addWidget(self.label3)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_scene)
        layout = QGridLayout(self.central_widget)
        layout.addWidget(self.box1,0,0,1,1)
        layout.addWidget(self.box2,1,0,1,1)
        layout.addWidget(self.box3,2,0,1,1)
        layout.addWidget(self.start_button,3,0,1,1)

        # Set initial colors (red)
        self.label1.setStyleSheet("background-color: red; border-radius: 25px;")
        self.label2.setStyleSheet("background-color: red; border-radius: 25px;")
        self.label3.setStyleSheet("background-color: green; border-radius: 25px;")
        self.image_display_widget = ImageDisplayWidget(self.label1)
        if(not self.display_image):
            self.image_box = QGroupBox("Thermal Display")
            self.image_layout = QVBoxLayout(self.image_box)
            self.image_layout.addWidget(self.image_display_widget)
            layout.addWidget(self.image_box,0, 1,4,3)
            # Create and start the image display thread
            self.image_thread = ImageDisplayThread()
            self.image_thread.update_signal.connect(self.image_display_widget.update_image)
            self.image_thread.start()

    
    def start_scene(self):
        if not self.running:
            self.running = True
            self.label1.setStyleSheet("background-color: green; border-radius: 25px;")
            self.start_button.setText('Pause')
        else:
            self.running = False
            self.label1.setStyleSheet("background-color: red; border-radius: 25px;")
            self.start_button.setText('Continue')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
