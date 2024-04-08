import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGroupBox, QVBoxLayout, QWidget,QPushButton,
                             QHBoxLayout,QGridLayout,QLabel)
from PyQt5.QtGui import QPixmap, QImage
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLGridItem,GLAxisItem,GLLinePlotItem
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from graphUtilities import * 
import csv
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from display_thermal import pithermalcam
import cv2 
import constants as const


class ScatterPlotWidget(QWidget):
    """

    A class that creates the Widget to display the pointcloud

    Attributes
    ----------
    scatter_item : GLScatterPlotItem or None
        Creates an scatter item and draws them
    
    currTime: int
        gets the time when a pointcloud was taken 
    paused : bool
        Used to pause the plot  of the  thermal images and the pointclouds
    pointclouds : dict
        dictionary that contain x,y,z and doppler
    coords : ndarray
        stores x,y,z and time from dictionary

    Methods
    -------
    __init__()
        Defines the atributes and the characteristics of the scatter widget
    
    read_data()
        reads pointcloud data stored on a csv file 
    
    point_transform_to_standard_axis()
        Transforms coordinates from mmWave frame to world coordinates
    
    update_points()
        updates the scatterplot

    proj3Dto2D()
        projects 3D pointcloud into 2D thermal image
    
    """
    def __init__(self):
        super().__init__()
        self.scatter_item = None
        self.currTime = 0
        self.paused = True
        self.pointclouds = {}
        self.coords = []
        # Create the 3D scatter plot widget
        self.plot_widget = GLViewWidget()
        self.plot_widget.opts['distance'] = 20
        self.plot_widget.opts['elevation'] = 30
        self.plot_widget.opts['azimuth'] = 45
        
        self.read_data()
        self.curr_frame = min(self.pointclouds.keys())

        # Add grid
        grid = GLGridItem()
        self.plot_widget.addItem(grid)

        axis = GLAxisItem(antialias=True, glOptions='additive', parentItem=None)
        axis.setSize(x=5, y=5, z=5)
        self.plot_widget.addItem(axis)

        # Add bounding box
        xl,yl,zl,xr,yr,zr = -2,0,0,2,6,3
        boundaryBoxViz =GLLinePlotItem()
        bottomSquare = GLLinePlotItem()
        boxLines = getBoxLines(xl,yl,zl,xr,yr,zr)
        squareLine = getSquareLines(xl,yl,xr,yr,zl)
        boundaryBoxViz.setData(pos=boxLines,color=pg.glColor('r'),width=2,antialias=True,mode='lines')
        bottomSquare.setData(pos=squareLine,color=pg.glColor('b'),width=2,antialias=True,mode='line_strip')
        self.plot_widget.addItem(boundaryBoxViz)
        self.plot_widget.addItem(bottomSquare)
        # Set layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_points)
        self.timer.start(120)  

    def read_data(self):
        """
        Reads stored data taken from mmWave sensor 
        
        """
        with open(const.P_EXPERIMENT_POINTCLOUD_READ, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                framenum = int(row[0])
                self.coords = [float(row[1]), float(row[2]), float(row[3]), float(row[4]),float(row[5])]

                world_coords = self.point_transform_to_standard_axis()

                if framenum in self.pointclouds:
                    # Append coordinates to the existing lists
                    for key, value in zip(["x", "y", "z", "doppler"],  world_coords):
                        self.pointclouds[framenum][key].append(value)
                else:
                    # If not, create a new dictionary for the framenum
                    self.pointclouds[framenum] = {
                        "x": [ world_coords[0]],
                        "y": [world_coords[1]],
                        "z": [world_coords[2]],
                        "doppler": [ self.coords[3]],
                        "time": [self.coords[4]],
                    }
   
    def point_transform_to_standard_axis(self):
        """
        Coordinate transformation from mmWaframe to real world frame, where the origin
        is in the bloor bellow the mmWave sensor
        """
        # Translation Matrix (T)
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, const.S_HEIGHT], [0, 0, 0, 1]])
        # Rotation Matrix (R_inv)
        ang_rad = np.deg2rad(const.S_TILT_ANGLE)
        R_inv = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(ang_rad), -np.sin(ang_rad), 0],
                [0, np.sin(ang_rad), np.cos(ang_rad), 0],
                [0, 0, 0, 1],
            ]
        )
        coordinates = np.concatenate((self.coords[:3], [1]))
        transformed_coords = np.dot(T, np.dot(R_inv, coordinates))

        return np.array(
            [
                transformed_coords[0],
                transformed_coords[1],
                transformed_coords[2],
                self.coords[3],
            ]
        )

    def update_points(self):
        """
        This method is used to iterate over the pointcloud, and plot the points 
        one frame at the time
        """
        if self.paused:
            return
        if self.scatter_item:
            self.plot_widget.removeItem(self.scatter_item)  # Remove previous group
        #print(self.curr_frame)
        if self.curr_frame in self.pointclouds:
            self.currTime = self.pointclouds[self.curr_frame]["time"][0]
            group_points = np.column_stack((self.pointclouds[self.curr_frame]["x"],self.pointclouds[self.curr_frame]["y"],self.pointclouds[self.curr_frame]["z"]))
            self.scatter_item = GLScatterPlotItem(pos=group_points, color=(1, 1, 1, 1), size=0.1, pxMode=False)
            self.plot_widget.addItem(self.scatter_item)    
        else:
            self.scatter_item = None
            #print("Frame not found")

        self.curr_frame = self.curr_frame+1

        if self.curr_frame>max(list(self.pointclouds.keys())):
            self.timer.stop()
    
    def proj3Dto2D(self):
        """
        Makes a projection of the 3D pointcould into the thermal image frame
        """
        yfov =np.deg2rad(55)
        xfov =np.deg2rad(35)
        ratio_x =12/np.tan(xfov/2)
        ratio_y =16/np.tan(yfov/2)
        desp = 0.01
        x_corr = 12
        y_corr = 16
        
        with open(const.P_EXPERIMENT_POINTCLOUD_TARGET, "r") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                framenum = int(row[0])
                target_coords = [float(row[1]), float(row[2]), float(row[3])]


        cords_2D=np.array([target_coords[0]*ratio_x/target_coords[2]+x_corr,
                           target_coords[1]*ratio_y/target_coords[2]+y_corr]) 
        return cords_2D

 
class ImageDisplayWidget(QWidget):
    """
    Widget for displaying thermal camera images with contours.

    Attributes:
        current_image_index (int): Index of the current image being displayed.
        paused (bool): Flag indicating whether the image display is paused or not.
        image_label (QLabel): QLabel to display the image.
        timestamp (int): Timestamp associated with the current image.
        thermal (pithermalcam): Instance of thermal camera.
    """
    def __init__(self):
        """
        Initializes the ImageDisplayWidget.

        Args:
            None
        """
        super().__init__()
        
        self.current_image_index = 0
        self.paused = True
        self.image_label = QLabel()
        self.timestamp = 0
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        self.thermal = pithermalcam(const.P_EXPERIMENT_THERMAL)
        
        
    def update_image(self):
        """
        Updates the displayed image.

        Args:
            None
        
        Returns:
            None
        """
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
                    x,y,w,h = cv2.boundingRect(cntr)
                    cv2.rectangle(image, (20*x, 20*y), (20*x+20*w, 20*y+20*h), (0, 0, 255), 2)
                    print("x,y,w,h:",x,y,w,h)
            
            text  =  'Frame: '+str(self.current_image_index)
            image= cv2.putText(image, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX , 1.0, (255,255,255), 2, cv2.LINE_AA)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0],QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.current_image_index += 1
        else:
            return
            

    def find_contours(self,index):
        """
        Finds contours in the thermal image.

        Args:
            index (int): Index of the image frame.
        
        Returns:
            list: List of contours found in the image.
        """
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
            self.msleep(1000)  # Wait for 1 second


class MainWindow(QMainWindow):
    """
    Main window class for the application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Scatter Plot and Thermal Display")
        self.resize(1500, 600)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.setStatsLayout()

        self.group_box = QGroupBox("3D Scatter Plot")
        self.group_layout = QVBoxLayout(self.group_box)
        self.scatter_plot_widget = ScatterPlotWidget()
        self.group_layout.addWidget(self.scatter_plot_widget)

        

        
        self.image_box = QGroupBox("Thermal Display")
        self.image_layout = QVBoxLayout(self.image_box)
        self.image_display_widget = ImageDisplayWidget()
        self.image_layout.addWidget(self.image_display_widget)



        # Create start button to start and pause plotting
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_plotting)
        layout = QGridLayout(self.central_widget)
        layout.addWidget(self.statBox,0,0,1,2)
        layout.addWidget(self.group_box,1,0,1,1)
        layout.addWidget(self.image_box,1, 1,1,1)
        layout.addWidget(self.start_button, 2, 0, 1, 2)


        # Create and start the image display thread
        self.image_thread = ImageDisplayThread()
        self.image_thread.update_signal.connect(self.graph_done)
        self.image_thread.start()
    
    def start_plotting(self):
        """
        Starts or pauses the plotting process based on the current state.
        """
        if self.scatter_plot_widget.paused:
            self.scatter_plot_widget.paused = False 
            self.image_display_widget.paused = False
            self.start_button.setText('Pause')
        else:
            self.scatter_plot_widget.paused = True
            self.image_display_widget.paused= True 
            self.start_button.setText('Continue')

    def graph_done(self):
        """
        Updates statistics and image display when the image display thread completes.
        """
        self.image_display_widget.update_image()
        self.framemmWave.setText('mmWave Frame:'+str(self.scatter_plot_widget.curr_frame))
        self.frameThermal.setText('Thermal Frame:'+str(self.image_display_widget.current_image_index))
        self.TimemmWave.setText('Time mmWave:'+str(self.scatter_plot_widget.currTime))
        self.TimeThermal.setText('Time thermal:'+ str(self.image_display_widget.timestamp))


    def setStatsLayout(self):
        """
        Sets up the layout for displaying point cloud statistics.

        """
        self.statBox = QGroupBox('PointCloud statistics')
        self.framemmWave = QLabel('mmWave Frame: 0')
        self.TimemmWave = QLabel('Time mmWave: 0 ms')
        self.frameThermal= QLabel('Thermal Frame: 0')
        self.TimeThermal= QLabel('Time thermal: 0ms')
        self.statsLayout = QHBoxLayout()
        self.statsLayout.addWidget(self.framemmWave)
        self.statsLayout.addWidget( self.TimemmWave)
        self.statsLayout.addWidget(self.frameThermal)
        self.statsLayout.addWidget(self.TimeThermal)
        self.statBox.setFixedHeight(50)
        self.statBox.setLayout(self.statsLayout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
