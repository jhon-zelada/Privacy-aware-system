import time
import numpy as np
import cv2
import cmapy
import constants as const
from GaussianModel import GMM
class pithermalcam:
    _colormap_list=['jet','gnuplot2','coolwarm','bwr','seismic','PiYG_r','tab10','tab20','brg']
    _interpolation_list =[cv2.INTER_NEAREST,cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_CUBIC,cv2.INTER_LANCZOS4,5,6]
    _interpolation_list_name = ['Nearest','Inter Linear','Inter Area','Inter Cubic','Inter Lanczos4','Pure Scipy', 'Scipy/CV2 Mixed']
    _current_frame_processed=False  # Tracks if the current processed image matches the current raw image
    mlx=None
    _temp_min=None
    _temp_max=None
    _raw_image=None
    _image=None
    _exit_requested=False
    _stoped= False
    _fps = 2
    def __init__(self, file_path,image_width:int=640, 
                image_height:int=480):
        self.image_width=image_width
        self.image_height=image_height
        self.file_path = file_path
        self.foreground= []
        self.frames = []
        self._colormap_index = 0
        self._interpolation_index = 2
        self.timestamps = []
        self.background =np.zeros(24*32) #np.zeros(shape = (train_frames.shape[1], train_frames.shape[2]))
        self.foreground = np.zeros(24*32)
        self._read_therm_cam()
        #self.background_extraction_average()
        self.background_extraction_GMM()
        self.foreground_estimation()
        self._t0 = time.time()
        self.update_image_frame()


    def _read_therm_cam(self):
        """Initialize data input"""
        with open(self.file_path) as data:
            line = data.read()
            data_term= list(eval(line))
            if (len(data_term[0])==2):
                frame_data = [np.array(x[1]) for x in data_term]
                self.timestamps = [np.array(x[0]) for x in data_term]
            else:
                frame_data = np.array(data_term)
        data.close()
        for frame in frame_data:
            for n in np.where(frame<-20): # finds values wrong read
                frame[n] = (frame[n-1]+frame[n-2])/2
            self.frames.append(frame)
        self.frames = np.array(self.frames)
        self.mlx = self.frames[0]
        


    def _pull_raw_image(self):
        """Get one pull of the raw image data, converting temp units if necessary"""
        # Get image
        self._raw_image = np.zeros((24*32,))
        try:
            self._raw_image = self.mlx # read mlx90640
            self._temp_min = np.min(self._raw_image)
            self._temp_max = np.max(self._raw_image)
            self._raw_image=self._temps_to_rescaled_uints(self._raw_image,self._temp_min,self._temp_max)
            self._current_frame_processed=False  # Note that the newly updated raw frame has not been processed
        except ValueError:
            print("Math error; continuing...")
            self._raw_image = np.zeros((24*32,))  # If something went wrong, make sure the raw image has numbers
            
        except OSError:
            print("IO Error; continuing...")
            self._raw_image = np.zeros((24*32,))  # If something went wrong, make sure the raw image has numbers
            

    def _process_raw_image(self):
        """Process the raw temp data to a colored image. Filter if necessary"""
        self._image = cv2.applyColorMap(self._raw_image, cmapy.cmap(self._colormap_list[self._colormap_index]))
        self._image = cv2.resize(self._image, (640,480), interpolation=self._interpolation_list[self._interpolation_index])
        self._image = cv2.flip(self._image, 0)

    def add_customized_text(self,text):
        """Add custom text to the center of the image, used mostly to notify user that server is off."""
        cv2.putText(self._image, text, (300,300),cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        time.sleep(0.1)

    def _show_processed_image(self):
        """Resize image window and display it"""
        cv2.namedWindow('Thermal Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Thermal Image', self.image_width,self.image_height)
        cv2.imshow('Thermal Image', self._image)
        time.sleep(1/self._fps)

    def _set_click_keyboard_events(self):
        key = cv2.waitKey(1) & 0xFF
        if  key==27:  # Exit nicely if escape key is used
            cv2.destroyAllWindows()
            self._displaying_onscreen = False
            print("Code Stopped by User")
            self._exit_requested=True
            
        elif key == ord("s"):
            self._stoped = True
            print("Press c to continue")
            
        elif key == ord("c"):
            self._stoped = False
            


    def display_next_frame_onscreen(self):
        """Display the camera live to the display"""
        # Display shortcuts reminder to user on first run
        for i,frame in enumerate(self.frames):
            self.mlx = frame
            self.update_image_frame()
            self.add_customized_text(f'Frame num: {i}')
            self._show_processed_image()
            while(self._stoped):
                self._set_click_keyboard_events()
                continue
            self._set_click_keyboard_events()
            

    def update_image_frame(self):
        """Pull raw temperature data, process it to an image, and update image text"""
        self._pull_raw_image()
        self._process_raw_image()
        #self._add_image_text()
        self._current_frame_processed=True
        return self._image

    def _temps_to_rescaled_uints(self,f,Tmin,Tmax):
        """Function to convert temperatures to pixels on image"""

        f=np.nan_to_num(f)
        norm = np.uint8((f - Tmin)*255/(Tmax-Tmin))
        norm.shape = (24,32)
        return norm
    
    def background_extraction_average(self):
        self.background = np.mean(self.frames[0:5], axis= 0)
    
    def background_extraction_GMM(self):
        n_samples = 70
        train_frames = self.frames[:n_samples,:]
        train_frames= train_frames.reshape([n_samples,24,32])
        print(train_frames.shape)
        gmm_background = np.zeros(shape = (train_frames.shape[1], train_frames.shape[2]))
        for i in range(train_frames.shape[1]):
            for j in range(train_frames.shape[2]):
                X = train_frames[:, i, j]
                X = X.reshape(X.shape[0], 1)
                gmm = GMM(2,max_iter=6)
                gmm.fit(X)
                means = gmm.means
                weights = gmm.weights
                idx = np.argmax(weights)
                gmm_background[i,j] = means[idx]
        self.background = gmm_background.reshape([32*24])
        print(self.background)
    def foreground_estimation(self):
        self._frame_diff = np.abs(self.frames-self.background)
        self._frame_diff= np.uint8(self._frame_diff*255/4)
        self.foreground = self._frame_diff.reshape((self._frame_diff.shape[0],24,32))
        


    def display_camera_onscreen(self):
        # Loop to display frames unless/until user requests exit
        while not self._exit_requested:
            self.display_next_frame_onscreen()

if __name__ == "__main__":
    # If class is run as main, read ini and set up a live feed displayed to screen
    thermcam = pithermalcam(const.P_EXPERIMENT_THERMAL)  # Instantiate class
    thermcam.display_camera_onscreen()