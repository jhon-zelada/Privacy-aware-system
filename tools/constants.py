import os 
# Paths for stored data 

# Paths and Ports
P_CONFIG_PATH = "./config_cases/ISK_6m_default.cfg"
P_LOG_PATH =  "./log/"
P_DATA_PATH = "./data/"

P_CLI_PORT = "COM3"
P_DATA_PORT = "COM4"

FILE_NAME = "outdoor_test"
P_EXPERIMENT_POINTCLOUD_READ= os.path.join(
        P_LOG_PATH,
        FILE_NAME+".csv")

P_EXPERIMENT_THERMAL=  os.path.join(
        P_DATA_PATH,
        FILE_NAME+".txt")

P_EXPERIMENT_POINTCLOUD_TARGET = os.path.join(
                                P_LOG_PATH,
                                FILE_NAME+"_target.csv")

P_EXPERIMENT_POINTCLOUD_WRITE = "from_center_data.csv"




# Sensor position 
S_TILT_ANGLE = -46
S_HEIGHT = 4.1

###### Frames and Buffering #######
FB_FRAMES_SKIP = 0
FB_WRITE_BUFFER_SIZE = 100
FB_READ_BUFFER_SIZE = 100





