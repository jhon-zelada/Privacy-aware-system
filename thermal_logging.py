import time
import board
import busio
import adafruit_mlx90640
import argparse 
import ntplib 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--tiempo", type= float, help= "Time of execution",default= 5)
parser.add_argument("--rate", type= int, help= "refresh rate",default = 8)
parser.add_argument("--fname", type = str, help = "file name",default = "data")
args = parser.parse_args()

syncronized = False

while not syncronized:
  try:
    os.system('sudo ntpdate 192.168.195.178')
    syncronized = True
  except Exception as e:
    print(f"Error connecting to {server}: {e}")
    print("Retrying......")
  
i2c = busio.I2C(board.SCL, board.SDA, frequency=700000)
mlx = adafruit_mlx90640.MLX90640(i2c)
print("MLX addr detected on I2C, Serial #", [hex(i) for i in mlx.serial_number])

if (args.rate == 2):
  mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ

elif (args.rate == 4) :
  mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ

else:
  mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
  
print("Refresh rate: ", pow(2, (mlx.refresh_rate - 1)), "Hz")


frame = [0] * 768
start_time = time.time()
frames = []

while (time.time()-start_time<args.tiempo):
    try:
        mlx.getFrame(frame)
        
        frames.append([time.time(),frame])
        frame = [0] * 768
        
    except ValueError:
        print("An Error has occurred")
        continue
    
print("Number of frames taken:", len(frames))
print("FPS:",len(frames)/args.tiempo)

with open(f"{args.fname}.txt","w") as data:
    data.write(str(frames))
    data.close

