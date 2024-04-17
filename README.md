# Privacy-aware-system-for-outdoor-people-detection-and-tracking
This project aims to develop a sophisticated yet privacy-conscious system for outdoor people detection and tracking. Leveraging cutting-edge mmWave technology in conjunction with a thermal array sensor, our system ensures robust detection and tracking capabilities while prioritizing user privacy.

## Key Features:
**mmWave Technology:** Utilizing mmWave technology for precise detection and tracking of individuals in outdoor environments, enabling accurate positioning even in adverse weather conditions.

**Thermal Array Sensor:** Integration of a thermal array sensor to complement mmWave technology, providing additional data points for enhanced detection accuracy, particularly in low-light or challenging visual conditions.

## Scripts general description:
### **thermal_logging:**
Script executed on a raspberry to read data from the thermal array

### **offline_display:** 
Script used for data visualization for thermal and mmWave

### **ReadDataIWR6843ISK:** 
Script used to parse and read data from mmWave

### **DataLogging_IWR6843:** 
Stores the data from mmWave radar to preprocess and display.

## Getting Started
### Installation and Execution

1. Clone this repository.
   ```sh
   git clone https://github.com/jhon-zelada/Privacy-aware-system-for-outdoor-people-detection-and-tracking
   ```

2. Install Dependencies.
   ```sh
   pip install -r requirements.txt
   ```
3. Run the program to visualize stored data.
    ```sh
    python3 ./offline_display.py
    ```
