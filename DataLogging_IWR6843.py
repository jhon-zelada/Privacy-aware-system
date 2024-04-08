import sys
import time
import os
from threading import Thread, Event
from queue import Queue
import ntplib
import pandas as pd
import constants as const
from ReadDataIWR6843ISK import uartParserSDK

def query_to_overwrite(
    question="An experiment file with the same name already exists. Overwrite?\n",
):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [y/N]: "
    default = False

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return default
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Invalid answer\n")

def read_thread(queue: Queue, IWR6843: uartParserSDK, SLEEPTIME,client: ntplib.NTPClient, stop_event: Event):
    frame_select = const.FB_FRAMES_SKIP + 1
    try:
        while not stop_event.is_set():
            t0 = time.time()
            #dataOk, frameNumber, detObj = IWR1443.read()
            Object,targets, _,numDetectedObj, _,frameNumber, fail, _= IWR6843.readAndParseUart()
            dataOk = not fail
            detObj = {}

            detObj["x"]= Object[0,:]
            detObj["y"] = Object[1,:]
            detObj["z"] = Object[2,:]
            detObj["doppler"] = Object[3,:]
            try:
                detObj["time"] = [client.request('0.nl.pool.ntp.org', version=4).tx_time]*len(Object[0,:])
            except Exception as e:
                print(frameNumber)
                print("Error:", e)


            target = targets[1:4].T

            if dataOk and frameNumber % frame_select == 0:
                queue.put((frameNumber, detObj))
                #queue.put((frameNumber, target))

            #sys.stdout.write(f"\rFrame Number: {frameNumber}")
            #sys.stdout.flush()

            t_code = time.time() - t0
            t_sleep = max(0, SLEEPTIME - t_code)
            time.sleep(t_sleep)

    except KeyboardInterrupt:
        stop_event.set()


def write_thread(queue: Queue, data_buffer, data_path, stop_event: Event):
    try:
        while not stop_event.is_set():
            frameNumber, detObj = queue.get()

            # Prepare data for logging
            #data = {}
            if "time" in detObj:
                data = {
                    "Frame": frameNumber,
                    "X": detObj["x"],
                    "Y": detObj["y"],
                    "Z": detObj["z"],
                    "Doppler": detObj["doppler"],
                    "Time" : detObj["time"]
                }
            # Store data in the data path
            df = pd.DataFrame(data)
            data_buffer = pd.concat([data_buffer, df], ignore_index=True)

            if len(data_buffer) >= const.FB_WRITE_BUFFER_SIZE:
                data_buffer.to_csv(
                    data_path,
                    mode="a",
                    index=False,
                    header=False,
                )

                # Clear the buffer
                data_buffer.drop(data_buffer.index, inplace=True)

    except KeyboardInterrupt:
        stop_event.set()


def data_log_mode():
    # Create the general folder system
    client = ntplib.NTPClient()
    if not os.path.exists(const.P_LOG_PATH):
        os.makedirs(const.P_LOG_PATH)

    # Create the logging file target
    data_path = os.path.join(
        const.P_LOG_PATH,
        const.P_EXPERIMENT_FILE_WRITE,
    )
    if os.path.exists(data_path):
        if query_to_overwrite():
            os.remove(data_path)
        else:
            return

    data_buffer = pd.DataFrame()
    queue = Queue()
    stop_event = Event()

    IWR6843 = uartParserSDK(
        const.P_CONFIG_PATH, CLIport=const.P_CLI_PORT, Dataport=const.P_DATA_PORT
    )
    print("reached")
    SLEEPTIME = 0.001 * 55  # Sleeping period (sec)

    # Create separate threads for reading and writing
    read_thread_instance = Thread(
        target=read_thread, args=(queue, IWR6843, SLEEPTIME,client, stop_event)
    )
    write_thread_instance = Thread(
        target=write_thread, args=(queue, data_buffer, data_path, stop_event)
    )

    try:
        read_thread_instance.start()
        write_thread_instance.start()

        read_thread_instance.join()
        write_thread_instance.join()

    except KeyboardInterrupt:
        stop_event.set()
        read_thread_instance.join()
        write_thread_instance.join()


# Call the main function
data_log_mode()
