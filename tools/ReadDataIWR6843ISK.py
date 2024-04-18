import struct
import serial
import time
import numpy as np
import math

class uartParserSDK():
    def __init__(self,configFileName, CLIport, Dataport,type='3D People Counting'):
        self.configFileName = configFileName
        self.uartCom = serial.Serial(CLIport,115200,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,timeout=0.3)
        self.dataCom = serial.Serial(Dataport,921600*1,parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,timeout=0.025)
        self.headerLength = 52
        self.magicWord = 0x708050603040102
        self.threeD = 0
        self.capon3D = 0
        self.aop = 0
        self.maxPoints = 1150
        if (type=='(Legacy): Overhead People Counting'):
            self.threeD = 1
        elif (type=='3D People Counting'):
            self.capon3D = 1
        elif (type == 'Capon3DAOP'): #unused
            self.capon3D = 1
            self.aop = 1
        #data storage
        self.pcPolar = np.zeros((5,self.maxPoints))
        self.pcBufPing = np.zeros((5,self.maxPoints))
        self.numDetectedObj = 0
        self.targetBufPing = np.ones((10,20))*-1
        self.indexBufPing = np.zeros((1,self.maxPoints))
        self.classifierOutput = []
        self.frameNum = 0
        self.missedFrames = 0
        self.byteData = bytes(1)
        self.oldData = []
        self.indexes = []
        self.numDetectedTarget = 0
        self.fail = 0
        self.unique = []
        self.savedData = []
        self.saveNum = 0
        self.saveNumTxt = 0
        self.replayData = []
        self.startTimeLast = 0
        self.saveReplay = 0
        self.savefHist = 0
        self.saveTextFile = 1 # previusly was 0
        self.fHistRT = np.empty((100,1), dtype=object)
        self.plotDimension = 0
        self.getUnique = 0
        self.CaponEC = 0
        self.profile = {'startFreq': 60.25, 'numLoops': 64, 'numTx': 3, 'sensorHeight':3, 'maxRange':10, 'az_tilt':0, 'elev_tilt':0}
        self.__parseCfg()
        self.sendCfg()
        self.printVerbosity = 0 #set 0 for limited logFile printing, 1 for more logging
        
        if (self.capon3D):
            self.textStructCapon3D = np.zeros(1000*3*self.maxPoints*10).reshape((1000,3,self.maxPoints,10))#[frame #][header,pt cloud data,target info]


    #convert 3D people counting polar to 3D cartesian
    def polar2Cart3D(self):
        self.pcBufPing = np.empty((5,self.numDetectedObj))
        for n in range(0, self.numDetectedObj):
            self.pcBufPing[2,n] = self.pcPolar[0,n]*math.sin(self.pcPolar[2,n]) #z
            self.pcBufPing[0,n] = self.pcPolar[0,n]*math.cos(self.pcPolar[2,n])*math.sin(self.pcPolar[1,n]) #x
            self.pcBufPing[1,n] = self.pcPolar[0,n]*math.cos(self.pcPolar[2,n])*math.cos(self.pcPolar[1,n]) #y
        self.pcBufPing[3,:] = self.pcPolar[3,0:self.numDetectedObj] #doppler
        self.pcBufPing[4,:] = self.pcPolar[4,0:self.numDetectedObj] #snr
        #print(self.pcBufPing[:,:10])

    #decode People Counting TLV Header
    def tlvHeaderDecode(self, data):
        #print(len(data))
        tlvType, tlvLength = struct.unpack('2I', data)
        return tlvType, tlvLength


    #decode 3D People Counting Point Cloud TLV
    def parseDetectedObjects3D(self, data, tlvLength):
        objStruct = '5f'
        objSize = struct.calcsize(objStruct)
        self.numDetectedObj = int(tlvLength/20)
        for i in range(self.numDetectedObj):
            try:
                self.pcPolar[0,i], self.pcPolar[1,i], self.pcPolar[2,i], self.pcPolar[3,i], self.pcPolar[4,i] = struct.unpack(objStruct,data[:objSize])
                data = data[20:]
            except:
                self.numDectedObj = i
                print('failed to get point cloud')
                break
        self.polar2Cart3D()

    #support for Capoin 3D point cloud
    #decode Capon 3D point Cloud TLV
    def parseCapon3DPolar(self, data, tlvLength):
        pUnitStruct = '5f'  #elev, azim, doppler, range, snr
        pUnitSize = struct.calcsize(pUnitStruct)
        pUnit = struct.unpack(pUnitStruct, data[:pUnitSize])
        data = data[pUnitSize:]
        objStruct = '2bh2H' #2 int8, 1 int16, 2 uint16
        objSize = struct.calcsize(objStruct)
        self.numDetectedObj = int((tlvLength-pUnitSize)/objSize)
        #if (self.printVerbosity == 1):
        #print('Parsed Points: ', self.numDetectedObj)
        for i in range(self.numDetectedObj):
            try:
                elev, az, doppler, ran, snr = struct.unpack(objStruct, data[:objSize])
                #print(elev, az, doppler, ran, snr)
                data = data[objSize:]
                #get range, azimuth, doppler, snr
                self.pcPolar[0,i] = ran*pUnit[3]           #range
                if (az >= 128):
                    print ('Az greater than 127')
                    az -= 256
                if (elev >= 128):
                    print ('Elev greater than 127')
                    elev -= 256
                if (doppler >= 32768):
                    print ('Doppler greater than 32768')
                    doppler -= 65536
                self.pcPolar[1,i] = az*pUnit[1]  #azimuth
                self.pcPolar[2,i] = elev*pUnit[0] #elevation
                self.pcPolar[3,i] = doppler*pUnit[2]       #doppler
                self.pcPolar[4,i] = snr*pUnit[4]           #snr
                
                #add pt cloud data to textStructCapon3DCapon3D for text file printing
                #self.textStructCapon3DCapon3D[,,,] = [frame #][header,pt cloud data,target info]
                #[][pt cloud = 0][pt index][#elev, azim, doppler, range, snr]
                self.textStructCapon3D[self.frameNum%1000,1,i,0] = self.pcPolar[2,i] #elev
                self.textStructCapon3D[self.frameNum%1000,1,i,1] = self.pcPolar[1,i] #az
                self.textStructCapon3D[self.frameNum%1000,1,i,2] = self.pcPolar[3,i] #doppler
                self.textStructCapon3D[self.frameNum%1000,1,i,3] = self.pcPolar[0,i] #range
                self.textStructCapon3D[self.frameNum%1000,1,i,4] = self.pcPolar[4,i] #snr
            except:
                self.numDetectedObj = i
                print('Point Cloud TLV Parser Failed')
                break
        self.polar2Cart3D()
    

    #decode 3D People Counting Target List TLV
    def parseDetectedTracks3D(self, data, tlvLength):
        targetStruct = 'I9f'
        targetSize = struct.calcsize(targetStruct)
        self.numDetectedTarget = int(tlvLength/targetSize)
        targets = np.empty((13,self.numDetectedTarget))
        for i in range(self.numDetectedTarget):
            targetData = struct.unpack(targetStruct,data[:targetSize])
            targets[0:7,i]=targetData[0:7]
            targets[7:10,i]=[0,0,0]
            targets[10:13,i] = targetData[7:10]
            data = data[targetSize:]
        self.targetBufPing = targets

    #decode Target Index TLV
    def parseTargetAssociations(self, data):
        targetStruct = 'B'
        targetSize = struct.calcsize(targetStruct)
        numIndexes = int(len(data)/targetSize)
        self.indexes = []
        self.unique = []
        try:
            for i in range(numIndexes):
                ind = struct.unpack(targetStruct, data[:targetSize])
                self.indexes.append(ind[0])
                data = data[targetSize:]
            if (self.getUnique):
                uTemp = self.indexes[math.ceil(numIndexes/2):]
                self.indexes = self.indexes[:math.ceil(numIndexes/2)]
                for i in range(math.ceil(numIndexes/8)):
                    for j in range(8):
                        self.unique.append(getBit(uTemp[i], j))
        except:
            print('TLV Index Parse Fail')

    #decode Classifier output
    def parseClassifierOutput(self, data):
        classifierDataStruct = 'Ii'
        clOutSize = struct.calcsize(classifierDataStruct)
        self.classifierOutput = np.zeros((2,self.numDetectedTarget))
        for i in range(self.numDetectedTarget):
            self.classifierOutput[0,i], self.classifierOutput[1,i] = struct.unpack(classifierDataStruct, data[:clOutSize])
            data = data[clOutSize:]

#below is for labs that are compliant with SDK 3.x  This code can parse the point cloud TLV and point cloud side info TLV from the OOB demo.  
#It can parse the SDK3.x Compliant People Counting demo "tracker_dpc"
    #get SDK3.x Cartesian Point Cloud
    def parseSDK3xPoints(self, dataIn, numObj):
        pointStruct = '4f'
        pointLength = struct.calcsize(pointStruct)
        try:
            for i in range(numObj):
                self.pcBufPing[0,i], self.pcBufPing[1,i], self.pcBufPing[2,i], self.pcBufPing[3,i] = struct.unpack(pointStruct, dataIn[:pointLength])
                dataIn = dataIn[pointLength:]
            self.pcBufPing = self.pcBufPing[:,:numObj]
        except Exception as e:
            print(e)
            self.fail = 1

    #get Side Info SDK 3.x
    def parseSDK3xSideInfo(self, dataIn, numObj):
        sideInfoStruct = '2h'
        sideInfoLength = struct.calcsize(sideInfoStruct)
        try:
            for i in range(numObj):
                self.pcBufPing[4,i], unused = struct.unpack(sideInfoStruct, dataIn[:sideInfoLength])
                dataIn = dataIn[sideInfoLength:]
        except Exception as e:
            print(e)
            self.fail = 1

    #convert SDK compliant Polar Point Cloud to Cartesian
    def polar2CartSDK3(self):
        self.pcBufPing = np.empty((5,self.numDetectedObj))
        for n in range(0, self.numDetectedObj):
            self.pcBufPing[2,n] = self.pcPolar[0,n]*math.sin(self.pcPolar[2,n]) #z
            self.pcBufPing[0,n] = self.pcPolar[0,n]*math.cos(self.pcPolar[2,n])*math.sin(self.pcPolar[1,n]) #x
            self.pcBufPing[1,n] = self.pcPolar[0,n]*math.cos(self.pcPolar[2,n])*math.cos(self.pcPolar[1,n]) #y
        self.pcBufPing[3,:] = self.pcPolar[3,0:self.numDetectedObj] #doppler

    #decode SDK3.x Format Point Cloud in Polar Coordinates
    def parseSDK3xPolar(self, dataIn, tlvLength):
        pointStruct = '4f'
        pointLength = struct.calcsize(pointStruct)
        self.numDetectedObj = int(tlvLength/pointLength)
        try:
            for i in range(self.numDetectedObj):
                self.pcPolar[0,i], self.pcPolar[1,i], self.pcPolar[2,i], self.pcPolar[3,i] = struct.unpack(pointStruct, dataIn[:pointLength])
                dataIn = dataIn[pointLength:]
        except:
            self.fail = 1
            return
        self.polar2CartSDK3()


    def parseDetectedTracksSDK3x(self, data, tlvLength):
        if (self.printVerbosity == 1):
            print(tlvLength)
        if (self.CaponEC):
            targetStruct = 'I27f'
        else:
            #targetStruct = 'I15f'
            targetStruct = 'I27f'
        targetSize = struct.calcsize(targetStruct)
        if (self.printVerbosity == 1):
            print('TargetSize=',targetSize)
        self.numDetectedTarget = int(tlvLength/targetSize)
        if (self.printVerbosity == 1):
            print('Num Detected Targets = ',self.numDetectedTarget)
        targets = np.empty((16,self.numDetectedTarget))
        rotTarget = [0,0,0]
        #theta = self.profile['elev_tilt']
        #print('theta = ',theta)
        #Rx = np.matrix([[ 1, 0           , 0           ],
        #           [ 0, math.cos(theta),-math.sin(theta)],
        #           [ 0, math.sin(theta), math.cos(theta)]])
        try:
            for i in range(self.numDetectedTarget):
                targetData = struct.unpack(targetStruct,data[:targetSize])
                if (self.printVerbosity == 1):
                    print(targetData)
                #tid, x, y
                if (self.CaponEC):
                    targets[0:13,i]=targetData[0:13]
                else:
                    #tid, pos x, pos y
                    targets[0:3,i]=targetData[0:3]
                    if (self.printVerbosity == 1):
                        print('Target Data TID,X,Y = ',targets[0:3,i])
                        print('i = ',i)
                    # pos z
                    targets[3,i] = targetData[3]
                    
                    #rotTargetDataX,rotTargetDataY,rotTargetDataZ = rotX (targetData[1],targetData[2],targetData[3],self.profile['elev_tilt'])
                    
                    #print('Target Data TID,X,Y = ',rotTargetDataX,', ',rotTargetDataY,', ',rotTargetDataZ)
                    #vel x, vel y
                    targets[4:6,i] = targetData[4:6]
                    #vel z
                    targets[6,i] = targetData[6]
                    # acc x, acc y
                    targets[7:9,i] = targetData[7:9]
                    # acc z
                    targets[9,i] = targetData[9]
                    #ec[16]
                    #targets[10:14,i]=targetData[10:14]
                    targets[10:13,i]=targetData[10:13]#Chris 2020-12-18
                    if (self.printVerbosity == 1):
                        print('ec = ',targets[10:13,i])
                    #g
                    #targets[14,i]=targetData[14]
                    targets[14,i]=targetData[26]
                    if (self.printVerbosity == 1):
                        print('g= ',targets[14,i])
                    #confidenceLevel
                    #targets[15,i]=targetData[15]
                    targets[15,i]=targetData[27]
                    if (self.printVerbosity == 1):
                        print('Confidence Level = ',targets[15,i])
                                        
                                        
                    #self.textStructCapon3D[[frame #],[header,pt cloud data,target info],index,data]
                    #[][header][magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum]
                    #[][pt cloud][pt index][#elev, azim, doppler, range, snr]
                    #[][target][Target #][TID,x,y,z,vx,vy,vz,ax,ay,az]                    
                    if (self.saveTextFile):
                        self.textStructCapon3D[self.frameNum%1000,2,i,0] = targets[0,i] #TID
                        self.textStructCapon3D[self.frameNum%1000,2,i,1] = targets[1,i] #x
                        self.textStructCapon3D[self.frameNum%1000,2,i,2] = targets[2,i] #y
                        self.textStructCapon3D[self.frameNum%1000,2,i,3] = targets[3,i] #z
                        self.textStructCapon3D[self.frameNum%1000,2,i,4] = targets[4,i] #vx
                        self.textStructCapon3D[self.frameNum%1000,2,i,5] = targets[5,i] #vy
                        self.textStructCapon3D[self.frameNum%1000,2,i,6] = targets[6,i] #vz
                        self.textStructCapon3D[self.frameNum%1000,2,i,7] = targets[7,i] #ax
                        self.textStructCapon3D[self.frameNum%1000,2,i,8] = targets[8,i] #ay
                        self.textStructCapon3D[self.frameNum%1000,2,i,9] = targets[9,i] #az
                        if (self.printVerbosity == 1):
                            print('target added to textStructCapon3D')
                data = data[targetSize:]
        except:
            print('Target TLV parse failed')
        self.targetBufPing = targets
        if (self.printVerbosity == 1):
            print(targets)
        
    #parsing for 3D People Counting lab
    def Capon3DHeader(self, dataIn):
        #reset point buffers
        self.pcBufPing = np.zeros((5,self.maxPoints))
        self.pcPolar = np.zeros((5,self.maxPoints))
        self.targetBufPing = np.zeros((13,20))
        self.numDetectedTarget = 0
        self.numDetectedObj = 0
        self.indexes = []
        tlvHeaderLength = 8
        headerLength = 48
        #stay in this loop until we find the magic word or run out of data to parse
        while (1):
            try:
                magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum =  struct.unpack('Q9I2H', dataIn[:headerLength])
            except Exception as e:
                #bad data, return
                #print("Cannot Read Frame Header")
                #print(e)
                self.fail = 1
                return dataIn
            if (magic != self.magicWord):
                #wrong magic word, increment pointer by 1 and try again
                dataIn = dataIn[1:]
            else:
                #got magic word, proceed to parse
                break
        

        dataIn = dataIn[headerLength:]
        remainingData = packetLength - len(dataIn) - headerLength
        #check to ensure we have all of the data
        #print('remaining data = ',remainingData)
        if (remainingData > 0):
            newData = self.dataCom.read(remainingData)
            remainingData = packetLength - len(dataIn) - headerLength - len(newData)
            dataIn += newData
        if (self.saveTextFile):
            self.textStructCapon3D[self.frameNum%1000,0,0,0] = magic
            self.textStructCapon3D[self.frameNum%1000,0,1,0] = version
            self.textStructCapon3D[self.frameNum%1000,0,2,0] = packetLength
            self.textStructCapon3D[self.frameNum%1000,0,3,0] = platform
            self.textStructCapon3D[self.frameNum%1000,0,4,0] = frameNum
            self.textStructCapon3D[self.frameNum%1000,0,5,0] = subFrameNum
            self.textStructCapon3D[self.frameNum%1000,0,6,0] = chirpMargin
            self.textStructCapon3D[self.frameNum%1000,0,7,0] = frameMargin
            self.textStructCapon3D[self.frameNum%1000,0,8,0] = uartSentTime
            self.textStructCapon3D[self.frameNum%1000,0,9,0] = trackProcessTime
            self.textStructCapon3D[self.frameNum%1000,0,10,0] = numTLVs
            self.textStructCapon3D[self.frameNum%1000,0,11,0] = checksum
            if (self.printVerbosity == 1):
                print('FrameNumber = ',self.textStructCapon3D[self.frameNum%1000,0,4,0])
         
        #now check TLVs
        for i in range(numTLVs):
            #try:
            #print("DataIn Type", type(dataIn))
            try:
                tlvType, tlvLength = self.tlvHeaderDecode(dataIn[:tlvHeaderLength])
                dataIn = dataIn[tlvHeaderLength:]
                dataLength = tlvLength-tlvHeaderLength
            except:
                print('TLV Header Parsing Failure')
                self.fail = 1
                return dataIn
            if (tlvType == 6):
                #DPIF Polar Coordinates
                self.parseCapon3DPolar(dataIn[:dataLength], dataLength)
            elif (tlvType == 7):
                #target 3D
                self.parseDetectedTracksSDK3x(dataIn[:dataLength], dataLength)
                
            elif (tlvType == 8):
                #target index
                self.parseTargetAssociations(dataIn[:dataLength])
            elif (tlvType == 9):
                if (self.printVerbosity == 1):
                    print('type9')
                #side info
                #self.parseSDK3xSideInfo(dataIn[:dataLength], self.numDetectedObj)
            dataIn = dataIn[dataLength:]
            #except Exception as e:
            #    print(e)
            #    print ('failed to read OOB SDK3.x TLV')
        if (self.frameNum + 1 != frameNum):
            self.missedFrames += frameNum - (self.frameNum + 1)
        self.frameNum = frameNum
        return dataIn


    # This function is always called - first read the UART, then call a function to parse the specific demo output
    # This will return 1 frame of data. This must be called for each frame of data that is expected. It will return a dict containing:
    #   1. Point Cloud
    #   2. Target List
    #   3. Target Indexes
    #   4. number of detected points in point cloud
    #   5. number of detected targets
    #   6. frame number
    #   7. Fail - if one, data is bad
    #   8. classifier output
    # Point Cloud and Target structure are liable to change based on the lab. Output is always cartesian.
    def readAndParseUart(self):
        self.fail = 0
        numBytes = 4666
        data = self.dataCom.read(numBytes)
        if (self.byteData is None):
            self.byteData = data
        else:
            self.byteData += data
        #try:
        if (self.capon3D == 1):
            self.byteData = self.Capon3DHeader(self.byteData)

        if (self.fail):
            return self.pcBufPing, self.targetBufPing, self.indexes, self.numDetectedObj, self.numDetectedTarget, self.frameNum, self.fail, self.classifierOutput

        if (self.saveTextFile):
            if (self.frameNum%1000 == 0):
                if (self.capon3D):
                    toSave = self.textStructCapon3D

                print('Saved data file ', self.saveNumTxt)
                fileName = 'pHistText_'+str(self.saveNumTxt)+'.csv'
                if (self.saveNumTxt < 75):
                    self.saveNumTxt += 1
                else:
                    self.saveNumTxt = 0
                tfile = open(fileName, 'w')
                tfile.write('This file contains parsed UART data in sensor centric coordinates\n')
                tfile.write('file format version 1.0\n')
                #tfile.write(str(toSave))
                
                if (self.capon3D):
                    #[frame #][header,pt cloud data,target info]
                    #[][header][magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum]
                    #[][pt cloud][pt index][#elev, azim, doppler, range, snr]
                    #[][target][Target #][TID,x,y,z,vx,vy,vz,ax,ay,az]

                    for i in range (1000):
                        tfile.write('magic, version, packetLength, platform, frameNum, subFrameNum, chirpMargin, frameMargin, uartSentTime, trackProcessTime, numTLVs, checksum\n')
                        for j in range (0,12):
                            tfile.write(str(self.textStructCapon3D[i,0,j,0]))
                            tfile.write(',')
                            #print(str(self.textStructCapon3D[i,0,j,0]))
                        tfile.write('\n')
                        tfile.write('elev, azim, doppler, range, snr\n')
                        for j in range (np.count_nonzero(self.textStructCapon3D[i,1,:,0])): #self.numDetectedObj):#len(self.textStructCapon3D[i,1,:,0]!=0)):
                            for k in range(5):
                                tfile.write(str(self.textStructCapon3D[i,1,j,k]))
                                tfile.write(',')
                            tfile.write('\n')

                        tfile.write('TID,x,y,z,vx,vy,vz,ax,ay,az\n')
                        for j in range (np.count_nonzero(self.textStructCapon3D[i,2,:,1])):
                            for k in range(10):
                                tfile.write(str(self.textStructCapon3D[i,2,j,k]))
                                tfile.write(',')
                            tfile.write('\n')
                    self.textStructCapon3D = np.zeros(1000*3*12*self.maxPoints).reshape((1000,3,12,self.maxPoints))#[frame #][header,pt cloud data,target info]
                    tfile.close
        parseEnd = int(round(time.time()*1000))
        #print (self.pcBufPing)
        return self.pcBufPing, self.targetBufPing, self.indexes, self.numDetectedObj, self.numDetectedTarget, self.frameNum, self.fail, self.classifierOutput

    def __parseCfg(self):
        cfg_file = open(self.configFileName, 'r')
        self.cfg = cfg_file.readlines()
        counter = 0
        chirpCount = 0
        for line in self.cfg:
            args = line.split()
            if (len(args) > 0):
                if (args[0] == 'cfarCfg'):
                    zy = 4
                    #self.cfarConfig = {args[10], args[11], '1'}
                elif (args[0] == 'AllocationParam'):
                    zy=3
                    #self.allocConfig = tuple(args[1:6])
                elif (args[0] == 'GatingParam'):
                    zy=2
                    #self.gatingConfig = tuple(args[1:4])
                elif (args[0] == 'SceneryParam' or args[0] == 'boundaryBox'):
                    self.boundaryLine = counter
                    self.profile['leftX'] = float(args[1])
                    self.profile['rightX'] = float(args[2])
                    self.profile['nearY'] = float(args[3])
                    self.profile['farY'] = float(args[4])
                    if (self.capon3D == 1):
                        self.profile['bottomZ'] = float(args[5])
                        self.profile['topZ'] = float(args[6])
                    else:
                        self.profile['bottomZ'] = float(-3)
                        self.profile['topZ'] = float(3)
                elif (args[0] == 'staticBoundaryBox'):
                    self.staticLine = counter
                elif (args[0] == 'profileCfg'):
                    self.profile['startFreq'] = float(args[2])
                    self.profile['idle'] = float(args[3])
                    self.profile['adcStart'] = float(args[4])
                    self.profile['rampEnd'] = float(args[5])
                    self.profile['slope'] = float(args[8])
                    self.profile['samples'] = float(args[10])
                    self.profile['sampleRate'] = float(args[11])
                    print(self.profile)
                elif (args[0] == 'frameCfg'):
                    self.profile['numLoops'] = float(args[3])
                    self.profile['numTx'] = float(args[2])+1
                elif (args[0] == 'chirpCfg'):
                    chirpCount += 1
                elif (args[0] == 'sensorPosition'):
                    self.profile['sensorHeight'] = float(args[1])
                    print('Sensor Height from cfg = ',str(self.profile['sensorHeight']))
                    self.profile['az_tilt'] = float(args[2])
                    self.profile['elev_tilt'] = float(args[3])
            counter += 1
        self.profile['maxRange'] = self.profile['sampleRate']*1e3*0.9*3e8/(2*self.profile['slope']*1e12)
        #update chirp table values
        bw = self.profile['samples']/(self.profile['sampleRate']*1e3)*self.profile['slope']*1e12
        rangeRes = 3e8/(2*bw)
        Tc = (self.profile['idle']*1e-6 + self.profile['rampEnd']*1e-6)*chirpCount
        lda = 3e8/(self.profile['startFreq']*1e9)
        maxVelocity = lda/(4*Tc)
        velocityRes = lda/(2*Tc*self.profile['numLoops']*self.profile['numTx'])

    #send cfg over uart
    def _sendCfg(self, cfg):
        for line in cfg:
            time.sleep(.1)
            self.uartCom.write(f"{line}".encode()) #need to be corrected
            ack = self.uartCom.readline()
            print(ack)
            ack = self.uartCom.readline()
            print(ack)
        time.sleep(3)
        self.uartCom.reset_input_buffer()
        self.uartCom.close()
    
    def sendCfg(self):
        try:
            if (self.capon3D == 1):
                self._sendCfg(self.cfg)
                self.configSent = 1
                print("Correct configuration....")
        except Exception as e:
            print(e)
            print ('No cfg file selected!')

    def __del__(self):
        self.dataCom.close()    
        
def getBit(byte, bitNum):
    mask = 1 << bitNum
    if (byte&mask):
        return 1
    else:
        return 0