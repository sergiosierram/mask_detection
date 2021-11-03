import socket
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

class Detector():
    def __init__(self, ip, device):
        self.ip = ip
        self.device = device
        self.initSocketServices()
        self.initDetector()
        self.main()

    def initSocketServices(self):
        port = 8081  # Make sure it's within the > 1024 $$ <65535 range
        self.s = socket.socket()
        self.s.connect((self.ip, port))

    def initDetector(self):
        self.video = cv2.VideoCapture(self.device)

        prototxtPath = "deploy.prototxt"
        weightsPath = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

        print("[INFO] loading face mask detector model...")
        self.model = load_model("mask_detect.model")

    def main(self):
        old_message = "10"
        msg = "101"
        msgs = []	
        while True:
            _, frame = self.video.read()
            h,w = frame.shape[:2]
            
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 117.0, 123.0),False,False)
            print("[INFO] computing face detections...")
            self.net.setInput(blob)
            detections = self.net.forward()
            
            try:
                for i in range(0 , detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x,y,W,H) = box.astype("int")
                        #print("sx: ",x,"ex",w,"sy",y,"ey",h)

                        crop = frame[y:H, x:W]
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        resize = cv2.resize(rgb, (224, 224))

                        img_array = img_to_array(resize)
                        process = preprocess_input(img_array)
                        face = np.expand_dims(process, axis=0)
                        
                        (mask, withoutMask) = self.model.predict(face)[0]
                        if mask > withoutMask:
                            label = "Mask Detected"
                            message = "1"
                            msgs.append(1) 
                        else:
                            label = "No Mask Detected"
                            message = "0"
                            msgs.append(0)
                        if label == "Mask Detected":
                            color = (0,255,0)  
                        else :
                            color =(0,0,255)
                        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,
                                    color,2)
                        cv2.rectangle(frame,(x,y),(W,H),color,2)
                        if len(msgs) == 3:
                            if sum(msgs) == 3:
                                msg = "1"
                            elif sum(msgs) == 0:
                                msg = "0"
                            if old_message != msg:
                                self.s.send(msg.encode('utf-8'))
                            old_message = msg
                            msgs = []
            except:
                print("[WARNING] skipping frame")    
            cv2.imshow("Output",frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()

        message = "q"
        self.s.send(message.encode('utf-8'))
        data = self.s.recv(1024).decode('utf-8')
        self.s.close()

if __name__ == '__main__':
  server = Detector('192.168.0.100', 2)  

  

