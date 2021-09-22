#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import time
import os
os.sys.path.insert(0, r"C:\Users\007al\OneDrive\Desktop\object-detection\custom")
os.sys.path.insert(0, r"C:\Users\007al\OneDrive\Desktop\object-detection\num-plate\train_yolo_to_detect_custom_object\yolo_custom_detection")
os.sys.path.insert(0, r"C:\Users\007al\OneDrive\Desktop\object-detection\yolo_helmet_detection")
#print(os.sys.path)
import object_detection_tutorial 
from Helmet_detection_YOLOV3 import detect_helemt as helmet_yolo
from num_detection import detect_num_plate
from glob import glob
import threading
# In[6]:

frame_global = None
stop = False
started = False
net = cv2.dnn.readNetFromDarknet("yolo/yolov3_pb.cfg","yolo/yolov3-obj_final.weights")
classes = []
with open("yolo/coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


# In[7]:


print(classes)


# In[8]:


layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print(outputlayers)
colors= np.random.uniform(0,255,size=(len(classes),3))


# In[10]:


#loading image
#cap=cv2.VideoCapture(0) #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
def start_detect(frame):
    starting_time= time.time()
    frame_id = 0

    for fn in glob('images/*.jpg'):#C:\Users\007al\OneDrive\Desktop\object-detection\num-plate\data
        # frame_id==0:
        # _,frame= cap.read() # 
        #frame = cv2.imread("car.jpg")
        #frame = cv2.imread("images/d3.jpg")#151
        frame_id+=1
        if frame_id > 1:
            break
        #frame = cv2.imread(fn)
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #reduce 416 to 200    

        net.setInput(blob)
        outs = net.forward(outputlayers)
        #print(outs)
        #print(outs[1])


        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence= confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
                

        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
        if len(indexes) != 0:
            
            #import object_detection_tutorial
            while not object_detection_tutorial.complete:
                do_nothing = 1

            import num_detection
            while not num_detection.complete:
                do_nothing = 1        

            import Helmet_detection_YOLOV3
            while not Helmet_detection_YOLOV3.complete:
                do_nothing =1

            time_end = time.time()
            print(time_end-starting_time)
            label = str(object_detection_tutorial.display)
            helemt = str(Helmet_detection_YOLOV3.label_global)
            diff = False
            print(label, helemt)
            if helemt != 'Helmet' and label != 'helmet_present' :
                diff = False
            elif helemt == 'Helmet' and label == 'helmet_present':
                diff = False
            else:
                diff = True 
            if not diff:
                if "not" in object_detection_tutorial.display:
                    img1 = cv2.resize(num_detection.img_global, None, fx=0.9, fy=0.9)
                    img2 = cv2.resize(object_detection_tutorial.image_np, None, fx=0.9, fy=0.9)
                    cv2.imshow("Number Plate", img1)
                    cv2.imshow("No Helmet", img2)
                else:
                    img1 = cv2.resize(object_detection_tutorial.image_np, None, fx=0.9, fy=0.9)
                    cv2.imshow("Helmet Detector", img1)
            else:
                if helemt != 'Helmet':
                    img1 = cv2.resize(num_detection.img_global, None, fx=0.9, fy=0.9)
                    img2 = cv2.resize(Helmet_detection_YOLOV3.img_global, None, fx=0.9, fy=0.9)
                    cv2.imshow("No Helmet", img2)
                    cv2.imshow("Number Plate", img1)
                else:
                    img1 = cv2.resize(Helmet_detection_YOLOV3.img_global, None, fx=0.9, fy=0.9)
                    cv2.imshow("Helmet Detector", img1)
        else:
            img1 = cv2.resize(frame, None, fx=0.9, fy=0.9)
            cv2.imshow("Person Detector", img1)
        key = cv2.waitKey() #wait 1ms the loop will start again and we will process the next frame
        
        #if key == 27: #esc key stops the process
            #break;
        
#cap.release()    
    cv2.destroyAllWindows()

def skipp_frames(cap):
    global frame_global, stop, started
    
    while cap.isOpened():
        while not stop:
            ret, frame_global = cap.read()
            started=True
            #print("here")

if __name__ == '__main__':
    #global stop, frame_global
    time_start = time.time()
    frame_id=0
    for fn in glob(r'.\num-plate\data\i2.png'):#.\\data\*.jpg
     
        frame = cv2.imread(fn)#151, napt,npp
        
        #try:
        #img1 = cv2.resize(frame, None, fx=0.4, fy=0.4)
        #cv2.imshow(str(frame_id),frame)
        #cv2.waitKey(5000)
        #frame_id = frame_id + 1
        thread1 = threading.Thread(target=object_detection_tutorial.detect_helemt, args=[frame])
        thread1.start()
        thread2 = threading.Thread(target=detect_num_plate, args=[frame])
        thread2.start()
        thread3 = threading.Thread(target=helmet_yolo, args=[frame])
        
        thread3.start()
        start_detect(frame)
        thread1.join()
        thread2.join()
        thread3.join()
         
            
    #stop = True
    #thread_vid.join()
    #cap.release()
    


# In[ ]:




