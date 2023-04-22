#!/usr/bin/env python3
import cv2
import numpy as np
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String, Int32, Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType

configPath = "/home/juancangaritan/Downloads/Repositorio_JuanC_Angarita_Noriega/YoloV4_Darknet/cfg/yolov4-tiny-ang.cfg"
weightsPath = "/home/juancangaritan/Downloads/Repositorio_JuanC_Angarita_Noriega/YoloV4_Darknet/backup/yolov4-tiny-ang_final.weights"
labelsPath = "/home/juancangaritan/Downloads/Repositorio_JuanC_Angarita_Noriega/YoloV4_Darknet/data/ang.names"
#pathImage_test = "/home/juancangaritan/Pictures/image40.jpg"
#img = cv2.imread(pathImage_test)
classes = open(labelsPath).read().rstrip('\n').split('\n')

# cargando red neuronal

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#net = cv2.dnn.readNet(weightsPath,configPath,'darknet')

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#qqqqqnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# <<<<<<------- Funcion dibujo de cajas ------------->>>>>>
def trazos_cajas(frame, boxes, confidences, class_ids, idxs):
    if len(idxs) > 0:
        for i in idxs.flatten(): #ubicacion de cada caja en matriz "boxes"
            left, top = boxes[i][0], boxes[i][1]  # coordenada esquina superior izquierda
            width, height = boxes[i][2], boxes[i][3] # alto y ancho de la caja

            # dibujando la caja con los datos hallados
            if classes[class_ids[i]] == 'Pad Aterrizaje':
                cv2.rectangle(frame,(left, top), (left + width, top + height), (0, 255, 0))
                label = "%s: %.2f"%(classes[class_ids[i]], confidences[i])
                cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                caja_pos = [left + width/2, top + height/2]
            if classes[class_ids[i]] == 'Vehiculo':
                cv2.rectangle(frame,(left, top), (left + width, top + height), (255, 200, 0))
                label = "%s: %.2f"%(classes[class_ids[i]], confidences[i])
                cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0))
            
    else:
        caja_pos = 'Sin detecciones'          
            
            
            #cv2.rectangle(frame,(left, top), (left + width, top + height), (0, 255, 0))
            #label = "%s: %.2f"%(classes[class_ids[i]], confidences[i])
            #cv2.putText(frame, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    return frame, caja_pos


# <<<<<<<------ Funcion detectar -------->>>>>>>

def detectar(net, outNames, classes, frame, conf_threshold, nms_threshold):
    boxes = []
    confidences = []
    class_ids = []
    frame_height, frame_width = frame.shape[:2]

    # creando un blob

   
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
        frame = cv2.resize(frame, (416, 416))
        net.setInput(np.array([[416, 416, 1.6]], dtype=np.float32), 'im_info')
    outputs = net.forward(outNames)
    #print(outputs)

    #r = blob[0, 0, :, :]
    #cv2.imshow('blob', r)
    #print(blob)    
    # obteniendo datos de las capas de salida

    for output in outputs:
        #print(output.shape)
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # aplicando el filtro "threshold"

            if confidence > conf_threshold:

                center_x = int(detection[0]*frame_width)
                center_y = int(detection[1]*frame_height)
                width = int(detection[2]*frame_width)
                height = int(detection[3]*frame_height)

                #coordenadas de la esquina izquierda
                left = int(center_x - (width/2))
                top = int(center_y - (height/2))

                # Guardando datos en boxes
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, confidences, class_ids, idxs


#### -------- < Inicio recoleccion de datos ----- >>>   

def callback(data):
    # Used to convert between ROS and OpenCV images
    image_bridge = CvBridge()

    # Output debugging information to the terminal
    #rospy.loginfo("receiving video frame")
    # Convert ROS Image message to OpenCV image
    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)
    outNames = net.getUnconnectedOutLayersNames()
    conf_threshold = 0.4
    nms_threshold = 0.4
    frame = image_bridge.imgmsg_to_cv2(data)
    boxes, confidences, class_ids, idxs = detectar(net, outNames, classes, frame, conf_threshold, nms_threshold)
    img, caja_pos = trazos_cajas(frame, boxes, confidences, class_ids, idxs)
    try:
        print (caja_pos)
    except UnboundLocalError:
        print('Sin detecciones')
    talker(caja_pos)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
    
    
def talker(caja_pos):
    
    send_distance = rospy.Publisher('Visual_distance', Int32MultiArray, queue_size= 10)
    
    
    #rospy.init_node('talker', anonymous= True)
    
    try:
        intposx = int(caja_pos[0])
        intposy = int(caja_pos[1])
        pos_enviar = Int32MultiArray()
        pos_enviar.data = [intposx, intposy]
        
    except ValueError:
        pos_enviar = Int32MultiArray()
        pos_enviar.data = [2022, 2022]
    send_distance.publish(pos_enviar)
    #rospy.loginfo(info)

def listener():
    rospy.init_node("video_sub_py", anonymous=True)
    rospy.Subscriber("camera/rgb/image_raw", Image, callback)
    

    rospy.spin()

    cv2.destroyAllWindows()




# Cargando los nombres de las clases


#------ metodo de seleccion de capas esandar --------
#layer_names = net.getUnconnectedOutLayersNames()
#layer_names = net.getLayerNames()
#layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
#print(layer_names)

# >>>>> ---------- Método de AlexeyAB -------- <<<<<<<

#frame = img

if __name__ == "__main__":
   

   listener()