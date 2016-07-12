import cv2
import numpy as np
import scipy
import time
import socket
import sys
def hist(a):
	hist, bin_edges = np.histogram(a, bins = range(64))
	return hist
def calcgrad(i):
	#i=cv2.imread("images.png",0)
	#i=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
	#i=np.array(i)
	height,width=i.shape
	first=np.pad(i,((0,0),(1,0)),'constant')
	second=np.pad(i,((0,1),(1,0)),'constant')
	third=np.pad(i,((0,1),(0,0)),'constant')
	fourth=np.pad(i,((0,1),(0,1)),'constant')
	first=first[:,0:width]
	second=second[1:height+1,0:width]
	third=third[1:height+1,:]
	fourth=fourth[1:height+1,1:width+1]
	first=i-first
	second=i-second
	third=i-third
	fourth=i-fourth
	combo1=32*np.array( first >= second, dtype=int)
	combo2=16*np.array( first >= third, dtype=int)
	combo3=8*np.array( first >= fourth, dtype=int)
	combo4=4*np.array( second >= third, dtype=int)
	combo5=2*np.array( second >= fourth, dtype=int)
	combo6=np.array( third >= fourth, dtype=int)
	ldgp=combo1+combo2+combo3+combo4+combo5+combo6



	return ldgp
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera=cv2.VideoCapture(0)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print s
port=sys.argv[1]
ret=s.connect(("localhost",int(port)))
#print ret
while True:
	ret,frame=camera.read()
	gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for (x,y,w,h) in faces:
		face=gray[y: y + h, x: x + w]
    		#face=cv2.resize(face,(240,240))
    		face=cv2.resize(face,(240,240))
    		face=face.flatten()
    		print face
    		print len(face)
    		face=face.tostring()
		#face=str(face)
		#print face
    		s.send(face)
    		
    		name=s.recv(4096)
    		if name == "corrupt_data":
    			continue
	    	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
	    	font = cv2.FONT_HERSHEY_SIMPLEX
	    	#cv2.putText(frame,naming[labels[0]],(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	    	cv2.putText(frame,name,(x,y), font, 0.5,(255,255,255),2,cv2.LINE_AA)
	    	print ""
	cv2.imshow("original",frame)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break
camera.release()
cv2.destroyAllWindows()
