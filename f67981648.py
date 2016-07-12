#!/usr/bin/python
import os
import sys
import cv2
from Tkinter import *
root=Tk()
import numpy
agree = IntVar()
path=sys.argv[1]
                        #contains labels
naming = file("naming","w")
csv = file("csv.ext","w")
model = cv2.createFisherFaceRecognizer()
def cancelled(event):
	print "cancelled"
	sys.exit()
def saidOK(event):
	root.destroy()
	data=[]                          #contains numpy arrays
	labels=[]
	if agree.get()==1:
		index=0
		faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		for folder in os.listdir(path):
			naming.write(str(index)+' '+folder+'\n')
			folders=path+'/'+folder

			for image in os.listdir(folders):
				images=folders+'/'+image
				print images
				frame=cv2.imread(images,0)
				height,width=frame.shape
				if height==240 and width == 240:
					#image is already processed
					csv.write(images+';'+str(index)+'\n')
					data.append(frame)
					labels.append(index)
					continue
				#print images
				#cv2.imshow("cropper",image)
				faces = faceCascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5,minSize=(30, 30),flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
				if len(faces)==0:
					print "no face detacted"
					os.remove(images)
				result = None
				for(x,y,w,h) in faces:
					x1=x+w
				        y1=y+h
					result = frame[y:y+h,x:x+w]
					result=cv2.resize(result,(240,240))
					csv.write(images+';'+str(index)+'\n')
					data.append(result)
					labels.append(index)
					os.chdir(path)
					os.chdir(folder)
					cv2.imwrite(image,result);
					#cv2.imshow("cropping",result)
				'''else:
					print "no face detected in "+images
					try: 
 						os.remove(images)
 						print images+' deleted'
					except: pass
					#print result+" written"'''
			index+=1

		cv2.waitKey(0)
		cv2.destroyAllWindows()
	labels=numpy.array(labels)
	model.train(data,labels)
	print 'model is trained'
	model.save("database.yml")
	print 'model is saved'



label_1=Label(root,text="Pls consider that images which will not contain any faces will be removed.\nHave a backup before proceeding");
label_1.grid(row=0,sticky=W)
c=Checkbutton(root,text="yes I want to proceed",variable=agree)
c.grid(row=1,sticky=W)

ok=Button(root,text="ok",fg="green")
ok.grid(row=2,column=0,sticky=W)
ok.bind("<Button-1>",saidOK)

cancel=Button(root,text="cancel",fg="red")
cancel.grid(row=2,column=1,sticky=E) 
cancel.bind("<Button-1>",cancelled)
root.mainloop()



