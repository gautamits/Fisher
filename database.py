import os
import sys
import numpy as np
import cv2
#from ldgp import calcgrad
#from ldgp import hist
path=sys.argv[1]
naming=[]
labels=[]
locations=[]
data=[]
k=0
model = cv2.face.createFisherFaceRecognizer()
for folders in os.listdir(path):
	folder=path+'/'+folders
	naming.append(folders)
	for images in os.listdir(folder):
		image=folder+'/'+images
		print image
		locations.append(image)
		labels.append(k)
		data.append(cv2.resize(cv2.imread(image,0),(240,240)))
	k+=1
labels=np.array(labels)
print "training"
model.train(data,labels)
print "model is trained"
model.save("database.yml")
np.save("naming",np.array(naming))
np.save("labels",np.array(labels))
np.save("locations",np.array(locations))
np.save("data",np.array(data))
