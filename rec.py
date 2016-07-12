import cv2
import sys
import numpy as np
model = cv2.face.createFisherFaceRecognizer()
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

data=[]
labels=[]

'''def histeq(im,nbr_bins=256):
	imhist,bins=histogram(im.flatten(),nbr_bins,normed=True)
	cdf=imhist.cumsum()
	cdf = 255 * cdf / cdf[-1] #normalize
	#use linear interpolation
	im2 = interp(im.flatten(),bins[:-1],cdf)
	return im2.reshape(im.shape), cdf'''
	
	
cam=int(sys.argv[1])
video_capture=cv2.VideoCapture(cam)
###############function to print name on screen #############

#################        function to train #################
'''l=file('csv.ext',"r").read().split()
for line in l:
	print line
	line=line.split(';')
	m=cv2.imread(line[0],0)
	data.append(m)
	label=int(line[1])
	labels.append(label)
labels=numpy.array(labels)
model.train(data,labels)'''


model.load("database.yml")
naming=np.load("naming.npy")

################ starts here ##############################
while True:
    ret,frame = video_capture.read()
    #print "image size is",frame.shape
    if not ret:
        print 'cannot acquire image'
        video_capture.release()
        sys.exit()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.equalizeHist(gray)
    faces = faceCascade.detectMultiScale(gray,1.8,5)
    result=None
    label=""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        new=gray[y:y+h,x:x+w]
        result=cv2.resize(new,(240,240))
        #new=cv2.equalizeHist(result)
       # new = histeq(im)
        #cv2.imshow("enhanced",new)
        prediction = model.predict(result)
        print prediction
        #(label,prediction)=prediction
        cv2.putText(frame,naming[prediction],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,500)

    # Display the resulting frame
    #gray2 = cv2.equalizeHist(gray)
    #cv2.imshow('enhanced',gray2)
    cv2.imshow("original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



	
	
