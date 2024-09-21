import cv2

def draw_boundary(img,classifier,saclefactor,minNeighbors,color,text):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img ,saclefactor ,minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        #id,_ = clf.predict(gray_img[y:y+h,x:x+w])
        cv2.putText(img,"Face",(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,1,cv2.LINE_AA)
        coords = [x,y,w,h]
    return coords

def detect(img,faceCascade,eyesCascade):
    color = {"blue":(255,0,0),"Red":(0,0,255),"green":(0,255,0)}

    coords = draw_boundary(img,faceCascade, 1.1 ,10,color['green'],"Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        coords = draw_boundary(roi_img,eyesCascade, 1.1 ,15,color['Red'],"eyes")

    return img
video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(r"C:\Users\avigu\OneDrive\Desktop\ml project\Face detection\Models\haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier(r"C:\Users\avigu\OneDrive\Desktop\ml project\Face detection\Models\haarcascade_eye.xml")
while True:
    _,img = video_capture.read()
    img = detect(img, faceCascade,eyesCascade)
    cv2.imshow("Face detation",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()