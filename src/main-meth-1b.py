


# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 as cv

#from google.colab.patches import cv2_imshow


# importing libraries 
import os 
import cv2  
from PIL import Image  
  
# Checking the current directory path 
# print(os.getcwd())  
  
# Folder which contains all the images 
# from which video is to be generated 



def main():


    # indor videos : 'patadas'
    #outdor videos : 'street' barras
    path="../input/"

    name_orignial_video='street'

    video_name="../output/"+"Clean_"+name_orignial_video+".avi"
    nvideo_name="../output/"+"del_"+name_orignial_video+".avi"
    novideo_name="../output/"+"nobady_"+name_orignial_video+".avi"

    cap = cv2.VideoCapture(path+name_orignial_video+'.mp4')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kernel = np.ones((5,5),np.uint8)
    fgbg = cv2.createBackgroundSubtractorMOG2() 


    img_array = []
    imgarray = []
    new_frame_array=[]
    nobady_frames=[]

      
    #while(1):
    for ii in range(3,n_frames):
        print('processing frame', ii, 'of', n_frames - 1)
        ret, frame = cap.read() 

        fgmask = fgbg.apply(frame)
        fgmask=np.where(fgmask ==0, fgmask, 255)
        fgmask2 = cv.dilate(fgmask,kernel,iterations = 1) 

        kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
        thresh = cv2.threshold(fgmask2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel7, iterations=1)

        # Find outer contour and fill with white
        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(close, cnts, [255,255,255])
        cv2.fillPoly(close, cnts, [255,255,255])
        close = cv.dilate(close,kernel,iterations = 1)
        cv2.fillPoly(close, cnts, [255,255,255])

        ######################################
        close1=close.copy()
        cnts1 = cv2.findContours(close1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
        cv2.fillPoly(close1, cnts, [255,255,255])
        cv2.fillPoly(close1, cnts, [255,255,255])
        cv2.fillPoly(close1, cnts, [255,255,255])

        img=close.copy()

        scale_percent = 20 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)


        dim10=(img.shape[1], img.shape[0])
        # resize image
        rclose = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        opening = cv.morphologyEx(rclose, cv.MORPH_OPEN, kernel)
        opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

        close2 = cv2.resize( opening, dim10, interpolation = cv2.INTER_AREA)
        cv2.fillPoly(close2, cnts, [255,255,255])


        opening2=np.where(opening ==255, opening, 0)

        close3 = cv2.resize( opening2, dim10, interpolation = cv2.INTER_AREA)

        tframe= np.zeros(frame.shape, dtype=np.uint8)
        tframe[:,:,0] = frame[:,:,0].copy()
        tframe[:,:,1] = frame[:,:,1].copy()
        tframe[:,:,2] = frame[:,:,2].copy()

        tframe[:,:,0]=np.where(close3 ==0, tframe[:,:,0], 0)
        tframe[:,:,1]=np.where(close3 ==0, tframe[:,:,1], 0)
        tframe[:,:,2]=np.where(close3 ==0, tframe[:,:,2], 0)

        height, width, layers = frame.shape     

        # Appending the images to the video one by one 
        cv2.imwrite("../temp/"+str(ii)+'img.jpg',tframe)

        #video.write(tframe)        
        img_array.append(img)  
        imgarray.append(tframe)  

        if ii<7:
            background_old=tframe
            background_prima=background_old

        else:
            background_new=tframe
            background_old=background_old+np.where(background_old ==0, background_new,0) 
            background_prima=background_new+np.where(background_new ==0, background_old,0)
            new_frame=tframe+np.where(tframe ==0, background_old,0)

            cv2.imwrite("../temp/"+str(ii)+'background_old.jpg',background_old)
            cv2.imwrite("../temp/"+str(ii)+'new_frame.jpg',new_frame)
            new_frame_array.append(new_frame)

    no_one_video=cv2.VideoWriter(novideo_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

    """
    #video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
    video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
    new_video = cv2.VideoWriter(nvideo_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))



     
    for i in range(len(img_array)):
        video.write(img_array[i])
    video.release()  # releasing the video genera
    
    for j in range(len(new_frame_array)):
        new_video.write(new_frame_array[j])
    new_video.release()  # releasing the video genera
    #video.write(tframe)         


    new_video.release()  # releasing the video genera
    """ 
    out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

    nobady_frames=[]
    for i in range(len(imgarray)):
        tempframe=imgarray[i]+np.where(imgarray[i] ==0, background_old,0)
        nobady_frames.append(tempframe)
        out.write(imgarray[i])
    out.release()
    
    for r in range(len(nobady_frames)):
        no_one_video.write(nobady_frames[r])
    
    no_one_video.release()  # releasing the video genera

if __name__ == "__main__":
    main()
