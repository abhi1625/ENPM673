"""
ENPM673 Spring 2019: Perception for Autonomous Robots
Project 2: Lane Detection


Author(s):
Abhinav Modi (abhi1625@umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park

Kamakshi Jain (kamakshi@terpmail.umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park

Rohan Singh (rohan42@terpmail.umd.edu)
Graduate Student pursuing Masters in Robotics,
University of Maryland, College Park
"""

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import cv2
import numpy as np
import argparse


global templ
global tempr
templ = 0
tempr = 0

K = np.array([[  1.15422732e+03,0.00000000e+00,6.71627794e+02],
              [  0.00000000e+00,1.14818221e+03,3.86046312e+02],
              [  0.00000000e+00,0.00000000e+00,1.00000000e+00]])
dist = np.array([ -2.42565104e-01,-4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
    2.20573263e-02])
def window_centers(image,out,window_width,window_height,margin,dst):#templ,tempr):
    left_windows=[]
    right_windows =[]
    w = image.shape[1]
    h = image.shape[0]
    mid = np.int((dst[0][0]+dst[1][0])/2)#np.int(w/2)#

    windows = int(h/window_height)
    upper = int(windows/2) +10
    leftHist = np.sum(image[:,:mid],axis=0)
    rightHist = np.sum(image[:,mid:],axis=0)
    cv2.imshow('d',image[:int(h-upper),:mid])
    print(rightHist)

    window = np.ones(window_width)
    left = np.argmax(np.convolve(window,leftHist))-int(window_width/2)
    right = np.argmax(np.convolve(window,rightHist))-int(window_width/2) +mid
    left_windows.append(left)
    right_windows.append(right)

    print(right)


    #Search for nonzero pixels in each window
    global templ
    global tempr

    for win in range(10,upper):
        level = np.sum(image[(h - (win+1)*window_height):(h - (win)*window_height),:],axis=0)
        conv = np.convolve(window,level)

        x_leftMin = int(max((left-margin+window_width/2),0))
        x_leftMax = int(min((left+margin+window_width/2),mid))
        if (np.argmax(conv[x_leftMin:x_leftMax]) > 0):
            l_center = np.argmax(conv[x_leftMin:x_leftMax])+int(x_leftMin-window_width/2)
            templ = np.argmax(conv[x_leftMin:x_leftMax])
        else:
            l_center = templ + int(x_leftMin-window_width/2)
        print(np.argmax(conv[x_leftMin:x_leftMax]))

        x_rightMin = int(max((right-margin+window_width/2),mid))
        x_rightMax = int(min((right+margin+window_width/2),w))
        if (np.argmax(conv[x_rightMin:x_rightMax]) > 0):
            r_center = np.argmax(conv[x_rightMin:x_rightMax])+int(x_rightMin-window_width/2)
            tempr = np.argmax(conv[x_rightMin:x_rightMax])
        else:
            r_center = tempr + int(x_rightMin-window_width/2)
            # continue


        y_min = h - (win+1)*window_height
        y_max = h - (win)*window_height


        cv2.rectangle(out,(int(l_center),y_min),(int(l_center+window_width),y_max),(0,255,255),2)
        cv2.rectangle(out,(int(r_center),y_min),(int(r_center+window_width),y_max),(255,0,255),2)

        left_windows.append(l_center)
        right_windows.append(r_center)
    # left_inds=[]
    left_inds = left_windows#np.average(left_windows[-15:],axis=0)
    # right_inds=[]
    right_inds = right_windows#np.average(right_windows[-15:],axis=0)

    return out,left_inds,right_inds#,templ,tempr


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(image, lookUpTable)
    return res
def preprocessing(frame):
    th = np.median(frame[int(frame.shape[0]/2):,:,:])
#     frame = adjust_gamma(frame,gamma=1.5)
#     undist = cv2.undistort(frame,K,dist)
#     blur = cv2.bilateralFilter(undist,9,75,75)
#     median = cv2.medianBlur(blur,5)


#     gray = cv2.cvtColor(median, cv2.COLOR_BGR2HLS)
#     s_channel = gray[:,:,2]
#     l_channel = gray[:,:,1]

    #Create mask for S Channel
    if th<60:
        s_channel = adjust_gamma(frame,gamma=25.0)
        undist = cv2.undistort(frame,K,dist)
        blur = cv2.bilateralFilter(undist,9,75,75)
        median = cv2.medianBlur(blur,5)


        gray = cv2.cvtColor(median, cv2.COLOR_RGB2LAB)
        s_channel = gray[:,:,2]
        s_channel = cv2.Sobel(s_channel,cv2.CV_64F,1,0)
        s_channel = np.absolute(s_channel)
        s_channel = np.uint8(255*s_channel/np.max(s_channel))
        l_channel = gray[:,:,0]
        s_min = 40
        s_max = 250

        l_min = 170
        l_max = 255

        mask1 = np.zeros([s_channel.shape[0],s_channel.shape[1]])
        mask1[(s_channel>=s_min)&(s_channel<=s_max)]=1

        mask2 = np.zeros([l_channel.shape[0],l_channel.shape[1]])
        mask2[(l_channel>=l_min)&(l_channel<=l_max)]=1

    elif th>130:
        s_channel = adjust_gamma(frame,gamma=1)
        undist = cv2.undistort(frame,K,dist)
        blur = cv2.bilateralFilter(undist,9,75,75)
        median = cv2.medianBlur(blur,5)


        gray = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
        s_channel = gray[:,:,2]
        s_channel = gray[:,:,2]
        s_channel = cv2.Sobel(s_channel,cv2.CV_64F,1,0)
        s_channel = np.absolute(s_channel)
        s_channel = np.uint8(255*s_channel/np.max(s_channel))
        l_channel = gray[:,:,0]
        s_min = 20
        s_max = 250

        l_min = 220
        l_max = 255

        mask1 = np.zeros([s_channel.shape[0],s_channel.shape[1]])
        mask1[(s_channel>=s_min)&(s_channel[1]<=s_max)]=1


        mask2 = np.zeros([l_channel.shape[0],l_channel.shape[1]])
        mask2[(l_channel>=l_min)&(l_channel<=l_max)]=1
    else:
        s_channel = adjust_gamma(frame,gamma=1.5)
        undist = cv2.undistort(frame,K,dist)
        blur = cv2.bilateralFilter(undist,9,75,75)
        median = cv2.medianBlur(blur,5)


        gray = cv2.cvtColor(median, cv2.COLOR_BGR2LAB)
        s_channel = gray[:,:,2]
        s_channel = cv2.Sobel(s_channel,cv2.CV_64F,1,0)
        s_channel = np.absolute(s_channel)
        s_channel = np.uint8(255*s_channel/np.max(s_channel))
        # s_channel = cv2.Canny(s_channel,10,300)
        l_channel = gray[:,:,0]
        s_min = 25
        s_max = 250

        l_min = 190
        l_max = 255

        mask1 = np.zeros([s_channel.shape[0],s_channel.shape[1]])
        mask1[(s_channel>=s_min)&(s_channel[1]<=s_max)]=1

        mask2 = np.zeros([l_channel.shape[0],l_channel.shape[1]])
        mask2[(l_channel>=l_min)&(l_channel<=l_max)]=1

    combined = np.zeros([frame.shape[0],frame.shape[1]])
    combined[(mask1==1)|(mask2==1)]=1
    mask = np.uint8(255*combined/np.max(combined))
#     mask = cv2.inRange(median,np.array([0,190,80]),np.array([255,255,150]))
    seg = cv2.bitwise_and(frame,median,mask=mask)
    return seg,combined,th

def main():
     #Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--video', default='project', help='Define name of the video you want to use')

    Args = Parser.parse_args()
    video = Args.video
    path = './data/'+str(video)+'_video'+'.mp4'

    cap = cv2.VideoCapture(path)

    K = np.array([[  1.15422732e+03,0.00000000e+00,6.71627794e+02],
                  [  0.00000000e+00,1.14818221e+03,3.86046312e+02],
                  [  0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    dist = np.array([ -2.42565104e-01,-4.77893070e-02,  -1.31388084e-03,  -8.79107779e-05,
        2.20573263e-02])
    while(True):
        ret,frame = cap.read()

        seg,combined,th = preprocessing(frame)
        #Extract the region of interest and warp to get bird's eye view
        h = frame.shape[0]
        w = frame.shape[1]
    #     src = np.float32([[w-(0.5-0.08/2),h*0.62],[w*(0.5+0.08/2),h*0.62],[w*(0.5+0.76/2),h*0.935],[w*(0.5-0.76/2),h*0.935]])
    #     dst = np.array([[0,0],[400,0],[400,600],[0,400]])
        src = np.array([[375,480],[905,480],[1811,685],[-531,685]])#np.array([[375,480],[905,480],[1811,685],[-531,685]])#np.array([[600,500],[770,500],[1050,680],[350,680]])#
    #     dst = np.float32([[w*0.25,0],[0.75*w,0],[0.75*w,h],[0.25*w,h]])
    #     src = np.array([[580,450],[160,h],[1150,h],[740,450]])
        dst = np.array([[100,0],[w,0],[w,h-100],[100,h-100]])#np.array([[300,300],[500,300],[500,600],[300,600]])#
        H,flag = cv2.findHomography(src,dst)
        Hinv,flag = cv2.findHomography(dst,src)
        out= cv2.warpPerspective(seg,H,(w,h-100))
        out2 = cv2.Canny(seg,0,300)
        out3 = cv2.warpPerspective(out2,H,(w,h-100))
        out_gray = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
        ret,thresh1 = cv2.threshold(out_gray,100,255,cv2.THRESH_BINARY)



    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(out,str(th),(10,500), font, 4,(255,0,0),2,cv2.LINE_AA)
    # #     cv2.putText(l_channel,str(th),(10,500), font, 4,(255,0,0),2,cv2.LINE_AA)
    #     cv2.putText(combined,str(th),(10,500), font, 4,(255,0,0),2,cv2.LINE_AA)
        ###########################################


    #     window_width = 20
    #     window_height = 50
    #     curve_centers = tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin = 25, My_ym = 10/720, My_xm = 4/384, Mysmooth_factor=15)
    #     window_centroids = curve_centers.find_window_centroids(thresh1)
    #     # Points used to draw all the left and right windows
    #     l_points = np.zeros([thresh1.shape[0],thresh1.shape[1]])
    #     r_points = np.zeros([thresh1.shape[0],thresh1.shape[1]])

    #     # points used to find the right & left lanes
    #     rightx = []
    #     leftx = []

    #     # Go through each level and draw the windows
    #     for level in range(0,len(window_centroids)/2):
    #         # Window_mask is a function to draw window areas
    #         # Add center value found in frame to the list of lane points per left, right
    #         leftx.append(window_centroids[level][0])
    #         rightx.append(window_centroids[level][1])

    #         l_mask = window_mask(window_width,window_height,thresh1,window_centroids[level][0],level)
    #         r_mask = window_mask(window_width,window_height,thresh1,window_centroids[level][1],level)
    #         # Add graphic points from window mask here to total pixels found
    #         l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
    #         r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    #     # Draw the results
    #     template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    #     zero_channel = np.zeros_like(template) # create a zero color channel
    #     template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    #     warpage = np.array(cv2.merge((thresh1,thresh1,thresh1)),np.uint8) # making the original road pixels 3 color channels
    #     result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the original road image with window results
        result,leftx,rightx = window_centers(thresh1,out,window_width=50,window_height=10,margin=25,dst=dst)

        # print(rightx)
        window_height = 20
        window_width=50
        res_yvals = np.int32(thresh1.shape[0]-np.arange(len(leftx))*window_height)

        # left_lane = np.hstack

        # p = np.polyfit()
        yvals = np.arange(0,thresh1.shape[0])
        # print(yvals)

        degree = 2
        print('l',leftx)
        print('r',rightx)
        left_fit = np.polyfit(res_yvals, leftx, degree)
        left_fitx = np.zeros(yvals.shape)
        for i in range(degree+1):
            left_fitx = left_fitx + left_fit[i]*(np.power(yvals,(degree-i)))
        left_fitx = np.array(left_fitx,np.int32)

        right_fit = np.polyfit(res_yvals, rightx, degree)
        right_fitx = np.zeros(yvals.shape)
        for i in range(degree+1):
            right_fitx = right_fitx + right_fit[i]*(np.power(yvals,(degree-i)))
        right_fitx = np.array(right_fitx,np.int32)

        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

        print(np.add(right_lane,left_lane)*0.5)
        road = np.zeros_like(frame)
        road_bkg = np.zeros_like(frame)
        cv2.fillPoly(road,[left_lane],color=[255,0,0])
        cv2.fillPoly(road,[right_lane],color=[0,0,255])
        cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
        cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,yvals])))])
        pts = np.hstack((left_line_window1, left_line_window2))
        pts = np.array(pts, dtype=np.int32)

        road_col = np.zeros_like(road_bkg).astype(np.uint8)
        cv2.fillPoly(road, pts, (0,255, 0))
    	# cv2.imshow('green', road_col)

        img_size = (frame.shape[1],frame.shape[0])

        road_warped = cv2.warpPerspective(road,Hinv,img_size,flags=cv2.INTER_LINEAR)
        road_warped_bkg= cv2.warpPerspective(road_bkg,Hinv,img_size,flags=cv2.INTER_LINEAR)

        base = cv2.addWeighted(frame,1.0,road_warped, -1.0, 0.0)
        result2 = cv2.addWeighted(base,1.0,road_warped, 1.0, 0.0)

        ym_per_pix = 30/720
        xm_per_pix = 3.7/700

        curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
        curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) /np.absolute(2*curve_fit_cr[0])#curverad = ((1 + (2*left_fit[0]*yvals[-1] + left_fit[1])**2)**1.5) /np.absolute(2*left_fit[0])

        # Calculate the offset of the car on the road
        threshold = 3000
        if curverad < threshold:
            turn = 'Right'
        else:
            turn = 'Left'
        print(turn)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result2,('Expected turn = '+str(turn)),(50,50), font, 1,(0,0,255),2,cv2.LINE_AA)
        # Visualize the results of iden
        # tified lane lines and overlapping them on to the original undistorted image
        # plt.figure(figsize = (30,20))
        # grid = gridspec.GridSpec(8,2)
        # # set the spacing between axes.
        # grid.update(wspace=0.05, hspace=0.05)
        # gidx=0
        # #img_plt = plt.subplot(grid[0])
        # plt.subplot(grid[gidx])
        # cv2.imshow('road',out3)
        # plt.title('Identified lane lines')

        #img_plt = plt.subplot(grid[1])
        # plt.subplot(grid[gidx+1])
        cv2.imshow('result',result)
        # plt.title('Lane lines overlapped on original image')

        # plt.show()

        cv2.imshow('s',result2)
        # cv2.imshow('l',thresh1)
    #     cv2.imshow('test',l_channel)
    #     cv2.imshow('mask',mask)
        # while(True):
        #     if cv2.waitKey(1)& 0xff==ord('q'):
        #         break
        if cv2.waitKey(1)& 0xff==ord('q'):
            cv2.destroyAllWindows()
            break
if __name__ == '__main__':
    main()
