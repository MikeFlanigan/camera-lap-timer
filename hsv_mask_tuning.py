import cv2 
import numpy as np

cap = cv2.VideoCapture(1)

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('frame',int(screensize[0]/2),int(screensize[1]/2))
cv2.namedWindow('mask',cv2.WINDOW_NORMAL)

def nothing():
    pass

#assign strings for ease of coding
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'
wnd = 'mask'
#Begin Creating trackbars for each
cv2.createTrackbar(hl, wnd,0,179,nothing)
cv2.createTrackbar(hh, wnd,0,179,nothing)
cv2.createTrackbar(sl, wnd,0,255,nothing)
cv2.createTrackbar(sh, wnd,0,255,nothing)
cv2.createTrackbar(vl, wnd,0,255,nothing)
cv2.createTrackbar(vh, wnd,0,255,nothing)

quit_flag = False

while True:
    ret, frame = cap.read() 
    frame_show = frame.copy()

    hsv_frame = cv2.cvtColor(frame_show, cv2.COLOR_BGR2HSV)
    hsv_frame=cv2.GaussianBlur(hsv_frame,(15,15),0)

    red_low = np.array((0,153,165))
    red_high = np.array((144,255,255))

    blue_low = np.array((102,108,63))
    blue_high = np.array((144,153,255))

    # low = (0,0,0)
    # high = (255,255,255)

    # color_mask = cv2.inRange(hsv_frame, low, high)
    # cv2.imshow('mask',color_mask)

    #read trackbar positions for each trackbar
    hul=cv2.getTrackbarPos(hl, wnd)
    huh=cv2.getTrackbarPos(hh, wnd)
    sal=cv2.getTrackbarPos(sl, wnd)
    sah=cv2.getTrackbarPos(sh, wnd)
    val=cv2.getTrackbarPos(vl, wnd)
    vah=cv2.getTrackbarPos(vh, wnd)
 
    #make array for final values
    HSVLOW=np.array([hul,sal,val])
    HSVHIGH=np.array([huh,sah,vah])
 
    #create a mask for that range
    color_mask = cv2.inRange(hsv_frame,HSVLOW, HSVHIGH)
    cv2.imshow('mask',color_mask)

    cv2.imshow('frame',frame_show)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        quit_flag = True
        break
    elif k == ord(' '):
        break
    

cv2.destroyAllWindows()