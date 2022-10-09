import ctypes 
import cv2
import numpy as np 
import datetime as dt 
import time 
import os 
import imutils
from threading import Thread
import threading
import matplotlib.pyplot as plt
from playsound import playsound

cwd = os.getcwd()

TRACK_CALIBRATION = 0
RACE_REGISTRATION = 1
RACING = 2

font_scale = 1.5
font = cv2.FONT_HERSHEY_SIMPLEX

BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)


cap = cv2.VideoCapture(1)

""" working resolutions for this webcam 
{'160.0x120.0': 'OK', '176.0x144.0': 'OK', '320.0x176.0': 'OK', '320.0x240.0': 'OK', '432.0x240.0': 'OK', 
'352.0x288.0': 'OK', '544.0x288.0': 'OK', '640.0x360.0': 'OK', '800.0x448.0': 'OK', '640.0x480.0': 'OK', 
'864.0x480.0': 'OK', '800.0x600.0': 'OK', '960.0x544.0': 'OK', '1024.0x576.0': 'OK', '960.0x720.0': 'OK', 
'1184.0x656.0': 'OK', '1280.0x720.0': 'OK', '1280.0x960.0': 'OK'}
TODO write a bit that only checks a couple relevant resolutions for a webcam... or don't
"""

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)


def get_user_screen_res():
    """Gets and returns the users screen resolution as a tuple"""
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screensize

screensize = get_user_screen_res() # for use in later fxns 
screen_w = screensize[0]
screen_h = screensize[1]


def play_clip(name):
    try:
        cwd = os.getcwd()
        clip_dir = cwd + "\\audio_clips\\"
        fp = clip_dir + name
        # fp = fp.replace("\\","/")
        playsound(fp)
    except Exception as e:
        pass 

def simple_intersect_check(array1, array2):
    """Accepts arrays of x's and y's, checks if there is any interesction"""

    print(array1)
    print(' ')
    print(array2)
    print(' ')
    intersection = False
    x1s = array1[:,0]
    x2s = array2[:,0]
    y1s = array1[:,1]
    y2s = array2[:,1]

    if x1s.min() > x2s.max() or x1s.max() < x2s.min():
        print('nope')
        pass
    else:
        if y1s.min() > y2s.max() or y1s.max() < y2s.min():
            print('y nope')
            pass
        else:
            intersection = True

    return intersection


posList = []
def onMouse(event, x, y, flags, param):
   global posList
   if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        # print(x,y)

detects_history = []
def time_pos_filter(detect):
    global detects_history 

    accepted = False 
    ts = dt.datetime.now()
    if len(detects_history) == 0:
        detects_history.append([detect, ts])
        accepted = True
    else:
        ts_last = detects_history[-1][1]
        pos_last = detects_history[-1][0]

        dist = ((detect[0] - pos_last[0])**2 + (detect[1] - pos_last[1])**2)**0.5
        # print("distance: ",dist)

        if (ts-ts_last).total_seconds() > 0.75:
            detects_history.pop(-1)
            detects_history.append([detect, ts])
            accepted = True
        else:
            if dist > 225:
                detects_history.pop(-1)
                detects_history.append([detect, ts])
                accepted = True

    return [accepted, detects_history]

def nothing(value):
    pass

def num_cars_num_laps():
    wnd = 'race register'
    cv2.namedWindow(wnd,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wnd,int(screen_w/2),int(screen_h/2))

    tot_laps = 5 
    tot_cars = 2
    lap_s = 'Laps #:'
    car_s = 'Cars #'

    cv2.createTrackbar(lap_s, wnd,1,20,nothing)
    cv2.setTrackbarMin(lap_s, wnd, 1)
    cv2.createTrackbar(car_s, wnd,1,2,nothing)
    cv2.setTrackbarMin(car_s, wnd, 1)

    filler = np.zeros((int(screen_w/2),int(screen_h/2),3))
    filler[:,:,:] = 220 # should be gray

    while True:
        #read trackbar positions for each trackbar
        tot_laps=cv2.getTrackbarPos(lap_s, wnd)
        tot_cars=cv2.getTrackbarPos(car_s, wnd)
        print("laps: ",tot_laps," cars: ",tot_cars)

        cv2.putText(filler, 'SPACE to continue. Q to quit.', (10, int(screen_h/4)), font, 0.5, BLACK, 2)

        cv2.imshow(wnd, filler)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            # quit_flag = True
            break
        elif k == ord(' '):
            break
        
    cv2.destroyAllWindows()
    return [tot_cars, tot_laps]


def user_bounding_box(frame, text_string, polygons = []):
    """Prompts user to click in four points define a bounding box
    returns the four points as an array
    
    frame is the input image
    text string is the user prompt(s)
    polygons is an array for drawing existing reference polygons on the image
    """

    global posList

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',int(screen_w/2),int(screen_h/2))

    cv2.setMouseCallback('frame',onMouse)

    last_pos_list_len = len(posList)

    quit_flag = False
    retry_flag = False
    finished = 0

    show_frame = frame.copy()

    cv2.putText(show_frame, text_string, (10, 200), font, font_scale, BLACK, 2)

    # print(polygons)
    for points in polygons:
        # print('ppoints: ',points)
        show_frame = cv2.polylines(show_frame, [np.array(points)], True, BLUE, 2)

    while True:
        

        if len(posList) > last_pos_list_len:
            last_pos_list_len = len(posList)
            cv2.circle(show_frame,posList[-1],10,RED,-1)
        
        if len(posList) > 1:
            cv2.line(show_frame, posList[-1], posList[-2], GREEN, 2) 

        if len(posList) == 4:
            cv2.line(show_frame, posList[-1], posList[0], GREEN, 2)
            finished += 1


        cv2.imshow('frame',show_frame)

        if finished > 5:
            break
        
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            quit_flag = True
            break
        if k == ord('x'):
            retry_flag = True
        if k == ord(' '):
            posList = None
            break
        
    cv2.destroyAllWindows()

    ret_posList = posList
    posList = [] # clear this
    return [ret_posList, retry_flag, quit_flag]



    position = (30, 30)
    text = "Some text including newline \n characters."
    color = (255, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y0 = position
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * line_height
        cv2.putText(frame,
                    line,
                    (x, y),
                    font,
                    font_scale,
                    color,
                    thickness,
                    line_type)


def raw_feed(text_string):

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',int(screen_w/2),int(screen_h/2))

    quit_flag = False

    while True:
        ret, frame = cap.read() 
        frame_show = frame.copy()

        # cv2.putText(frame_show, text_string, (10, 200), font, font_scale, BLACK, 2)
        text_size, _ = cv2.getTextSize(text_string, font, font_scale, 2)
        line_height = text_size[1] + 5
        x, y0 = (10,100)
        for i, line in enumerate(text_string.split("\n")):
            y = y0 + i * line_height
            cv2.putText(frame_show, line, (x, y), font, font_scale, BLACK, 2)

        cv2.imshow('frame',frame_show)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            quit_flag = True
            break
        elif k == ord(' '):
            break
        
    
    cv2.destroyAllWindows()

    return([frame, quit_flag])


def process_and_save_race_results(car_race_stats):

    red_times = []
    blue_times = []
    x_ticks = []

    red_tot = 0
    blue_tot = 0
    cars_racing = len(car_race_stats.keys())
    for i in range(len(car_race_stats["Blue"])-1):
        bs = (car_race_stats["Blue"][i+1]-car_race_stats["Blue"][i]).total_seconds()
        blue_times.append(round(bs,1))

        if cars_racing > 1:
            rs = (car_race_stats["Red"][i+1]-car_race_stats["Red"][i]).total_seconds()
            red_times.append(round(rs,1))
            red_tot += rs
       
        x_ticks.append("Lap "+str(i+1))
        blue_tot += bs

    if cars_racing > 1:
        red_times.append(round(red_tot,1))
    blue_times.append(round(blue_tot,1))
    x_ticks.append("Total time")
    
    width = 0.2
    
    x = np.arange(len(blue_times))
     
    # plot data in grouped manner of bar type
    fig, ax = plt.subplots()
    rects1 = ax.bar(x+0.2, blue_times, width, color='b',label='blue')
    if cars_racing > 1:
        rects2 = ax.bar(x-0.2, red_times, width, color='r',label='red')
    
    # plt.bar_label()
    ax.set_xticks(x, x_ticks)
    ax.set_xlabel("Laps")
    ax.set_ylabel("Time [seconds]")
    if cars_racing > 1:
        ax.legend(["Red car", "Blue car"])
    ax.bar_label(rects1, padding=3)
    if cars_racing >1:
        ax.bar_label(rects2,padding=3)
    im_full_name = cwd + "\\images\\race_results-"+str(dt.datetime.now().hour)+"-"+str(dt.datetime.now().minute)+".png"
    fig.savefig(im_full_name)
    time.sleep(0.5)
    return im_full_name


def match_hsv_finisher(cars, masked_finisher):

    masked_finisher_hsv = cv2.cvtColor(masked_finisher, cv2.COLOR_BGR2HSV)
    masked_finisher_hsv = cv2.GaussianBlur(masked_finisher_hsv,(15,15),0)
    # create another mask 
    # a blue one and a red one 
    red_low = np.array((0,153,165))
    red_high = np.array((144,255,255))

    blue_low = np.array((102,108,63))
    blue_high = np.array((144,153,255))

    # TODO a green one

    #create a mask for that range
    red_mask = cv2.inRange(masked_finisher_hsv,red_low, red_high)
    # cv2.imshow('red',red_mask)

    blue_mask = cv2.inRange(masked_finisher_hsv,blue_low, blue_high)
    # cv2.imshow('blue',blue_mask)

    blue_match_score = blue_mask.sum()
    red_match_score = red_mask.sum()

    # print("BLUE MASK: ",blue_match_score)
    # print("RED MASK: ",red_match_score)

    min_score_match = 1500

    # compare to the list of cars to return which car it was... if any 
    if blue_match_score > red_match_score:
        if blue_match_score > min_score_match:
            best_ind = 0
        else:
            best_ind = None
    else:
        if red_match_score > min_score_match:
            best_ind = 1
        else:
            best_ind = None

    return best_ind


def get_race_places(car_race_stats, num_laps):

    print_string = "" 
    cars_racing = len(car_race_stats.keys())

    
    blue_laps = len(car_race_stats["Blue"])
    if cars_racing > 1:
        red_laps = len(car_race_stats["Red"])
        
    if cars_racing > 1:
        if red_laps > 1:
            red_lap_time = round((car_race_stats["Red"][-1] - car_race_stats["Red"][-2]).total_seconds(),1)
            red_string = "Red car lap " + str(red_laps -1) + " last lap time: " + str(red_lap_time)
            if red_laps == num_laps+1:
                red_string = "Red car FINISHED"
        else: 
            red_lap_time = 999
            red_string = "Red car lap " + str(red_laps) + " last lap time: NA" 

    
    if blue_laps > 1:
        blue_lap_time = round((car_race_stats["Blue"][-1] - car_race_stats["Blue"][-2]).total_seconds(),1)
        blue_string = "Blue car lap " + str(blue_laps-1) + " last lap time: " + str(blue_lap_time)
        if blue_laps == num_laps+1:
            blue_string = "Blue car FINISHED"
    else:
        blue_lap_time = 999
        blue_string = "Blue car lap " + str(blue_laps) + " last lap time: NA" 

    if cars_racing > 1:
        if red_laps > blue_laps:
            # red in the lead
            print_string = red_string + " \n " + blue_string

        elif red_laps < blue_laps:
            # blue in the lead
            print_string = blue_string + " \n " + red_string
        elif red_laps == blue_laps:
            if blue_lap_time < red_lap_time:
                # blue in the lead
                print_string = blue_string + " \n " + red_string
            else:
                # red in the lead
                print_string = red_string + " \n " + blue_string
    else:
        print_string = blue_string

    return print_string


def race_time(text_string, polygons, cars, num_laps):

    score_bd_w = int(screen_w/3)
    score_bd_h = int(screen_h*3/4)

    cv2.namedWindow('Score_Board',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Score_Board',score_bd_w,score_bd_h)
    cv2.moveWindow('Score_Board',0,0)

    cv2.namedWindow('camera',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('camera',int(screen_w/2),int(screen_h/2))
    cv2.moveWindow('camera',score_bd_w+5,0)

    # cv2.namedWindow('red',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('blue',cv2.WINDOW_NORMAL)


    quit_flag = False

    last_frame = None

    filler = np.zeros((score_bd_h,score_bd_w,3))
    filler[:,:,:] = 220 # should be gray
    cv2.imshow('Score_Board',filler)

    showing_text = "Waiting to start. \n 1st Place NA \n 2nd Place NA"

    car_race_stats ={}
    for c in cars:
        car_race_stats[c] = [] # empty list of lap timestamps

    finishers = []
    lap_cross = 0
    race_over = False
    re_reg_flag = False
    while True:
        ret, frame = cap.read() 
        frame_show = frame.copy()

        for points in polygons:
            frame_show = cv2.polylines(frame_show, [np.array(points)], True, GREEN, 2)
        sf_line = np.array(polygons[0]) # inefficient to do this here but w/e for now

        # change detect bit
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0) # blur it
        if not isinstance(last_frame, type(None)):

            ###########################
            # draw filled contour on black background
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [sf_line], 255)
            masked = cv2.bitwise_and(gray, gray, mask=mask)
            last_masked = cv2.bitwise_and(last_frame, last_frame, mask=mask)

            frameDelta = cv2.absdiff(masked, last_masked)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2) # ditch small noise
            thresh = cv2.dilate(thresh, None, iterations=12)

            contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # RETR_EXTERNAL
            contours = imutils.grab_contours(contours)

            for c in contours:
                M = cv2.moments(c)
                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                accepted, detects = time_pos_filter([cX,cY])
                if accepted:
                    cross_pos = (detects[-1][0][0], detects[-1][0][1])
                    cv2.circle(frame_show, cross_pos, 5, WHITE, -1)
                    # print(' ')
                    # print('detects ts: ', detects[-1][1])
                    
                    # get a masked area
                    # save a masked image of each car for color comparison later
                    new_mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

                    win_size = 75
                    AOI = [
                        [cross_pos[0] + win_size, cross_pos[1] + win_size],
                        [cross_pos[0] + win_size, cross_pos[1] - win_size],
                        [cross_pos[0] - win_size, cross_pos[1] - win_size],
                        [cross_pos[0] - win_size, cross_pos[1] + win_size]
                    ]
                    AOI = np.clip(np.array(AOI), 0, 100000).astype(int) #TODO set max to be window size
                    cv2.fillPoly(new_mask, [AOI], 255)
                    masked_finisher = cv2.bitwise_and(frame, frame, mask=new_mask)

                    best_ind = match_hsv_finisher(cars, masked_finisher)

                    if not isinstance(best_ind, type(None)):
                        # print(car_race_stats)
                        # print(best_ind)
                        curr_lapper_line_stamps = len(car_race_stats[cars[best_ind]])
                        if curr_lapper_line_stamps < num_laps +1:
                            if curr_lapper_line_stamps == 0:
                                print("Car ",cars[best_ind], " started")
                                car_race_stats[cars[best_ind]].append(dt.datetime.now())

                                # play lap audio sound
                                audio_thread = Thread(target=play_clip, args=('car_lap.mp3',))
                                audio_thread.setDaemon(True)
                                audio_thread.start()

                            elif curr_lapper_line_stamps > 0:
                                print("Car ",cars[best_ind]," finished lap ",str(curr_lapper_line_stamps))
                                car_race_stats[cars[best_ind]].append(dt.datetime.now())
                                new_lap_time = round((car_race_stats[cars[best_ind]][-1] - car_race_stats[cars[best_ind]][-2]).total_seconds(),1)

                                min_lap_time = 4 # seconds
                                if new_lap_time < min_lap_time:
                                    # too fast to be a lap
                                    car_race_stats[cars[best_ind]].pop(-1)
                                else:
                                    print("Car ",cars[best_ind]," finished lap ",str(curr_lapper_line_stamps))

                                    # play lap audio sound
                                    audio_thread = Thread(target=play_clip, args=('car_lap.mp3',))
                                    audio_thread.setDaemon(True)
                                    audio_thread.start()

                                    # save a picture of the line crosser
                                    frame_show = cv2.polylines(frame_show, [AOI], True, RED, 2)
                                    cv2.imwrite(cwd + "\\images\\"+'finisher_'+str(lap_cross)+'.png',frame_show)
                                    lap_cross += 1

                            # check if finished
                            if len(car_race_stats[cars[best_ind]]) == num_laps +1:
                                print("FINISHED!")

                                # check if the winner
                                if len(finishers) == 0:
                                    finishers.append(cars[best_ind])
                                    # Play winner audio
                                    audio_thread = Thread(target=play_clip, args=('finish-cheer.mp3',))
                                    audio_thread.setDaemon(True)
                                    audio_thread.start()
                                else:
                                    finishers.append(cars[best_ind])
                                    # Play boo audio
                                    audio_thread = Thread(target=play_clip, args=('boo.mp3',))
                                    audio_thread.setDaemon(True)
                                    audio_thread.start()

                                if len(finishers) == len(cars):
                                    print("All cars finished")
                                    race_over = True
                                    result_im_file_name = process_and_save_race_results(car_race_stats)
                                    results_im = cv2.imread(result_im_file_name)

                        else:
                            print("This car already finished...")
                    else:
                        print("Couldn't match finisher to a known car")


            last_frame = gray
        else:
            last_frame = gray


        showing_text = get_race_places(car_race_stats, num_laps)


        # printing bit
        filler[:,:,:] = 20 # should be gray
        text_size, _ = cv2.getTextSize(showing_text, font, font_scale, 2)
        line_height = text_size[1] + 5
        x, y0 = (0,100)
        for i, line in enumerate(showing_text.split("\n")):
            y = y0 + i * line_height
            if "red".lower() in line.lower():
                fc = RED
            elif "blue".lower() in line.lower():
                fc = BLUE
            else:
                fc = BLACK
            sc_font_scale = 0.5

            cv2.putText(filler, line, (x, y), font, sc_font_scale, fc, 2)
        
        cv2.imshow('Score_Board',filler)

        if race_over:
            cv2.imshow('camera',results_im)
        else:
            cv2.imshow('camera',frame_show)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            quit_flag = True
            break
        elif k == ord(' '):
            if race_over:
                # resetting to race again
                car_race_stats ={}
                for c in cars:
                    car_race_stats[c] = [] # empty list of lap timestamps

                finishers = []
                lap_cross = 0
                race_over = False
            break
        elif k == ord('r'):
            re_reg_flag = True
            break
        
    
    cv2.destroyAllWindows()

    return([frame, quit_flag, re_reg_flag])



def clean_up():
    cap.release()
    cv2.destroyAllWindows()
