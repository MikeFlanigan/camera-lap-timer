import cv2
import numpy as np 
import datetime as dt 
import time 
import os 
from lap_timer_fxns import *

screensize = get_user_screen_res()

cwd = os.getcwd()

state = TRACK_CALIBRATION
quit_flag = False

while True:

    if state == TRACK_CALIBRATION:
        print("TRACK_CALIBRATION")
        # TODO change text string prints to be able to take a list of strings and use this post 
        # https://stackoverflow.com/questions/27647424/opencv-puttext-new-line-character
        frame, quit_flag = raw_feed('SPACE BAR when the track is in view. "q" to quit')

        print("Passing to user bounding box")

        SF_line_points, retry_flag, quit_flag = user_bounding_box(frame, 'Click to bound the S/F line. "q" to quit')

        if not retry_flag and not quit_flag:
            state = RACE_REGISTRATION
            print("done")
            print("sf line points: ", SF_line_points)
        
    if state == RACE_REGISTRATION:
        print("RACE_REGISTRATION")

        # get number of races and number of cars racing
        tot_cars, tot_laps = num_cars_num_laps()
        cars = []
        colors = ["Blue", "Red", "Green"]
        for i in range(tot_cars):
            cars.append(colors[i])
        state = RACING

    if state == RACING:
        print("RACING") 
        print() # up to 4 segments
        # start the change detector

        frame, quit_flag, re_reg_flag = race_time('', [SF_line_points], cars, tot_laps)

        if re_reg_flag:
            state = RACE_REGISTRATION

    if quit_flag:
        break


clean_up()
