README for the vision based lap timer.

# background
This is intended to be an rc car lap timer thanks to max and jack being into rc car racing and there being a bit of an office bet
as to who could make a lap time the fastest. Honestly this wasn't fast. But it seems to work.

# How to use it (once it's been setup)
- generally at any time press 'q' to quit the program.

1. View webcam stream. If the start/finish line is in view then press space bar to continue.
2. Left click in four corners to mark out the start / finish line.
    - press space to continue
    - Note that this app should work for cars racing in either direction. It doesn't know about directionality. 
3. Enter race info
    - Drag the sliders to the desired number of laps and number of race cars.
    - * Required, if only one car is racing it must be marked blue. If two cars are racing then the second must be marked red.
    - Press the space bar to continue
4. Race!
    - Cross the line and the computer will register (hopefully) your car and print an update on the score board 
    and make an audio noise (volume up!)
    - When the winner finishes their last lap you'll hear a distinct thing.
    - When everyone finishs lap times will be posted and shown.
    - Note, finish pictures and laptime results are saved in the sub folder called images
5. Restarting
    - When looking at the results:
        - press space bar to race again with the same settings
        - press 'r' to return to the race info entering stage (step 3)
        - And as always press 'q' to quit

# How to set it up
- Non programmers
    - Windows only
    - Copy the lap_timer_executable folder to your desktop and try to run the exe (lap_timer_v1.exe)
        - note this is super not recommended from a cyber security standpoint, so only do it if you trust me, or if you have a burner or air gapped laptop
    - hopefully it works!
    - if it does i recommend making a short cut for the exe on your desktop or somewhere so you don't have to dig in that folder
- Programmers
    - You could run the python program lap_timer_v1.py
        - Note the requirements.txt file for installing with pip could well be incomplete
    - You can create fresh exe's (for a more cyber friendly option) with the command "python setup.py py2exe" within this repo
        - after running that you need to put the audio clips folder and a blank images folder in the same directory as the exe

# How to make it better
Feel free to submit a pull request! A ton of this was done as fast as possible and so is kinda shoddy.
Ideas for improvements:
- make it work with different numbers of cars 
- expose the min  lap time setting to users
- expose an option to change the color thresholds
- change it so the first car doesn't always have to be blue
- general code clean up and refactoring
- auto matching (NN?) so don't need color stickers

# hours log :) 
- 0.25 hours writing 9/30 night
- 0.3 hours writing 10/1 
- 93 minutes coding 10/1
- 55 minutes 10/2
- 6 hours 10/2 to 10/3 ... dang.
- 3 hours, 50 minutes, 10/3 ... mostly working. Needs GUI and user flow work. 
- 