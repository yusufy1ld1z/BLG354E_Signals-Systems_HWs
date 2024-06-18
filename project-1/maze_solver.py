'''
    BLG 354E - Signal&Systems for Comp.Eng.
    Project-1 
    CRN: 21350
    
    Yusuf YILDIZ
    150210006
    Part-3
'''

import soundcard as sc
import pyautogui as pg
import time
import matplotlib.pyplot as plt
import numpy as np

# ---------- PART 3 ----------
"""
I tested the simulation on more than one PCs and OSs, and it managed to finish everytime. 
If it can not finish in your PC, you may adjust the parameters in the code such as lookup_time, change_tile_time, etc.
"""

for mic in sc.all_microphones(include_loopback=True):
    print(mic)

input_audio_device = 4 # Adjust this value based on the microphone you want to use
mic = sc.all_microphones(include_loopback=True)[input_audio_device]
print("[INFO] Using microphone:", mic)

time.sleep(2) # Wait for 2 seconds for initialization

sample_rate = 16000 # Sample rate for audio recording
num_frames = 16  # Number of frames to record for each audio sample

lookup_time = 0.59 # If needed you may adjust these parameters, if it is too high, it may result in an explosing while checking
change_tile_time = 1.87
move_up = "w"
move_left = "a"
move_down = "s"
move_right = "d"

decisions = []

def main():
    '''
    Main function to run the simulation.

    It first checks the 'down' direction to initialize the environment.
    It is tested and confirmed that when the monster moves down once at the beginning, 
    remaining results are more reliable.
    Then, it calls the simulate function to start the simulation.

    '''
    check('down')
    simulate() # Call the simulate function to start the simulation
    
def simulate():
    '''
    Simulates the decision-making process.

    The function repeatedly calls the make_decision function to determine the next action to take.
    It prints the current action and stops when the make_decision function returns -1.

    '''
    print("[SIMULATOR] Simulation started...")
    time.sleep(2) # Wait for 2 seconds for initialization
    
    current = 1
    directions = ['RIGHT', 'UP', 'DOWN', 'LEFT', "FINISH"]
    try:
        while True:
            current = make_decision(current)
            print('Current:', directions[current])
            if current == -1: # If the make_decision function returns -1 indicating the end of the game, break the loop
                break
    except KeyboardInterrupt: # Handle KeyboardInterrupt (Ctrl+C)
        print("[SIMULATOR] Keybord Interrupt. Exiting...")
        exit()
    print('[SIMULATOR] Simulation finished...')    

def make_decision(prev):
    '''
    Makes a decision based on the previous action.

    Parameters:
        prev (int): The previous action taken.
            0: Right
            1: Up
            2: Down
            3: Left

    Returns:
        int: The action to take next.
            0: Right
            1: Up
            2: Down
            3: Left
            -1: Indicates that no action should be taken.
    '''
    print('[DECISION MAKER] Making decision...')

    # Check if the maximum number of decisions has been reached
    if len(decisions) == 15 or len(decisions) == 16:
        return -1

    # Try moving right if the previous action was not left
    if prev != 3:
        if check('right'):
            move('right')
            return 0

    # Try moving up if the previous action was not down and the number of decisions is less than 12
    if prev != 2 and len(decisions) < 12: # These extra conditions are for the monster not to be fall from the platfom
        if check('up'):
            move('up')
            return 1

    # Try moving down if the previous action was not up and the number of decisions is within a certain range
    if prev != 1 and not (len(decisions) > 7 and len(decisions) < 11): # These extra conditions are for the monster not to be fall from the platfom
        if check('down'):
            move('down')
            return 2

    # Try moving left if the previous action was not right
    if prev != 0:
        if check('left'):
            move('left')
            return 3

    # If none of the above conditions are met, return the previous action
    return prev 

def move(direction):
    '''
    Moves the agent in the specified direction.

    Parameters:
        direction (str): The direction to move. Can be one of 'right', 'up', 'down', or 'left'.

    '''
    if direction == 'right':
        print('[MOVER] Moving right...')
        decisions.append('right')       # These sleep durations are necessary for a reliable simulation
        pg.keyDown(move_right)
        time.sleep(change_tile_time)
        pg.keyUp(move_right)
        time.sleep(0.4)
        
    elif direction == 'up':
        print('[MOVER] Moving up...')
        decisions.append('up')
        pg.keyDown(move_up)
        time.sleep(change_tile_time)
        pg.keyUp(move_up)
        time.sleep(0.4)
    
    elif direction == 'down':
        print('[MOVER] Moving down...')
        decisions.append('down')
        pg.keyDown(move_down)
        time.sleep(change_tile_time)
        pg.keyUp(move_down)
        time.sleep(0.4)
    
    elif direction == 'left':
        print('[MOVER] Moving left...')
        decisions.append('left')
        pg.keyDown(move_left)
        time.sleep(change_tile_time)
        pg.keyUp(move_left)
        time.sleep(0.4)
        
def check(direction):
    '''
    Checks for obstacles or valid paths in the specified direction.

    Parameters:
        direction (str): The direction to check. Can be one of 'right', 'up', 'down', or 'left'.

    Returns:
        bool: The result of the check, indicating whether the direction is clear or blocked.

    '''
    if direction == 'right':
        print('[CHECKER] Checking right...')
        pg.keyDown(move_right) # Simulate pressing the key for moving in a given direction
        time.sleep(lookup_time)
        pg.keyUp(move_right)
        time.sleep(0.4)
    
        result = listen() # Listen for the result of the check
        
        pg.keyDown(move_left) # Simulate pressing the key for moving backwards to return to the initial position
        time.sleep(lookup_time)
        pg.keyUp(move_left)
        time.sleep(0.4)
    
    elif direction == 'up':
        print('[CHECKER] Checking up...')
        pg.keyDown(move_up)
        time.sleep(lookup_time)
        pg.keyUp(move_up)
        time.sleep(0.4)
        
        result = listen()
        
        pg.keyDown(move_down)
        time.sleep(lookup_time)
        pg.keyUp(move_down)
        time.sleep(0.4)
        
    elif direction == 'down':
        print('[CHECKER] Checking down...')
        pg.keyDown(move_down)
        time.sleep(lookup_time)
        pg.keyUp(move_down)
        time.sleep(0.4)
        
        result = listen()
        
        pg.keyDown(move_up)
        time.sleep(lookup_time)
        pg.keyUp(move_up)
        time.sleep(0.4)
        
    elif direction == 'left':
        print('[CHECKER] Checking left...')
        pg.keyDown(move_left)
        time.sleep(lookup_time)
        pg.keyUp(move_left)
        time.sleep(0.4)
        
        result = listen()
        
        pg.keyDown(move_right)
        time.sleep(lookup_time)
        pg.keyUp(move_right)
        time.sleep(0.4)
    
    return result # Return the result of the check

def listen():
    '''
    Listens for audio input and determines if there is an obstacle based on the mean of the audio sample.

    Returns:
        bool: True if an squarewave is detected indicating the safe, False otherwise indicating the danger.
    '''
    print('[LISTENER] Listening...')
    
    # IMPORTANT NOTE: If you use mono microphone, you may not need to slice the data.
    # But in our tests we could not get reliable results with mono microphones.
    with mic.recorder(samplerate=sample_rate) as rec: # Record audio using the microphone
        data = rec.record(numframes=num_frames)[:,1] # Since we are using stereo, we only need one channel.
        
    max_ = np.max(data)
    min_ = np.min(data)
    norm = (data - min_) / (max_ - min_) * 2 - 1 # Normalize the data to (-1, 1) interval
    
    mean = np.mean(norm) # Calculate the mean of the normalized data

    print('[LISTENER] Mean of the sample:', mean)
    
    if mean > 0.4: # If the mean is greater than 0.4, return True indicating a safe path
        return False
    elif mean < -0.4: # If the mean is less than -0.4, return False indicating a dangerous path
        return True
    else: return False # Otherwise, return False indicating a dangerous path

if __name__ == '__main__':
    main()