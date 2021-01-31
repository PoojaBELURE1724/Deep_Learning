import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
#Pooja BELURE added this library
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
#Pooja BELURE

import cv2
path = 'pooja.png'
ncol=200
nrow=200
ncolreal=5
square_size=40
step_size=5
targetx= 16
targety=16
nA=4
w = 0
h = 0
#nS= int( ncol/square_size*(ncol/square_size))
nS=int((ncol-square_size)/step_size)*int((nrow-square_size)/step_size)
import cv2
precision = [40,40]
trackingFace=0
first_frame=True
largeur_capture=ncol
hauteur_capture=nrow

final_x = 100
final_y= 100

class PendulumEnv(gym.Env):

    metadata = {

        'render.modes': ['human', 'rgb_array'],

        'video.frames_per_second': 30

    }
    position_x=10
    position_y=10
    ncol=ncol
    nrow=nrow
    step_size=step_size
    
    def __init__(self, g=10.0):

        self.max_speed = 8

        self.state = 0

        self.dt = .05

        # self.x = 0

        # self.y = 0

        self.l = 1.

        self.viewer = None
        
        self.reward_range = (0, 1)

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
         

        self.observation_space = spaces.Box(

            low=-high,

            high=high,

            dtype=np.float32

        )



        self.seed()

    # def pos_to_state(self,row,col):
        #print("pos to state row",row,"col",col,row*(self.ncol-square_size)/step_size + col)
        # return int(row*(self.ncol-square_size)/step_size + col)
    def in_range(self,x,y,z):
        if x>y-z and x<y+z:  
            return(True)
        return(False)
    def pos_to_state(self,row,col):
        new_col=int(col/step_size)
        new_row=int(row/step_size)
       # print("new_col",new_col,"new_row",new_row)
       # print("pos to state row",row,"col",col,row*(self.ncol-square_size)/step_size + col)
        return int(new_row*((self.ncol-square_size)/step_size) + new_col)
    
    def state_to_pos(self):
        #print("stat to pos",self.state,"x", int(self.state // ((ncol-square_size)/step_size)), "y", int(self.state % ((ncol-square_size)/step_size)))
        return int(step_size*(self.state // ((ncol-square_size)/step_size))), int(step_size*(self.state % ((ncol-square_size)/step_size)))
        

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def target_reached(self,x,y):
        error=50
        if self.in_range(x ,final_x ,error) and self.in_range(y ,final_y ,error):
            return 1
        return 0
    
    
    def step(self, u):
        reward=0
        done=0
        x,y=self.state_to_pos()
#------------- For no external device -------------------------------------#
        #if ext==False :
        if u==0 :   #right
            #print("right......................")
            x = min(x+step_size,(ncol-square_size)-1)
        if u==1 :    #left
            x= max(x-step_size,0)
        if u==2 : # up
            y= max(y-step_size,0)
        if u==3 :   #down
            y = min(y+step_size,(nrow-square_size)-1)
         
        if self.target_reached(x,y)==1:
            done=1
            reward=1
            print("target reached")

        self.state = self.pos_to_state(x,y)
        #print("step",x,y)
        return self.state, reward, done, {}



    def reset(self):

         

        #self.state = np.random.randint((ncol-square_size)*(nrow-square_size))
        self.state = np.random.randint(int((ncol-square_size)/step_size)*int((nrow-square_size)/step_size))

        self.last_u = None
       # print("self.state",self.state)
        
        return int(self.state)



     

    def render(self, mode='human'):
        image = cv2.imread(path) 
        # Window name in which image is displayed 
        window_name = 'Image'
          
        # Start coordinate, here (5, 5) 
        # represents the top left corner of rectangle 
        x,y=self.state_to_pos()
        print("x,y",x,y)
        start_point = (x, y) 
        #print("step",x,y)
        # Ending coordinate, here (220, 220) 
        # represents the bottom right corner of rectangle 
        end_point = (x+32, y+32) 
        #print("end point rectangle x,y",x+32,y+32)
          
        # Blue color in BGR 
        color = (255, 0, 0) 
          
        # Line thickness of 2 px 
        thickness = 2
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        image = cv2.rectangle(image, start_point, end_point, color, thickness) 
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        value=5
        #image = cv2.rectangle(image, (int(value),int(value)), end_point, color, thickness) 
        imgcpy=image.copy()
        img = cv2.resize(imgcpy,None,fx=0.5,fy=0.5) 
        cv2.imshow(window_name, image)
        # while True:
            # if cv2.waitKey(0) & 0xFF == ord('q'):
                # break
        # cv2.destroyAllWindows()
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

        return 0



    def close(self):

        if self.viewer:

            self.viewer.close()

            self.viewer = None
        cv2.waitKey(1)
        cv2.destroyAllWindows()


def angle_normalize(x):

    return (((x+np.pi) % (2*np.pi)) - np.pi)
