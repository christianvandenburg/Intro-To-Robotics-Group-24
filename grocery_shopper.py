"""grocery controller."""

# Apr 1, 2025

from controller import Robot
import math
import numpy as np

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

map = None



# ------------------------------------------------------------------
# Helper Functions
def getPoint(eqs,t):#get one point from equation for possible elbow positions
    return (f(t) for f in eqs)    
def formulateEquations(goal,r1,r2):#gx is goal_x, r1 is first semgent len, r2 is second
    #circle of results
    (xg,yg,zg)=goal
    d2=xg*xg+yg*yg+zg*zg# distance of goal from origin, squared
    d=math.sqrt(d2)
    rDiff=(r1-r2)*(r1+r2)# difference of squares of segment lengths
    bigRoot=math.sqrt(4*d2*r1*r1-(d2+rDiff)**2)
    smallRoot=math.sqrt(yg*yg+zg*zg)
    term1=rDiff/(2*d2)+0.5# for X,Y,Z
    term2=bigRoot/(2*d*smallRoot)# for Y,Z
    #print("pre-vals:",d2,d,rDiff,bigRoot,smallRoot,term1,term2,xg,yg,zg,r1,r2)
    def X(t): return xg*term1+math.sin(t)*(smallRoot*bigRoot/(2*d2))
    def Y(t): return yg*term1-(zg*math.cos(t)+yg*xg*math.sin(t)/d)*term2
    def Z(t): return zg*term1+(yg*math.cos(t)-zg*xg*math.sin(t)/d)*term2
    #t_=4.56
    #print(f"test t={t_}, output=({X(t_)},{Y(t_)},{Z(t_)})")
    return X,Y,Z
def pickUpCube(goal,randomizeElbow=False):
    global part_names
    #goal coords are relative coordinates to robot arm origin
    r1=.8 #first segment length
    r2=.8 #second segment length
    eqs = formulateEquations(goal,r1,r2)
    import random
    t=random.randint(0,60)/10 if randomizeElbow else 0 #0 will probably be best in most situations
    xe,ye,ze = getPoint(eqs,t)
    #now just turn that into angle reqs
    joint1=math.atan2(ye,xe)
    joint2=math.pi/2-math.asin(ze/r1)#maybe no (maybe yes?) math.pi/2- on actual robot implementation
    if joint1>=0.07 and joint2<=1.02:
        print(f"input t={t}, output=({xe},{ye},{ze})")
        print("angles:",joint1,"and",joint2)
        robot.getDevice("arm_1_joint").setPosition(joint1)
        robot.getDevice("arm_2_joint").setPosition(joint2)
    else:
        print("beyond joint range")

# mode='grabby'
mode='IK_demo'

gripper_status="closed"

# Main Loop
while robot.step(timestep) != -1:
    
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if mode=='grabby':
        #robot_parts["wheel_left_joint"].setVelocity(vL)
        #robot_parts["wheel_right_joint"].setVelocity(vR)
        if(gripper_status=="open"):
            # Close gripper, note that this takes multiple time steps...
            robot_parts["gripper_left_finger_joint"].setPosition(0)
            robot_parts["gripper_right_finger_joint"].setPosition(0)
            if right_gripper_enc.getValue()<=0.005:
                gripper_status="closed"
        else:
            # Open gripper
            robot_parts["gripper_left_finger_joint"].setPosition(0.045)
            robot_parts["gripper_right_finger_joint"].setPosition(0.045)
            if left_gripper_enc.getValue()>=0.044:
                gripper_status="open"
    elif mode=='IK_demo':#demoing reaching the same point in multiple ways
        #note, coordinates centered on first arm joint
        goal=(.1,1,.1)#any point in range (vals > 0)
        pickUpCube(goal,randomizeElbow=True)
        #print("test complete",flush=True)
        robot.step(500)
    else:
        print("no mode set")
        robot.step(5000)
