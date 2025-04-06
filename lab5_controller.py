from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)


##### vvv [Begin] Do Not Modify vvv #####

robot = Robot()
timestep = int(robot.getBasicTimeStep())

part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

range_finder = robot.getDevice('range-finder')
range_finder.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

keyboard = robot.getKeyboard()
keyboard.enable(timestep)

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

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'
# mode = 'picknplace'



###################
# Planner
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = (-8.46, -4.88) # (Pose_X, Pose_Y) in meters
    end_w = (-6, -10.2) # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start =  (146, 254) # (x, y) in 360x360 map
    end = (306, 180) # (x, y) in 360x360 map
    
    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def path_planner(map, start, end):
        '''
        :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
        :param start: A tuple of indices representing the start cell in the map
        :param end: A tuple of indices representing the end cell in the map
        :return: A list of tuples as a path from the given start to the given end in the given maze
        '''
        rows, cols = map.shape
        open_set = [(heuristic(start, end), start)]  # Use a list for the open set
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        while open_set:
            # Get the node with the lowest f_score
            open_set.sort()  # Sort to simulate a priority queue
            _, current = open_set.pop(0)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse the path to start-to-end

        # Explore neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and map[neighbor[0], neighbor[1]] == 0:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                        open_set.append((f_score[neighbor], neighbor))
    
        return []

    # Part 2.1: Load map (map.npy) from disk and visualize it
    map = np.load('map.npy')
    map = np.rot90(map, k=-1)
    map = np.fliplr(map)
    

    plt.imshow(map, cmap='gray')
    # plt.title("Map Visualization (Flipped)")
    # plt.show()

    # Part 2.2: Compute an approximation of the “configuration space”
    kernel = np.ones((10,10))  # Quadratic kernel of ones
    config_space = convolve2d(map, kernel, mode='same', boundary='fill', fillvalue=0)
    config_space = np.clip(config_space, 0, 1)  # Ensure binary values (0 or 1)

    # Visualize configuration space
    plt.imshow(config_space, cmap='gray')
    plt.title("Configuration Space")


    # plt.show()
    # Part 2.3 continuation: Call path_planner
    start_map = (150, 112)  # Example starting point
    end_map = (288, 320)  # Example ending point

    # Plan the path
    path = path_planner(config_space, start_map, end_map)

    if path:
        # Convert map coordinates to world coordinates
        waypoints = [((-x[1] / 360) * 12, (-x[0] / 360) * 12) for x in path]

        # Save path to disk
        np.save('path.npy', waypoints)

        # Visualize path
        for coord in path:
            config_space[coord[0], coord[1]] = 0.5  # Mark path
        plt.imshow(config_space, cmap='magma')
        plt.title("Path Visualization")
        plt.show()
        # plt.show()  # Comment this out before submitting
        print('here')
    else:
        print("No path found.")


    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    # done above
# waypoints = []

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map = np.zeros(shape=[360,360])
inc = 5e-3
waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = np.load('path.npy') # Replace with code to load your path
    plt.figure()
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'o-', label="Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Waypoints")
    plt.legend()
    plt.show()
    index = 0

    
    # index = 0
    # while( index != len(waypoints) - 1 ):
        # xr = gps.getValues()[0]
        # yr = gps.getValues()[1]
        # theta = np.atan2(compass.getValues()[0], compass.getValues()[1])
        
        # distance = np.sqrt(((xr-waypoints[index][0])*(xr-waypoints[index][0]))+((yr-waypoints[index][1])*(yr-waypoints[index][1])))
        # bearing = np.atan2((waypoints[index][1]-yr),(waypoints[index][0]-xr))-theta
        # if index+2 <= len(waypoints):
            # heading = np.atan2((waypoints[index+1][1]-yr),(waypoints[index+1][0]-xr))-theta
        
        # if distance > .05:
            # if bearing > .2:
                # print('turning left')
                # rightMotor.setVelocity(MAX_SPEED)
                # leftMotor.setVelocity(-MAX_SPEED)
            # elif bearing < -.2:
                # print('turning right')
                # rightMotor.setVelocity(-MAX_SPEED)
                # leftMotor.setVelocity(MAX_SPEED)
            # elif distance > 0:
                # print('forward')
                # rightMotor.setVelocity(MAX_SPEED)
                # leftMotor.setVelocity(MAX_SPEED)
        # else:
            # print('here')
            # if heading > .2:
                # print('adjusting left')
                # rightMotor.setVelocity(MAX_SPEED)
                # leftMotor.setVelocity(-MAX_SPEED)
            # elif heading < -.2:
                # print('adjusting right')
                # rightMotor.setVelocity(-MAX_SPEED)
                # leftMotor.setVelocity(MAX_SPEED)
            # if heading < .2 and heading > -.2:
                # if index+2 > len(waypoints):
                    # print('done')
                    # index = 0
                # else:    
                    # print('next')
                    # index+=1

# state = 0 # use this to iterate through your path

if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    pass

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]

    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            coordX=360-abs(int(wx*30))
            coordY=abs(int(wy*30))
            if coordX<360 and coordY<360:
                map[coordX][coordY]=min(1,map[coordX][coordY]+inc)
                g=map[coordX][coordY]
                #color=int((g*256**2+g*256+g)*255)
                shade=int(g*255)
                color=(shade<<16)|(shade<<8)|shade# I think this is what you meant to do
                display.setColor(color)
                display.drawPixel(coordX,coordY)

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))
    threshold_map=map>0.5


    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            # new_map = map>.5
            # new_map = np.multiply(new_map, 1)
            new_map = (map > 0.5).astype(int)

            np.save('map.npy', new_map) 
            
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            map = np.load("map.npy")
            plt.imshow(map, cmap='gray')
            plt.show()
            
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    elif mode == 'autonomous': # not manual mode
        length = len(waypoints) - 1
        print('started', flush = True)
        if( index != length):
            xr = gps.getValues()[0]
            yr = gps.getValues()[1]
            theta = np.atan2(compass.getValues()[0], compass.getValues()[1])
            
            distance = np.sqrt(((xr-waypoints[index][0])*(xr-waypoints[index][0]))+((yr-waypoints[index][1])*(yr-waypoints[index][1])))
            bearing = np.atan2((waypoints[index][1]-yr),(waypoints[index][0]-xr))-theta
            if index+1 <= length:
                heading = np.atan2((waypoints[index+1][1]-yr),(waypoints[index+1][0]-xr))-theta
            
            if distance > .05:
                if bearing > .2:
                    print('turning left')
                    vR = MAX_SPEED
                    vL = -MAX_SPEED
                elif bearing < -.2:
                    print('turning right')
                    vR = -MAX_SPEED
                    vL = MAX_SPEED
                elif distance > 0:
                    print('forward')
                    vR = MAX_SPEED
                    vL = MAX_SPEED
            else:
                print('here')
                if heading > .2:
                    print('adjusting left')
                    vR = MAX_SPEED
                    vL = -MAX_SPEED
                elif heading < -.2:
                    print('adjusting right')
                    vR = -MAX_SPEED
                    vL = MAX_SPEED
                if heading < .2 and heading > -.2:
                    if index+1 > length:
                        print('done')
                        # index = 0
                    else:    
                        print('next')
                        index+=1
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        rho = 0
        alpha = 0

        #STEP 2: Controller
        dX = 0
        dTheta = 0

        #STEP 3: Compute wheelspeeds
        vL = 0
        vR = 0

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
