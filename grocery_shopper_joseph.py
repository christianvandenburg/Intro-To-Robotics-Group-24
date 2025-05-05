from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils
MAX_SPEED = 7.0
MAX_SPEED_MS = 0.633
AXLE_LENGTH = 0.4044
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
    robot_parts[i].setVelocity(0)
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
pose_x     = 0
pose_y     = 0
pose_theta = 0
vL = 0
vR = 0
lidar_sensor_readings = []
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83]
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
    x = gps.getValues()[0]
    y = gps.getValues()[1]
    start_w = (x, y)
    end_w = (-1.47, -9.68)
    start =  (146, 254)
    end = (306, 180)
    def heuristic(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    def path_planner(map, start, end):
        rows, cols = map.shape
        open_set = [(heuristic(start, end), start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}
        while open_set:
            open_set.sort()
            _, current = open_set.pop(0)
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
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
    map = np.load('map.npy')
    map = np.rot90(map, k=-1)
    map = np.fliplr(map)
    plt.imshow(map, cmap='gray')
    kernel = np.ones((10,10))
    config_space = convolve2d(map, kernel, mode='same', boundary='fill', fillvalue=0)
    config_space = np.clip(config_space, 0, 1)
    plt.imshow(config_space, cmap='gray')
    plt.title("Configuration Space")
    start_map = (150, 112)
    end_map = (288, 320)
    path = path_planner(config_space, start_map, end_map)
    if path:
        waypoints = [(-12 - (((-x[1] / 360) * 12)), (-x[0] / 360) * 12) for x in path]
        np.save('path.npy', waypoints)
        for coord in path:
            config_space[coord[0], coord[1]] = 0.5
        # plt.imshow(config_space, cmap='magma')
        # plt.title("Path Visualization")
        # plt.show()
        print('here')
    else:
        print("No path found.")
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
    #waypoints=[[(12-xy[0]),xy[1]] for xy in waypoints]
    #plt.figure()
    #plt.plot(waypoints[:, 0], waypoints[:, 1], 'o-', label="Path")
    #plt.xlabel("X")
    #plt.ylabel("Y")
    #plt.title("Waypoints")
    #plt.legend()
    #plt.show()
    index = 0
# state = 0 # use this to iterate through your path

if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    # they forgot to put the loose code in lab5_joint into a __main__ method, so copying the necessary methods here
    target_item_list = ["orange"]


    vrb = True
    # Enable Camera
    camera = robot.getDevice('camera')
    camera.enable(timestep)
    camera.recognitionEnable(timestep)
    
    # We are using a GPS and compass to disentangle mapping and localization
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    compass = robot.getDevice("compass")
    compass.enable(timestep)
    
    ## fix file paths
    ################ v [Begin] Do not modify v ##################
    
    base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
    my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])
    
    print(my_chain.links)
    
    part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
                "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
                "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")
    
    for link_id in range(len(my_chain.links)):
    
        # This is the actual link object
        link = my_chain.links[link_id]
        
        # I've disabled "torso_lift_joint" manually as it can cause
        # the TIAGO to become unstable.
        if link.name not in part_names or  link.name =="torso_lift_joint":
            print("Disabling {}".format(link.name))
            my_chain.active_links_mask[link_id] = False
            
    # Initialize the arm motors and encoders.
    motors = []
    for link in my_chain.links:
        if link.name in part_names and link.name != "torso_lift_joint":
            motor = robot.getDevice(link.name)
    
            # Make sure to account for any motors that
            # require a different maximum velocity!
            if link.name == "torso_lift_joint":
                motor.setVelocity(0.07)
            else:
                motor.setVelocity(1)
                
            position_sensor = motor.getPositionSensor()
            position_sensor.enable(timestep)
            motors.append(motor)
    
    def rotate_y(x,y,z,theta):
        new_x = x*np.cos(theta) + y*np.sin(theta)
        new_z = z
        new_y = y*-np.sin(theta) + x*np.cos(theta)
        return [-new_x, new_y, new_z]
    
    def lookForTarget(recognized_objects):
        if len(recognized_objects) > 0:
    
            for item in recognized_objects:
                if "orange" in str(item.get_model()):
                #if str(item.get_model()) in recognized_objects:
    
                    target = recognized_objects[0].get_position()
                    dist = abs(target[2])
    
                    if dist < 5:
                        return True
    
    def checkArmAtPosition(ikResults, cutoff=0.00005):
        '''Checks if arm at position, given ikResults'''
        
        # Get the initial position of the motors
        initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]
    
        # Calculate the arm
        arm_error = 0
        for item in range(14):
            arm_error += (initial_position[item] - ikResults[item])**2
        arm_error = math.sqrt(arm_error)
    
        if arm_error < cutoff:
            if vrb:
                print("Arm at position.")
            return True
        return False
    
    def moveArmToTarget(ikResults):
        '''Moves arm given ikResults'''
        # Set the robot motors
        for res in range(len(ikResults)):
            if my_chain.links[res].name in part_names:
                # This code was used to wait for the trunk, but now unnecessary.
                # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
                robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
                if vrb:
                    print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))
    
    def calculateIk(offset_target,  orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
        '''
        Parameters
        ----------
        offset_target : list
            A vector specifying the target position of the end effector
        orient : bool, optional
            Whether or not to orient, default True
        orientation_mode : str, optional
            Either "X", "Y", or "Z", default "Y"
        target_orientation : list, optional
            The target orientation vector, default [0,0,1]
        
        Returns
        -------
        list
            The calculated joint angles from inverse kinematics
        '''
        
        # Get the number of links in the chain
        num_links = len(my_chain.links)
        
        # Create initial position array with the correct size
        initial_position = [0] * num_links
        
        # Map each motor to its corresponding link position
        motor_idx = 0
        for i in range(num_links):
            link_name = my_chain.links[i].name
            if link_name in part_names and link_name != "torso_lift_joint":
                if motor_idx < len(motors):
                    initial_position[i] = motors[motor_idx].getPositionSensor().getValue()
                    motor_idx += 1
        
        # Calculate IK
        ikResults = my_chain.inverse_kinematics(
            offset_target, 
            initial_position=initial_position,
            target_orientation=target_orientation, 
            orientation_mode=orientation_mode
        )
        
        # Validate result
        position = my_chain.forward_kinematics(ikResults)
        squared_distance = math.sqrt(
            (position[0, 3] - offset_target[0])**2 + 
            (position[1, 3] - offset_target[1])**2 + 
            (position[2, 3] - offset_target[2])**2
        )
        print(f"IK calculated with error - {squared_distance}")
        
        return ikResults

    # Legacy code for visualizing
        # import matplotlib.pyplot
        # from mpl_toolkits.mplot3d import Axes3D
        # ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')

        # my_chain.plot(ikResults, ax, target=ikTarget)
        # matplotlib.pyplot.show()
            
    def getTargetFromObject(recognized_objects):
        ''' Gets a target vector from a list of recognized objects '''
    
        # Get the first valid target
        target = recognized_objects[0].get_position()
    
        # Convert camera coordinates to IK/Robot coordinates
        # offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
        offset_target = [-(target[2])+0.22, -target[0]+0.06, (target[1])+0.97+0.2]
    
        return offset_target
    
    def reachArm(target, previous_target, ikResults, cutoff=0.00005):
        '''
        This code is used to reach the arm over an object and pick it up.
        '''
    
        # Calculate the error using the ikTarget
        error = 0
        ikTargetCopy = previous_target
    
        # Make sure ikTarget is defined
        if previous_target is None:
            error = 100
        else:
            for item in range(3):
                error += (target[item] - previous_target[item])**2
            error = math.sqrt(error)
    
        
        # If error greater than margin
        if error > 0.05:
            print("Recalculating IK, error too high {}...".format(error))
            ikResults = calculateIk(target)
            ikTargetCopy = target
            moveArmToTarget(ikResults)
    
        # Exit Condition
        if checkArmAtPosition(ikResults, cutoff=cutoff):
            if vrb:
                print("NOW SWIPING")
            return [True, ikTargetCopy, ikResults]
        else:
            if vrb:
                print("ARM NOT AT POSITION")
    
        # Return ikResults
        return [False, ikTargetCopy, ikResults]
    
    def closeGrip():
        robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
        robot.getDevice("gripper_left_finger_joint").setPosition(0.0)
    
        # r_error = abs(robot.getDevice("gripper_right_finger_joint").getPositionSensor().getValue() - 0.01)
        # l_error = abs(robot.getDevice("gripper_left_finger_joint").getPositionSensor().getValue() - 0.01)
        
        # print("ERRORS")
        # print(r_error)
        # print(l_error)
    
        # if r_error+l_error > 0.0001:
        #     return False
        # else:
        #     return True
    
    def openGrip():
        robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
        robot.getDevice("gripper_left_finger_joint").setPosition(0.045)
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    pass


debug=0
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
    #print("X:",pose_x,"Y:",pose_y,"ROT",pose_theta)#

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
        #print(pose_x,pose_x-0.25*math.cos(pose_theta),pose_x-0.25*math.cos(pose_theta+math.pi/2),pose_x-0.25*math.cos(pose_theta+math.pi),pose_x-0.25*math.cos(pose_theta+math.pi*3/2))
        #print(pose_y,pose_y-0.25*math.cos(pose_theta),pose_y-0.25*math.cos(pose_theta+math.pi/2),pose_y-0.25*math.cos(pose_theta+math.pi),pose_y-0.25*math.cos(pose_theta+math.pi*3/2))
        
        #centering GPS (why is it not centered already?????)
        pose_x-=0.25*math.cos(pose_theta+math.pi/2)
        pose_y-=0.25*math.cos(pose_theta)
        pose_theta+=math.pi/2
        
        if index<len(waypoints):
            #waypoints[index][0]=--5.06283#DEBUG TEST
            #waypoints[index][1]=-4.83086#DEBUG TEST
        
            dist=((pose_x-waypoints[index][0])**2+(pose_y-waypoints[index][1])**2)**.5
            bear=math.atan2(waypoints[index][1]-pose_y,waypoints[index][0]-pose_x)-pose_theta
            print("goal:",waypoints[index][0],waypoints[index][1])#
            print("  xy:",pose_x,pose_y,pose_theta)
            while bear>math.pi:
                bear-=2*math.pi
            while bear<-math.pi:
                bear+=2*math.pi
            print("\tbear:",bear,"dist:",dist)#
            #bear will be heading when dist~=0
            turn_scalar=3#d/r
            forward_scalar=4
            rot_threshold=0.015
            dist_threshold=0.5
            if dist<=dist_threshold:
                index+=5
            elif abs(bear)>rot_threshold:
                vL=-bear*turn_scalar
                vR=bear*turn_scalar
                print("left" if vL<0 else "right")#
            elif dist>dist_threshold:
                vL=dist*forward_scalar
                vR=dist*forward_scalar
                print("forward")#
            vL=min(vL,MAX_SPEED/4)
            vL=max(vL,-MAX_SPEED/4)
            vR=min(vR,MAX_SPEED/4)
            vR=max(vR,-MAX_SPEED/4)
            #print(f"waypoint[{index}]:",waypoints[index],"vL:",vL,"vR:",vR)#
        else:
            vL=0
            vR=0
            print("done")
            STATE=None

        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        #rho = 0
        #alpha = 0

        #STEP 2: Controller
        #dX = 0
        #dTheta = 0

        #STEP 3: Compute wheelspeeds
        #vL = 0
        #vR = 0

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)
    elif mode=='picknplace':
        if debug==0:
            (vL,vR)=(1,1)
            #robot.step(2000)
            debug=1
        #(vL,vR)=(1,1)
        print("debug",debug)#
        #if lookForTarget(["orange"]):
        #    (vL,vR)=(0,0)
        openGrip()
        orange=[-8.40456,-6.08132,1.11976]
        try:
            data=calculateIk(orange)
            print(data)
            (vL,vR)=(0,0)
        except ValueError as err:
            print("their method had a whoopsie again :(")
            print("\t",err)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    #pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    #pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    #pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    print("vL",vL,"vR",vR)
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
