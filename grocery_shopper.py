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
def setArm(joint1,joint2,joint3,joint4,joint5,joint6,joint7,time=0):
    robot.getDevice("arm_1_joint").setPosition(joint1)
    robot.getDevice("arm_2_joint").setPosition(joint2)
    robot.getDevice("arm_3_joint").setPosition(joint3)
    robot.getDevice("arm_4_joint").setPosition(joint4)
    robot.getDevice("arm_5_joint").setPosition(joint5)
    robot.getDevice("arm_6_joint").setPosition(joint6)
    robot.getDevice("arm_7_joint").setPosition(joint7)
    robot.step(time)

def getPoint(eqs,t):#get one point from equation for possible elbow positions
    return (f(t) for f in eqs)    
def formulateEquations(goal,r1,r2):#gx is goal_x, r1 is first semgent len, r2 is second
    #circle of results
    (xg,yg,zg)=goal
    d2=xg*xg+yg*yg+zg*zg# distance of goal from origin, squared
    d=math.sqrt(d2)
    if d>r1+r2:
        print("out of reach")#
        return 0,0,0#out of reach
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
def moveArmToPoint(true_goal,wrist_goal=None,randomizeElbow=False):
    global part_names
    r1=0.311105247641 #first segment length
    r2=0.370789854306 #second segment length (maybe actually =0.315780840772)
    r3=0.2 #claw (just a guess)
    
    if wrist_goal==None:
        wrist_goal=(true_goal[0],true_goal[1]-r3,true_goal[2])
    #goal = (true_goal[0],true_goal[1],true_goal[2]+r3)#"goal" is goal for wrist
    goal=wrist_goal#don't worry about it
    
    #if goal[1]<0:
    #    print("need y>=0 (probably)")
    #    return 3#scoot back
    dist = ((goal[0]*goal[0])+(goal[1]*goal[1])+(goal[2]*goal[2]))**0.5
    #print("distance:",dist,flush=True)#
    if goal[1]==0 and goal[2]==0:
        goal[2]=0.000001
    #goal coords are relative coordinates to robot arm origin
    #displacement between j1,j2 = 0.130038455851
    #irl robot official arm reach: 87cm "without end effector"
    if dist>r1+r2:
        print("out of reach")
        return 1#out of reach
    
    joint4=math.acos((sum(k*k for k in goal)-r1*r1-r2*r2)/(2*r1*r2))#law of cosines
    inner_acute=sum(k*k for k in goal)<r1*r1+r2*r2
    if inner_acute and joint4<math.pi/2:#joint4 is outer angle
        joint4=math.pi-joint4
    elif not inner_acute and joint4>math.pi/2:
        joint4=math.pi-joint4    
        
    if joint4>2.29:
        print("elbow bend beyond joint range")
        return 2#can't bend that way
    eqs = formulateEquations(goal,r1,r2)
    import random
    tStart=random.randint(0,60)/10 if randomizeElbow else 1 #0 will probably be best in most situations
    t=tStart
    inJointRange=False
    reachedClawAngles=False
    while not inJointRange:
        inJointRange=True
        xe,ye,ze = getPoint(eqs,t)
        #print(f"t={t}, elbow=({(xe,ye,ze)})")
        #now just turn that into angle reqs
        joint1=math.atan2(ye,xe)
        joint2=math.asin(ze/r1)#maybe no (maybe yes?) math.pi/2- on actual robot implementation
        inJointRange=joint1>=0.07 and joint1<=2.68 and joint2>=-1.5 and joint2<=1.02
        if inJointRange:
            def rotateWorld(E,G):
                #part 1: rotate xy
                xy_rot = math.atan2(E[1],E[0])# ==joint1 if E,G are elbow,goal
                G=(G[0]*math.cos(-xy_rot)-G[1]*math.sin(-xy_rot), G[0]*math.sin(-xy_rot)+G[1]*math.cos(-xy_rot), G[2])
                #part 2: rotate xz
                E_temp=(E[0]*math.cos(-xy_rot)-E[1]*math.sin(-xy_rot), E[0]*math.sin(-xy_rot)+E[1]*math.cos(-xy_rot), E[2])
                #print("E_temp:",E_temp)#
                xz_rot=math.atan2(E_temp[2],E_temp[0])# ==joint2 if E,G are elbow,goal
                    
                #E_temp=(E_temp[0]*math.cos(-xz_rot)-E_temp[2]*math.sin(-xz_rot), E_temp[1], E_temp[0]*math.sin(-xz_rot)+E_temp[2]*math.cos(-xz_rot))#
                #print("E_{rot}="+str(E_temp))#for debugging (should be (k,0,0) )
                
                G=(G[0]*math.cos(-xz_rot)-G[2]*math.sin(-xz_rot), G[1], G[0]*math.sin(-xz_rot)+G[2]*math.cos(-xz_rot))
                return G
            goal_rot=rotateWorld((xe,ye,ze),goal)
            joint3=math.atan2(goal_rot[1],goal_rot[2])+math.pi #atan(y/z)
            #if joint3>math.pi:#better version below
            #    joint3-=2*math.pi
            
            #joint3=math.atan2(ye,xe)-0.5*math.pi
            if joint3>1.5:
                joint3-=2*math.pi
                #if joint3>=-3.46:#
                #    print("\tjoint3 saved!")#
            if joint3<-3.46:
                inJointRange=False
                #print("joint3 out of range")
            else:#joint6
                print("goal_rot:",goal_rot," joint3:",joint3,flush=True)#
                reachedClawAngles=True
                #pointing claw forward (positive y)
                claw_goal=(goal[0],goal[1]+r3,goal[2])#true_goal
                claw_elbow_dist2=((claw_goal[0]-xe)**2+(claw_goal[1]-ye)**2+(claw_goal[2]-ze)**2)
                joint6=math.acos((claw_elbow_dist2-r3*r3-r2*r2)/(2*r3*r2))#law of cosines
                inner_acute=claw_elbow_dist2<r3*r3+r2*r2
                if inner_acute and joint6<math.pi/2:#joint4 is outer angle
                    joint6=math.pi-joint6
                elif not inner_acute and joint6>math.pi/2:
                    joint6=math.pi-joint6
                if joint6>1.39:
                    print(f"joint6 out of range ({joint6}>1.39)")#
                    inJointRange=False
                else:#joint5
                    #print("\telbow:",(xe,ye,ze),"
                    #claw_goal_rot=rotateWorld((goal[0]-xe,goal[1]-ye,goal[2]-ze),(claw_goal[0]-xe,claw_goal[1]-ye,claw_goal[2]-ze))
                    #joint5=math.atan2(claw_goal_rot[1],claw_goal_rot[2])+math.pi/2-joint3 #same logic as joint3, minus joint3 
                    
                    #use the adjusted rot values from earlier
                    elbow_rot=(r1,0,0)
                    #goal_rot=goal_rot
                    true_rot=rotateWorld((xe,ye,ze),true_goal)
                    #print(f"debug1 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    def desmosPrint(elbow_rot,goal_rot,true_rot,var_num=0,spacing=False):#fancy debug tool
                        (elbow_rot,goal_rot,true_rot)=("\\left("+str(["{:f}".format(i) for i in point])[1:-1]+"\\right)" for point in (elbow_rot,goal_rot,true_rot))
                        if spacing:
                            print("")
                        print(f"D_{var_num}=",end="")
                        print(("\\left["+elbow_rot+","+goal_rot+","+true_rot+"\\right]").replace("'",""))
                        if spacing:
                            print("")
                    def rotateXY(G,xy_rot):
                        return (G[0]*math.cos(-xy_rot)-G[1]*math.sin(-xy_rot), G[0]*math.sin(-xy_rot)+G[1]*math.cos(-xy_rot), G[2])
                    def rotateXZ(G,xz_rot):#z is y
                        return (G[0]*math.cos(-xz_rot)-G[2]*math.sin(-xz_rot), G[1], G[0]*math.sin(-xz_rot)+G[2]*math.cos(-xz_rot))
                    def rotateYZ(G,xz_rot):#y is x
                        return (G[0], G[1]*math.cos(-xz_rot)-G[2]*math.sin(-xz_rot), G[1]*math.sin(-xz_rot)+G[2]*math.cos(-xz_rot))
                    
                    #elbow_rot=(r1,0,0)    
                    #goal_rot=(0.27209659807614145, 0.3302318858857273, -0.16404372240273835)#placeholder for rotateWorld((xe,ye,ze),goal)
                    #true_rot=(0.4541767276390809, 0.4020098322591067, -0.2052110982331633)#placeholder for rotateWorld((xe,ye,ze),true_goal)
                    #joint3=math.atan2(goal_rot[1],goal_rot[2])+math.pi #evals to 5.173435720066505 (-2pi = -1.1097495871130816)
                    #joint3 = joint3-2*math.pi if joint3>1.5 else joint3 #shouldn't affect math i don't think
                    #print(f"debug1 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    #print(f"\tjoint3={joint3}")
                    #desmosPrint(elbow_rot,goal_rot,true_rot,1)
                    
                    #step 1: "undo" rotation so it doesn't affect j5
                    (elbow_rot,goal_rot,true_rot) = (rotateYZ(point,-joint3) for point in (elbow_rot,goal_rot,true_rot))
                    #print(f"debug2 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    #desmosPrint(elbow_rot,goal_rot,true_rot,2)
                    
                    #step 2: move elbow_rot to origin
                    (elbow_rot,goal_rot,true_rot) = ((point[0]-r1,point[1],point[2]) for point in (elbow_rot,goal_rot,true_rot))
                    #print(f"debug3 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    #desmosPrint(elbow_rot,goal_rot,true_rot,3)
                    
                    #step 3: align goal_rot to positive z axis
                    temp_angle=-math.atan2(goal_rot[0],goal_rot[2])
                    (elbow_rot,goal_rot,true_rot) = (rotateXZ(point,temp_angle) for point in (elbow_rot,goal_rot,true_rot))
                    #print(f"debug4 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    #desmosPrint(elbow_rot,goal_rot,true_rot,4)
                    
                    #step 4: align true_rot to positive xz plane
                    joint5=-math.atan2(true_rot[0],true_rot[1]) #possibly + some number of quarter rotations
                    #print(f"debug5 E={elbow_rot}, G={goal_rot}, T={true_rot}")#
                    #desmosPrint(elbow_rot,goal_rot,true_rot,5)
                    #print(f"joint5={joint5}")
                    
                    
                    while joint5>2.07:
                        joint5-=math.pi
                        joint6*=-1
                        #print("(flipping j6)")#
                    while joint5<-2.07:
                        joint5+=math.pi
                        joint6*=-1
                        #print("(flipping j6)")#
                    if joint5<-2.07 or joint5>2.07:
                        print("j5 out of range")#impossible, but just in case
                        inJointRange=False
                    else:#joint7
                        joint7=joint3-joint5+math.pi/2
                        while joint7>2.07:
                            joint7-=math.pi
                        while joint7<-2.07:
                            joint7+=math.pi
                        if joint7>2.07 or joint7<-2.07:
                            print("j7 out of range")#impossible, but just in case
                            joint7=2.07 if joint7>2.07 else -2.07
                    
                    
        if not inJointRange:
            #print("angles:",joint1,"and",joint2)#
            #print("beyond joint range")
            t-=0.1
            #if (abs(tStart-2*math.pi-t)<0.01):
            if (t<tStart-2*math.pi):
                print("no option in joint range")
                return 2#can't bend that way
    #https://cyberbotics.com/doc/guide/tiago-steel?version=R2023a
    #d2=xg*xg+yg*yg+zg*zg# distance of goal from origin, squared
    #print(f"t={t}, elbow=({(xe,ye,ze)})")#
    print("angles:",joint1,joint2,joint3,joint4)#,joint5,joint6,joint7)#
    #setArm(joint1,joint2,joint3,joint4,joint5,joint6,joint7)
    return (joint1,joint2,joint3,joint4,joint5,joint6,joint7)
        
def identifyCubes(yellow=True,green=False): 
    obj_coords=[]
    for obj in camera.getRecognitionObjects():
        if obj.getNumberOfColors()!=1:
            continue#the things we want are only one color
        color=obj.getColors()
        RGB=(color[0],color[1],color[2])
        if yellow and RGB==(1.0,1.0,0.0):
            obj_coords.append(obj.getPosition())
        if green and RGB==(0.0,1.0,0.0):
            obj_coords.append(obj.getPosition())
            
    if len(obj_coords)>1:
        obj_coords.sort(key=lambda x:x[0]**2+x[1]**2+x[2]**2)#closest first
    return obj_coords

def collectCube(goal):
    r1=0.311105247641 #first segment length
    r2=0.370789854306 #second segment length (maybe actually =0.315780840772)
    r3=0.2 #claw (estimated)
    
    pose_angles=moveArmToPoint(goal)
    if (pose_angles==1):#out of reach
        return False
    elif (pose_angles==2):#arm can't bend that way
        return False
    
    #assuming robot's been lined up already
    SCOOT_SPEED=0.5#max is 7
    SCOOT_TIME=6000
    #0.5,6000 is roughly 0.3 meters
    
    #(x0,y0,z0) = gps.getValues()
    #step 1: scoot back
    robot_parts["wheel_left_joint"].setVelocity(-SCOOT_SPEED)#max is 7
    robot_parts["wheel_right_joint"].setVelocity(-SCOOT_SPEED)
    robot.step(SCOOT_TIME)
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    #(x1,y1,z1) = gps.getValues()
    
    #step 2: pose arm and small adjustment:
    #TODO: turn entire robot CW (right) by exactly (math.pi-joint1)
    j1_j2_displacement=0.130038455851
    #TODO: move entire robot backwards by exactly j1_j2_displacement
    #TODO: turn entire robot CCW (left) by exactly (math.pi-joint1)
    setArm(*pose_angles)
    robot.step(4000)
    
    #step 3: scoot forward:
    robot_parts["wheel_left_joint"].setVelocity(SCOOT_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(SCOOT_SPEED)
    robot.step(SCOOT_TIME)
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    #robot.step(2000)
    #(x0,y0,z0) = gps.getValues()
    #dist=((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**.5
    #print("dist2:",dist,"\n")
    
    #step 4: grab!
    robot_parts["gripper_left_finger_joint"].setPosition(0)
    robot_parts["gripper_right_finger_joint"].setPosition(0)
    robot.step(4000)
    
    #step 5: scoot back again
    robot_parts["wheel_left_joint"].setVelocity(-SCOOT_SPEED)
    robot_parts["wheel_right_joint"].setVelocity(-SCOOT_SPEED)
    robot.step(SCOOT_TIME)
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)
    
    #step 5.5 (optional): verify
    robot.getDevice("arm_1_joint").setPosition(0.65)
    robot.getDevice("arm_2_joint").setPosition(-0.25)
    robot.getDevice("arm_3_joint").setPosition(-2.5)
    robot.getDevice("arm_4_joint").setPosition(2.29)
    robot.getDevice("arm_5_joint").setPosition(0)
    robot.getDevice("arm_6_joint").setPosition(0)
    robot.getDevice("arm_7_joint").setPosition(-0.7)
    robot.step(3000)
    cube_coords=identifyCubes(True,True)
    if cube_coords==[] or cube_coords[0][0]**2+cube_coords[0][1]**2+cube_coords[0][2]**2>0.4**2:
        print("the cube has eluded my grasp")
        return False
        
    #step 6: position above basket:
    robot.getDevice("arm_1_joint").setPosition(0.07)
    robot.getDevice("arm_2_joint").setPosition(0)
    robot.getDevice("arm_3_joint").setPosition(0)
    robot.getDevice("arm_4_joint").setPosition(2.29)
    robot.getDevice("arm_5_joint").setPosition(0)
    robot.getDevice("arm_6_joint").setPosition(0)
    robot.getDevice("arm_7_joint").setPosition(-1.5)
    robot.step(3000)
    
    #step 7: ungrab
    robot_parts["gripper_left_finger_joint"].setPosition(0.045)
    robot_parts["gripper_right_finger_joint"].setPosition(0.045)
    robot.step(2000)
    
    #step 8: return out to rest (out of the way)
    #from initialization: 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41
    robot.getDevice("arm_1_joint").setPosition(0.07)
    robot.getDevice("arm_2_joint").setPosition(1.02)
    robot.getDevice("arm_3_joint").setPosition(-3.16)
    robot.getDevice("arm_4_joint").setPosition(1.27)
    robot.getDevice("arm_5_joint").setPosition(1.32)
    robot.getDevice("arm_6_joint").setPosition(0)
    robot.getDevice("arm_7_joint").setPosition(1.41)
    robot.step(2000)
    
    #step 9 (optional): scoot forwards again to not screw up odometry (too much)
    robot_parts["wheel_left_joint"].setVelocity(SCOOT_SPEED)#max is 7
    robot_parts["wheel_right_joint"].setVelocity(SCOOT_SPEED)
    robot.step(SCOOT_TIME)
    robot_parts["wheel_left_joint"].setVelocity(0)
    robot_parts["wheel_right_joint"].setVelocity(0)

    return True#success

# mode='grabby'
# mode='IK_demo'
# mode='shopping'
# mode='object_dist_demo'
mode='testing'

gripper_status="closed"

# Main Loop
#Xtest,Ytest,Ztest=-3,1,-3
cube_coords_global=[]#not currently used
while robot.step(timestep) != -1:
    
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    if mode=='shopping':
        r1=0.311105247641 #first segment length
        r2=0.370789854306 #second segment length (maybe actually =0.315780840772)
        r3=0.2 #claw (just a guess)
        
        pos=gps.getValues()#TODO: replace with odometry version
        
        #step 1: try to spot yellow cubes
        cube_coords=identifyCubes(yellow=True,green=False)
        if False:#might be nice to have object permanence (incomplete)
            cube_coords_local=cube_coords
            new_cubes=False
            for ccl in cube_coords_local:#i can speed this up if it's too slow
                ccl=(ccl[0]+pos[0],ccl[1]+pos[1],ccl[2]+pos[2])#TODO: account for rotation
                for ccg in cube_coords_global:#object permenance
                    if ((ccl[0]-ccg[0])**2+(ccl[1]-ccg[1])**2+(ccl[2]-ccg[2])**2)**0.5 < 0.01:#same cube
                        break
                else:#if didn't break from inner loop
                    cube_coords_global.append(ccl)
                    new_cubes=True
            if new_cubes:
                cube_coords_global.sort(lambda x:(x[0]-pos[0])**2+(x[1]-pos[1])**2+(x[2]-pos[2])**2)
            cube_coords=cube_coords_global 
        
        #step 2: move to nearest cube (or explore if no cubes)
        if cube_coords==[]:
            #TODO: explore
            continue#skip steps 3+
        goal=cube_coords[0]
        #adj_goal=(goal[1]+0.0137,goal[0]+0.137,goal[2]+0.326797)#what IK methods will want
        adj_goal=(goal[1]+0.0137,goal[0]+0.137,goal[2]+0.4)#alt version (slightly higher)
        test_goal=(adj_goal[0],adj_goal[1]-r3,adj_goal[2])#where robot wrist is aiming (in IK perspective)
        test_goal_dist=sum(g*g for g in test_goal)**0.5#
        #dist = ((test_goal[0]*test_goal[0])+(test_goal[1]*test_goal[1])+(test_goal[2]*test_goal[2]))**0.5
            
        
        print("test_goal is "+("in range " if test_goal_dist<=r1+r2 else "out of range ")+f"({test_goal_dist})",flush=True)
        if test_goal_dist>r1+r2:#maybe >=, or -0.01>
            pass#TODO: move towards cube_coords[0]
            robot_parts["wheel_left_joint"].setVelocity(1)#placeholder
            robot_parts["wheel_right_joint"].setVelocity(1)
            robot.step(500)
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)
            #TODO: change torso_lift_joint to 0 if lower (second) shelf, or 0.35 if top shelf
            #if adj_goal[2]>robot_parts["torso_lift_joint"].getPosition()[2]:
            continue#remove after replacing placeholder
        else:
            robot_parts["wheel_left_joint"].setVelocity(0)
            robot_parts["wheel_right_joint"].setVelocity(0)
            robot.step(5000)
        
        #step 5: the engrabbening
        #adj_goal=(goal[1],goal[0],goal[2])#relative to start of arm, assuming robot is facing positive Y (swap x/y)
        #camera: 1.79165, 1.30525, 1.05289
        #j1 (origin-ish): 1.77795, 1.16825, 0.726093
        #translation: 0.0137, 0.137, 0.326797
        
        #(already made it)#adj_goal=(goal[1]+0.0137,goal[0]+0.137,goal[2]+0.326797)
        print("and thus begins the engrabbening",flush=True)#
        success=collectCube(adj_goal)
        if success:
            print("WE DID IT",flush=True)#
            cube_coords.pop(0)
        else:
            pass#wiggle around and try again?
        
    elif mode=='testing':
        robot.step(1000)#move to a spot on the shelf, then test cube pick up process
        print("going to the store",flush=True)
        #go to cube (set up test)
        robot_parts["wheel_left_joint"].setVelocity(-7)
        robot_parts["wheel_right_joint"].setVelocity(7)
        robot.step(500)
        robot_parts["wheel_left_joint"].setVelocity(7)
        robot.step(2000)
        robot_parts["wheel_right_joint"].setVelocity(-7)
        robot.step(1200)
        robot_parts["wheel_right_joint"].setVelocity(7)
        robot.step(2000)
        robot_parts["wheel_right_joint"].setVelocity(-7)
        robot.step(1200)
        robot_parts["wheel_right_joint"].setVelocity(7)
        robot.step(650)
        robot_parts["wheel_left_joint"].setVelocity(0)
        robot_parts["wheel_right_joint"].setVelocity(0)
        print("Cube Time",flush=True)
        robot.step(1000)
        """
        robot.getDevice("arm_1_joint").setPosition(0.65)
        robot.getDevice("arm_2_joint").setPosition(-0.25)
        robot.getDevice("arm_3_joint").setPosition(-2.5)
        robot.getDevice("arm_4_joint").setPosition(2.29)
        robot.getDevice("arm_5_joint").setPosition(0)
        robot.getDevice("arm_6_joint").setPosition(0)
        robot.getDevice("arm_7_joint").setPosition(-0.7)
        """
        moveArmToPoint((0,0.25,0),(0,0.25,0.2))
        #moveArmToPoint((-0.2,0.25,0),(0,0.25,0))
        #robot.getDevice("torso_lift_joint").setPosition(0)#testing lower shelf (move higher cube off shelf first)
        robot.step(12000)
        mode='shopping'
    elif mode=='grabby':#came with the project
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
        #print(f"X={Xtest}, Y={Ytest}, Z={Ztest}",flush=True)#
        #goal=(Xtest/10,Ytest/10,Ztest/10)
        print("starting test",flush=True)
        moveArmToPoint((-.2,.6,-.1),randomizeElbow=True)#(0.64818,.33,-0.2553)
        robot.step(5000)
    elif mode=='object_dist_demo':#demoing getting yellow cubes and their positions
        #torso range: [0, 0.35]
        #camera: width=240, height=135, FOV=2
        #calculated focal length: 1.438839533241073
        print("focal length:",robot.getDevice('camera').getFocalLength())
        print("focal distance:",robot.getDevice('camera').getFocalDistance())
        print("FOV:",robot.getDevice('camera').getFov())
        focal_length=2 * math.atan(math.tan(camera.getFov() * 0.5) / (240 / 135))
        print("calculated:",focal_length)
        #horiz_dist=focal_length*(-0.35)/abs(y2-y1)
        seen_objects=camera.getRecognitionObjects()
        print(len(seen_objects),"objects recognized")
        yellow_objects=[]
        for obj in seen_objects:
            colors = obj.getColors()
            obj_pos=obj.getPosition()
            #print(obj.getNumberOfColors(),"colors")
            if obj.getNumberOfColors()>1:
                print("MULTIPLE COLORS!",obj.getNumberOfColors())
                #for some reason it's a C_array, where every three indices are a color
                colorsList=[(colors[i*3],colors[i*3+1],colors[i*3+2]) for i in range(obj.getNumberOfColors())]
                print("\t",colorsList)#
            if colors:
                #print("\t",colors[0],colors[1],colors[2])
                RGB=(colors[0],colors[1],colors[2])
                if RGB==(1.0,1.0,0.0):
                    print("yellow")
                    yellow_objects.append(obj.getPosition())
                if RGB==(0.0,1.0,0.0):
                    print("green")#append anyways?
                else:
                    pass#print("not yellow (nor green)")
                #print(list(colors))
                #first_color = colors[1]
                #print("first color:",first_color)
                #print(f"    -> First color: R={first_color[0]:.2f}, G={first_color[1]:.2f}, B={first_color[2]:.2f}")
            #print("\tcolors:",*obj_colors)
            #print("\trel_pos:",*obj_pos)
            #to convert to IK goal point, swap x,y and adjust for "origin"
            #break
        if len(yellow_objects)>1:
            yellow_objects.sort(key=lambda x:x[0]**2+x[1]**2+x[2]**2)#closest first
    else:
        print("no mode set")
        robot.step(5000)

#https://cyberbotics.com/doc/guide/tiago-steel?version=R2023a