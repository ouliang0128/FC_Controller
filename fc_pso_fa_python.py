"""fc_pso_python controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import math
import random
import struct
import numpy as np
from numpy import matlib

INFMAX = 1.0e6
INFMIN = 1.0e-6
_pi = 3.1415926
RAND_MAX = 32767


def min(a, b):
    return a if (a < b) else b


def max(a, b):
    return a if (a > b) else b


# random value generato
def randn(nmin, nmax):
    # print(str(random.random())+"qqqq")
    # print(str(nmax)+"max"+str(nmin))
    thisRand = ((random.randint(0, RAND_MAX) / (RAND_MAX + 1.0))) * (nmax - nmin) + nmin;
    # print(thisRand)
    return thisRand


# control setting
_MaxTimestep = 2000
_ConstantSpeed = 4

# Fuzzy
_InVarl = 4  # input dimension
_OutVarl = 1  # output dimension
_NumRule = 10  # the number of fuzzy rules
_CenLimit_max = 5  # max center value of a fuzzy set
_CenLimit_min = 0  # min           ...
_WidLimit_max = 2  # max width value of a fuzzy set
_WidLimit_min = 0.1  # min           ...
_ConqLimit_max = 3.14  # max consequence value of a fuzzy controller
_ConqLimit_min = -3.14  # min           ...

# Arguments for PSO
_SwarmSize = 50  # the size of the population
_MaxIter = 10000  # Maximum Number of Iterations
_CT1 = 1.0  # Personal Learning Coefficient
_CT2 = 1.0  # Global Learning Coefficient
_IW = 0.8  # Inertia Weight
_IW_Damp = 0.999  # Inertia Weight Damping Ratio
_PH1 = 2.05  # constant value for velocity update
_PH2 = 2.05
_LenChrom = _NumRule * (2 * _InVarl + _OutVarl)  # length of an individual

# Variables
_timestep = 0.0  # the time interval between each two control step in simulation
_w, _wdamp, _c1, _c2 = 0.0, 0.0, 0.0, 0.0  # parameters for PSO
_posmin = [0.0 for i in range(0, _LenChrom + 1)]
_posmax = [0.0 for i in range(0, _LenChrom + 1)]
_velmin = [0.0 for i in range(0, _LenChrom + 1)]
_velmax = [0.0 for i in range(0, _LenChrom + 1)]
# _posmin[_LenChrom+1], _posmax[_LenChrom+1], #boundaries for position
# _velmin[_LenChrom+1], _velmax[_LenChrom+1]; #boundaries for velocity
_pop = None  # _pop[_SwarmSize+1],  // Particles
_gbest = None  # global best
# Particle _pop[_SwarmSize+1],  #Particles
#        _gbest; #global best

# Device tag for all devices of the robot in the simulation
left_wheel, right_wheel, tag_emitter_ch01, gps, inertial, lidar_lms291, robot = None, None, None, None, None, None, None;


# --------------------------------------------------------------------------------------
# Classes
class Particle:

    def __init__(self):
        self.pos = [0.0 for i in range(0, _LenChrom + 1)]  # _LenChrom+1 position of an individual in PSO
        self.v = [0.0 for i in range(0, _LenChrom + 1)]  # _LenChrom+1 velocity of an individual in PSO
        self.pbest = [0.0 for i in range(0, _LenChrom + 1)]  # _LenChrom+1 personal best of an individual
        self.fitness = 0.0  # fitness value
        self.pbestfit = 0.0  # fitness value of pbest

    def update_velocity(self):  # velocity update equation
        for k in range(1, _LenChrom + 1):
            # velocity update equation
            self.v[k] = _w * self.v[k] + _c1 * randn(0, 1) * (self.pbest[k] - self.pos[k]) + _c2 * randn(0, 1) * (
                        _gbest.pos[k] - self.pos[k])
            # Apply Velocity Limits
            self.v[k] = min(self.v[k], _velmax[k])
            self.v[k] = max(self.v[k], _velmin[k])

    def update_position(self):  # position update equation
        for k in range(1, _LenChrom + 1):
            self.pos[k] = self.v[k] + self.pos[k]
            # apply position limits and velocity mirror effect
            if self.pos[k] > _posmax[k]:
                self.pos[k] = _posmax[k]
                self.v[k] = -self.v[k]
            elif self.pos[k] < _posmin[k]:
                self.pos[k] = _posmin[k]
                self.v[k] = -self.v[k]

    def initpop(self):  # initialise each particle
        for i in range(1, _LenChrom + 1):
            self.pos[i] = randn(_posmin[i], _posmax[i]);
            self.v[i] = 0.0;
        self.fitness = INFMAX;


    def FA_pop(self, position, fitness):  # initialise each particle from FA
        for i in range(1, _LenChrom + 1):
            self.pos[i] = position[i - 1];
            self.v[i] = 0.0;
        self.fitness = fitness;


_pop = [Particle() for i in range(0, _SwarmSize + 1)]
_gbest = Particle();


# Particle _pop[_SwarmSize+1],  #Particles
#        _gbest; #global best

class FuzzyRule:
    # center is the centre of Gaussian membership function
    # width is the width of Gaussian membership function
    # conq is the consequence values of fuzzy rules

    def __init__(self):
        self.bound = [0 for i in range(0, _InVarl + 1)]  # [_InVarl+1];  // boundary mark of centre
        self.centre = [0.0 for i in range(0, _InVarl + 1)]  # [_InVarl+1],
        self.width = [0.0 for i in range(0, _InVarl + 1)]  # [_InVarl+1],
        self.conq = [0.0 for i in range(0, _OutVarl + 1)]  # [_OutVarl+1]; //ai
        self.mu = [0.0 for i in range(0, _InVarl + 1)]  # [_InVarl+1],   // membership values
        self.phi = 0.0  # ;  // firing strength values 就是所有的mu的乘积

    def memfun(self, x, j):  # // fuzzy membership funcion
        self.mu[j] = math.exp(-0.5 * ((x - self.centre[j]) * (x - self.centre[j]) / (self.width[j] * self.width[j])));
        if self.bound[j] == 1:
            if x > self.centre[j]:
                self.mu[j] = 1;
        elif self.bound[j] == -1:
            if x < self.centre[j]:
                self.mu[j] = 1;


#  Functions for robot control
#  --------------------------------------------------------------------------------------

#  Initialize all devices of the robot
def InitRobot():
    global robot, left_wheel, right_wheel, tag_emitter_ch01, gps, inertial, lidar_lms291
    left_wheel = robot.getMotor("left wheel motor");
    right_wheel = robot.getMotor("right wheel motor");
    tag_emitter_ch01 = robot.getEmitter("emitter_ch01");
    gps = robot.getGPS("pioneer_gps");  # 获取gps 用于导航？
    inertial = robot.getInertialUnit("pioneer_inertial");
    lidar_lms291 = robot.getLidar("lms291");  # 获取所有的雷达

    gps.enable(int(_timestep / 2));  # 启动gps并设置速度为0.5timestep
    inertial.enable(int(_timestep / 2));  # 惯性启动
    lidar_lms291.enable(int(_timestep / 2));  #

    # Enable wheels rolling
    left_wheel.setPosition(float("inf"));
    right_wheel.setPosition(float("inf"));
    left_wheel.setVelocity(0.0);
    right_wheel.setVelocity(0.0);

    robot.step(_timestep);


# Send data to simulation environment supervisor
def robot_send_data(a, ldata):
    # buffersize=(ldata+1)*sizeof(float);
    sent = 0;
    outbuffer = [0.0 for i in range(ldata + 1)]
    outbuffer[0] = ldata;
    for i in range(0, ldata):
        outbuffer[i + 1] = a[i];

    message = struct.pack("iif", outbuffer[0], outbuffer[1], outbuffer[2])
    sent = tag_emitter_ch01.send(message);
    robot.step(_timestep);
    return sent;


# get the distance values to the obstacles from lidar information
def robot_get_lidar_range(lidar_tag):
    area = [[0, 20], [20, 50], [50, 70], [70, 90],  # left scanning area
            [90, 110], [110, 130], [130, 160], [160, 179]];  # Right scanning area 这些其实都是设定好所在的角度
    obs_dist = [0.0 for i in range(8 + 1)]

    lidar_values = lidar_tag.getRangeImage()  # 获取到雷达的值
    for i in range(0, 8):  # 遍历八个雷达左4右4
        obs_dist[i + 1] = 10
        for j in range(area[i][0], area[i][1]):  # 分别获取到扫描角度
            obs_dist[i + 1] = min(lidar_values[j], obs_dist[i + 1])
            # obs_dist[i+1] = min(lidar_values[j], lidar_values[j+1]);//这个代码有问题他应该是想获取到最小值，但是这样的话。。。获取不了啊
    return obs_dist


# Judge whether the robot colliding with or moving far away from an obstacle.
def JudgeRobot(dist):  # 输入一个距离的数组
    min_dist = INFMAX  # 初始化最大值
    robot_failure = 0
    for i in range(1, 8):
        min_dist = min(min_dist, dist[i])  # 获取这个距离数组的最小值

    if min_dist < 0.5:  # define collision condition
        robot_failure = 2  # 如果最小距离小于0.5 那么失败状态是2

    if dist[1] > 3.0:  # define far-away condition
        robot_failure = 1  # 如果距离数组的第一个的值大于3.0那么，失败状态是1
    return robot_failure;


# Robot delay function
def robot_wait(waitime):
    for i in range(1, waitime + 1):
        left_wheel.setVelocity(0)
        right_wheel.setVelocity(0)
        robot.step(_timestep)


# // Functions for fuzzy system
# // --------------------------------------------------------------------------------------

# // Extract the parameters of fuzzy controller from a particle
def FuzzyController(inArr, rule):
    fout = [0.0 for i in range(0, _OutVarl + 1)]  # 定义一个_OutVarl+1 的输出的数组
    num = [0.0 for i in range(0, _OutVarl + 1)]
    den = [0.0 for i in
           range(0, _OutVarl + 1)]  # num[_OutVarl+1], den[_OutVarl+1];定义这两个数组num用来算phi与ai相乘后的累加， den是用来放phi的累加
    for i in range(1, _NumRule + 1):  # 开始循环遍历所有的Rule的数量
        rule[i].phi = 1;  # 初始化phi
        for j in range(1, _InVarl + 1):  # 开始遍历第i个Rule里面的所有输入

            # calculate membership value of a input data point
            rule[i].memfun(inArr[j], j);  # 计算membership function的值，这个时候会把值保存在FuzzyRule 的mu里面

            # calculate firing strength value of a rule (AND operation)
            rule[i].phi = rule[i].phi * rule[i].mu[j];  # 根据公式第i个rule的第j个输入的phi等于这个~~~因为phi初始化是1, 所以其实也就等于mu[j]

    # weighted average
    for k in range(1, _OutVarl + 1):  # 遍歷所有的输出的数量
        den[k] = 0.0;  # 初始化当前输出的den
        num[k] = 0.0;  # 初始化当前输出的num
        for i in range(1, _NumRule + 1):  # 遍历所有的Rule的数量
            den[k] = den[k] + rule[i].phi;  # 计算 第k个输出的所有rule 的 phi的总和
            num[k] = num[k] + rule[i].phi * rule[i].conq[k];  # 计算 第k个输出的所有rule 的 phi*rule的结果的总和

        if den[k] < INFMIN:
            den[k] = INFMIN;  # 如果当前的den小于输入的最小值, 那么赋值为最小值

        fout[k] = num[k] / den[k];  # 计算fuzzy controller 输出

    return fout;  # 这个就是y(x)


def Report(iter):
    print("Iteration[" + str(iter) + "]: " + "gbest=" + str(_gbest.fitness) + "\n");
    # save the so far best fitness velue
    ps = open("BestCost__Phase1.dat", "a");
    ps.write("{}\n".format(_gbest.fitness));
    ps.close();

    # save global best
    ps = open("GlobalBest_Phase1.dat", "w");
    for i in range(1, _LenChrom + 1):
        ps.write("{}\n".format(_gbest.pos[i]));
    ps.close();


# setting contrain condition
def AssignConstrain():
    n = 2 * _InVarl + _OutVarl;
    # Assign maximum and minimum values for position
    for i in range(1, _NumRule + 1):
        for j in range(1, _InVarl + 1):
            # constrain condition for the centre of fuzzy sets
            _posmax[(i - 1) * n + 2 * j - 1] = _CenLimit_max;
            _posmin[(i - 1) * n + 2 * j - 1] = _CenLimit_min;

            # constrain condition for the width of fuzzy sets
            _posmax[(i - 1) * n + 2 * j] = _WidLimit_max;
            _posmin[(i - 1) * n + 2 * j] = _WidLimit_min;

        for k in range(1, _OutVarl + 1):
            # constrain condition for the consequence parts
            _posmax[(i - 1) * n + 2 * _InVarl + k] = _ConqLimit_max;
            _posmin[(i - 1) * n + 2 * _InVarl + k] = _ConqLimit_min;

    # Assign maximum and minimum values for velocity
    for i in range(1, _LenChrom + 1):
        _velmax[i] = 0.5 * (_posmax[i] - _posmin[i]);
        _velmin[i] = -_velmax[i];


# Extract free parameters from a particle
def ExtractFuzzy(evapop):
    rule = [FuzzyRule() for i in range(0, _NumRule + 1)];  # 初始化 N FuzzyRule
    maxcen = INFMIN
    mincen = INFMAX;
    max_id = 0
    min_id = 0;
    n = 2 * _InVarl + _OutVarl;
    for i in range(1, _NumRule + 1):
        for j in range(1, _InVarl + 1):
            rule[i].bound[j] = 0;
            rule[i].centre[j] = evapop.pos[(i - 1) * n + 2 * j - 1];  # centre of fuzzy set
            rule[i].width[j] = evapop.pos[(i - 1) * n + 2 * j];  # width of fuzzy set

            if rule[i].centre[j] > maxcen:
                maxcen = rule[i].centre[j];
                max_id = j;

            if rule[i].centre[j] < mincen:
                mincen = rule[i].centre[j];
                min_id = j;

        rule[i].bound[max_id] = 1;
        rule[i].bound[min_id] = -1;

        for k in range(1, _OutVarl + 1):
            rule[i].conq[k] = evapop.pos[(i - 1) * n + 2 * _InVarl + k];  # consequence

    return rule;


# Extract free parameters from a array
def ExtractFuzzy_array(position):
    # 坐标后移一位
    pos = np.zeros(len(position)+1)
    for i in range(len(position)):
        pos[i+1]=position[i]

    rule = [FuzzyRule() for i in range(0, _NumRule + 1)];  # 初始化 N FuzzyRule
    maxcen = INFMIN
    mincen = INFMAX;
    max_id = 0
    min_id = 0;
    n = 2 * _InVarl + _OutVarl;
    for i in range(1, _NumRule + 1):
        for j in range(1, _InVarl + 1):
            rule[i].bound[j] = 0;
            rule[i].centre[j] = pos[(i - 1) * n + 2 * j - 1];  # centre of fuzzy set
            rule[i].width[j] = pos[(i - 1) * n + 2 * j];  # width of fuzzy set

            if rule[i].centre[j] > maxcen:
                maxcen = rule[i].centre[j];
                max_id = j;

            if rule[i].centre[j] < mincen:
                mincen = rule[i].centre[j];
                min_id = j;

        rule[i].bound[max_id] = 1;
        rule[i].bound[min_id] = -1;

        for k in range(1, _OutVarl + 1):
            rule[i].conq[k] = pos[(i - 1) * n + 2 * _InVarl + k];  # consequence

    return rule;


# Funcitons for PSO algorithm
# --------------------------------------------------------------------------------------

# Initialization
def InitSwarm():
    AssignConstrain();

    _gbest.fitness = INFMAX;

    for i in range(1, _SwarmSize + 1):
        _pop[i].initpop();
        _pop[i].fitness = Evaluation(_pop[i]);

        # initialise personal best
        _pop[i].pbestfit = _pop[i].fitness;
        for j in range(1, _LenChrom):
            _pop[i].pbest[j] = _pop[i].pos[j];

        # initialise global best
        if _pop[i].fitness < _gbest.fitness:
            _gbest.fitness = _pop[i].fitness;
            for j in range(1, _LenChrom + 1):
                _gbest.pos[j] = _pop[i].pos[j];


# Evaluate the performance of a solution (fuzzy controller)
def Evaluation(evapop):  # Particle
    global robot
    step, robot_failure, i = 0, 0, 0
    fout, obs_dist = 0.0, 0.0
    vr = [0.0 for i in range(0, 3)]  # 2+1
    inArr = [0.0 for i in range(0, _InVarl + 1)]  # [_InVarl+1],
    cost = 0.0
    outbuffer = [0.0 for i in range(0, 5)]  # 5
    if type(evapop) == type(_pop[0]):
        rule = ExtractFuzzy(evapop)
    else:
        rule = ExtractFuzzy_array(evapop)

    step = 0;
    robot_failure = 0;

    # robot control loop
    while (step <= _MaxTimestep) and (robot_failure == 0):
        step = step + 1;

        # measure distance to obstacles
        obs_dist = robot_get_lidar_range(lidar_lms291);
        robot.step(_timestep);

        # input data
        inArr[1] = obs_dist[1];
        inArr[2] = obs_dist[2];
        inArr[3] = obs_dist[3];
        inArr[4] = obs_dist[4];

        # fuzzy controller
        fout = FuzzyController(inArr, rule);

        # calculate left and right wheel speeds.
        # 16.03 is the half distance between the left and right wheels.
        # 9.75 (cm) is the radius of the wheels
        vr[1] = _ConstantSpeed - (16.3 * fout[1]) / 9.75;
        vr[2] = _ConstantSpeed + (16.3 * fout[1]) / 9.75;

        # constrain maximum wheel speed. maximum wheel speed is 5.24 (0.524 m/s).
        for i in range(1, 3):
            if vr[i] > 5.24:
                vr[i] = 5.24;
            elif vr[i] < -5.24:
                vr[i] = -5.24;

        # send command to the robot
        left_wheel.setVelocity(vr[1]);
        right_wheel.setVelocity(vr[2]);

        # measure distance to obstacles
        obs_dist = robot_get_lidar_range(lidar_lms291);
        robot.step(_timestep);

        # Judge whether the robot colliding with or moving far away from an  obstacle.
        # If robot_failure=0, robot is moving in the constrain path.
        #   robot_failure=1, robot is moving far away from an obstacle.
        #   robot_failure=2, robot is colliding with an obstacle
        robot_failure = JudgeRobot(obs_dist);
    pass

    # fitness function
    cost = 1.0 / float(step);

    # reset the environment.
    outbuffer[0] = robot_failure;  # reset command
    outbuffer[1] = cost;  # fitness value
    robot_send_data(outbuffer, 2);
    robot_wait(20);

    # robot_send_data(outbuffer, 2);
    return cost;





 # FA function
 # 04/09/2019

def FWA(pop, explosion_rate=1):
    # if shape(X)[0] != shape(Y)[0]:
    #     print("The line of X and Y must be same!")
    # n = shape(X)[1]; l = shape(Y)[1]
    print ("into firework explosion")
    #初始化算法参数

    #烟花个数
    nn = int((len(pop)-1)/explosion_rate) # 爆炸率越高，取火花数越少

    #火花总数
    m = nn*explosion_rate
    #上限和下限

    a=0.04; b=0.8
    # 烟花高斯爆炸个数
    mm = nn
        #烟花维度
    dimension = _LenChrom

        # #最大、最小边界
        #爆炸幅度 等于当前值和最大最小值之间差距最小的那个
    # A = 40
    A=np.zeros([nn, dimension])
    for i in range(nn):
        for j in range(dimension):
            A[i][j]= min((_posmax[i+1]-pop[i+1].pos[j+1]),(pop[i+1].pos[j+1]-_posmin[i+1]))

    #新fitness
    new_fitness=np.zeros([nn])
    #初始化烟花
    fireworks = np.zeros([nn, dimension])
    for i in range(nn):
        for j in range(dimension):
            fireworks[i][j] = pop[i+1].pos[j+1] #改成_CenLimit_max和_CenLimit_min

    #初始化新烟花
    fireworks_new = np.zeros([nn, 100, dimension])

    #初始化高斯火花
    fireworks_rbf = np.zeros([nn,dimension])

    #产生火花
        #每个烟花产生火花的数量
    Si = np.zeros([nn])
        #每个烟花的爆炸半径
    Ai = np.zeros([nn, dimension])
        #火花限制
    si = np.zeros([nn])
        #计算每个烟花的适应度函数值
    f = np.zeros([nn])
        #最大最小适应度
    fmax = f[0]; fmin = 1
        #误差函数初始化
    # E = np.zeros([5000, 1])
    #
    #
    # #烟花算法迭代过程
    # for delta_num in range(5000):

    # 普通爆炸产生的火花总数
    sum_new_fireworks = 0
    # 总适应度
    sum = 0
    #计算适应度,并求出最大值和最小值
    for i in range(nn):
        # f[i] = calculatef(X, Y, fireworks[i], n, h, l) #Evaluation(evapop)
        # f[i] = Evaluation(pop[i+1])
        f[i] = pop[i+1].fitness
        if f[i] > fmax:
            fmax = f[i]
        if f[i] < fmin:
            fmin = f[i]
        sum += f[i]

        #求每个烟花的爆炸半径和火花数
    for i in range(nn):
            #计算火花数, 根据fitness决定火花数量
        Si[i] = m * (fmax - f[i] + 0.0001) / (nn * fmax - sum + 0.0001) #这个解和最好的差距/所有解和最好的差距
        Si[i] = round(Si[i])
        if Si[i] < a * m:
            si[i] = round(a * m)
        elif Si[i] > b * m:
            si[i] = round(b * m)
        else:
            si[i] = round(Si[i])
            #不能超过火花数限制
        if Si[i] > si[i]:      #可以取消
            Si[i] = si[i]

            #计算普通爆炸产生的火花总数
        sum_new_fireworks += int(Si[i])

        #计算爆炸半径
        for i in range(nn):
            for j in range(dimension):
                Ai[i][j] = A[i][j] * (f[i] - fmin + 0.0001) / (sum - nn * fmin + 0.0001) #距离是通过fitness记录的

            #产生新火花
        for j in range(int(Si[i])):
                #初始化新火花
            fireworks_new[i][j] = fireworks[i] #新的sparks首先赋值为旧的fireworks
                #随机选择z个维度
            z = random.randint(1, dimension)
                #打乱随机选择前z个
            # zz = range(dimension)    #这行有错，可以改成：
            zz = [zzi for zzi in range(dimension)]
            random.shuffle(zz)               #从dimension里挑选zz个

            # 产生新火花
            for k in range(z):
                fireworks_new[i][j][zz[k]] += random.uniform(0, Ai[i][j])


    #产生高斯火花（每个烟花产生一个高斯火花）
        # 随机选择z个维度
    z = random.randint(1, dimension)
    # zz = range(dimension) # 也得改
    zz = [zzi for zzi in range(dimension)]
    random.shuffle(zz)
        #高斯随机数
    g = randn(1,1)

        #高斯爆炸算子
    for i in range(mm):
        for j in range(z):
            fireworks_rbf[i][zz[j]] = g * fireworks[i][zz[j]] #每个烟花只炸一个高斯


        #构造总烟花
    sum_fireworks = nn + sum_new_fireworks + mm
    print ("numbmer of new particles is: ", sum_fireworks)
    fireworks_final = np.zeros([sum_fireworks, dimension])
    for i in range(nn):
        fireworks_final[i] = fireworks[i]
    #
    # for j in range(Si[0]):
    #     fireworks_final[nn+j] = fireworks_new[0][j] # 第1一个firework周围的sparks

    count=0
    for i in range(nn):
        for j in range(int(Si[i])):
            fireworks_final[int(nn+j+count)] = fireworks_new[i][j]
        count += Si[i]





    for i in range(mm):
        fireworks_final[int(nn+sum_new_fireworks+i)] = fireworks_rbf[i]


    #映射规则,每一个参数限制在[-5,5]范围内
    for i in range(sum_fireworks):
        for j in range(dimension):
            if fireworks_final[i][j] > _posmax[j+1] or fireworks_final[i][j] < _posmin[j+1]:
                fireworks_final[i][j] =_posmin[j+1] + np.mod(abs(fireworks_final[i][j]), #最小值加上最大最小相减取余，限制在[-5,5]之间
                                       (_posmax[j+1] - _posmin[j+1]))

    # 选择策略
        #爆炸后新种群适应度 （这个地方会有问题，每一个都要跑一次，太费时间了，可以改成similaritys）
    f_new = np.zeros([sum_fireworks, 1])
    f_new_min = f_new[0]
    #print f_new_min
        #初始化最优适应度下标
    min_i = 0
        #选出下一代nn个个体，由最大适应度个体与距离更远的nn-1个个体组成
        #求最优适应度
    for i in range(sum_fireworks):
        #print fireworks_final[i]
        # f_new[i] = calculatef(X, Y, fireworks_final[i], n, h, l)
        f_new[i] = Evaluation(fireworks_final[i])
        if f_new[i] < f_new_min:
            f_new_min = f_new[i]
            min_i = i
    print ("Fireworks new min fitness is ",f_new_min)

        #求出每个个体被选择的概率
        #初始化两两个体之间的距离
    D = np.zeros([sum_fireworks, sum_fireworks])
        #计算两两个体之间的距离
    for i in range(sum_fireworks):
        for j in range(sum_fireworks):
            D[i][j] = np.dot((fireworks_final[i] - fireworks_final[j]), \
                          (fireworks_final[i] - fireworks_final[j])) / 2

        #初始化每个个体与其他个体之间的距离之和
    Ri = np.zeros([sum_fireworks, 1])
        #初始化距离矩阵的副本
    RRi = np.zeros([sum_fireworks, 1])
        #计算每个个体与其他个体的距离之和
    for i in range(sum_fireworks):
        for j in range(sum_fireworks):
            Ri[i] += D[i][j]
    RRi = Ri

        #选出距离最远的nn-1个个体，即对距离矩阵进行排序 （排序可优化）
    for i in range(sum_fireworks-1):
        for j in range(i, sum_fireworks):
            if Ri[i] < Ri[j]:
                tmp = Ri[i]
                Ri[i] = Ri[j]
                Ri[j] = tmp

        #构造新种群
    fireworks[0] = fireworks_final[min_i]
    new_fitness[0] = f_new[min_i]

    for i in range(sum_fireworks):
        for k in range((len(pop))-1):
            if Ri[k] == RRi[i]:
                fireworks[k+1] = fireworks_final[i]
                new_fitness[k+1]=f_new[i]


    return {"fireworks":fireworks,"new_fitness":new_fitness}


# --------------------------------------------------------------------------------------
def main():
    global robot, _timestep, _pop, _gbest, _w, _wdamp, _c1, _c2, iter
    ph = 0.0
    chi = 0.0
    outbuffer = [0.0 for i in range(0, 3)]
    # setting parameters for PSO
    _w = _IW
    _wdamp = _IW_Damp
    _c1 = _CT1
    _c2 = _CT2
    # If you would like to use Constriction Coefficients for PSO,
    # uncomment the following block and comment the above set of parameters.
    # Constriction Coefficients
    # -------------------------------------------------------
    ph = _PH1 + _PH2  # was _PH1+_PH1
    chi = 2.0 / (ph - 2 + math.sqrt(ph * ph - 4 * ph))  # velocity constrain factor
    _w = chi  # inertia Weight
    _wdamp = 1  # inertia Weight Damping Ratio
    _c1 = chi * _PH1  # personal Learning Coefficient
    _c2 = chi * _PH2  # global Learning Coefficient was _c1=chi*_PH2
    # -------------------------------------------------------

    # create the Robot instance.
    robot = Robot()

    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    _timestep = timestep;  # 每16个基本步算一步
    InitRobot();

    # Main loop:

    InitSwarm();
    Report(0);

    for iter in range(1, _MaxIter + 1):  # 开始 实验 _MaxIter 次看看会不会成功
        for i in range(1, _SwarmSize + 1):  # 我们每一次Iter 其实都有_SwarmSize 个 particle
            _pop[i].update_velocity();  # 公式15
            _pop[i].update_position();  # 公式16
            # Evaluation
            _pop[i].fitness = Evaluation(_pop[i]);  # evaluate 获取fitness


            # Update Global Best
            if _pop[i].fitness < _gbest.fitness:
                _gbest.fitness = _pop[i].fitness;
                for j in range(1, _LenChrom + 1):
                    _gbest.pos[j] = _pop[i].pos[j];
            else: #如果globle fitness没有变化或者还变差了，那就用firework炸一波
                # 用FA更新swarm
                FA_results = FWA(_pop,5)
                for i in range(len(_pop) - 1):
                    _pop[i + 1].FA_pop(FA_results['fireworks'][i], FA_results['new_fitness'][i])
                _gbest.fitness = _pop[1].fitness;
                for j in range(1, _LenChrom + 1):
                    _gbest.pos[j] = _pop[1].pos[j];


        # update inertia weight with damping ratio
        _w = _w * _wdamp;  # update inertia weight with damping ratio
        # save and report information of gbest
        Report(iter);
        # If the robot is capable of following the boundary more than one round,
        # then PSO terminates.
        if _gbest.fitness <= 1.0 / _MaxTimestep:
            iter = _MaxIter + 1;
            outbuffer[0] = 100;  # terminating simulation command
            outbuffer[1] = _gbest.fitness;  # fitness value
            robot_send_data(outbuffer, 2);
            break;
        # loop for iteration ends

    #  reload the environment.
    outbuffer[0] = -1;  # reset command
    outbuffer[1] = 0;  # fitness value
    robot_send_data(outbuffer, 2);
    return 0;

main();

