import numpy as np
import dill
import math
def D_AOA_cal(Target_loc):
    fc = 3.9936e+9
    c = 299792458
    radius = 0.0732
    Square_Target_loc = np.power(Target_loc,2)
    real_D = np.sqrt((Square_Target_loc[:,0] + Square_Target_loc[:,1]).reshape(-1))

    real_Angle = np.arctan2(Target_loc[:,1],Target_loc[:,0])
    lambda_0 = c/fc
    antenna_angle = []
    phi = []
    real_Pdoa = np.zeros([len(Target_loc),8])
    for k in range(8):
        antenna_angle.append(wrapToPi(k * 2 * np.pi / 8))
    std_phi = (2 * np.pi * fc / c * radius * np.cos(real_Angle)).reshape(-1,1)

    for k in range(8):
        phi = (2 * np.pi * fc / c * radius * np.cos(real_Angle - antenna_angle[k]))
        phi = phi.reshape(-1,1)
        real_Pdoa[:,k] = (wrapToPi(phi - std_phi)).reshape(-1)
    real_D_Pdoa = np.hstack((real_D.reshape(-1,1),real_Pdoa.reshape(-1,8)))
    return real_D_Pdoa

def wrapToPi(angle):
    angle = np.array(angle).reshape(-1,1)
    wraped_angle = np.zeros([len(angle),1])
    for k in range(len(angle)):
        wraped_angle[k,0] = math.remainder(angle[k,0],math.tau)
    return wraped_angle