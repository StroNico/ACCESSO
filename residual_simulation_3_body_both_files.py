"""
This includes two asteroids: a perturbed and a perturbing. It is a coarser version of the script.
We define the SV at encounter epoch of the perturbing, and relatively the perturbed with the Broucke parameters 
(see https://doi.org/10.3390/aerospace11080647). Then propagation and synthetic observations of the perturbed and
unperturbed solutions, saved as ADES_3B_pert_m_{mu:.4e}.txt and ADES_3B_unpert_m_{mu:.4e}.txt. In the unperturbed 
the mass of the perturbing body is set as 0. Noise can be set and applied to the solution to resemble real-life observation errors.
"""

import os
import numpy as np
import math
from astropy.time import Time
from astropy import units as u
import synthetic_obs_l as SOL
from synthetic_obs_l import L1,L3

from MD_library import RMS_from_FO
from MD_auxiliary import check_mass_det_available

def daAUaKm(dist_AU):
    fatt_trans=149597870.691 
    return dist_AU*fatt_trans;

noise_flag = 1
noise_RES_0_1_arcs = .0001/5
noise_RES_0_5_arcs = .0001


Saving_Path = 'Synth_obs/'
os.makedirs(Saving_Path, exist_ok=True)

#### CONSTANTS

K = 0.01720209895 
mu_sun =  1.327124381789709e+11 
M_sun =  1.9891E+30 
G =     mu_sun/ 1.9891E+30  #


##### VARIABLES TO CHANGE


Deltat = (7*u.year).to(u.second) #set propagation span (bi-directional)
epoch0 = Time("2014-12-31 02:00:00", scale="utc") #set epoch of simulated close encounter, in the middle of the propagation arc

r_ref = daAUaKm(2) # Distance form the Sun at which the flyby takes place
v_circ_P = np.sqrt(mu_sun / r_ref)

# State vectors of the perturbing body
RP = np.array([1.,0.,0.]) * r_ref
# RP = np.array([  -229969655.4800489  ,  194402434.5038561   ,  -4823886.372822])  #Alternative way of defining it

VP = np.array([0.,1.,0.]) * v_circ_P
# VP = np.array([   -11.8055871295280 ,   -15.3198965336977  ,   -6.2258821528704    ])


#Definition of the perturbing body's orbital elements, either calculating from the function in SOL or manually definition
# [a_P,e_P,i_P,Omega_P,omega_P,nu_P,phi_P]  = SOL.el_orbs_from_rv(RP,VP,mu_sun)  

#Manually:
a_P = 1 * r_ref 
i_P = 0
e_P=0
Omega_P=0
omega_P=0
nu_P=0
# set the mass parameter of the perturbing body
mu =  2.255e-14 
m_P = mu * M_sun
#Now we have to generate the SOI of the perturbing body, but taking care that the mass is not zero. If it is, it acquires an arbitrary value
if mu > 0:
    r_SOI = (mu/3)**(1/3) * r_ref
else:
    r_SOI = (9.823e-13/3)**(1/3) * r_ref

# Set rotation matrices
L_IJK_to_XYZ = L3(omega_P+nu_P)*L1(i_P)*L3(Omega_P)
L_XYZ_to_IJK = L_IJK_to_XYZ.T

#####################Encounter parameters in Broucke 3D representation
alpha =  np.pi/2 
beta = 0 
gamma = 0
r_P =  0.35  * r_SOI
v_P =  0.1  * v_circ_P 

#Definition of the orbital helio state vectors
rP_XYZ = np.array([[r_P*math.cos(beta)*math.cos(alpha)],[r_P*math.cos(beta)*math.sin(alpha)],[r_P*math.sin(beta)]])
rP_IJK = L_XYZ_to_IJK * rP_XYZ
R_small_helio = np.squeeze(np.asarray( RP.reshape((-1,1)) + rP_IJK))

v_P_XYZ = np.array([[-v_P * math.sin(gamma)*math.sin(beta)*math.cos(alpha) - v_P * math.cos(gamma) * math.sin(alpha)],
          [-v_P * math.sin(gamma) *math.sin(beta) * math.sin(alpha) + v_P * math.cos(gamma)*math.cos(alpha)],[v_P*math.cos(beta)*math.sin(gamma)]])
v_P_IJK = L_XYZ_to_IJK * v_P_XYZ
V_helio = np.squeeze(np.asarray( VP.reshape((-1,1)) + v_P_IJK))

"""
START OF THE 2 ITEMS ITERATION
First:  perturbed
Second unperturbed solution (change m_P = 0 if i=1) 
Produtcion of 2 different files, but at least ther distances form the centre are the same
The name of the files will distinguidh b/w pert/unpert and have the mu_P in the name
"""

for i in range(2):
    
    if i == 0:
        
    
        File_name = f"ADES_3B_pert_m_{mu:.4e}.txt"
        Ast_Name = "2024 P3R"
        
        with open( f'{Saving_Path}{File_name}', 'w') as f:
        
            f.write("permID |provID     |trkSub  |mode|stn|obsTime                 |ra           |dec          |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat|photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks   \n")
            f.close()
            
    elif i== 1:
        
        File_name = f"ADES_3B_unpert_m_{mu:.4e}.txt"
        Ast_Name = "2024 UN9"
        
        with open( f'{Saving_Path}{File_name}', 'w') as f:
        
            f.write("permID |provID     |trkSub  |mode|stn|obsTime                 |ra           |dec          |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat|photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks   \n")
            f.close()
            
        m_P = 0
        
        
    
    
    
    """
    SIMULATION AND TRAJECTORY PROPAGATION

    """

    print("Start pre-propagation...")
    pos_post, vel_post, t_post = SOL.RK_3_body(R_small_helio, V_helio, RP, VP, epoch0,  Deltat, m_P , M_sun, G) 
    print("Start post propagation...")
    pos_pre, vel_pre,   t_pre  = SOL.RK_3_body(R_small_helio, V_helio, RP, VP, epoch0, -Deltat, m_P , M_sun, G)
    print("Propagation complete!")
    state_vec_total = np.hstack((pos_pre[:,::-1],pos_post))
    vel_total =       np.hstack((vel_pre[:,::-1],vel_post))
    time_total = np.concatenate((np.flip(t_pre), t_post))
    
    
    """
    OBSERVATIONS SAMPLING IF THE SAMPLING IS NOT DONE ALREADY IN THE PROPAGATOR 
    """
    
    # if i == 0:
        
    L = len(time_total)
    S = 15 if L > 30 else int(np.floor(L/2)-1)
    H = 18 if L > 18*3 else 0
    
    random_indices_first_half = np.random.choice(range(L//2-H), size=S, replace=False)
    random_indices_second_half = np.random.choice(range(L//2+H, L), size=S, replace=False)
    


    random_indices = np.concatenate((random_indices_first_half, random_indices_second_half))
    random_indices = np.sort(random_indices)

#We create a function that takes the positions and times of the desired observables, and makes the observations


    if noise_flag == 1:
        noise_ra  = np.random.normal(0, math.radians(noise_RES_0_1_arcs), S * 2)
        noise_dec = np.random.normal(0, math.radians(noise_RES_0_1_arcs), S * 2)
    else:
        noise_ra  = np.zeros(S * 2)
        noise_dec = np.zeros(S * 2)
        
    
    SOL.observations(state_vec_total[:,random_indices], vel_total[:,random_indices], time_total[random_indices], Saving_Path, File_name, noise_ra, noise_dec, Ast_Name) 
    

    
    """
    AUTOMATIC SOLUTION IN LINUX TO GET THE RESIDUALS IMMEDIATELY (if wanted, uncomment the following lines)
    """
    
    # local_database_folder="/mnt/c/Synth_obs/"
    # subfolder = "linux/home/folder/containing/.find_orb/"
    # wsl_name = "-d wsl-name "
    # environ_no_perts =  "/environ/environ_totally_unperturbed.dat"
    # folder_linux="//wsl.localhost\\wsl-name\\home\\folder_containing_.find_orb\\.find_orb"
    
    
    # Res = RMS_from_FO(wsl_name, local_database_folder + File_name, subfolder, environ_no_perts, folder_linux)[1]
    
    # print(f"Res = {Res}")
    
    