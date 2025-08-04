"""
Orbit propagator and synthetic observations generator that allows, with the use of flags to be manually set in the beginning, 
the desired output of the script.
"""

import os
import numpy as np
import math
from astropy.time import Time
from astropy import units as u
import synthetic_obs_l as SOL
import ephem_generation_lib as EGL
from synthetic_obs_l import L1,L3
from poliastro.ephem import Ephem, ephem_interpolant
from poliastro.frames.enums import Planes
import spiceypy as spice
import time
from poliastro.bodies import  Sun
import replicate_obs_file_l as ROFL


def daAUaKm(dist_AU):
    fatt_trans=149597870.691 
    return dist_AU*fatt_trans;




"""
FIXED CONSTANTS SIMULATION PARAMETERS SETUP

"""

# noise_RES_0_1_arcs = .0001/5
noise_RES_0_5_arcs = .0001
# n_pert_ast = 105 #  Eros in order of file mu.txt 151
# n_pert_ast_MPC = 324   #433


spice.unload('de430.bsp')
spice.unload('naif0012.tls')
spice.unload('codes_300ast_20100725.tf')
spice.unload('codes_300ast_20100725.bsp')
spice.furnsh('de430.bsp')
spice.furnsh('naif0012.tls')
spice.furnsh('codes_300ast_20100725.tf')
spice.furnsh('codes_300ast_20100725.bsp')

K =       0.01720209895 
mu_sun =  1.327124381789709e+11
M_sun =   1.9891E+30
G =       mu_sun/ 1.9891E+30 

Delta_days = 30
leapsecond = 67* u.second

"""
SIMULATION PARAMETERS MANUAL SETUP. Flags and parameters to set

"""
noise_flag = 1  #1 yes noise, 0 no noise in observations
noise_RMS =  .4  #set the desired background noise of the final RMS solution
noise_RES = noise_RMS * noise_RES_0_5_arcs / 0.5 # DO NOT TOUCH
topocentric_obs = 1  #1 observations from actual ground observatories, 0 observations from "500" geocentric
print_min_dist = 1   #1 print mnimum distance between the 2 asteroids, 0 do not print it
Generate_ephemerides = 1  #1 generate and update ephemerides for the perturbing body, 0 do not do it (in general to save computational time if the ephemerides have already been produced and saved)
Existing_asteroids = 1    #1 to use actual asteroids (and actual encounter) for the propagation, 0 to manually set the initial state vectors of desired synthetic asteroids
Generate_unp_file = 0   #Set 0 if you do not want the unperturbed file to be generated and 1 if you want the file
max_observations_desired = 350  #maximum number of observations desired (the code will try to achieve it, if it is a sensible number)

mu = 1.79e-12   #mass parameter of the perturbing body (mass_perturbing/mass_sun)


perturbers_list =  ['1','2','3','4','5','6','7','8','c','p','v']       #['1','2','3','4','5','6','7','8','c','p','v']


####################### MAIN BODY ########################################

'''
Still to define the initial and boundary conditions below
'''


if topocentric_obs == 1:
        
    File_obs = "Observatories.txt"
    obs_code_list = []
    
    with  open(File_obs, 'r') as file_surveys:
        for linea_surveys in file_surveys:
            obs_code_list.append(linea_surveys[0:3])
            
elif topocentric_obs == 0:
    obs_code_list = ['500']



   
epoch0 = Time("2001-02-01T00:00:00.000000", scale="utc")  #Select
Deltat  = (22*u.year).to(u.second)                        #Select total propagation time


"""
DEFINITION OF THE INITIAL STATE VECTORS
Set Existing_asteroids = 1 if we want the SV from ephemerides of a real couple of asteroids. We set the MPC codes and the epoch. The propagation will be only forward of Deltat
Set Existing_asteroids = 0 if we want to define an orbit for the perturbing body and the perturbed with the Broucke parameter of the encounter. epoch0 is the encounter epoch and the propagation is 2 ways of Deltat/2
"""


if Existing_asteroids == 1:


    perturbing_asteroid = '747'
    perturbed_asteroid  = '209638'
    
    #SAVING PATH DEFINITION BASED ON THE NAME OF THE ASTEROIDS STUDIED
    Saving_Path = f'Observations/{perturbing_asteroid}_{perturbed_asteroid}/'
    os.makedirs(Saving_Path, exist_ok=True)

    """
    SIMULATION PARAMETERS DERIVATION
    
    """
    
    ephem_perturbing = Ephem.from_horizons(perturbing_asteroid, id_type='smallbody', epochs= (epoch0 ), attractor= Sun , plane=Planes.EARTH_ECLIPTIC)
    VP               = ephem_perturbing.rv(epoch0)[1].to(u.km/u.s).value
    RP               = ephem_perturbing.rv(epoch0)[0].to(u.km).value
    #Orbital elements of the perturbing body
    a_P,e_P,i_P,Omega_P,omega_P,nu_P,phi_P = SOL.el_orbs_from_rv(RP,VP,mu_sun)
    r_ref = np.linalg.norm(RP)
    
    ephem_perturbed = Ephem.from_horizons(perturbed_asteroid, id_type='smallbody', epochs= epoch0 , attractor= Sun ,  plane=Planes.EARTH_ECLIPTIC)
    V_helio         = ephem_perturbed.rv(epoch0 )[1].to(u.km/u.s).value
    R_small_helio   = ephem_perturbed.rv(epoch0)[0].to(u.km).value

    
    m_P = mu * M_sun
    #SOI of the perturbing body. If the mass is set to zero, an arbitrary value is put just for this computation (can't be a zero SOI)
    if mu > 0:
        r_SOI = (mu/3)**(1/3) * r_ref
    else:
        r_SOI = (9.823e-13/3)**(1/3) * r_ref
    
    # Set rotation matrixes
    L_IJK_to_XYZ = L3(omega_P+nu_P)*L1(i_P)*L3(Omega_P)
    L_XYZ_to_IJK = L_IJK_to_XYZ.T
        



elif Existing_asteroids == 0:
    
    Deltat = Deltat / 2
    
    perturbing_asteroid = '0000'
    perturbed_asteroid = "2024AAA"
    
    r_ref = daAUaKm(2)
    
    #You can define the SV from a reference distance (and velocity), or form the full vector. The reference distance wil be the magnitude of the RP
    
    # RP = np.array([1.,0.1,0.2]) * r_ref
    RP = np.array([  -229969655.4800489  ,  194402434.5038561   ,  -4823886.372822])

    r_ref = np.linalg.norm(RP)
    
    v_circ_P = np.sqrt(mu_sun / r_ref)
    # VP = np.array([0.,1.,0.04]) * v_circ_P
    VP = np.array([   -11.8055871295280 ,   -15.3198965336977  ,   -6.2258821528704    ])
    # v_circ_P = v_circ_P/np.linalg.norm(VP)
    
    #Orb elems perturbing body
    a_P,e_P,i_P,Omega_P,omega_P,nu_P,phi_P = SOL.el_orbs_from_rv(RP,VP,mu_sun)
    
    m_P = mu * M_sun
    
    #SOI of the perturbing body. If the mass is set to zero, an arbitrary value is put just for this computation (can't be a zero SOI)
    if mu > 0:
        r_SOI = (mu/3)**(1/3) * r_ref
    else:
        r_SOI = (9.823e-13/3)**(1/3) * r_ref

    # Set rotation matrices
    L_IJK_to_XYZ = L3(omega_P+nu_P)*L1(i_P)*L3(Omega_P)
    L_XYZ_to_IJK = L_IJK_to_XYZ.T
    
    #TO BE SET TO DEFINE THE PERTURBED ASTEROIDS POSITION AT THE MOMENT OF CLOSEST ENCOUNTER:
    #Encounter parameters in Broucke 3D representation
    alpha =  np.pi /2 
    beta = 0 
    gamma = 0
    # r_P_factor =  0.2     #   THE MINIMUM ENCOUNTER DISTANCE CAN BE SET AS A NOMINAL DISTANCE OR AS A DISTANCE WITH RESPECT TO THE SOI
    r_P = 10000 #r_P_factor * r_SOI
    r_P_factor = r_P / r_SOI
    v_P =  0.05  * v_circ_P 
    
    #Check escape velocity
    vE = np.sqrt(2 * m_P * G / r_P)
    if v_P < vE:
        raise ValueError("Velocity at encounter lower than minimum escape velocity required")

    #Definition of the orbital helio state vectors
    rP_XYZ = np.array([[r_P*math.cos(beta)*math.cos(alpha)],[r_P*math.cos(beta)*math.sin(alpha)],[r_P*math.sin(beta)]])
    rP_IJK = L_XYZ_to_IJK * rP_XYZ
    R_small_helio = np.squeeze(np.asarray( RP.reshape((-1,1)) + rP_IJK))

    v_P_XYZ = np.array([[-v_P * math.sin(gamma)*math.sin(beta)*math.cos(alpha) - v_P * math.cos(gamma) * math.sin(alpha)],
              [-v_P * math.sin(gamma) *math.sin(beta) * math.sin(alpha) + v_P * math.cos(gamma)*math.cos(alpha)],[v_P*math.cos(beta)*math.sin(gamma)]])
    v_P_IJK = L_XYZ_to_IJK * v_P_XYZ
    V_helio = np.squeeze(np.asarray( VP.reshape((-1,1)) + v_P_IJK))
    
    Saving_Path = f'Observations/rP_{r_P_factor}_mu_{mu}/'
    os.makedirs(Saving_Path, exist_ok=True)
        
    
"""
GENERATION OF THE EPHEMERIDES OF THE PERTURBING BODY (if activated)
We use the library "ephem_generation_lib" to change the ephemerides of Eros (ast 151th in the order) 

We also change the mass on mu.txt file
"""    
    

if Generate_ephemerides == 1:


    
    EGL.modify_mass_file_mu1(433, mu)
    
    start = time.time()
    print("Start ephemeris creation...")
    EGL.Ephem_gen(RP, VP, epoch0, Deltat,  m_P , M_sun, G, perturbing_asteroid, Existing_asteroids, perturbers_list)
    print(time.time()-start)


"""
START OF THE 2 ITEMS ITERATION
First:  perturbed
Second unperturbed solution (change m_P = 0 if i=1) 
Produtcion of 2 different files
The name of the files will distinguidh b/w pert/unpert and have the mu_P in the name

if Generate_unp_file == 0 we do not do the second part of the loop
"""

for i in range(Generate_unp_file + 1):
    
    if i == 0:
        
    
        File_name = f"ADES_{perturbed_asteroid}_pert_m_{mu:.4e}.txt"
        Ast_Name = f"{perturbed_asteroid}PE"
        
        with open( f'{Saving_Path}{File_name}', 'w') as f:
        
            f.write("permID |provID     |trkSub  |mode|stn|obsTime                 |ra           |dec          |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat|photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks   \n")
            f.close()
            
    elif i== 1:
        
        File_name = f"ADES_{perturbed_asteroid}_unpert_m_{mu:.4e}.txt"
        Ast_Name = f"{perturbed_asteroid}UN"
        
        with open( f'{Saving_Path}{File_name}', 'w') as f:
        
            f.write("permID |provID     |trkSub  |mode|stn|obsTime                 |ra           |dec          |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat|photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks   \n")
            f.close()
            
        m_P = 0
        print_min_dist = 0
        
        
    
    
    
    """
    SIMULATION AND TRAJECTORY PROPAGATION

    """
    
    if Existing_asteroids == 1:
        print("Start propagation...")
        state_vec_total, vel_total, time_total, epoch_min_dist = SOL.RK_3_body(R_small_helio, V_helio, RP, VP, epoch0,  Deltat, m_P , M_sun, G, perturbers_list, print_min_dist,  max_dt = 3*24*60*60) 
        print("Propagation complete!")
    elif Existing_asteroids == 0:
        print("Start pre-propagation...")
        pos_pre, vel_pre,   t_pre, epoch_min_dist  = SOL.RK_3_body(R_small_helio, V_helio, RP, VP, epoch0, -Deltat, m_P , M_sun, G, perturbers_list, print_min_dist_flag = 0, max_dt=2*24*60*60) #print_min_dist = 0 enforced, and epoch_min_dist useless but needs to be provided (it is epoch0)
        print("Start post propagation...")
        pos_post, vel_post, t_post, epoch_min_dist = SOL.RK_3_body(R_small_helio, V_helio, RP, VP, epoch0,  Deltat, m_P , M_sun, G, perturbers_list, print_min_dist_flag = 0, max_dt=2*24*60*60)
        print("Propagation complete!")

        
        if i == Generate_unp_file: #and print_min_dist == 1:
            print(f"\nClosest approach from Horizons at {r_P:.2f} km on {epoch0}  ")
            print(f"Distance in R_H = {r_P_factor:.2f}")
            
        state_vec_total = np.hstack((pos_pre[:,::-1],pos_post))
        vel_total =       np.hstack((vel_pre[:,::-1],vel_post))
        time_total = Time(np.concatenate((np.flip(t_pre), t_post)))


    
    """
    OBSERVATIONS SAMPLING IF THE SAMPLING IS NOT DONE ALREADY IN THE PROPAGATOR 
    """
    
    # if i == 0:
    
        
    L = len(time_total)
    S = int(math.floor(max_observations_desired/2)) if L > max_observations_desired else int(np.floor(L/2)-1)
    H = 18 if L//2-18 > S else 0
    
    random_indices_first_half = np.random.choice(range(L//2-H), size=S, replace=False)
    random_indices_second_half = np.random.choice(range(L//2+H, L), size=S, replace=False)
    

    random_indices = np.concatenate((random_indices_first_half, random_indices_second_half))
    random_indices = np.sort(random_indices)
    
    


    if noise_flag == 1:
        noise_ra  = np.random.normal(0, math.radians(noise_RES), S * 2)
        noise_dec = np.random.normal(0, math.radians(noise_RES), S * 2)
    else:
        noise_ra  = np.zeros(S * 2)
        noise_dec = np.zeros(S * 2)
        
    
    SOL.observations(state_vec_total[:,random_indices], vel_total[:,random_indices], time_total[random_indices], Saving_Path, File_name, noise_ra = noise_ra, noise_dec = noise_dec, Ast_Name = Ast_Name, obs_code_list = obs_code_list)  #[:,random_indices]

    if Existing_asteroids == 1:
        ROFL.plot_obs_distrib(time_total[random_indices].jd,perturbing_asteroid,perturbed_asteroid,epoch0)

    else:
        ROFL.plot_obs_distrib_sim_ast(time_total[random_indices].jd, epoch0, Deltat) #Deltat already divided by 2


    