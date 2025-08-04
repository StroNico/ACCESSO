# -*- coding: utf-8 -*-
"""
This code creates the observation file organised in this way:
    
    Topocentric observatories
    Real distribution with magnitude/visibility/daylight check on observatories
    Symmetric number of observations pre/post
    Symmetric time distribution pre/post
    Proper N obs
    Ideally obs close to the CA
    
"""



import os
import numpy as np
import math
from astropy.time import Time 
from astropy import units as u
import synthetic_obs_l as SOL
import ephem_generation_lib as EGL
from synthetic_obs_l import L1,L3
from MD_library import RMS_from_FO
from MD_auxiliary import check_mass_det_available
from poliastro.ephem import Ephem, ephem_interpolant
from poliastro.frames.enums import Planes
import spiceypy as spice
from scipy.optimize import minimize_scalar
from ephemerides_libr import SV_sun_405
import time
from poliastro.bodies import  Sun
import replicate_obs_file_l as ROFL
import random
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import matplotlib
matplotlib.use('Agg')  


plt.close('all')




def daAUaKm(dist_AU):
    fatt_trans=149597870.691 
    return dist_AU*fatt_trans;

def daKmaAU(dist_Km):
    fatt_trans=149597870.691 
    return dist_Km / fatt_trans;

#Function to parallelise propagations in 2 different directions in time
def propagate(direction, R_small_helio, V_helio, RP, VP, epoch0, Deltat, m_P, M_sun, G, perturbers_list, N_obs):
    if direction == 'pre':  # Backward in time
        return ROFL.RK_3_body_check_visibility(R_small_helio, V_helio, RP, VP, epoch0, -Deltat, m_P , M_sun, G, perturbers_list, print_min_dist_flag = 0, max_dt = 2*24*60*60, N = N_obs/2)
    elif direction == 'post':  # Forward in time
        return ROFL.RK_3_body_check_visibility(R_small_helio, V_helio, RP, VP, epoch0,  Deltat, m_P , M_sun, G, perturbers_list, print_min_dist_flag = 0, max_dt = 2*24*60*60,  N = N_obs/2)




"""
FIXED CONSTANTS SIMULATION PARAMETERS SETUP

"""

# noise_RES_0_1_arcs = .0001/5
noise_RES_0_5_arcs = .0001


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
G =       mu_sun/ 1.9891E+30  #m

Delta_days = 30
leapsecond = 67* u.second

"""
SIMULATION PARAMETERS MANUAL SETUP

"""
noise_flag = 0
noise_RMS = .20/2  #5*4 #set the desired background noise of the final RMS solution
noise_RES = noise_RMS * noise_RES_0_5_arcs / 0.5 # DO NOT TOUCH
topocentric_obs = 1
print_min_dist = 1
Generate_ephemerides = 1
Existing_asteroids = 0
perturbing_manually_defined = 1   #1 if you ewant to set manually the state vectors, or 0 if you want to define the desired orbital parameters, the epoch of encounter and then the state vectors will be generated to match the requirments

max_observations_desired = 386 #np.ceil(35 * 25) # 19.35
fraction = 0 #FOR OFFSET

mu =  2.255e-14   


perturbers_list =  []      #['1','2','3','4','5','6','7','8','c','p','v']

if topocentric_obs == 1:
        
    File_obs = "Observatories.txt"
    obs_code_list = []
    
    with  open(File_obs, 'r') as file_surveys:
        for linea_surveys in file_surveys:
            obs_code_list.append(linea_surveys[0:3])
            
elif topocentric_obs == 0:
    obs_code_list = ['500']



   
# epoch0 = Time("2014-11-10T22:50:00.000000", scale="utc")
epoch0 = Time("2024-11-10T22:50:00.000000", scale="utc")
Deltat  = (20*u.year).to(u.second)


"""
DEFINITION OF THE INITIAL STATE VECTORS
Set Existing_asteroids = 1 if we want the SV from ephemerides of a real couple of asteroids. We set the MPC codes and the epoch. The propagation will be only forward of Deltat
Set Existing_asteroids = 0 if we want to define an orbit for the perturbing body and the perturbed with the Broucke parameter of the encounter. epoch0 is the encounter epoch and the propagation is 2 ways of Deltat/2
"""

if Existing_asteroids == 1:


    perturbing_asteroid = '32'
    perturbed_asteroid  = '113675'
    
    #SAVING PATH DEFINITION BASED ON THE NAME OF THE ASTEROIDS STUDIED
    Saving_Path = f'{perturbing_asteroid}_{perturbed_asteroid}/'
    os.makedirs(Saving_Path, exist_ok=True)

    """
    SIMULATION PARAMETERS DERIVATION
    
    """
    
    ephem_perturbing = Ephem.from_horizons(perturbing_asteroid, id_type='smallbody', epochs= (epoch0 ), attractor= Sun , plane=Planes.EARTH_ECLIPTIC)
    VP               = ephem_perturbing.rv(epoch0)[1].to(u.km/u.s).value# - SV_sun_405(epoch0)[1]
    RP               = ephem_perturbing.rv(epoch0)[0].to(u.km).value#- SV_sun_405(epoch0)[0]
    #Orbital elements of the perturbing body
    a_P,e_P,i_P,Omega_P,omega_P,nu_P,phi_P = SOL.el_orbs_from_rv(RP,VP,mu_sun)
    r_ref = np.linalg.norm(RP)
    
    ephem_perturbed = Ephem.from_horizons(perturbed_asteroid, id_type='smallbody', epochs= epoch0 , attractor= Sun ,  plane=Planes.EARTH_ECLIPTIC)
    V_helio         = ephem_perturbed.rv(epoch0 )[1].to(u.km/u.s).value# - SV_sun_405(epoch0)[1]
    R_small_helio   = ephem_perturbed.rv(epoch0)[0].to(u.km).value# - SV_sun_405(epoch0)[0]

    
    m_P = mu * M_sun
    #Now we have to generate the SOI of the perturbing body, but taking care that the mass is not zero, so in case I put it 
    #in a arbitrary value
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
    
    r_ref = daAUaKm(2.82)
    
    if perturbing_manually_defined == 1:
        # RP = np.array([1.,0.1,0.2]) * r_ref
        RP = np.array([  2.30465421e+08,  3.36969677e+08, -6.60765926e+07])
        
        r_ref = np.linalg.norm(RP)
        
        v_circ_P = np.sqrt(mu_sun / r_ref)
        # VP = np.array([0.,1.,0.04]) * v_circ_P
        VP = np.array([-14.7470096 ,  10.40500707,   1.21985104])
        # v_circ_P = v_circ_P/np.linalg.norm(VP)
    else:
        #Function that takes desired orbital elementse and epoch0 to define an encounter in visibility of the earth
        desired_aP = r_ref
        desired_eP = 0.02                                                         
        desired_iP = math.radians(10)
        v_circ_P = np.sqrt(mu_sun / r_ref)
        
        RP, VP = ROFL.get_initial_conditions(desired_aP, desired_eP, desired_iP, epoch0, mu_sun)
        
        
        
        
    
    a_P,e_P,i_P,Omega_P,omega_P,nu_P,phi_P = SOL.el_orbs_from_rv(RP,VP,mu_sun)
    
    
    
    m_P = mu * M_sun
    #Now we have to generate the SOI of the perturbing body, but taking care that the mass is not zero, so in case I put it 
    #in a arbitrary value
    if mu > 0:
        r_SOI = (mu/3)**(1/3) * r_ref
    else:
        r_SOI = (9.823e-13/3)**(1/3) * r_ref
    
    # Set rotation matrices
    L_IJK_to_XYZ = L3(omega_P+nu_P)*L1(i_P)*L3(Omega_P)
    L_XYZ_to_IJK = L_IJK_to_XYZ.T
    
    #Encounter parameters in Broucke 3D representation
    alpha =  np.pi /2 
    beta =  0 
    gamma =    0 
    r_P_factor =  1.2
    r_P = r_P_factor * r_SOI  
    # r_P_factor = r_P / r_SOI
    v_P =  0.05 * v_circ_P  
    
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
    
    Saving_Path = f'rP_{r_P_factor}_mu_{mu}/'
    os.makedirs(Saving_Path, exist_ok=True)
        
    
    a_ast,e_ast,i_ast,Omega_ast,omega_ast,nu_ast,phi_ast = SOL.el_orbs_from_rv(R_small_helio,V_helio,mu_sun)

    print('Orb elems:  PERTURBING / PERTURBED')
    print(f'a: {daKmaAU(a_P)}  /  {daKmaAU(a_ast)}')
    print(f'e: {e_P}  /  {e_ast}')
    print(f'i: {math.degrees(i_P)}  /  {math.degrees(i_ast)}')

# response = input("Do you want to continue? Press Y to continue, N to abort: ").strip().lower()


if __name__ == '__main__' :

        
            
        """
        GENERATION OF THE EPHEMERIDES OF THE PERTURBING BODY (if activated)
    

        """    
            
        
        if Generate_ephemerides == 1:
        
        
            
            EGL.modify_mass_file_mu1(433, mu)
            
            start = time.time()
            print("Start ephemeris creation...")
            EGL.Ephem_gen(RP, VP, epoch0, Deltat,  m_P , M_sun, G, perturbing_asteroid, Existing_asteroids, perturbers_list)
            print(time.time()-start)
        
        
        """
        SET THE OBSERVATION EPOCHS
        Based on the number of observations desired, and on how many of them should be batched, we need to define the observation epochs to be given
        to the propagator
        """
                
        # if  max_observations_desired % 8 == 0:
        #     N = max_observations_desired
        # else:
        #     N = max_observations_desired - (max_observations_desired % 8)
            
        
        #Generate the time vec for evaluation
        
        #First evaluate the synodic period
        T_asteroid = 2 * math.pi * np.sqrt( SOL.el_orbs_from_rv(R_small_helio,V_helio ,mu_sun)[0]**3 / mu_sun )
        T_earth = 365.2425*24*60*60
        T_syn = 1 / (1/T_earth-1/T_asteroid)
        
        frac_obs_time_vec = np.linspace(0.85,1,4)
        
        
        offset = fraction * T_syn*u.s  #useless if we set offset = 0
        
        period_start = epoch0 - Deltat + offset
        period_end   = epoch0 + Deltat
        
        


   
        
        
        """
        FILE GENERATION
        """
        
        File_name = f"ADES_{perturbed_asteroid}_obs_true_visibility_{daKmaAU(a_P):.1f}_aP_{max_observations_desired}_Nobs_{noise_RMS*2}_RMS0.txt"
        Im_name =  f"ADES_{perturbed_asteroid}_obs_true_visibility_{daKmaAU(a_P):.1f}_aP_{max_observations_desired}_Nobs_{noise_RMS*2}_RMS0.png"
        Ast_Name = f"{perturbed_asteroid}PE"
       
        with open( f'{Saving_Path}{File_name}', 'w') as f:
       
            f.write("permID |provID     |trkSub  |mode|stn|obsTime                 |ra           |dec          |rmsRA|rmsDec|astCat  |mag  |rmsMag|band|photCat|photAp|logSNR|seeing|exp |rmsFit|nStars|notes|remarks   \n")
            f.close()
                             
                             
    
    
    
    
        """
        SIMULATION AND TRAJECTORY PROPAGATION
    
        """
        
        
        
        #Parallel orbit propagation  = executor.submit(
        if Existing_asteroids == 1:
            state_vec_total, vel_total, time_total, epoch_min_dist, t_plot, distances = propagate('post', R_small_helio, V_helio, RP, VP, epoch0, Deltat, m_P, M_sun, G, perturbers_list, max_observations_desired*2)
            
            plt.figure(figsize=(10, 6))
            plt.plot(t_plot, distances) #, label='Backward Propagation', color='blue')
            plt.title('Distance Between Asteroids Over Time', fontsize=20)
            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Relative Distance', fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{Saving_Path}_distance_plot.png') 
            plt.show()
                        
        
        else:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                
                future_pre = executor.submit( propagate, 'pre' , R_small_helio, V_helio, RP, VP, epoch0, Deltat, m_P, M_sun, G, perturbers_list, max_observations_desired)
                future_post = executor.submit( propagate,  'post', R_small_helio, V_helio, RP, VP, epoch0, Deltat, m_P, M_sun, G, perturbers_list, max_observations_desired)
                
                pos_pre, vel_pre, t_pre, epoch_min_dist_pre, t_plot_pre, distances_pre        = future_pre.result()
                pos_post, vel_post, t_post, epoch_min_dist_post, t_plot_post, distances_post  = future_post.result()
            
                print("Propagation complete!")
                
                
            plt.figure(figsize=(10, 6))
            plt.plot(t_plot_pre, distances_pre) #, label='Backward Propagation', color='blue')
            plt.plot(t_plot_post, distances_post) # , label='Forward Propagation', color='orange')
            plt.title('Distance Between Asteroids Over Time', fontsize=20)
            plt.xlabel('Time', fontsize=18)
            plt.ylabel('Relative Distance', fontsize=18)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{Saving_Path}_distance_plot.png') 
            plt.show()
                        
                    
            
            state_vec_total = np.hstack((pos_pre,pos_post))
            vel_total =       np.hstack((vel_pre,vel_post))
            time_total = Time(np.concatenate((t_pre, t_post)))
        """
        OBSERVATIONS PRINTING (they are already sampled with visibility and magnitude verification in the propagation function)
        """
        
        # if i == 0:
        
            
        S = len(time_total)
        
        print(f'N observations = {S}')
       
        
        if noise_flag == 1:
            noise_ra  = np.random.normal(0, math.radians(noise_RES), S)
            noise_dec = np.random.normal(0, math.radians(noise_RES), S)
        else:
            noise_ra  = np.zeros(S)
            noise_dec = np.zeros(S)
        
  
        SOL.observations(state_vec_total, vel_total, time_total, Saving_Path, File_name, noise_ra = noise_ra, noise_dec = noise_dec, Ast_Name = Ast_Name, obs_code_list = obs_code_list)  #[:,random_indices]
        
        
        fig = ROFL.plot_obs_distrib_sim_ast(time_total.jd, epoch0, Deltat) #Deltat already divided by 2
            
        
        
        fig.savefig(f'{Saving_Path}{Im_name}')
