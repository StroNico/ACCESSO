# -*- coding: utf-8 -*-
"""
Library for the synthetic observations

Propagation
Sampling
Positioning
Earth positioning (ephem)
"""


import numpy as np
import math
from scipy.integrate import solve_ivp
from astropy import units as u
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates.earth_orientation import obliquity
import spiceypy as spice
from ephemerides_libr import SV_sun_405, SV_earth_405
import equations_of_motion as EOM
from poliastro._math.interpolate import interp1d
from astropy.coordinates import Longitude, EarthLocation
from scipy.optimize import minimize_scalar
import topocentric_observations as TO
from astropy import units as uts
import replicate_obs_file_l as ROFL
import matplotlib.pyplot as plt
import random


solar_system_ephemeris.set("jpl")

c_light = 299792.458    #km/s
leapsecond = 67*u.second
###############################################################################################################################
obliquity_j2000 = obliquity( 2451545.) # mean obliquity of the ecliptic at J2000.0
sin_obliq_2000 = 0.397777155931913701597179975942380896684
cos_obliq_2000 = 0.917482062069181825744000384639406458043
# rot_equat_to_eclip = rotation_matrix( obliquity_j2000, 'x') #rotation matrix from equatorial to ecliptic frames
# rot_eclip_to_equat = rotation_matrix(-obliquity_j2000, 'x')
rot_eclip_to_equat =  np.array([[1, 0, 0],  
                                [0, cos_obliq_2000, -sin_obliq_2000], 
                                [0, sin_obliq_2000, cos_obliq_2000]])

###############################################################################################################################



###############################################################################################################################

# ###############################################################################################################################
# def rot_eclip_to_equat(JD):
#     obliquity_j2000 = obliquity(JD) # mean obliquity of the ecliptic at J2000.0
#     # rot_equat_to_eclip = rotation_matrix( obliquity_j2000, 'x') #rotation matrix from equatorial to ecliptic frames
#     return rotation_matrix(-obliquity_j2000, 'x')
# ###############################################################################################################################

bary = 0  #leave it, it will always be heliocentric (sun in 0,0,0)



MAX_FILES = 5300

def L1(x):
    return np.matrix([[1, 0, 0] ,[0, math.cos(x) ,math.sin(x)] ,[ 0 ,-math.sin(x), math.cos(x)]])


def L3(x):
    return np.matrix([[math.cos(x), math.sin(x), 0] ,[-math.sin(x) ,math.cos(x), 0] ,[0, 0 ,1]])

def daKmaAU(dist_km):
    fatt_trans=149597870.691 
    return dist_km/fatt_trans;



def zeroto2pi(alpha_rad):
    if alpha_rad >= 0:
        while alpha_rad >= 2*math.pi:
            alpha_rad = alpha_rad - 2*math.pi
    else:
        while alpha_rad < 0:
            alpha_rad = alpha_rad + 2*math.pi
    return alpha_rad




#THE USE OF THIS FUCNTION DEPENDS ON THE VERSION OF FIND_ORB USED (OR ANY OTHER ORBIT DETERMINATION SOFTWARE)

def include_thrown_in_planets(r):
    M_central_body = 0 
    m_planets = np.array([1, 1.660114153054348e-07  ,  2.447838287784771e-06  ,   3.040432648022641e-06 , 
                          3.227156037554996e-07 ,9.547919101886966e-04,2.858856727222416e-04, 4.366249662744965e-05,
                          5.151383772628673e-05 ,  7.350487833457740e-09   ])
    dist_planets_au = np.array([0., .38709927, .72333566, 1.00000261,
                  1.52371034, 5.20288799, 9.53667594,  19.18916464,  30.06992276])
    
    r = daKmaAU(r)
    
    for i, dist in enumerate (dist_planets_au):
        if r > 1.2*dist:
            M_central_body += m_planets[i]
        elif r > dist:
            
            f =    ((1.2-r/dist)/0.2)  #if it is with a recent find_orb non interactive (r / dist - 1.) / 0.2
            M_central_body += m_planets[i] * f
            # print(r/dist)
            # print(f)
    
    return M_central_body



################################################################################################################################



def RK_2_body(R, V, epoch0, DeltaT, m, M, G,  perturbers_vec = [], max_dt=np.inf):

    
    masses = [0, M]
   
    
    if bary == 1 :

        r_sun = SV_sun_405(epoch0)[0]
        vSUN =  SV_sun_405(epoch0)[1]
        initial_state =  np.concatenate([ R + r_sun  ,  r_sun  , V  + vSUN, vSUN])  #   # Initial positions and velocities   
    else:
        r_sun = np.array([0.,0.,0.])
        initial_state =  np.concatenate([ R, r_sun, V,  np.zeros(3)])  # Initial positions and velocities
            
    
    time_span = (0, DeltaT.value)   #Time span for the propagation from 0 to Delta t in seconds
    
    sol = solve_ivp(EOM.fun_2body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-12, atol=3e-12) 
    

    if bary == 1 :
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[3,:],sol.y[4,:],sol.y[5,:]])
        vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) - np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]])

        
    else:
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) 
        vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) 
    
    num_cols = pos.shape[1]
    
    time_vec = epoch0 + sol.t*u.second
    
    return pos, vel, time_vec
    


def RK_3_body(R, V, RP, VP, epoch0, DeltaT, m, M, G,  perturbers_vec = [], print_min_dist_flag = 0, max_dt=np.inf):

    
    masses = [0, m, M]
    rel_pos = []
    
    if bary == 1:

        r_sun = SV_sun_405(epoch0)[0]
        vSUN =  SV_sun_405(epoch0)[1]
        initial_state =  np.concatenate([ R + r_sun  ,  r_sun  , V  + vSUN, vSUN])    # Initial positions and velocities    
    else:
        r_sun = np.array([0.,0.,0.])
        initial_state =  np.concatenate([ R, RP, r_sun, V, VP, np.zeros(3)])  # Initial positions and velocities
            
    
    time_span = (0, DeltaT.value)   #Time span for the propagation from 0 to Delta t in seconds
    
    sol = solve_ivp(EOM.fun_3body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-14, atol=3e-14) 
    
    if bary == 1:
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]])
        vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) - np.array([sol.y[15,:],sol.y[16,:],sol.y[17,:]])

        
    else:
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]])
        vel = np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]]) - np.array([sol.y[15,:],sol.y[16,:],sol.y[17,:]])
    
    num_cols = pos.shape[1]
    
    for i in range(num_cols):
        

        rel_pos.append(np.linalg.norm(np.array([sol.y[0,i],sol.y[1,i],sol.y[2,i]]) - np.array([sol.y[3,i],sol.y[4,i],sol.y[5,i]])))  # Here we have the vector of the relative positions and below...

    time_vec = epoch0 + sol.t*u.second
    
    if print_min_dist_flag == 1: #...we have to perform an interpolation to see the true moment of the closest approach
       
        CA_epoch_0_index = np.argmin(rel_pos)
        # We need to inteprolate the distance 4 indexes before and after this one...
        Delta_index = 4
        Delta_days = min(abs((time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd[0]),
        abs((time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd[-1]))
        
        interpolant = interp1d(
            (time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd,
            rel_pos[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1],
        )
        
        def objective_function(x):
            return interpolant(x)
        
        result = minimize_scalar(objective_function, bounds=(-Delta_days, Delta_days))
        min_dist = result.fun
        Delta_days_min_dist = result.x
        
        print(f"\nClosest approach from Horizons at {min_dist:.2f} km on {time_vec[np.argmin(rel_pos)] + Delta_days_min_dist * u.day}  ")
        if m > 0:
            r_SOI = (m/M/3)**(1/3) * np.linalg.norm(pos[:,CA_epoch_0_index])
            print(f"Distance in R_H = {min_dist/r_SOI:.2f}")
    
    


    return pos, vel, time_vec, time_vec[np.argmin(rel_pos)]
    


def RK_3_body_1_way(R, V, RP, VP, epoch0, DeltaT, m, M, G,  perturbers_vec = [], print_min_dist_flag = 0, max_dt=np.inf):
    
    masses = [0, m, M]
    rel_pos = []
    
    if bary == 1:

        r_sun = SV_sun_405(epoch0)[0]
        vSUN =  SV_sun_405(epoch0)[1]
        initial_state =  np.concatenate([ R + r_sun  ,  r_sun  , V  + vSUN, vSUN])  # Initial positions and velocities     
    else:
        r_sun = np.array([0.,0.,0.])
        initial_state =  np.concatenate([ R, RP, r_sun, V, VP, np.zeros(3)])  # Initial positions and velocitie
            
    
    time_span = (0, -DeltaT.value)   #Time span for the propagation from 0 to Delta t in seconds
    
    sol = solve_ivp(EOM.fun_3body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-15, atol=3e-15) #max_dt
    
    initial_state =   sol.y[:,-1]
    time_vec = epoch0 + sol.t*u.second
    epoch0 = time_vec[-1]
    time_span = (0, 2 * DeltaT.value)   #Time span for the propagation
    
    sol = solve_ivp(EOM.fun_3body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-16, atol=3e-16) #max_dt
    
    
    
    
    if bary == 1:
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]])
        vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) - np.array([sol.y[15,:],sol.y[16,:],sol.y[17,:]])

        
    else:
        pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]])
        vel = np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]]) - np.array([sol.y[15,:],sol.y[16,:],sol.y[17,:]])
    
    num_cols = pos.shape[1]
    
    for i in range(num_cols):
        
       
        rel_pos.append(np.linalg.norm(np.array([sol.y[0,i],sol.y[1,i],sol.y[2,i]]) - np.array([sol.y[3,i],sol.y[4,i],sol.y[5,i]])))  # Here we have the vector of the relative positions and below...

    time_vec = epoch0 + sol.t*u.second
    
    if print_min_dist_flag == 1: #...we have to perform an interpolation to see the true moment of the closest approach
        
        CA_epoch_0_index = np.argmin(rel_pos)
      
        Delta_index = 4
        Delta_days = min(abs((time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd[0]),
        abs((time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd[-1]))
        
        interpolant = interp1d(
            (time_vec[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1] - time_vec[CA_epoch_0_index]).jd,
            rel_pos[CA_epoch_0_index - Delta_index:CA_epoch_0_index + Delta_index + 1],
        )
        
        def objective_function(x):
            return interpolant(x)
        
        result = minimize_scalar(objective_function, bounds=(-Delta_days, Delta_days))
        min_dist = result.fun
        Delta_days_min_dist = result.x
        
        print(f"\nClosest approach from Horizons at {min_dist:.2f} km on {time_vec[np.argmin(rel_pos)] + Delta_days_min_dist * u.day}  ")
        if m > 0:
            r_SOI = (m/M/3)**(1/3) * np.linalg.norm(pos[:,CA_epoch_0_index])
            print(f"Distance in R_H = {min_dist/r_SOI:.2f}")
    
    


    return pos, vel, time_vec, time_vec[np.argmin(rel_pos)]
    
################################################################################################################################
  
    
def plot_earth_asteroid(RE, pos, r_obs):
    # Calculate rho_vec = pos - RE (the vector connecting the Earth to the asteroid)
    rho_vec = pos - RE


    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plot circles representing the orbits of the Earth and the asteroid
    earth_circle = plt.Circle((0, 0), np.linalg.norm(RE), color='blue', fill=False, linestyle='--', label='Earth Orbit')
    asteroid_circle = plt.Circle((0, 0), np.linalg.norm(pos), color='red', fill=False, linestyle='--', label='Asteroid Orbit')
    ax.add_artist(earth_circle)
    ax.add_artist(asteroid_circle)

    # Plot Earth and asteroid positions
    ax.scatter(RE[0], RE[1], color='blue', label='Earth (RE)')
    ax.scatter(pos[0], pos[1], color='red', label='Asteroid (pos)')

    # Draw the radius vectors from the origin to RE and pos
    ax.plot([0, RE[0]], [0, RE[1]], color='blue', linestyle='-', label='Radius vector to Earth (RE)')
    ax.plot([0, pos[0]], [0, pos[1]], color='red', linestyle='-', label='Radius vector to Asteroid (pos)')

    # Plot rho_vec connecting Earth to asteroid
    ax.plot([RE[0], pos[0]], [RE[1], pos[1]], color='green', linestyle='-', label='Vector rho (pos - RE)')
    
    # Plot the r_obs vector, centered at RE
    r_obs_scaled = r_obs * 1000000  # Scale for visualization purposes
    ax.quiver(RE[0], RE[1], r_obs_scaled[0], r_obs_scaled[1], color='purple', scale=1, scale_units='xy', angles='xy', label='r_obs (surface normal)')


    # Set equal scaling and add grid
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Set labels and title
    ax.set_xlabel('X (AU or km)')
    ax.set_ylabel('Y (AU or km)')
    ax.set_title('Earth and Asteroid Positions in XY Plane')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()
  


################################################################################################################################
    
def radec_wrto_earth(pos_0, vel, date, obs_code_list_original):
    
    "We start a loop on the obs_code_list ot find one obs_code that allows for having an observation in the desired epoch (no under horizon)"
    
    if bary == 1:
        RE_centre  = SV_earth_405(date)[0] - SV_sun_405(date)[0]
       
    else:
        RE_centre  = SV_earth_405(date)[0] - SV_sun_405(date)[0]
       
    #equatorial
    # rho_vec = np.matmul(rot_eclip_to_equat, pos) - RE 
    # pos = pos + SV_sun_405(date)[0]
    # vel = vel + SV_sun_405(date)[1]
    
    #IF YOU WANT TO PLOT:
    # plot_earth_asteroid(RE_centre, pos)
    if not isinstance(obs_code_list_original, list):
        obs_code_list_original = [obs_code_list_original]

    random.shuffle(obs_code_list_original)
    obs_code_list = obs_code_list_original[:]
    obs_code_list.append('500')
    
    for obs_code in obs_code_list:
        
        
    
        
        RE = RE_centre  +  TO.observerpos_mpc(obs_code, date - leapsecond) 
        
    
        #ecliptic
        rho_vec = pos_0 - RE
        
        "Still in ecliptic frame I add a light-time correction, with a first order approximation of the motion of the asteroid"
        rel_dist = np.linalg.norm(rho_vec)
        Dt_light = rel_dist / c_light
        # if bary == 1:
        pos = pos_0 - vel * Dt_light
        rho_vec = pos - RE
        
        
        "End light correction"
        
        
        
        rho_vec = np.matmul(rot_eclip_to_equat, rho_vec)
        
        rho_vec_norm = np.linalg.norm(rho_vec, ord=2)
        rho_vec_unit = rho_vec/rho_vec_norm
        
        cosd_cosa = rho_vec_unit[0]
        cosd_sina = rho_vec_unit[1]
        sind = rho_vec_unit[2]
        
        ra_comp = np.mod(np.arctan2(cosd_sina, cosd_cosa), 2.0*np.pi)
        dec_comp = np.arcsin(sind)
        
        # r_obs = RE - RE_centre
        # plot_earth_asteroid(RE_centre, pos, r_obs / np.linalg.norm(r_obs))
        
        if obs_code != '500':
        
            #CHECK VISIBILITY FROM STATION
            
            # # Observer's latitude (in radians)
            # lat = math.radians(float(TO.find_values(obs_code)["Latitude"]))
            # #lONGITUDE
            # long_site = Longitude( float(TO.find_values(obs_code)["Longitude"]), uts.degree, wrap_angle=360.0*uts.degree)
            # # Local Sidereal Time (LST) in radians (already computed in your observerpos_mpc function)
            # lst = (date - leapsecond).sidereal_time('mean', longitude=long_site).rad
            # # Hour Angle (HA) in radians
            # ha = lst - ra_comp
            # # Altitude calculation (above horizon)
            # altitude = np.arcsin(np.sin(dec_comp) * np.sin(lat) + np.cos(dec_comp) * np.cos(lat) * np.cos(ha))
            # # Convert altitude to degrees (optional)
            # altitude_deg = math.degrees(altitude)
            
            # if altitude_deg > 20:
            r_obs = RE - RE_centre
                
            
            n_obs = r_obs / np.linalg.norm(r_obs) #normal to he surface at the station in helio ecliptic
            n_obs = np.matmul(rot_eclip_to_equat, n_obs)  #put in equatorial
            
            
            
            cos_theta = np.dot(n_obs, rho_vec_unit)
            
            #check visibility with a ground elevation of 10 degrees
            if cos_theta > np.sin(math.radians(10)):
                
                # plot_earth_asteroid(RE_centre, pos, r_obs / np.linalg.norm(r_obs))
                # print(ra_comp, dec_comp, obs_code)
                return ra_comp, dec_comp, obs_code
            
        else:
            #If nothing is achieving visibility, the observation goes in "500": geocentric
            return ra_comp, dec_comp, obs_code
            
    

    return ra_comp, dec_comp

def print_file_obs(date,ra,dec,Master_path, File_Name, Ast_Name = "2024 AA01", obs_code = "500", mag = 0):
    
    date = date - leapsecond
    
    filename = f'{Master_path}{File_Name}'
    
    with open(filename, 'a') as f:

        
        f.write(("       |{}  |        | CCD|{}|{}Z|{:13.7f}|{:13.7f}|     |      |        |{:5.1f}|      |    |       |      |      |      |    |      |      |     |\n").format(Ast_Name,obs_code,date.isot,math.degrees(ra),math.degrees(dec),mag))
        

        f.close()
    
    
    
def observations(pos_vec,  vel_vec, epoch_vec,  Save_path, File_Name, obs_code_list = ['500'], noise_ra = 0, noise_dec = 0, Ast_Name = "2024 AA01"):
    
    if isinstance(noise_ra, (int, float, complex)) or isinstance(noise_dec, (int, float, complex)):
        for i, epoch in enumerate(epoch_vec):
            #for every time-position sampled we find the RADEC positions
            
            # SORT AND DECIDE THE OBSERVER CODE FROM THE OBSERVER CODE VECTOR...
            
            obs_code = random.choice(obs_code_list)
            
            RA, DEC = radec_wrto_earth(pos_vec[:,i], vel_vec[:,i], epoch, obs_code)
            
            
            mag = ROFL.magnitude(pos_vec[:,i], 0, epoch)
            #We print these RADEC positions in a file
            
            print_file_obs(epoch, RA, DEC, Save_path, File_Name, Ast_Name, obs_code, mag)
    else:
        #(this side of the if turns out to be obsolete with the condition noise_ra = 0, but still stands)
        
        for i, epoch in enumerate(epoch_vec):
            #for every time-position sampled we find the RADEC positions
            
            # SORT AND DECIDE THE OBSERVER CODE FROM THE OBSERVER CODE VECTOR...
            
            # obs_code = random.choice(obs_code_list)
            
            RA, DEC, obs_code = radec_wrto_earth(pos_vec[:,i], vel_vec[:,i], epoch, obs_code_list)
            mag = ROFL.magnitude(pos_vec[:,i], 0, epoch)
            
            #We print these RADEC positions in a file
            # print( RA, DEC, obs_code)
            print_file_obs(epoch, RA + noise_ra[i], DEC + noise_dec[i], Save_path, File_Name, Ast_Name, obs_code, mag)


spice.unload('naif0012.tls')



########################################################################################################################################################################


def el_orbs_from_rv(r_vec,v_vec,mu):
    

    K=np.array([0,0,1])

    r=np.linalg.norm(r_vec)
    v=np.linalg.norm(v_vec)
    
    a=-mu/2*(1/(v**2/2-mu/r))
    phi=math.asin(np.dot(r_vec,v_vec)/(r*v))
    h_ijk=np.cross(r_vec,v_vec)
    h=np.linalg.norm(h_ijk)
    p=h**2/mu
    
    if abs(p-a)/a < 1e-8:
        ecc = 0
    else:
        
        ecc=np.sqrt(1-p/(a))
        
    if ecc == 0:
        nu = 0
    else:
        
        if abs(phi) <  1e-8:
            if abs(1-p/(1+ecc)) < 1e-8:
                nu = 0
            else:
                nu = math.pi
        else:

            x=(p-r)/(ecc*r)
        
            nu=math.acos(x)
            if phi<0:
                nu=-nu
        
    n=np.cross(K,h_ijk)
      
    i=math.acos(h_ijk[2]/h)
    
   
    if i != 0.0:
        Omega=math.acos(n[0]/np.linalg.norm(n))
        if n[1]<0:
            Omega=2*math.pi-Omega
    else:
        Omega = 0
        n = np.array([1,0,0])
        
    if ecc == 0:
        omega = 0
    else:
        
            
        e=1/mu*((v**2-mu/r)*r_vec-(np.dot(r_vec,v_vec))*v_vec) 
        
    
        omega=math.acos(np.dot(n,e)/(np.linalg.norm(e)*np.linalg.norm(n)))
        if e[2]<0:
          omega=2*math.pi-omega

    return a,ecc,i,Omega,omega,nu,phi    

