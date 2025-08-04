# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:26:39 2024

@author: s368232
"""

import os
import numpy as np
from astropy.time import Time
import shutil
from astropy import units as u
import math
from poliastro.ephem import Ephem, ephem_interpolant
from poliastro.frames.enums import Planes
from poliastro.bodies import  Sun
import equations_of_motion as EOM
import synthetic_obs_l as SOL
from scipy.integrate import solve_ivp
import spiceypy as spice
import time
import fileinput
from decimal import Decimal
import matplotlib.pyplot as plt


# n_ast = 151 #Eros in order of file mu.txt
folder_fo = "Folder/to/find_orb/find_c64"
folder_ephem_file = "Ast_ephem"

filename = "asteroid_ephemeris_mod.txt"

def find_ephem_line(n_ast, epoch_index):
    n_ast -= 1
    a_line = 300*epoch_index*6 + 6*n_ast 
    return a_line

def E_from_nu(nu,ecc):
    
    cos_E=(ecc+math.cos(nu))/(1+ecc*math.cos(nu))
    
    E=math.acos(cos_E)
    
    if nu < 0:
        E=2*math.pi-E
        
    return E

def get_precision(file_line):
    value_str = file_line.strip().split('/')[0]
    if '.' in value_str:
        return len(value_str.split('.')[1])  # Number of digits after the decimal point
    else:
        return 0  # No decimal point, so no digits after the decimal point


def M_from_nu(nu,ecc):
    
    M = E_from_nu(nu,ecc) - ecc*math.sin(E_from_nu(nu,ecc))
    
    return M

def daKmaAU(dist_km):
    fatt_trans=149597870.691 
    return dist_km/fatt_trans;


def find_closest_value_and_index(value, vector):
    # Calculate the absolute differences between the value and each element in the vector
    
    target_jd = value.jd
    time_jd = np.array([t for t in vector])
    
    absolute_diff = np.abs(time_jd - target_jd)
    
    # Find the index of the minimum absolute difference
    closest_index = np.argmin(absolute_diff)
    
    # Get the value corresponding to the closest index
    closest_value_jd = vector[closest_index]
    
    return closest_value_jd, closest_index


def copy_and_rename_file(source_file, new_filename):
    # Extract the destination directory from the source file path
    destination_directory = os.path.dirname(source_file)
    
    # Copy the file to the destination directory
    copied_file = os.path.join(destination_directory, new_filename)
    
    shutil.copy(source_file, copied_file)


def Ephem_gen(R, V, epoch0, DeltaT, m, M, G, perturbing_asteroid, Existing_asteroids,  perturbers_vec = [], max_dt=np.inf):
    
    #Define paramters
    epoch1 = Time("1799-12-30 12:00:00", scale="utc")
    epoch2 = Time("2200-01-22 12:00:00", scale="utc")
    step_size = 40*u.day
    epoch_vector = np.arange(epoch1.jd, epoch2.jd+40, 40)
    
    
    #I need to find the points of epoch0-Deltat-10 years (and then +) to evaluate the ephemerides. The points have to be the same in epoch_vector
    
    if   Existing_asteroids == 1:
        start_date, st_date_index = find_closest_value_and_index((epoch0 - 1*u.year), epoch_vector)
        end_date,  end_date_index = find_closest_value_and_index((epoch0 + DeltaT + 1*u.year), epoch_vector)
    elif Existing_asteroids == 0:
        start_date, st_date_index = find_closest_value_and_index((epoch0 - DeltaT - 1*u.year), epoch_vector)
        end_date,  end_date_index = find_closest_value_and_index((epoch0 + DeltaT + 1*u.year), epoch_vector)

    masses = [m, M]
    # PROPAGATION 1: back in time until the start_date
   
    r_sun = np.array([0.,0.,0.])
    initial_state =  np.concatenate([ R, r_sun, V,  np.zeros(3)])  # Initial positions and velocities
    
    time_span = (0, ((Time(start_date, format = "jd") - epoch0).to(u.second)).value)   #Time span for the propagation from 0 to epochf - epoch0 in seconds
    

    
    sol = solve_ivp(EOM.fun_2body, time_span, initial_state, method='DOP853',
                    args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-10, atol=3e-10) 
    
    # PROPAGATION 2: from this new starting point to 10 years after the end to get the points
    
    initial_state =  sol.y[:,-1]  # Initial positions and velocities
    time_points = np.array([((t - start_date)*u.day.to(u.second)) for t in epoch_vector[st_date_index:end_date_index]])
    time_span = (0, ((end_date - start_date)*u.day.to(u.second)))   #Time span for the propagation

    start_epoch = Time(start_date, format = "jd")
    start_epoch.format = 'datetime'

    sol = solve_ivp(EOM.fun_2body, time_span, initial_state, method='DOP853',
                    args = (G, masses, start_epoch ,perturbers_vec),max_step=max_dt,  rtol=3e-10, atol=3e-10,  t_eval=time_points) 


 
    pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[3,:],sol.y[4,:],sol.y[5,:]])
    vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) - np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]])
    
    num_cols = pos.shape[1]
    
    #Take the ephemerides from pos-vel  ---> Heliocentric elements
    time_vec = Time(start_date, format = "jd") + sol.t*u.second
    
    with open(folder_ephem_file + '/asteroid_ephemeris_original.txt','r')  as file:
        lines = file.readlines()
        
        
        for i in range(num_cols):
            
            
            el_orbs = SOL.el_orbs_from_rv(pos[:,i],vel[:,i],G*M)
            
            #Now we have to print the ephemerides in a file 
            # Then we have to print these elements in the correct place on the mu.txt
            
         
            
            new_values = [daKmaAU(el_orbs[0]), el_orbs[1], el_orbs[2], el_orbs[4], el_orbs[3], M_from_nu(el_orbs[5],el_orbs[1])]
            
            

            for j in range(len(new_values)):

                
                
                lines[find_ephem_line(151, st_date_index) + i*300*6 + j ] = "{:.{}g}\n".format(Decimal(new_values[j]), 16)

                        
    with open(folder_ephem_file + '/asteroid_ephemeris.txt', 'w') as file:
        file.writelines(lines)
        
    # plt.plot(epoch_vector,M_vec)
    # plt.show()
        
    copy_and_rename_file(folder_ephem_file + '/asteroid_ephemeris.txt', folder_ephem_file + f'/asteroid_ephemeris_{perturbing_asteroid}_short.txt')
            
        
def modify_mass_file_mu1(asteroid_number, new_mass):
    
    
    os.remove(folder_fo + '/mu1.txt')
    shutil.copyfile(folder_fo + '/mu1_original.txt', folder_fo + '/mu1.txt')
    
    
    
    with open(folder_fo + '/mu1.txt','r') as file:
    
    
        lines = file.readlines()

    # Find the line containing the desired asteroid number
    for i, line in enumerate(lines):
        if i > 7:
            parts = line.split()
            current_number = int(parts[0])
            if current_number == asteroid_number:
                old_mass = parts[1]
                parts[1] = f'{new_mass: .3e}'  # Modify the mass value
                line = line.replace(old_mass, parts[1])  # Update the line with modified mass value
                lines[i] = line  # Replace the line in the lines list
                break  # Stop searching furthe
    
    # Write the modified lines back to the file
    with open(folder_fo + '/mu1.txt' ,'w') as file:
        file.writelines(lines)
                

########################### GENERATION "LONG" WITHOUT COSTRAINING WITHIN THE ACTUAL EPOCHS OF PROPAGATION OF THE 3 BODY PROBLEM ############

def Ephem_gen_long(R, V, epoch0, DeltaT, m, M, G, perturbing_asteroid, Existing_asteroids,  perturbers_vec = [], max_dt=np.inf):
    
    #Define paramters
    epoch1 = Time("1799-12-30 12:00:00", scale="utc")
    epoch2 = Time("2200-01-22 12:00:00", scale="utc")
    step_size = 40*u.day
    epoch_vector = np.arange(epoch1.jd, epoch2.jd+40, 40)
    
    
    #I need to find the points of epoch0-Deltat-10 years (and then +) to evaluate the ephemerides. The points have to be the same in epoch_vector
    
    start_date, st_date_index = epoch1, 0 
    end_date,  end_date_index = epoch2, len(epoch_vector)-1

    masses = [m, M]
    # PROPAGATION 1: back in time until the start_date
   
    r_sun = np.array([0.,0.,0.])
    initial_state =  np.concatenate([ R, r_sun, V,  np.zeros(3)])  # Initial positions and velocitie
    
    time_span = (0, ((start_date-epoch0).to(u.second)).value)   #Time span for the propagation from 0 to epochf - epoch0 in seconds
    

    
    sol = solve_ivp(EOM.fun_2body, time_span, initial_state, method='DOP853',
                    args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-10, atol=3e-10) #max_dt
    
    # time_vec = epoch0 + sol.t*u.second
    # PROPAGATION 2: from this new starting point to 10 years after the end to get the points
    
    initial_state =   sol.y[:,-1]  # Initial positions and velocitie
    time_points = np.array([((Time(t, format = "jd",scale="utc") - start_date).to(u.second).value) for t in epoch_vector]) 
    time_span = (0, ((end_date - start_date).to(u.second)).value)   #Time span for the propagation

    
    sol = solve_ivp(EOM.fun_2body, time_span, initial_state, method='DOP853',
                    args = (G,masses,start_date,perturbers_vec),max_step=max_dt,  rtol=3e-10, atol=3e-10,  t_eval=time_points) #max_dt


 
    pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[3,:],sol.y[4,:],sol.y[5,:]])
    vel = np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) - np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]])
    
    num_cols = pos.shape[1]
    
    #Take the ephemerides from pos-vel  ---> Heliocentric elements
    # time_vec = time_vec[-1] + sol.t*u.second
    
   
    
    with open(folder_ephem_file + '/asteroid_ephemeris_original.txt','r')  as file:
        lines = file.readlines()
        
        
        for i in range(num_cols):
            
            # print(el_orbs_from_rv(pos[:,i],vel[:,i],G*M))
            el_orbs = SOL.el_orbs_from_rv(pos[:,i],vel[:,i],G*M)
            
            #Now we have to print the ephemerides in a file (just to test if everything is ok)
            # Then we have to print these elements in the correct place on the mu.txt
            
         
            
            new_values = [daKmaAU(el_orbs[0]), el_orbs[1], el_orbs[2], el_orbs[4], el_orbs[3], M_from_nu(el_orbs[5],el_orbs[1])]
            
            

            for j in range(len(new_values)):
                # if find_ephem_line(151, st_date_index) + i*300*6 + j == 901:
                #     print(find_ephem_line(151, st_date_index) + i*300*6 + j)
                
                
                lines[find_ephem_line(151, st_date_index) + i*300*6 + j ] = "{:.{}g}\n".format(Decimal(new_values[j]), 16)
                        
    with open(folder_ephem_file + '/asteroid_ephemeris.txt', 'w') as file:
        file.writelines(lines)
        
    # plt.plot(epoch_vector,M_vec)
    # plt.show()
        
    copy_and_rename_file(folder_ephem_file + '/asteroid_ephemeris.txt', folder_ephem_file + f'/asteroid_ephemeris_{perturbing_asteroid}_long.txt')