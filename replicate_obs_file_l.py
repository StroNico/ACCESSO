# -*- coding: utf-8 -*-
"""
Library that improves some functions of SOL, allowing for rading and replicating actual observations files. Included functions to plot the distances and observation distribution.
(I'd prefer using RK_3body from here than from SOL)
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
import equations_of_motion as EOM
from astropy import units as u
from poliastro._math.interpolate import interp1d
from scipy.optimize import minimize_scalar
import topocentric_observations as TO
import spiceypy as spice
import synthetic_obs_l as SOL
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import random
from ephemerides_libr import SV_sun_405, SV_earth_405
from scipy.optimize import minimize


##############################################################################################################################################################################

local_database_folder = "mpcobs_database/"

def dakmaAU(dist_km):
    fatt_trans=149597870.691 
    return dist_km / fatt_trans;

magn_threshold = 220

##############################################################################################################################################################################

def Julianday_greg(D,M,Y):
    a=math.floor((14-M)/12);
    y=Y+4800-a;
    m=M+12*a-3;

    return D+math.floor((153*m+2)/5)+365*y+math.floor(y/4)-math.floor(y/100)+math.floor(y/400)-32045-0.5;

def elongation(sol, t, epoch_0):
    
    date = epoch_0 + t*u.s
    r_ast = np.array([sol[0],sol[1],sol[2]]) - np.array([sol[6],sol[7],sol[8]])
    RE  = SV_earth_405(date)[0] - SV_sun_405(date)[0]
    
    r_earth_ast = r_ast - RE
    
    eps = math.acos(np.dot(-RE , r_earth_ast) / np.linalg.norm(RE) / np.linalg.norm(r_earth_ast))
    
    if eps > math.radians(160)  :#imput the desired angle
        return 1
    else:
        return 0
    
def magnitude(sol, t, epoch_0):
    
    date = epoch_0 + t*u.s
    if sol.shape[0] == 3:
        r_ast = sol
    else:
        r_ast = np.array([sol[0],sol[1],sol[2]]) - np.array([sol[6],sol[7],sol[8]])
    RE  = SV_earth_405(date)[0] - SV_sun_405(date)[0]
    r_earth_ast = r_ast - RE
    
    
    H = 17.20 #H of (209638) 
    G = 0.15
    
    alpha =  math.acos(np.dot(-r_ast , -r_earth_ast) / np.linalg.norm(r_ast) / np.linalg.norm(r_earth_ast))
    
    phi_1 = np.exp(-3.33 * (np.tan(np.radians(alpha) / 2))**0.63)
    phi_2 = np.exp(-1.87 * (np.tan(np.radians(alpha) / 2))**1.22)
    
    r_earth_ast = dakmaAU( np.linalg.norm(r_earth_ast) )
    r_ast = dakmaAU( np.linalg.norm(r_ast) )
    
    m = H + 5 * np.log10(r_ast * r_earth_ast) - 2.5 * np.log10((1 - G) * phi_1 + G * phi_2)
    
    return m
    
##############################################################################################################################################################################



def get_initial_conditions(aP, eP, iP, date, mu_sun):
# Function to get the initial state vector for the perturbing body, given its orbital parameters,
# in order that the close encounter, happening on 'date' is in visibility with Earth
#(this condition can be changed by manipulating possible_span, new_K, new_IJ) 
    

    RE  = SV_earth_405(date)[0] - SV_sun_405(date)[0]
    RE_IJ = np.linalg.norm(RE[0:1])
    ang_K = math.atan2(RE[2], RE_IJ)
    ang_IJ = math.atan2(RE[1], RE[0])
    
    possible_span = np.linspace(math.radians(-10), math.radians(10), 100)
    new_K = ang_K + random.choice(possible_span)
    new_IJ = ang_IJ + random.choice(possible_span)
    new_length_vec = np.linspace(1-eP,1+eP,100)
    new_length = random.choice(new_length_vec)
    
    RP = aP * new_length * np.array([math.cos(new_K)*math.cos(new_IJ), math.cos(new_K)*math.sin(new_IJ), math.sin(new_K)])
    
    # We find the components of vector h
    h_mag = np.sqrt(mu_sun * aP * (1 - eP**2))
    hK = h_mag * math.cos(iP)
    hI = (-RP[0]*RP[2]*hK + np.sqrt((RP[0]*RP[2]*hK)**2 - (RP[1]**2+RP[0]**2) * (RP[1]**2*(hK**2-h_mag**2) + (RP[2]*hK)**2)))/(RP[1]**2+RP[0]**2)
    hJ = - (RP[0]*hI + RP[2]*hK) / RP[1]
    
    h_vec = np.array([hI, hJ, hK])
    
    #We find VP as the inverse of h = r x v
    
    
    v_mag = np.sqrt(2*(mu_sun/(aP * new_length ) - mu_sun/2/aP))
    
    
    
    r_cross = np.array([
    [0, -RP[2], RP[1]],
    [RP[2], 0, -RP[0]],
    [-RP[1], RP[0], 0]
                    ])
    
    # v_initial = np.linalg.pinv(r_cross) @ h_vec
    
    def objective(v):
        v = np.array(v)
        # Constraint: r x v = h
        cross_product_diff = np.cross(RP/(aP * new_length ), v/np.linalg.norm(v)) - h_vec / h_mag
        # Constraint: |v| = v_mag
        magnitude_diff = (np.linalg.norm(v) - v_mag)/v_mag
        # Minimize the sum of squares of the constraint residuals
        return np.sum(cross_product_diff**2) + magnitude_diff**2
    
    v_initial = np.array([1,1,1])
    # Use optimization to find a solution
    result = minimize(objective, v_initial, method='BFGS', options={'gtol': 1e-7})
    if result.success:
        VP = result.x
    else:
        raise ValueError("Velocity search didn't converge")
    
    
    
    return RP, VP




##############################################################################################################################################################################


def read_obs_file(obsFile):
    
    jd_confr = []
    obs_code = []

    start_line = 6
    with open(obsFile) as Data_File:
        
        line_origin=0
        for x in Data_File:

            if line_origin > start_line:
                
                if len(x) > 120 :
                    if x[180:183] != 'C51' and x[180:183] != 'P07':
                    # print(x)
                    # TAKE THE TIME AND CONVERT IN THE SAME FORMAT
                        Date_string=x[17:35]
                        year=float(Date_string.split()[0])
                        month=float(Date_string.split()[1])
                        day=float(Date_string.split()[2])
                        jd_confr.append(Julianday_greg(day,month,year))
                        
                        obs_code.append(x[180:183])
 
            line_origin=line_origin+1
            
    return jd_confr, obs_code
            
def obs_epoch_batch(obsFile):
    jd_confr = []
    obs_code =[]

    start_line = 6
    with open(obsFile) as Data_File:
        
        line_origin=0
        for x in Data_File:

            if line_origin > start_line:
                
                if len(x) > 120 :
                    if x[180:183] != 'C51' and x[180:183] != 'P07':
                    # print(x)
                    # TAKE THE TIME AND CONVERT IN THE SAME FORMAT
                        Date_string=x[17:35]
                        year=float(Date_string.split()[0])
                        month=float(Date_string.split()[1])
                        day=float(Date_string.split()[2])
                        jd_confr.append(Julianday_greg(day,month,year))
                        
                        obs_code.append(x[180:183])
 
            line_origin=line_origin+1
     
            
    N_obs = len(jd_confr)
    N_nights = math.floor(N_obs/4)
    nights_index = np.random.choice(range(len(jd_confr)), size=N_nights, replace=False)
    nights_index = np.sort(nights_index)
    DeltaT = 0.01*u.day
    enlarged_epochs = [epoch + i * DeltaT.value for epoch in np.array(jd_confr)[nights_index] for i in range(4)]
    obs_code = np.array(obs_code)[nights_index]
    enlarged_obs = [obs_code[i // 4] for i in range(len(enlarged_epochs))]
        
     
        #
        
        
    return enlarged_epochs, enlarged_obs
    



def perturbed_file_string(AstName):
    
    if AstName.isdigit():
        #The asteroid is numbered, we get the number, from it the name of the folder, of the file, and the rwo itself
        
        AstNum=int(AstName)
        
        Ast_folder=math.floor(AstNum/1000)
        
        Ast_fileName=(local_database_folder+"numbered/{:04d}/{}.rwo").format(Ast_folder,AstNum)
        # print(AstNum_string)
    
    else:
        
        AstName=((AstName).strip()).replace(" ","")
        # print(AstName)
        
        if "P-L" in AstName:
            #So ther directory is unnumbered//unusual
        
            Ast_fileName=local_database_folder+"unnumbered/unusual/P-L/"+AstName+".rwo"
            
        else:
            
            #We must take the firs 5 digits of AstName
            Ast_folder=AstName[0:5]
            
            Ast_fileName=local_database_folder+"unnumbered/"+Ast_folder+"/"+AstName+".rwo"
            
    return Ast_fileName



def RK_3_body(R, V, RP, VP, epoch0, DeltaT, m, M, G,  perturbers_vec = [], print_min_dist_flag = 0, max_dt=np.inf, evaluation_span = []):

    #Similar to the function defined in SOL, but with the upgrade that it is possible to pre-define the peoch where we want to generate the observations (evolution_span).
    #Particularly useful if we want to replicate the same observations of an actual observation file (e.g., for verification)
    
    masses = [0, m, M]
    rel_pos = []
    
    
    r_sun = np.array([0.,0.,0.])
    initial_state =  np.concatenate([ R, RP, r_sun, V, VP, np.zeros(3)])  # Initial positions and velocities
            
    
    time_span = (0, DeltaT.value)   #Time span for the propagation from 0 to Delta t in seconds
    
    sol = solve_ivp(EOM.fun_3body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec),max_step=max_dt,  rtol=3e-14, atol=3e-14, t_eval = evaluation_span) #max_dt
    

    pos = np.array([sol.y[0,:],sol.y[1,:],sol.y[2,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]])
    vel = np.array([sol.y[9,:],sol.y[10,:],sol.y[11,:]]) - np.array([sol.y[15,:],sol.y[16,:],sol.y[17,:]])
    
    num_cols = pos.shape[1]
    
    for i in range(num_cols):
        
        # print(el_orbs_from_rv(pos[:,i],vel[:,i],G*M))
        # print(pos[:,i],  (epoch0 + sol.t[i]*u.second).jd)
        rel_pos.append(np.linalg.norm(np.array([sol.y[0,i],sol.y[1,i],sol.y[2,i]]) - np.array([sol.y[3,i],sol.y[4,i],sol.y[5,i]])))  # Here we have the vector of the relative positions and below...

    time_vec = epoch0 + sol.t*u.second
    
    if print_min_dist_flag == 1: #...we have to perform an interpolation to see the true moment of the closest approach
        # pos_perturbing = np.array([sol.y[3,:],sol.y[4,:],sol.y[5,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) 
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
        
        print(f"\nClosest approach from Horizons at {min_dist:.2f} km on {time_vec[np.argmin(rel_pos)] + Delta_days_min_dist * u.day}  CANNOT WORK ")
        if m > 0:
            r_SOI = (m/M/3)**(1/3) * np.linalg.norm(pos[:,CA_epoch_0_index])
            print(f"Distance in R_H = {min_dist/r_SOI:.2f}")
    
    


    return pos, vel, time_vec, time_vec[np.argmin(rel_pos)]


def RK_3_body_check_visibility(R, V, RP, VP, epoch0, DeltaT, m, M, G,  perturbers_vec = [], print_min_dist_flag = 0, max_dt=np.inf,  N = 0):
# This function putputs observable epochs only if those satisfy visibility requirements
    
    masses = [0, m, M]
    rel_pos = []
    distances = []
    
    
    r_sun = np.array([0.,0.,0.])
    initial_state =  np.concatenate([ R, RP, r_sun, V, VP, np.zeros(3)])  # Initial positions and velocitie
            
    
    time_span = (0, DeltaT.value)   #Time span for the propagation from 0 to Delta t in seconds 
    sol = solve_ivp(EOM.fun_3body, time_span, initial_state, method='DOP853', args = (G,masses,epoch0,perturbers_vec), dense_output = True, max_step=max_dt,  rtol=3e-14, atol=3e-14) #max_dt
    
    #Here we go densing the output solution to find the acceptable dates for observation and herein weselect the N random observations
    
    timestep_12h = 12 * 3600  # 12 hours in seconds
    if time_span[1] < 0:
        timestep_12h = - timestep_12h

    # Generate time grid spaced every 12 hours
    fine_t = np.arange(time_span[0], time_span[1], timestep_12h)
    
    acceptable_times = []
    for t in fine_t:
        pos = sol.sol(t)  # Interpolated solution at time t
        distance = np.linalg.norm(pos[0:3] - pos[3:6]) 
        distances.append(distance)
        if elongation(pos, t, epoch0) and magnitude(pos,  t, epoch0) < magn_threshold:  
            acceptable_times.append(t)
            # print(magnitude(pos,  t, epoch0))
    # print(fine_t)        
    print(N)        
    if len(acceptable_times) < N:
        N = len(acceptable_times)
    N = int(N)
    print(N)
    
    #plot of the relative distance between the 2 asteroids
    distances = np.array(distances)
    
    # plt.figure(figsize=(10, 6))
    # plt.plot(fine_t, distances, label='Propagation', color='blue')
    # plt.title('Distance Between Asteroids Over Time', fontsize=20)
    # plt.xlabel('Time', fontsize=18)
    # plt.ylabel('Distance', fontsize=18)
    # plt.legend(fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f'distance_plot_{DeltaT}.png')  # Save with a different name based on the mode (pre or post)
    # plt.close()  # Close the plot to free up memory and prevent blocking
    
    
    selected_times = random.sample(acceptable_times, N)
    selected_times = sorted(selected_times)
    
    # Retrieve the corresponding interpolated solutions (sol.y) for selected times
    selected_solutions = [sol.sol(t) for t in selected_times]  # List of position/velocity vectors
    
    selected_solutions = np.array(selected_solutions).T      
    

    pos = np.array([selected_solutions[0,:],selected_solutions[1,:],selected_solutions[2,:]]) - np.array([selected_solutions[6,:],selected_solutions[7,:],selected_solutions[8,:]])
    vel = np.array([selected_solutions[9,:],selected_solutions[10,:],selected_solutions[11,:]]) - np.array([selected_solutions[15,:],selected_solutions[16,:],selected_solutions[17,:]])
    
    num_cols = pos.shape[1]
    
    for i in range(num_cols):
        
        # print(el_orbs_from_rv(pos[:,i],vel[:,i],G*M))
        # print(pos[:,i],  (epoch0 + sol.t[i]*u.second).jd)
        rel_pos.append(np.linalg.norm(np.array([sol.y[0,i],sol.y[1,i],sol.y[2,i]]) - np.array([sol.y[3,i],sol.y[4,i],sol.y[5,i]])))  # Here we have the vector of the relative positions and below...

    time_vec = epoch0 + selected_times*u.second
    
    if print_min_dist_flag == 1: #...we have to perform an interpolation to see the true moment of the closest approach
        # pos_perturbing = np.array([sol.y[3,:],sol.y[4,:],sol.y[5,:]]) - np.array([sol.y[6,:],sol.y[7,:],sol.y[8,:]]) 
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
        
        print(f"\nClosest approach from Horizons at {min_dist:.2f} km on {time_vec[np.argmin(rel_pos)] + Delta_days_min_dist * u.day}  CANNOT WORK ")
        if m > 0:
            r_SOI = (m/M/3)**(1/3) * np.linalg.norm(pos[:,CA_epoch_0_index])
            print(f"Distance in R_H = {min_dist/r_SOI:.2f}")
    
    


    return pos, vel, time_vec, time_vec[np.argmin(rel_pos)], fine_t/(365.2425 * 24 * 60 * 60), distances
    

def observations(pos_vec,  vel_vec, epoch_vec,  Save_path, File_Name, obs_code_list = ['500'], noise_ra = 0, noise_dec = 0, Ast_Name = "2024 AA01"):
    
    if isinstance(noise_ra, (int, float, complex)) or isinstance(noise_dec, (int, float, complex)):
        for i, epoch in enumerate(epoch_vec):
            #for every time-position sampled we find the RADEC positions
            
            # SORT AND DECIDE THE OBSERVER CODE FROM THE OBSERVER CODE VECTOR...
            
            obs_code = obs_code_list[i]
            
            RA, DEC, obs_code = SOL.radec_wrto_earth(pos_vec[:,i], vel_vec[:,i], epoch, obs_code)
            
            mag = magnitude(pos_vec[:,i], 0, epoch)
            #We print these RADEC positions in a file
            
            SOL.print_file_obs(epoch, RA, DEC, Save_path, File_Name, Ast_Name, obs_code, mag)
    else:
        for i, epoch in enumerate(epoch_vec):
            #for every time-position sampled we find the RADEC positions
            
            # SORT AND DECIDE THE OBSERVER CODE FROM THE OBSERVER CODE VECTOR...
            
            obs_code = obs_code_list[i]
            
            RA, DEC, obs_code = SOL.radec_wrto_earth(pos_vec[:,i], vel_vec[:,i], epoch, obs_code)
            
            mag = magnitude(pos_vec[:,i], 0, epoch)
            #We print these RADEC positions in a file
            
            SOL.print_file_obs(epoch, RA + noise_ra[i], DEC + noise_dec[i], Save_path, File_Name, Ast_Name, obs_code, mag)
            
            
def plot_obs_distrib(observation_epochs, perturbing_asteroid, perturbed_asteroid, epoch0):
    
    plt.close('all')
    
    epochi = epoch0
    D_observation_epochs = (observation_epochs - epoch0.jd)*u.day.to(u.second) 
    Deltat = (D_observation_epochs[-1])*u.second
    epochf = epoch0 + Deltat
    epochfin = epochf
    
    # Convert JD days to a pandas DataFrame
    df = pd.DataFrame(observation_epochs, columns=['JD'])

    # Convert JD to integer days for week calculation
    df['Day'] = df['JD'].astype(int)

    # Calculate the week number for each JD (assuming JD starts at day 1)
    df['Week'] = ((df['Day'] - df['Day'].min()) // 7) + 1

    # Count the number of events per week
    weekly_counts = df['Week'].value_counts().sort_index()

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({'Week': range(1, weekly_counts.index.max() + 1)})
    plot_df = plot_df.merge(weekly_counts.rename('Count'), how='left', left_on='Week', right_index=True).fillna(0)

    # Plotting
    fig, ax = plt.subplots(figsize=(70, 6))
    ax.plot(plot_df['Week'], plot_df['Count'], label='Events per Week', color='blue')
    ax.scatter(plot_df['Week'], plot_df['Count'], s=plot_df['Count']*50, color='white', edgecolor='black', linewidth=1.5, label='Events')


    # Adding labels and title
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Events Over Time (per 7-day interval)')
    ax.legend()
    ax.grid(True)
    
    
    '2 - Find the CA date to be added to the plot with hotrizons'

    step = ['24h','6h','5m']
    Delta_central_time = [3*u.day,12*u.hour]

    for i in range(3):

        perturbed = Horizons(id_type='smallbody', id=perturbed_asteroid, location=None, epochs= {'start': epoch0.value, 'stop': epochf.value, 'step': step[i]})
        # eph_perturbed = perturbed.ephemerides()
        sv_perturbed = perturbed.vectors() 
        
        perturbing = Horizons(id_type='smallbody', id=perturbing_asteroid, location=None, epochs= {'start': epoch0.value, 'stop': epochf.value, 'step': step[i]})
        # eph_perturbing = perturbing.ephemerides()
        sv_perturbing = perturbing.vectors()
        
        
        dist = np.sqrt((sv_perturbed['x']-sv_perturbing['x'])**2 +  (sv_perturbed['y']-sv_perturbing['y'])**2 +  (sv_perturbed['z']-sv_perturbing['z'])**2)
        # print(f"{perturbing_asteroid}    {perturbed_asteroid}    {daAUaKm(min(dist)):.4f}  km")
        
        
        # Find the minimum distance and its index
        min_dist_index = np.argmin(dist)
        min_dist = dist[min_dist_index]
        
        # Get the corresponding date for the minimum distance
        min_dist_date = sv_perturbed['datetime_str'][min_dist_index]
        min_dist_date = min_dist_date.replace('A.D. ', '')
        custom_format = '%Y-%b-%d %H:%M:%S.%f'
        min_dist_date = Time.strptime(min_dist_date, custom_format)
        
        if i<2:
        
            epoch0 = min_dist_date - Delta_central_time[i]
            epochf = min_dist_date + Delta_central_time[i]
            
        else:
            min_dist_week = int((min_dist_date.jd -  df['Day'].min()) // 7) + 1
            ax.scatter([min_dist_week], [0], s=200, color='red', label='Minimum Distance', zorder=5)
            ax.annotate(f'CA Date: {min_dist_date.iso}', (min_dist_week, 0), textcoords="offset points", xytext=(10,10), ha='center', fontsize=15, color='red')

        
    before_ca_count = df[df['JD'] < min_dist_date.jd].shape[0]
    after_ca_count = df[df['JD'] >= min_dist_date.jd].shape[0]
    # Add the number of observations before and after the CA date
    ax.annotate(f'Observations before CA: {before_ca_count}\nObservations after CA: {after_ca_count}',
                 (min_dist_week, min(4,max(plot_df['Count']))), textcoords="offset points", xytext=(10,10), ha='center', fontsize=15, color='red')   
    
    
        # Add dashed lines for each year
    start_year = Time(epochi, format='jd').datetime.year
    end_year = Time(epochfin, format='jd').datetime.year
    
    for year in range(start_year, end_year + 1):
        jan_1_date = Time(f"{year}-01-01T00:00:00.000", format='isot', scale='utc')
        jan_1_week = int((jan_1_date.jd -  df['Day'].min()) // 7) + 1
        ax.axvline(x=jan_1_week, color='gray', linestyle='--', linewidth=1.5)
        ax.text(jan_1_week, max(plot_df['Count']), str(year), rotation=90, verticalalignment='bottom', fontsize=11)
        
    return fig

    
def plot_obs_distrib_sim_ast(observation_epochs, epoch0, DeltaT):
    
    # plt.close('all')
    
    epoch_CA = epoch0
    epoch0 = epoch0 - DeltaT
    epochi = epoch0
    D_observation_epochs = (observation_epochs - epoch0.jd)*u.day.to(u.second) 
    Deltat = (D_observation_epochs[-1])*u.second
    epochf = epoch0 + Deltat
    epochfin = epochf
    
    # Convert JD days to a pandas DataFrame
    df = pd.DataFrame(observation_epochs, columns=['JD'])

    # Convert JD to integer days for week calculation
    df['Day'] = df['JD'].astype(int)

    # Calculate the week number for each JD (assuming JD starts at day 1)
    df['Week'] = ((df['Day'] - df['Day'].min()) // 7) + 1

    # Count the number of events per week
    weekly_counts = df['Week'].value_counts().sort_index()

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({'Week': range(1, weekly_counts.index.max() + 1)})
    plot_df = plot_df.merge(weekly_counts.rename('Count'), how='left', left_on='Week', right_index=True).fillna(0)

    # Plotting
    fig, ax = plt.subplots (figsize=(70, 6))
    ax.plot(plot_df['Week'], plot_df['Count'], label='Events per Week', color='blue')
    ax.scatter(plot_df['Week'], plot_df['Count'], s=plot_df['Count']*50, color='white', edgecolor='black', linewidth=1.5, label='Events')


    # Adding labels and title
    ax.set_xlabel('Week Number')
    ax.set_ylabel('Number of Events')
    ax.set_title('Distribution of Events Over Time (per 7-day interval)')
    ax.legend()
    ax.grid(True)
    # ax.show()
    
    '2 - Find the CA date to be added to the plot with hotrizons'

    min_dist_date = epoch_CA
    min_dist_week = int((min_dist_date.jd -  df['Day'].min()) // 7) + 1
    ax.scatter([min_dist_week], [0], s=200, color='red', label='Minimum Distance', zorder=5)
    ax.annotate(f'CA Date: {min_dist_date.iso}', (min_dist_week, 0), textcoords="offset points", xytext=(10,10), ha='center', fontsize=15, color='red')

        
    before_ca_count = df[df['JD'] < min_dist_date.jd].shape[0]
    after_ca_count = df[df['JD'] >= min_dist_date.jd].shape[0]
    # Add the number of observations before and after the CA date
    ax.annotate(f'Observations before CA: {before_ca_count}\nObservations after CA: {after_ca_count}',
                 (min_dist_week, min(4,max(plot_df['Count']))), textcoords="offset points", xytext=(10,10), ha='center', fontsize=15, color='red')   
    
    
        # Add dashed lines for each year
    start_year = Time(epochi, format='jd').datetime.year
    end_year = Time(epochfin, format='jd').datetime.year
    
    for year in range(start_year, end_year + 1):
        jan_1_date = Time(f"{year}-01-01T00:00:00.000", format='isot', scale='utc')
        jan_1_week = int((jan_1_date.jd -  df['Day'].min()) // 7) + 1
        ax.axvline(x=jan_1_week, color='gray', linestyle='--', linewidth=1.5)
        ax.text(jan_1_week, max(plot_df['Count']), str(year), rotation=90, verticalalignment='bottom', fontsize=11)
        
    
    return fig

    
       
    