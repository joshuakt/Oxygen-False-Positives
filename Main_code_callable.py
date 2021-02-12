# This script contains the forward model

#####################
import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import scipy.optimize 
from radiative_functions import *
from other_functions import *
from stellar_funs import main_sun_fun
from carbon_cycle_model import *
from escape_functions import *
from all_classes import *
from outgassing_module import *
from Albedo_module import *
from thermodynamic_variables import *
import time
from numba import jit
#####################


def forward_model(Switch_Inputs,Planet_inputs,Init_conditions,Numerics,Stellar_inputs,MC_inputs):
    
    plot_switch = "n" # change to "y" to plot individual model runs for diagnostic purposes
    print_switch = Switch_Inputs.print_switch # This controls whether outputs print during calculations (slows things down, but useful for diagnostics)
    speedup_flag = Switch_Inputs.speedup_flag # Redundant - does not do anything in current version
    start_speed =  Switch_Inputs.start_speed  # Redundant - does not do anything in current version
    fin_speed = Switch_Inputs.fin_speed # Redundant - does not do anything in current version
    heating_switch = Switch_Inputs.heating_switch # Controls locus of internal heating, keep default values
    C_cycle_switch = Switch_Inputs.C_cycle_switch # Turns carbon cycle on or off, keep default values
    
    RE = Planet_inputs.RE #Planet radius relative Earth
    ME = Planet_inputs.ME #Planet mass relative Earth
    pm = Planet_inputs.pm #Average mantle density
    rc = Planet_inputs.rc #Metallic core radius (m)
    Total_Fe_mol_fraction = Planet_inputs.Total_Fe_mol_fraction # iron mol fraction in mantle
    
    Planet_sep = Planet_inputs.Planet_sep #planet-star separation (AU)
    albedoC = Planet_inputs.albedoC    #cold state albedo
    albedoH = Planet_inputs.albedoH    #hot state albedo
   
    #Stellar parameters
    tsat_XUV = Stellar_inputs.tsat_XUV #XUV saturation time
    Stellar_Mass = Stellar_inputs.Stellar_Mass #stellar mass (relative sun)
    fsat = Stellar_inputs.fsat 
    beta0 = Stellar_inputs.beta0
    epsilon = Stellar_inputs.epsilon
    
    #generate random seed for this forward model call
    np.random.seed(int(time.time()))
    seed_save = np.random.randint(1,1e9)

    ## Initial volatlie and redox conditions:
    Init_solid_H2O = Init_conditions.Init_solid_H2O
    Init_fluid_H2O = Init_conditions.Init_fluid_H2O
    Init_solid_O= Init_conditions.Init_solid_O
    Init_fluid_O = Init_conditions.Init_fluid_O
    Init_solid_FeO1_5 = Init_conditions.Init_solid_FeO1_5
    Init_solid_FeO = Init_conditions.Init_solid_FeO
    Init_fluid_CO2 = Init_conditions.Init_fluid_CO2
    Init_solid_CO2= Init_conditions.Init_solid_CO2

    #Oxidation parameters
    wet_oxid_eff = MC_inputs.interiord
    MFrac_hydrated = MC_inputs.interiorb
    dry_oxid_frac = MC_inputs.interiorc 
    surface_magma_fr = MC_inputs.surface_magma_frac

    #ocean chemistry and weathering parameters
    ocean_Ca = MC_inputs.ocean_a 
    omega_ocean = MC_inputs.ocean_b 
    efold_weath = MC_inputs.ccycle_a
    alpha_exp = MC_inputs.ccycle_b
    supp_lim = MC_inputs.supp_lim    

    #Escape parameters
    mult = MC_inputs.esc_c ## for when transition from diffusion to XUV
    mix_epsilon = MC_inputs.esc_d # fraction energy goes into escape above O-drag
    Te_input_escape = MC_inputs.Tstrat

    #Interior parameters
    visc_offset = MC_inputs.interiora 
    heatscale = MC_inputs.interiore

    #Impact parameters
    imp_coef = MC_inputs.esc_a 
    tdc = MC_inputs.esc_b  

    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    #kCO2 = 0.99 #sensitivity test reduced mantle (CO2 retained in interior)
    G = 6.67e-11 #gravitational constant
    cp = 1.2e3 # silicate heat capacity
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    delHf = 4e5 #Latent heat of silicates
    g = G*Mp/(rp**2) # gravity (m/s2)
    Tsolidus = sol_liq(rp,g,pm,rp,0.0,0.0) #Solidus for magma ocean evolution
    Tliquid = Tsolidus + 600 #Liquidus for magma ocean evolution
    alpha = 2e-5 #Thermal expansion coefficient (per K)
    k = 4.2 #Thermal conductivity, W/m/K)
    kappa = 1e-6 #Thermal diffusivity of silicates, m2/s)
    Racr = 1.1e3 #Critical Rayeligh number
    kH2O = 0.01 #Crystal-melt partition coefficent for water
    a1 = 104.42e-9 #Solidus coefficient
    b1 = 1420 - 80 #Solidus coefficient
    a2 = 26.53e-9 #Solidus coefficient
    b2 = 1825 + 0.000 #Solidus coefficient 
    min_Te = 150.0 ## Minimum Te for purposes of OLR/ASR calculations and escape calculations
    min_ASR = 5.67e-8 * (min_Te/(0.5**0.25))**4.0 ## Minimum Absorbed Shortwave Radiation (ASR)
    min_Te = 207.14285714 # Threshold to prevent skin temperature from getting too low where OLR grid contains errors. Note this lower limit does not apply to stratosphere temperatures used for escape calculations.

    Max_mantle_H2O = 1.4e21 * MC_inputs.interiorf * (rp**3 - rc**3) / ((6.371e6)**3 - (3.4e6)**3) ## Max mantle water content (kg)
    
    Start_time = Switch_Inputs.Start_time #Model start time (relative to stellar evolution track)
    Max_time=np.max([Numerics.tfin0,Numerics.tfin1,Numerics.tfin2,Numerics.tfin3,Numerics.tfin4]) #Model end time
    test_time = np.linspace(Start_time*365*24*60*60,Max_time*365*24*60*60,10000)
    new_t = np.linspace(Start_time/1e9,Max_time/1e9,100000)

    [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum] = main_sun_fun(new_t,Stellar_Mass,tsat_XUV,beta0,fsat) #Calculate stellar evolution
    ASR_new = (Absolute_total_Lum/(16*3.14159*(Planet_sep*1.496e11)**2) ) #ASR flux through time (not accounting for bond albedo)
    
    for ij in range(0,len(ASR_new)): # do not permit ASR outside of interpolation grid
        if (ASR_new[ij] < min_ASR):
            ASR_new[ij] = min_ASR
    Te_ar = (ASR_new/5.67e-8)**0.25
    Tskin_ar = Te_ar*(0.5**0.25) ## Skin temperature through time
    for ij in range(0,len(Tskin_ar)): #Don't permit skin temperature to exceed range min_Te - 350 due to errors in grid (does not apply to stratospheric temperature used to calculae escape fluxes)
        if Tskin_ar[ij] > 350:
            Tskin_ar[ij] = 350.0
        if Tskin_ar[ij] < min_Te:
            Tskin_ar[ij] = min_Te
    Te_fun = interp1d(new_t*1e9*365*24*60*60,Tskin_ar) #Skin temperature function, used in OLR calculations
    ASR_new_fun = interp1d(new_t*1e9*365*24*60*60, ASR_new) #ASR function, used to calculate shortwave radiation fluxes through time
    AbsXUV = interp1d(new_t*1e9*365*24*60*60 , Absolute_XUV_Lum/(4*np.pi*(Planet_sep*1.496e11)**2)) #XUV function, used to calculate XUV-driven escape

    #@jit(nopython=True) # function for finding surface temperature that balances ASR and interior heatflow
    def funTs_general(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 
        if max(Ts)<Tp:
            Ra =alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts) * (Ra/Racr)**beta
        else:
            qm = - 2.0 * (Ts - Tp) / (rp-rc)        
        Ts_in= max(Ts)
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return (qm - heat_atm)**2      
    
    #@jit(nopython=True) # alternative function for finding surface temperature that balances ASR and interior heatflow (does exact same thing, hardly used)
    def funTs_scalar(Ts,Tp,ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf):
        if Ts<Tp:
            Ra =alpha * g * (Tp -Ts) * ll**3 / (kappa * visc)
            qm = (k/ll) * (Tp - Ts) * (Ra/Racr)**beta           
        else:
            qm = - 2.0 * (Ts - Tp) / (rp-rc)        
        Ts_in= Ts
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,MMW,PO2_surf)  
        heat_atm =   OLR_new - ASR_input   
        return (qm - heat_atm)**2      

    def FeO_mass_frac(Total_Fe): # Convert total iron mole fraction to mass fraction
        XAl2O3 = 0.022423
        XCaO = 0.0335
        XNa2O = 0.0024 
        XK2O = 0.0001077 
        XMgO = 0.478144 
        XSiO2 =  0.4034    
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)
        return (Total_Fe) * (56.0+16.0)/m_sil       
    Total_Fe_mass_fraction = FeO_mass_frac(Total_Fe_mol_fraction)  
           
    global solid_switch,phi_final, ynewIC,switch_counter,liquid_switch,solid_counter
    # Define switches used to keep track of volatile transfers between magma ocean and solid mantle convection phases (and vice versa)
    solid_switch = 0
    solid_switch = 1 #for starting as solid only
    liquid_switch = 0
    liquid_switch_worked = 0
    switch_counter = 0
    solid_counter = 0

    model_run_time = time.time()

    def system_of_equations(t0,y):
        tic = time.time()
        if (tic - model_run_time)>60*60*1.5: ## time out forward model attempt after 90 minutes
            print ("TIMED OUT")
            return np.nan*[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

        Va = 0.0
        global solid_switch,liquid_switch,liquid_switch_worked
        global phi_final,ynewIC,switch_counter,solid_counter
       
        ocean_CO3 = float(omega_ocean * Sol_prod(y[8]) / ocean_Ca)
        Mantle_mass = (4./3. * np.pi * pm * (y[2]**3 - rc**3))
        
        if print_switch == "y":
            print (t0/(365*24*60*60))

        #################################################################################
        #### If in magma ocean phase
        if  (y[8] > Tsolidus):
            beta = 1./3. #Convective heatflow exponent
            if print_switch== "y":
                print('still molten',t0/(365*24*60*60),np.transpose(y))        

            #For switching from solid to magma ocean       
            if liquid_switch == 1:
                if y[2]+1 < rp:
                    liquid_switch_worked = 1.0
                    liquid_switch = 0.0
                else:
                    T_partition = np.max([y[7],Tsolidus+0.01])
                    rad_check = optimize.minimize(find_r,x0=float(y[2]),args = (T_partition,alpha,g,cp,pm,rp,0.0,0.0))
                    y[2] = np.max([rad_check.x[0],0.0])
        
            #Calculate surface melt fraction
            if y[8] > Tliquid:
                actual_phi_surf = 1.0
            elif y[8] < Tsolidus:
                actual_phi_surf = 0.0
            else:
                actual_phi_surf =( y[8] -Tsolidus)/(Tliquid - Tsolidus)
       
            ll = np.max([rp - y[2],1.0]) ## length scale is depth of magma ocean pre-solidification (even if melt <0.4)
            Qr = qr(t0,Start_time,heatscale)+np.exp(-(t0/(1e9*365*24*60*60)-4.5)/5.0)*20e12/((4./3.)* np.pi * pm *  (rp**3.0-rc**3.0))    
            Mliq = Mliq_fun(y[1],rp,y[2],pm)
            Mcrystal = (1-actual_phi_surf)*Mliq
            phi_final = actual_phi_surf
            [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
            [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) #molten so can ignore aqueous CO2
            AB = AB_fun(float(y[8]),H2O_Pressure_surface,float(y[1]+y[4]+y[12]),albedoC,albedoH)

            Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),float(0*y[0]/Mantle_mass))   # Need to change for hydrous melting sensitivity test 
            Tsolidus_visc = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),0.0)        
            visc =  viscosity_fun(y[7],pm,visc_offset,y[8],float(Tsolidus_visc))     

            if print_switch=="y":
                print ('visc',visc)
            if np.isnan(visc):
                print ('Viscosity error')
                return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

            ASR_input = float((1-AB)*ASR_new_fun(t0))
            if (ASR_input < min_ASR):
                ASR_input = min_ASR            
            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te         

                             
            if (2>1):
                initialize_fast = np.min([y[8],y[7],y[7]-1])
                if y[8] > y[7]:
                    initialize_fast = y[8]

                ace1= optimize.minimize(funTs_general,x0=initialize_fast,args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='COBYLA',bounds = ((100,4500),),options={'maxiter':1000})
                SurfT = float(ace1.x)  # COBYA 

                if ace1.fun > 0.1:
                    lower = 180.0
                    upper = float(y[7])+10
                    ace1=scipy.optimize.minimize_scalar(funTs_scalar, args=(float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),bounds=[lower,upper],tol=1e-10,method='bounded',options={'maxiter':1000,'xatol':1e-10})
                    SurfT = float(ace1.x)
                
                y[8] = SurfT
                
                                                                                                                                                          
                if ace1.fun > 0.1: 
                    differ_ace = 10.0
                    counter = 0
                    while differ_ace > 0.1:
                        counter = counter + 1
                        ran_num = np.min([150 + 2000*np.random.uniform(),3999])
                                      
                        try:
                            ace = optimize.basinhopping(funTs_general, x0=ran_num, niter=100, T=100, stepsize=100.0, minimizer_kwargs={"bounds" : ((274,4500),), "method": 'L-BFGS-B' , "args":(float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),"options": {'maxiter':1000}}, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=100, seed=None)
                            differ_ace = ace.fun[0]
                            SurfT = ace.x[0]
                        except:
                            print('not workin')
                        if counter == 3:
                            differ_ace = -2e-3 
                            SurfT = float(ace1.x) 
                         
                Ra =np.max([-1e15,alpha * g * (y[7] -SurfT) * ll**3 / (kappa * visc)  ])

                qm = (k/ll) * (y[7] - SurfT) * (Ra/Racr)**beta
        
                thermal_cond = 2 # W/m2/K
                qc = thermal_cond * (SurfT - y[7]) / (rp-rc)
        
                T_base = y[16]
                T_base = adiabat(rc,y[7],alpha,g,cp,rp) 
                T_solidus = adiabat(float(y[2]),y[7],alpha,g,cp,rp)
                visc_solid = (80e9 / (2*5.3e15)) * (1e-3 / 0.5e-9)**2.5 * np.exp((240e3+0*100e3)/(8.314*T_solidus)) * np.exp( - 26 * 0.0)/pm  
                ll_solid = y[2]-rc
                
        
                Ra_solid = alpha * g * (T_base -T_solidus) * ll_solid**3 / (kappa * visc_solid) 
                q_below = (k/ll_solid) * (T_base - T_solidus) * (Ra_solid/Racr)**beta
                delta_u = k * (y[7] - SurfT) / qm
                
            ## check to see if anywhere near solidus
            if adiabat(rc,y[7],alpha,g,cp,rp) > sol_liq(rc,g,pm,rp,0.0,0.0): #ignore pressure overburden when magma ocean solidifying (water mostly in mantle anyway)
                rs_term = 0.0
            else: # if at solidus, start increasing it according to mantle temperature cooling
                rs_term = rs_term_fun(float(y[2]),a1,b1,a2,b2,g,alpha,cp,pm,rp,y[7],0.0) #doesnt seem to affect things
            
        
            if (2>1):          
                if y[7]>SurfT:
                    if heating_switch == 1:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) -  4.0*np.pi*((rp)**2)*qm 
                    elif heating_switch == 0 :
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) -  4*np.pi*((rp)**2)*qm  
                    else:
                        print ('ERROR')
                    denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                    dy7_dt = numerator/denominator #this is Tp
                else:
                    if heating_switch == 1:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-y[2]**3.0) +  4.0*np.pi*((rp)**2)*qc 
                    elif heating_switch == 0:
                        numerator =  (4./3.)* np.pi * pm * Qr * (rp**3.0-rc**3.0) +  4*np.pi*((rp)**2)*qc
                    else:
                        print ('ERROR')
                    denominator = (4./3.) * np.pi * pm * cp *(rp**3 - y[2]**3) - 4*np.pi*(y[2]**2)* delHf*pm * rs_term
                    dy7_dt = numerator/denominator
                
        
            [OLR,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(float(SurfT),float(Te_input),H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,float(y[24]),float(y[22]))
            CO2_Pressure_surface = newpCO2


            ASR = ASR_input

            y[8] = SurfT
            heat_atm = OLR - ASR  
        
            if (2>1):
                if SurfT > y[7]:
                    true_balance = - heat_atm - qc
                else:
                    true_balance = - heat_atm + qm
        
            if print_switch == "y":        
                print ("phi",actual_phi_surf,"visc",visc)
                print ("time",t0/(365*24*60*60*1e6),"Ra",Ra)
                print (OLR,ASR)
                print ("Heat balance",true_balance)
                print (" ")
            solid_switch = 0.0
        
            if liquid_switch == 1 : ## If transitioning from liquid to solid, adjust inventories
                T_partition = np.max([y[7],Tsolidus+0.01])
                [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,y[2],rp,y[24])
                mu_O = 16.0
                mu_FeO_1_5 = 56.0 + 1.5*16.0
                if liquid_switch_worked==0: #first time or hasn't worked yet

                    y[3] = y[3] - Mliq * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[4] = y[4] + Mliq * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[5] = y[5] -  Mliq * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))     
                    y[6] = y[6] - Mliq *  (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))   
                    y[0] = y[0] - Mliq * (y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[1] = y[1] + Mliq * (y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[13] = y[13] - Mliq * (y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))
                    y[12] = y[12] + Mliq * (y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))

                    switch_name = "switch_garbage/switch_IC_%d" %seed_save
                    load_name2 = switch_name+".npy"
                    if switch_counter == 0 :
                        np.save(switch_name,y)
                    else:
                        y = np.load(load_name2)
                    switch_counter = switch_counter + 1
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                else: #switch has worked, back to normal, reset for next magma ocean transition
                    liquid_switch_worked = 0.0
                    liquid_switch = 0.0  
                    switch_counter = 0.0
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] 

        #################################################################################
        #### If in solid mantle phase     
        else: 
            if print_switch =="y":
                print('Mantle fully solid',t0/(365*24*60*60),y)
            beta = 1./3.
            ll = rp - rc #this is just mantle thickness in general
        

            if solid_switch == 0: ## If switching from solid to magma ocean
                Mliq = Mliq_fun(y[1],rp,y[2],pm) 
                Mcrystal = (1-phi_final)* Mliq 
                [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
                [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) 
                T_partition = np.max([y[7],Tsolidus+0.01]) 
                [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,y[2],rp,y[24])
                mu_O = 16.0
                mu_FeO_1_5 = 56.0 + 1.5*16.0
                if y[2] < rp:
                                
                    y[3] =  y[3] + (4./3.) *np.pi * pm *F_FeO1_5 * (rp**3 - y[2]**3) * 0.5*mu_O / (mu_FeO_1_5)
                    y[4] = y[4] - (4./3.) *np.pi * pm *F_FeO1_5 * (rp**3 - y[2]**3) * 0.5*mu_O / (mu_FeO_1_5)
                    y[5] = y[5] + (4./3.) *np.pi * pm * F_FeO1_5  *(rp**3 - y[2]**3)  
                    y[6] = y[6] + (4./3.) *np.pi * pm *  F_FeO * (rp**3 - y[2]**3)
                    Water_transfer = np.min([Max_mantle_H2O-y[0],FH2O * kH2O * Mcrystal + FH2O * (Mliq-Mcrystal)])
                    CO2_transfer = np.min([Max_mantle_H2O-y[13],kCO2 * FCO2 * Mcrystal  + FCO2 * (Mliq-Mcrystal) ]) 
                    y[0] = y[0] + Water_transfer
                    y[1] = y[1] - Water_transfer
                    y[13] = y[13] + CO2_transfer
                    y[12] = y[12] - CO2_transfer

                             
                    y[2] = rp
                    liquid_name = "switch_garbage/liquid_IC_%d" %seed_save
                    load_name = liquid_name + ".npy"
                    if solid_counter == 0:
                        np.save(liquid_name,y)
                    else:
                        y = np.load(load_name)
                    solid_counter = solid_counter + 1
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                else:
                    solid_switch = 1.0  
                    solid_counter = 0.0
                    return [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]        

        
            y[2] = rp
            Mliq = 0.0 
            Mcrystal = 0.0 
            
            [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24]) 
            [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],Mliq,Mcrystal,rp,g,kCO2,y[24]) 
            
            Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),float(0*y[0]/Mantle_mass))  # Need to change for hydrous melting sensitivity test 
            Tsolidus_visc = sol_liq(rp,g,pm,rp,float(H2O_Pressure_surface+CO2_Pressure_surface+1e5),0.0)                           
            visc =  viscosity_fun(y[7],pm,visc_offset,y[8],float(Tsolidus_visc))
            
            initialize_fast = np.min([y[8],y[7],y[7]-1])   

            AB = AB_fun(float(y[8]),H2O_Pressure_surface,float(y[1]+y[4]+y[12]),albedoC,albedoH)
            ASR_input = float((1-AB)*ASR_new_fun(t0))
            if (ASR_input < min_ASR):
                ASR_input = min_ASR

            Te_ar = (ASR_input/5.67e-8)**0.25
            Te_input = Te_ar*(0.5**0.25)
            if Te_input > 350:
                Te_input = 350.0
            if Te_input < min_Te:
                Te_input = min_Te 
 
            ace1= optimize.minimize(funTs_general,x0=initialize_fast,args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='Nelder-Mead',bounds = ((150,4000),))
            SurfT = ace1.x[0]

            if ace1.fun > 1.0:#0.1:
                ace1= optimize.minimize(funTs_general,x0=y[8],args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='L-BFGS-B',bounds = ((275,550),))
                SurfT = ace1.x[0]
            
            if ace1.fun > 1.0:#0.1:
                differ_ace = 10.0
                counter = 0
                while differ_ace > 0.1:
                    counter = counter + 1
                                
                    ran_num = np.min([150 + 2000*np.random.uniform(),3999])
                    try:
                        ace = optimize.basinhopping(funTs_general, x0=ran_num, niter=100, T=100, stepsize=100.0, minimizer_kwargs={"bounds" : ((150,4000),), "method": 'L-BFGS-B' , "args":(float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22]))}, take_step=None, accept_test=None, callback=None, interval=50, disp=False, niter_success=100, seed=None)                  
                        differ_ace = ace.fun[0]
                        SurfT = ace.x[0]

                        if counter == 2:
                            surft_new = np.linspace(274,3999,1000)
                            OLR_ar = np.copy(surft_new)
                            ASR_ar = ASR_new_fun(t0) 
                            Ra_ar =alpha * g * (y[7] -surft_new) * ll**3 / (kappa * visc) 
                            qm_ar = (k/ll) * (y[7] - surft_new) * (Ra_ar/Racr)**beta
                            qm_plot = np.copy(surft_new)
                            for kkk in range(0,len(qm_plot)):
                                surft_input = surft_new[kkk]
                                te_input = Te_fun(t0)
                                OLR_ar[kkk] = my_interp(float(surft_input),float(te_input),H2O_Pressure_surface,CO2_Pressure_surface)
                                if surft_new[kkk] < y[7]:
                                    qm_plot[kkk] = qm_ar[kkk]
                                else:
                                    qm_plot[kkk] = - 2.0 * (surft_new[kkk] - y[7]) / (rp-rc)

                            differ_ace = -1e-3
                    except:
                        print('not workin')
                    if counter == 3:
                        differ_ace = -2e-3 
                        SurfT = ace1.x[0]
            else:
                SurfT = ace1.x[0]                  

            T_diff = y[7]-SurfT
            Ra = np.max([-1e15,alpha * g * T_diff * ll**3 / (kappa * visc)])
            qm = (k/ll) * T_diff * (Ra/Racr)**beta
            Qr = qr(t0,Start_time,heatscale)+np.exp(-(t0/(1e9*365*24*60*60)-4.5)/5.0)*20e12/((4./3.)* np.pi * pm * (rp**3.0-rc**3.0))

            [OLR,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(float(SurfT),float(Te_input),H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,1e5,float(y[24]),float(y[22]))
            CO2_Pressure_surface = newpCO2 
            
            ASR = ASR_input
            heat_atm = OLR - ASR
            delta_u = k * (y[7] - SurfT) / qm
    
            numerator = - 4*np.pi*(rp**2)*qm  + (4./3.) * np.pi * pm * Qr * (rp**3 - rc**3) 
            denominator = pm * cp * (4./3.) *np.pi * (rp**3 - rc**3)
        
            thermal_cond = 2 # W/m2/K
            qc = thermal_cond * (SurfT - y[7]) / (rp-rc)
            if SurfT > y[7]:
                numerator =  4*np.pi*(rp**2)*qc  + (4./3.) * np.pi * pm * Qr * (rp**3 - rc**3) 
                denominator = pm * cp * (4./3.) *np.pi * (rp**3 - rc**3)   
        
            rs_term = 0.0  
            dy7_dt = numerator/denominator
                
            y[8] = SurfT
            liquid_switch = 1.0
            liquid_switch_worked = 0.0
       
        ####################################################################################  
        ## end magma ocean/solid mantle portion. The rest of the code applies to both phases
        ####################################################################################  

        y[9] = OLR
        y[10] = ASR
        if y[8]<=y[7]:
            y[11] = qm
        else:
            y[11] = -qc

        dy2_dt = rs_term * dy7_dt
        drs_dt = dy2_dt
        rs = np.min([y[2],rp])     

        [FH2O,H2O_Pressure_surface] = H2O_partition_function( y[1],Mliq,Mcrystal,rp,g,kH2O,y[24])
        if y[1]< 0:
            H2O_Pressure_surface = 0.0
            FH2O = 0.0

        water_frac = my_water_frac(float(y[8]),Te_input,H2O_Pressure_surface,CO2_Pressure_surface)
        
        atmo_H2O = np.max([H2O_Pressure_surface*water_frac,0.0])
        
        if solid_switch == 0:
            T_partition = np.max([y[7],Tsolidus+0.01])
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,T_partition,Total_Fe_mol_fraction,Mliq,rs,rp,y[24]) #
        else:
            [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(y[4],H2O_Pressure_surface,y[8],Total_Fe_mol_fraction,Mliq,rs,rp,y[24]) #

        fO2_pos = np.max([0,fO2])
        Pressure_surface = fO2_pos +atmo_H2O + CO2_Pressure_surface + 1e5 
        
        
        frac_h2o_upper = my_fH2O(float(SurfT),Te_input_escape,H2O_Pressure_surface,CO2_Pressure_surface)
        frac_h2o_upper = np.min([atmo_H2O / Pressure_surface, frac_h2o_upper])
        if (H2O_Pressure_surface<1e-5)and(frac_h2o_upper<1e-9): # lower threshold zero out NEW TRAPPIST THING
            frac_h2o_upper = 0.0
            atmo_H2O = 0.0
            H2O_Pressure_surface = 0.0
        
        #######################
        ## Atmosphsic escape calculations
        ## diffusion limited escape:
        fCO2_p = (1- frac_h2o_upper)*CO2_Pressure_surface / (CO2_Pressure_surface+fO2_pos+1e5)
        fO2_p = (1- frac_h2o_upper)*fO2_pos / (CO2_Pressure_surface+fO2_pos+1e5)
        fN2_p = (1- frac_h2o_upper)*1e5 / (CO2_Pressure_surface+fO2_pos+1e5)
        mol_diff_H2O_flux = better_diffusion(frac_h2o_upper,Te_input_escape,g,fCO2_p,fO2_p,fN2_p) #mol H2O/m2/s
        
        #XUV-driven escape
        XH_upper = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p) ## assumes CO2 and N2 don't dissociate
        XH = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XO = (2*fO2_p+frac_h2o_upper) / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XC = fCO2_p / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        true_epsilon = find_epsilon(Te_input,RE,ME,float(AbsXUV(t0)), XO, XH, XC,epsilon,mix_epsilon)
        if (XC < 0)or(y[12]<1e10):
            XC = 0.0
        [mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Te_input_escape,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XC) #kg/m2/s
        numH = ( mH_Odert3 / 0.001 ) # mol H / m2/ s
        numO = ( mO_Odert3 / 0.016 ) # mol O / m2/ s
        numC = ( mC_Odert3 / 0.044 ) # mol CO2 / m2/ s
        
        if 2*mol_diff_H2O_flux> numH: ## if diffusion limit exceeds XUV-driven, shift diffusion-limit downward
             mol_diff_H2O_flux = 0.5*np.copy(numH)

        ## Combined escape flux, weighted by H abundance:
        w1 = mult*(2./3 -  XH_upper)**4
        w2 = XH_upper**4
        Mixer_H = (w1*2*mol_diff_H2O_flux + w2 * numH ) / (w1+w2) # mol H / m2 /s
        Mixer_O = (w1*0.0 + w2 * numO ) / (w1+w2)
        Mixer_C = (w1*0.0 + w2 * numC ) / (w1+w2)
        Mixer_Build = 0.5* Mixer_H - Mixer_O  ## CO2 drag doesn't affect redox (CO2 drag negligible anyway)

        escape = 4*np.pi*rp**2 * Mixer_H*0.018/2 ## kg H2O /s
        net_escape = 4*np.pi*rp**2 * Mixer_Build*0.016 ## kg O2/s
        CO2_loss =  4*np.pi*rp**2 * Mixer_C*0.044 ## kg CO2/s
        # done with escape calculations
        #######################

        ## Find ocean depth and land fraction:
        Ocean_depth = (0.018/y[24]) * (1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
        if Ocean_depth > Max_depth:
            Linear_Ocean_fraction = 1.0
            Ocean_fraction = 1.0
        else:
            Linear_Ocean_fraction = (Ocean_depth/Max_depth) ## Crude approximation to Earth hypsometric curve
            Ocean_fraction = (Ocean_depth/Max_depth)**0.25
    
        ## Melt and crustal oxidation variables:
        actual_phi_surf_melt = 0.0
        F_CO2 = 0.0
        F_H2O = 0.0
        O2_consumption = 0.0
        OG_O2_consumption =0.0
        Plate_velocity = 0.0
        crustal_depth = 0.0
        Melt_volume = 0.0
        Poverburd = fO2_pos +H2O_Pressure_surface + CO2_Pressure_surface + 1e5
        Tsolidus_Pmod = sol_liq(rp,g,pm,rp,float(Poverburd),float(0*y[0]/Mantle_mass)) # Need to change for hydrous melting sensitivity test 

        ## If solid mantle, calculate crutal production and melt volume: 
        if (y[2]>=rp)and(Ra>0)and(y[8]<=Tsolidus):
            T_for_melting = float(y[7])
            if T_for_melting <= Tsolidus_Pmod: # no melting if potential temperature is below solidus
                [actual_phi_surf_melt, Va] = [0.0, 0.0]
                Melt_volume = 0.0
                Plate_velocity = 0.0
            else: #mantle temperature above solidus, calculate melt production
                mantleH2Ofrac = float(0*y[0]/Mantle_mass) #mantle water content, need to change for hydrous melting sensitivity test 
                rdck = optimize.minimize(find_r,x0=float(y[2]),args = (T_for_melting,alpha,g,cp,pm,rp,float(Poverburd),mantleH2Ofrac))
                rad_check = float(rdck.x[0]) # find radius where partial melting begins
                if rad_check>rp:
                    rad_check = rp
                [actual_phi_surf_melt,actual_visc,Va] = temp_meltfrac(0.99998*rad_check,rp,alpha,pm,T_for_melting,cp,g,Poverburd,mantleH2Ofrac) #calculate melt production
                crustal_depth = rp - (rp**3 - actual_phi_surf_melt * Va*3./(4.*np.pi))**(1./3.) #calulate crustal depth (depth of melt)

                Ra_melt =alpha * g * (y[7] - SurfT) * ll**3 / (kappa * visc) 
                Q =  (k/ll) * (y[7] - SurfT) * (Ra_melt/Racr)**beta
                Aoc = 4*np.pi*rp**2
                Melt_volume = (Q*4*np.pi*rp**2)**2/(2*k*(y[7] - SurfT))**2 * (np.pi*kappa)/(Aoc)  * crustal_depth #eq 18 kite
                Plate_velocity = Melt_volume/(crustal_depth * 3 * np.pi*rp) 
            
            ## Fresh crust production
            dmelt_dt = pm * Melt_volume #kg/s over whole planet
            iron3 = y[5]*56/(56.0+1.5*16.0) #kg fe203 * 56 g fe/mol / x g fe2o3 /mol fe2o3
            iron2 = y[6]*56/(56.0+16.0)
            iron_ratio_mantle = 0.5*iron3/iron2

            ## Given melt production and land fraction, calculate outgassing from seafloor and continents
            if (Ocean_fraction<1.0) and (T_for_melting > Tsolidus_Pmod): # Land outgassing
                [F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L] = outgas_flux_cal(Tsolidus,Pressure_surface,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)
            else: 
                [F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L] = [0.0,0.0,0.0,0.0,0.0,0.0]   
           
            if (T_for_melting > Tsolidus_Pmod): #Seafloor outgassing
                [F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O] = outgas_flux_cal(Tsolidus,Poverburd,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)
            else: 
                [F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O] = [0.0,0.0,0.0,0.0,0.0,0.0]

            # Total outgassing weighted by land fraction
            [F_H2O,F_CO2,F_H2,F_CO,F_CH4,OG_O2_consumption] = np.array([F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O])*Ocean_fraction+np.array([F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L])*(1-Ocean_fraction)
            O2_consumption = np.copy(OG_O2_consumption)
        else:
            dmelt_dt = 0.0                  
        if rs<rp:
            fudge = 1.0-(rs/rp)**1000.0 ## helps with numerical issues
        else:
            fudge = 0.0

        y[17] = Mcrystal
        mu_O = 16.0
        mu_FeO_1_5 = 56.0 + 1.5*16.0 
        mu_FeO = 56.0 + 16.0   
        # Calculate max surface emplacement available for oxidation
        # 1e7 ~ 80 km3/yr emplacement, 1e8 ~ 800 km3/yr emplacement = 1mm/yr oxidized over whole surface
        surface_emplacement =np.min([dmelt_dt,surface_magma_fr*1e9/dry_oxid_frac]) 

        
        if y[8]< 973: #If surface temperature below serpentine stability, calculate crustal hydration
            hydration_depth = np.max([0.0,delta_u * (973 - y[8])/(y[7] - y[8])])
            frac_hydrat = np.min([hydration_depth/crustal_depth,1.0])
            water_c = Linear_Ocean_fraction*MFrac_hydrated *frac_hydrat * dmelt_dt 
            water_crust = water_c * np.max([0.0,1 - y[0]/Max_mantle_H2O])
            wet_oxidation = wet_oxid_eff *(water_c/MFrac_hydrated) * Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) # kg FeO/s of the hydrated crust, what fraction iron oxidized
            # so this means, of the crust that is hydrated, how much iron oxidized
            ## additionally, restrict water to be less than surface inventory
            total_water_loss_surf = water_crust + wet_oxidation*18.0/(3.0* mu_FeO) #kg H2O/s
            total_water_gain_interior = water_crust
            new_wet_oxidation = wet_oxidation * 16.0 / (3.0* mu_FeO)  #kg O/s added solid
            total_wet_FeO_lost = wet_oxidation * (2.0/3.0) # kg FeO/s, leaving behind 1 mol FeO for every 2 FeO1.5
            total_wet_FeO1_5 = wet_oxidation * (2.0/3.0) * mu_FeO_1_5/mu_FeO #kg FeO1_5 /s  
            total_wet_H2_gained = wet_oxidation * 2.0/(3.0* mu_FeO) #kg H2/s
        else:
            wet_oxidation = 0.0
            water_crust = 0.0
            [total_water_loss_surf,total_water_gain_interior,new_wet_oxidation,total_wet_FeO_lost,total_wet_FeO1_5,total_wet_H2_gained] = [0,0,0,0,0,0]
        #wet_oxidation * (2.0/3.0)  # is kg/s FeO lost
        #wet_oxidation * (2.0/3.0) * mu_FeO_1_5/mu_FeO # is kg/s FeO1.5 gained
        #wet_oxidation * 18.0/(3.0* mu_FeO) # is kg/s H2O lost from surface
        #wet_oxidation * 2.0/(3.0* mu_FeO) # is kg/s H2 gained
        #wet_oxidation * 16.0 / (3.0* mu_FeO) # is kg/s O consumed 
        #water_crust is kg/s H2O gained from interior, lost from surface

        y[19] = net_escape 
        
        #impacts (only used in sensitivity tests)
        imp_flu = imp_coef/(365*24*60*60)*np.exp(-(t0/(365*24*60*60*1e9))/tdc) #kg/s
        if imp_flu < 1e7/(365*24*60*60):
            imp_flu = 0.0
        if fO2>0:
            O_imp_sink = imp_flu * 0.3 * 8.0/(56.0+16.0) #kg O2/s for 30% FeO
            O_imp_sink = imp_flu * 0.3 * 24.0/56.0 #kg O2/s for 30% metallic iron
        else:
            O_imp_sink = 0.0
        O_imp_sink = 0.0 ## Zero out impacts for nominal model

        if (fO2>0) and (y[2] >=rp) and(Ra>0.0)and(y[8]<=Tsolidus): ## Calculate oxygen sinks for solid surface
            ## Assumes dry magma oxidation only happens on non-submerged surface: instant cooling of magma in water precludes significant oxidation
            O2_dry_magma_oxidation  = (1- Ocean_fraction)*dry_oxid_frac*Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) * surface_emplacement *  0.5*mu_O / mu_FeO 
            Fe_dry_magma_oxidation = (1- Ocean_fraction)*dry_oxid_frac*Total_Fe_mass_fraction * (y[6]/(y[6]+y[5])) * surface_emplacement  
            O2_magma_oxidation = new_wet_oxidation + O2_dry_magma_oxidation
            Fe_magma_oxidation = Fe_dry_magma_oxidation + total_wet_FeO_lost
            y[18] = -O2_dry_magma_oxidation
        else:
            y[18] = 0.0
            O2_dry_magma_oxidation  = 0.0
            Fe_dry_magma_oxidation = 0.0
            O2_magma_oxidation = 0.0
            Fe_magma_oxidation = 0.0
            O2_consumption = 0.0
            new_wet_oxidation = 0.0 
               
        y[18] = y[18] - O_imp_sink 
        y[20] =-new_wet_oxidation 
        y[21] = -O2_consumption*0.032
        y[22] = fO2
        y[23] = CO2_Pressure_surface
        
        O2_magma_oxidation_solid = np.copy(O2_magma_oxidation)+ O2_consumption*0.032
        O2_magma_oxidation_volatile = np.copy(O2_magma_oxidation) + O2_consumption*0.032 + O_imp_sink
        Fe_magma_oxidation_solid = np.copy(Fe_magma_oxidation) + 4*O2_consumption*(0.056+0.016) # kg FeO/s
        Fe_magma_oxidation_solid_1_5 = (mu_FeO_1_5/mu_FeO) * np.copy(Fe_magma_oxidation) +  2*O2_consumption*(0.016*3 + 0.056*2) # kg FeO1.5/s
        
        # If anoxic atmosphere, adjust oxygen sinks accordingly 
        if (fO2_pos<1e-1)and(fO2>=0)and(net_escape<O2_magma_oxidation_volatile)and(y[2]>=rp)and(Ra>0.0)and(y[8]<=Tsolidus): # only do this solid state
            O2_magma_oxidation_volatile = net_escape
            if net_escape<O2_consumption*0.032+new_wet_oxidation+O_imp_sink: #fast outgassing sinks win, no crustal oxidation except via H2 escape
                Fe_dry_magma_oxidation = 0.0
                O2_dry_magma_oxidation = 0.0
                O2_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation - net_escape) ## gain Kg O/s as excess reductants escape
                Fe_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation - net_escape) * 2*mu_FeO/ mu_O # loss Kg FeO/s as excess reductant escape
                Fe_magma_oxidation_solid_1_5 = (O2_consumption*0.032+new_wet_oxidation - net_escape) * 2*mu_FeO_1_5/ mu_O # need factor of 2 bceause 4 mol FeO (or FeO1.5) for every 
            elif (net_escape > O2_consumption*0.032+new_wet_oxidation+O_imp_sink): # oxygen produced exceeds fast sinks, mopped up by magma, rate oxidation crust is just H loss
                O2_dry_magma_oxidation =   net_escape - (new_wet_oxidation + O2_consumption*0.032+O_imp_sink)
                O2_magma_oxidation_solid = (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation+O_imp_sink) ## gain Kg O/s as excess reductants escape
                Fe_magma_oxidation_solid =   (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation)* 2*mu_FeO/ mu_O # loss Kg FeO/s as excess reductant escape
                Fe_magma_oxidation_solid_1_5 =  (O2_consumption*0.032+new_wet_oxidation+O2_dry_magma_oxidation) * 2*mu_FeO_1_5/ mu_O # need factor of 2 bceause 4 mol FeO (or FeO1.5) for every 



        ###########################################################################################
        ### Time-evolution of reservoirs
        dy0_dt = fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  + total_water_gain_interior - F_H2O*0.018
        if drs_dt < 0.0:
            dy0_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[0] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) 
            dy1_dt = - dy0_dt - escape
        else:
            dy1_dt = - fudge * 4*np.pi * pm * kH2O * FH2O * rs**2 * drs_dt  - escape  - total_water_loss_surf + F_H2O*0.018    
        

        dy3_dt = fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) + O2_magma_oxidation_solid ## free O in solid , factor of half because only half is free oxygen
    ### this is correct, only 8 molecular mass free O2 for 56+1.5*16 molecular mass solidified
    ### QUERY: Maybe these negative drst_dt fuck it up
        if drs_dt < 0.0:
            dy3_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[3] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)))# * 0.5*mu_O / (mu_FeO_1_5) 
            dy4_dt = net_escape - dy3_dt
        else:
            dy4_dt = net_escape - fudge * 4 *np.pi * pm *F_FeO1_5 * rs**2 * drs_dt * 0.5*mu_O / (mu_FeO_1_5) - O2_magma_oxidation_volatile  ## magma ocean and atmo, free O   
        dy5_dt = fudge * 4 *np.pi * pm * F_FeO1_5 * rs**2 * drs_dt +  Fe_magma_oxidation_solid_1_5 ## mass FeO1_5 flux, FeO + O2 = 2Fe2O3
        dy6_dt = fudge * 4 * np.pi * pm * F_FeO * rs**2 * drs_dt - Fe_magma_oxidation_solid #F_FeO flux
        # O2_consumption is mol of free O2
        # 4FeO + O2 -> 2Fe2O3, so 1 mol O2_consumption -> 2 mol Fe2O3 = 2*O2_consumption * M(Fe2O3) for kg/s Fe2O3 = kgFeO1.5
        # similarly 1 mol O2_consumption > 4 mol FeO = 4 * O2_consumption * M(FeO) for kg/s FeO 

        if drs_dt < 0.0:
            dy5_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[5] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 
            dy6_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * (y[6] / (4./3. * np.pi * pm * (y[2]**3 - rc**3))) 

        dy13_dt = fudge * 4*np.pi * pm * kCO2 * FCO2 * rs**2 * drs_dt ## mass solid CO2
        if drs_dt < 0.0:
            dy13_dt = fudge * 4*np.pi * pm *  rs**2 * drs_dt * y[13] / (4./3. * np.pi * pm * (y[2]**3 - rc**3)) ##mass solid CO2
    
        dy12_dt = -dy13_dt - CO2_loss ##  escape CO2 assumed
        
        Weather = 0.0
        Outgas = 0.0
        if (C_cycle_switch == "y")and(SurfT<Tsolidus): # If carbon cycle operating, add silicate weathering fluxes

            Outgas = F_CO2*0.044
            Weather = weathering_flux(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,float(y[24]))
            if (CO2_Pressure_surface<50)and(CO2_Pressure_surface>0)and(water_frac<0.9999999)and(SurfT<647)and(Weather>Outgas): 
                Weather = Outgas
            dy12_dt = dy12_dt - Weather + Outgas
            dy13_dt = dy13_dt + Weather - Outgas

        ###########################################################################################

        y[16] = rp - (rp**3 - actual_phi_surf_melt * Va*3./(4.*np.pi))**(1./3.) #crustal depth
        y[15] = 1e-12 * Outgas*365*24*60*60/0.044 #convert outgassing flux (Kg/s) to Tmol CO2/yr
        y[14] = 1e-12 * Weather*365*24*60*60/0.044 #convert weathering flux (Kg/s) to Tmol CO2/yr
        y[24] = (fO2_pos*0.032 + atmo_H2O*0.018 + CO2_Pressure_surface*0.044 + 1e5*0.028)/Pressure_surface #Mean molecular weight
        toc = time.time()
        return [dy0_dt,dy1_dt,dy2_dt,dy3_dt,dy4_dt,dy5_dt,dy6_dt,dy7_dt,0.0,0.0,0.0,0.0,dy12_dt,dy13_dt,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    ########################################################################################### end forward model function
    ######################################################################################################################
    ######################################################################################################################

    ##############################################
    # Iron speciation and oxygen fugacity functions
    @jit(nopython=True) 
    def fff(logXFe2O3,XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW):
        XFe2O3 = np.exp(logXFe2O3)
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + (Total_Fe-2*XFe2O3) * (56.0+16.0)
        if (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2+3*16)/m_sil) <0:
            return 1e8
        ## g per mol of BSE, so on next line mol Xfe / mol BSE * gXfe/molXfe / g/mol BSE = g Xfe / mol BSe / g/mol BSE = gXfe/g BSE
        terms1 = 0.196*np.log( 1e-5*(MMW/0.032) * (y4 - Mliq *  (0.5*16/(56+1.5*16))*XFe2O3*(56*2.0+3.0*16.0)/m_sil) / (4*np.pi*(rp**2)/g)) + 11492.0/T - 6.675 - 2.243*XAl2O3  ## fO2 in bar not Pa
        terms2 = 3.201*XCaO + 5.854 * XNa2O
        terms3 = 6.215*XK2O - 3.36 * (1 - 1673.0/T - np.log(T/1673.0))
        terms4 = -7.01e-7 * P/T - 1.54e-10 * P * (T - 1673)/T + 3.85e-17 * P**2 / T
        terms = terms1+terms2+terms3+terms4  
        return (np.log((XFe2O3 /(Total_Fe -2*XFe2O3))) + 1.828 * Total_Fe - terms)**2.0 
        
    def solve_fO2_F_redo(y4,P,T,Total_Fe,Mliq,rs,rp,MMW): 
        if T > Tsolidus:
            XAl2O3 = 0.022423 
            XCaO = 0.0335
            XNa2O = 0.0024 
            XK2O = 0.0001077 
            XMgO = 0.478144  
            XSiO2 =  0.4034   

            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)
            logXFe2O3 = scipy.optimize.minimize_scalar(fff, args=(XMgO,XSiO2,XAl2O3,XCaO,XNa2O,XK2O,y4,P,T,Total_Fe,Mliq,rs,rp,MMW),bounds=[-100,0.0],method='bounded',options={'maxiter':1000,'xatol': 1e-20})
            XFe2O3 = np.exp(logXFe2O3.x)
            XFeO =(Total_Fe - 2*XFe2O3)
            m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) + XFe2O3 * (56.*2 + 16.*3) + XFeO * (56.0+16.0)
            F_FeO1_5 = XFe2O3*(56.0*2.0+3.0*16.0)/m_sil 
            F_FeO = XFeO * (56.0 + 16.0) / m_sil 
            fO2_out =  (MMW/0.032) *(y4 - (0.5*16/(56+1.5*16)) * Mliq * XFe2O3*(56.0*2.0+3.0*16.0)/m_sil ) / (4*np.pi*(rp**2)/g)

        else:
            fO2_out =  (MMW/0.032) *(y4 / (4*np.pi*(rp**2)/g))
            XFeO = 0.0
            XFe2O3 = 0.0
            F_FeO1_5 = 0.0
            F_FeO = 0.0
        return [XFeO,XFe2O3,fO2_out,F_FeO1_5,F_FeO]
    ##############################################
    ##############################################

    ### Initialize forward model
    ICs = [Init_solid_H2O,Init_fluid_H2O, rc, Init_solid_O,Init_fluid_O,Init_solid_FeO1_5,Init_solid_FeO,4000,3999,0.0,0.0,0.0,Init_fluid_CO2,Init_solid_CO2,0.0,0.0,3999.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.044]

    ### Various numerical inputs 
    if Numerics.total_steps==3:
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60) 
        ## the next sol2 ought to be changed from RK23 to RK45 to reduce model failure for waterworlds, but RK23 works better for nominal
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1], Numerics.tfin2*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        sol4 = solve_ivp(system_of_equations, [sol3.t[-1], Max_time*365*24*60*60], sol3.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step3*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t,sol4.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y,sol4.y),axis=1)

    elif Numerics.total_steps==9: # same as above, but slightly different numerical solver that works better for waterworld calculations
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60) 
        ## the next sol2 ought to be changed from RK23 to RK45 to reduce model failure for waterworlds, but RK23 works better for nominal
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK45', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1], Numerics.tfin2*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        sol4 = solve_ivp(system_of_equations, [sol3.t[-1], Max_time*365*24*60*60], sol3.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step3*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t,sol4.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y,sol4.y),axis=1)

    elif Numerics.total_steps ==2:
        sol = solve_ivp(system_of_equations, [Start_time*365*24*60*60, Numerics.tfin0*365*24*60*60], ICs,dense_output=True, method = 'RK45',max_step=Numerics.step0*365*24*60*60)
        sol2 = solve_ivp(system_of_equations, [sol.t[-1], Numerics.tfin1*365*24*60*60], sol.y[:,-1], method = 'RK45', vectorized=False, max_step=Numerics.step1*365*24*60*60)
        sol3 = solve_ivp(system_of_equations, [sol2.t[-1],Max_time*365*24*60*60], sol2.y[:,-1], method = 'RK23', vectorized=False, max_step=Numerics.step2*365*24*60*60)
        total_time = np.concatenate((sol.t,sol2.t,sol3.t))
        total_y = np.concatenate((sol.y,sol2.y,sol3.y),axis=1)

    ###################################
    ## Filling in output arrays (post-processing)

    t_array = total_time 
    FH2O_array= 0.0 * np.copy(t_array)
    FCO2_array= 0.0 * np.copy(t_array)
    MH2O_liq = np.copy(FH2O_array)
    MH2O_crystal = np.copy(FH2O_array)
    MCO2_liq = np.copy(FH2O_array)
    MCO2_crystal = np.copy(FH2O_array)
    Pressre_H2O = np.copy(FH2O_array)
    CO2_Pressure_array = np.copy(FH2O_array)
    CO2_Pressure_array_atmo = np.copy(FH2O_array)
    fO2_array = np.copy(FH2O_array)
    Mass_O_atm =  np.copy(FH2O_array)
    Mass_O_dissolved = np.copy(FH2O_array)
    water_frac = np.copy(FH2O_array)
    Ocean_depth = np.copy(FH2O_array)
    Max_depth = 11400 * (9.8 / g)  + 0.0*Ocean_depth
    Ocean_fraction = np.copy(FH2O_array)
    f_O2_FMQ = np.copy(FH2O_array)
    f_O2_IW = np.copy(FH2O_array)
    f_O2_MH = np.copy(FH2O_array)
    f_O2_mantle = np.copy(FH2O_array)


    for i in range(0,len(t_array)):
        rs = total_y[2][i]
        Mliq = Mliq_fun(total_y[2][i],rp,rs,pm)
        Mcrystal = 0.0
        Mcrystal = total_y[17][i]
        [FH2O_array[i],Pressre_H2O[i]] = H2O_partition_function( total_y[1][i],Mliq,Mcrystal,rp,g,kH2O,total_y[24][i])
        [FCO2_array[i],CO2_Pressure_array[i]] = CO2_partition_function( total_y[12][i],Mliq,Mcrystal,rp,g,kCO2,total_y[24][i])
        ocean_CO3 = float(omega_ocean * Sol_prod(total_y[8][i]) / ocean_Ca)
        
        [heat_atm,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(float(total_y[8][i]),float(Te_fun(t_array[i])),float(Pressre_H2O[i]),float(CO2_Pressure_array[i]),rp,g,ocean_CO3,1e5,float(total_y[24][i]),float(total_y[22][i])) 
        CO2_Pressure_array_atmo[i] = newpCO2

        MH2O_liq[i] = (Mliq - Mcrystal) * FH2O_array[i]
        MH2O_crystal[i] = kH2O * Mcrystal * FH2O_array[i]
    
        MCO2_liq[i] = (Mliq- Mcrystal) * FCO2_array[i]
        MCO2_crystal[i] = kCO2 * Mcrystal* FCO2_array[i]
    
        water_frac[i] = my_water_frac(float(total_y[8][i]),float(Te_fun(t_array[i])),float(Pressre_H2O[i]),float(CO2_Pressure_array_atmo[i]))
        Ocean_depth[i] = (0.018/total_y[24][i])*(1-water_frac[i]) * Pressre_H2O[i] / (g*1000)
        
        Ocean_fraction[i] = np.min([1.0,(Ocean_depth[i]/Max_depth[0])**0.25 ])
        
        [XFeO,XFe2O3,fO2,F_FeO1_5,F_FeO] = solve_fO2_F_redo(total_y[4][i],Pressre_H2O[i],total_y[8][i],Total_Fe_mol_fraction,Mliq,rs,rp,total_y[24][i])
        fO2_array[i] = total_y[22][i]
        fO2  =  total_y[22][i]
        Pressure_surface = fO2 + Pressre_H2O[i]*water_frac[i] + CO2_Pressure_array_atmo[i] + 1e5

        Mass_O_dissolved[i] = Mliq * F_FeO1_5*0.5*16/(56+1.5*16)
        Mass_O_atm[i] = fO2*4 *(0.032/total_y[24][i])*np.pi * rp**2 / g
        
        f_O2_FMQ[i] = buffer_fO2(total_y[7][i],Pressure_surface/1e5,'FMQ')
        f_O2_IW[i] = buffer_fO2(total_y[7][i],Pressure_surface/1e5,'IW')
        f_O2_MH[i] =  buffer_fO2(total_y[7][i],Pressure_surface/1e5,'MH')
        iron3 = total_y[5][i]*56/(56.0+1.5*16.0)
        iron2 = total_y[6][i]*56/(56.0+16.0)
        iron_ratio = iron3/iron2
        f_O2_mantle[i] = get_fO2(0.5*iron3/iron2,Pressure_surface,total_y[7][i],Total_Fe_mol_fraction)

    total_time = total_time/(365*24*60*60)-Start_time ## Start at time = zero

#########################################################################################################################################################
#########################################################################################################################################################
#### Everything is done - the rest is optional plotting of individual model runs for diagnostic purposes ################################################
#########################################################################################################################################################
#########################################################################################################################################################
    
    if plot_switch == "y":
 
        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.semilogx(new_t*1e9-1e7,Absolute_total_Lum)
        pylab.ylabel('Stellar flux (W)')
        pylab.subplot(2,1,2)
        pylab.semilogx(new_t*1e9-1e7,Absolute_XUV_Lum/Absolute_total_Lum)
        pylab.ylabel('XUV/Total Lum.')
        pylab.xlabel('Time (yrs)')

        pylab.figure()
        pylab.ylabel('MMW')
        pylab.semilogx(total_time,total_y[24])
             
        pylab.figure()
        pylab.subplot(6,1,1)
        pylab.ylabel('Mass H2O solid, kg')
        pylab.semilogx(total_time,total_y[0])
        pylab.subplot(6,1,2)
        pylab.ylabel('H2O reservoir (kg)')
        pylab.semilogx(total_time,total_y[1],'r',label='Mass H2O, magma ocean + atmo')
        pylab.semilogx(total_time,MH2O_liq,'b',label='Mass H2O, magma ocean')
        pylab.semilogx(total_time,Pressre_H2O *4 * (0.018/total_y[24]) * np.pi * (rp**2/g) ,label='Mass H2O atmosphere')
        pylab.semilogx(total_time,MH2O_crystal,label= 'crystal H2O')
        pylab.semilogx(total_time,total_y[0] +MH2O_liq + Pressre_H2O *4 * (0.018/total_y[24]) * np.pi * (rp**2/g)+MH2O_crystal ,'g--' ,label='Total H2O (kg)')
        pylab.legend()
        pylab.subplot(6,1,3)
        pylab.ylabel('Radius of solidification (m)')
        pylab.semilogx(total_time,total_y[2])
        pylab.subplot(6,1,4)
        pylab.ylabel("Pressure (bar)")
        pylab.loglog(total_time,Mass_O_atm*g/(4*(0.032/total_y[24])*np.pi*rp**2*1e5),label='fO2')
        pylab.loglog(total_time,Pressre_H2O/1e5,label='fH2O')
        pylab.loglog(total_time,total_y[23]/1e5,label='fCO2')
        pylab.loglog(total_time,CO2_Pressure_array/1e5,label='fCO2 total')
        pylab.legend()
        pylab.subplot(6,1,5)
        pylab.ylabel('O reservoir (kg)')
        pylab.semilogx(total_time,total_y[3],'k' ,label = 'Oxygen in solid' )
        pylab.semilogx(total_time,total_y[4],'r' ,label = 'Oxygen in magma ocean + atmo')
        pylab.legend()
        pylab.subplot(6,1,6)
        pylab.ylabel('Solid Fe reservoir (kg)')
        pylab.semilogx(total_time,total_y[5],'k' ,label = 'FeO1.5')
        pylab.semilogx(total_time,total_y[6],'r'  ,label = 'FeO') # iron 2+
        pylab.legend()
        pylab.xlabel('Time (yrs)')
        
        pylab.figure()
        pylab.subplot(3,1,1)
        pylab.plot(total_time,Mass_O_dissolved,'k',label='Magma oc')
        pylab.plot(total_time,Mass_O_atm,'r--', label= "Atmo")
        pylab.plot(total_time,Mass_O_dissolved+Mass_O_atm,'y*',label='Magma oc + atm')
        pylab.semilogx(total_time,total_y[3],'c' ,label = 'Oxygen in solid' )
        pylab.semilogx(total_time,Mass_O_dissolved+Mass_O_atm + total_y[3],'m' ,label = 'Total free O' )
        pylab.xlabel('Time (yrs)')
        pylab.ylabel('Free oxgen reservoir (kg)')
        pylab.legend()
        
        iron3 = total_y[5]*56/(56.0+1.5*16.0)
        iron2 = total_y[6]*56/(56.0+16.0)
        iron_ratio = iron3/(iron3+iron2)
        pylab.subplot(3,1,2)
        pylab.title("Fe3+/TotalFe in solid, mol ratio")
        pylab.semilogx(total_time,iron_ratio,'k')
        pylab.xlabel('Time (yrs)')
        
        pylab.subplot(3,1,3)
        pylab.ylabel('CO2 reservoir (kg)')
        pylab.semilogx(total_time,total_y[12],'r',label='Mass CO2, magma ocean + atmo')
        pylab.semilogx(total_time,MCO2_liq,'b',label='Mass CO2, magma ocean')
        pylab.semilogx(total_time,total_y[23] *4 *(0.044/total_y[24]) * np.pi * (rp**2/g) ,label='Mass CO2 atmosphere')
        pylab.semilogx(total_time,CO2_Pressure_array* 4 *(0.044/total_y[24])* np.pi* (rp**2/g),label= 'Mass CO2 volatiles')
        pylab.semilogx(total_time,MCO2_crystal,label= 'crystal CO2')
        pylab.semilogx(total_time,total_y[13],label= 'Mantle CO2')
        pylab.semilogx(total_time,total_y[13] +MCO2_liq + CO2_Pressure_array *4 *(0.044/total_y[24])* np.pi * (rp**2/g) +MCO2_crystal,'g--' ,label='Total CO2 (kg)')
        pylab.legend()
        
        pylab.figure()
        pylab.subplot(7,1,1)
        pylab.semilogx(total_time,total_y[7])
        pylab.ylabel("Mantle potential temperature (K)")

        pylab.subplot(7,1,2)
        pylab.semilogx(total_time,total_y[2])
        pylab.ylabel("Radius of solidification (m)")

        pylab.subplot(7,1,3)
        pylab.semilogx(total_time,total_y[8])
        pylab.ylabel("Surface temperature (K)")

        pylab.xlabel("Time (yrs)")
        pylab.subplot(7,1,4)
        pylab.semilogx(total_time,total_y[7],'b')
        pylab.semilogx(total_time,total_y[8],'r')

        pylab.subplot(7,1,5)
        pylab.semilogx(total_time,water_frac)
        pylab.subplot(7,1,6)
        pylab.semilogx(total_time,water_frac*Pressre_H2O *4 *(0.018/total_y[24])* np.pi * (rp**2/g) , 'g' ,label = 'Atmospheric H2O' )
        pylab.semilogx(total_time,(1 - water_frac)*Pressre_H2O *4 *(0.018/total_y[24])* np.pi * (rp**2/g),'m' ,label = 'Ocean H2O' )
        pylab.ylabel("Surface water (kg)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()
        pylab.subplot(7,1,7)
        pylab.loglog(total_time,total_y[9] , 'b' ,label = 'OLR' )
        pylab.loglog(total_time,total_y[10] , 'r' ,label = 'ASR' )
        pylab.loglog(total_time,total_y[11] , 'g' ,label = 'qm' )
        pylab.ylabel("OLR and ASR (W/m2)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()

        pylab.figure()
        pylab.subplot(3,1,1)
        pylab.title('qm')
        pylab.loglog(total_time,total_y[11])

        pylab.xlim([1.0, np.max(total_time)])
        pylab.subplot(3,1,2)
        pylab.loglog(total_time,total_y[14]/1000.0)
        pylab.title('delta_u')

        pylab.xlim([1.0, np.max(total_time)])
        pylab.subplot(3,1,3)
        pylab.title('Ra')
        pylab.loglog(total_time,total_y[15])

        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.figure()
        pylab.semilogx(total_time,total_y[9] - total_y[10] - total_y[11] , 'k' ,label = 'net loss' )
        pylab.ylabel("Radiation balance (W/m2)")
        pylab.xlabel("Time (yrs)")
        pylab.legend()
        
        
        pylab.figure()
        pylab.subplot(2,1,1)
        pylab.semilogx(total_time,Max_depth/1000.0,'r-',label='Max depth')
        pylab.semilogx(total_time,Ocean_depth/1000.0,'b',label='Ocean depth')
        pylab.legend()
        pylab.subplot(2,1,2)
        pylab.semilogx(total_time,Ocean_fraction,'k-',label='Ocean fraction')
        pylab.semilogx(total_time,total_time*0 + (2.5/11.4)**0.25,'b-',label='Modern Earth')
        pylab.legend()
        
        pylab.figure()
        pylab.subplot(5,1,1)
        pylab.semilogx(total_time,total_y[7],'b',label = "Mantle Temp.")
        pylab.semilogx(total_time,total_y[8],'r',label = "Surface Temp.")
        pylab.ylabel("Temperature (K)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.subplot(5,1,2)
        pylab.semilogx(total_time,total_y[2])
        pylab.ylabel("Radius of solidification (m)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.subplot(5,1,3)
        pylab.ylabel("Pressure (bar)")
        pylab.loglog(total_time,Mass_O_atm*g/(4*(0.032/total_y[24])*np.pi*rp**2*1e5),label='fO2')
        pylab.loglog(total_time,water_frac*Pressre_H2O/1e5,label='fH2O')
        pylab.loglog(total_time,total_y[23]/1e5,label='fCO2')
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.subplot(5,1,4)
        pylab.ylabel('Solid Fe reservoir (kg)')
        pylab.semilogx(total_time,total_y[5],'k' ,label = 'FeO1.5')
        pylab.semilogx(total_time,total_y[6],'r'  ,label = 'FeO') # iron 2+
        pylab.xlabel('Time (yrs)')
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.subplot(5,1,5)
        pylab.ylabel('C fluxes')
        pylab.semilogx(total_time,total_y[14],'k' ,label = 'Weathering/Escape')
        pylab.semilogx(total_time,total_y[15],'r' ,label = 'Outgassing')
        pylab.xlabel('Time (yrs)')
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
             
        pylab.figure()
        pylab.subplot(4,1,1)
        pylab.title('CO2 and H2O atmosphere')
        pylab.semilogx(total_time,total_y[7],'b',label = "Tp, my model")
        pylab.semilogx(total_time,total_y[8],'r',label = "Ts, my model")
        pylab.ylabel("Temperature (K)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.legend()
        
        pylab.subplot(4,1,2)
        pylab.semilogx(total_time,total_y[2]/1000,'b')
        pylab.semilogx(total_time,total_y[16]/1000,'r') #radius check
        pylab.ylabel("Solidus radius (km)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        
        pylab.subplot(4,1,3)
        pylab.loglog(total_time,f_O2_FMQ,'b',label = 'FMQ')
        pylab.loglog(total_time,f_O2_IW,'r',label = 'IW') 
        pylab.loglog(total_time,f_O2_MH,'k',label = 'MH') 
        pylab.loglog(total_time,f_O2_mantle,'g--', label = 'fO2') 
        pylab.ylabel("Oxygen fugacity mantle")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()

        pylab.subplot(4,1,4)
        pylab.title('Redox Budget')
        pylab.semilogx(total_time,total_y[18]*365*24*60*60/(0.032*1e12),'g' ,label = 'Dry crustal')
        pylab.semilogx(total_time,total_y[19]*365*24*60*60/(0.032*1e12),'k' ,label = 'Net Escape')
        pylab.semilogx(total_time,total_y[20]*365*24*60*60/(0.032*1e12),'b' ,label = 'Wet crustal oxidation')
        pylab.semilogx(total_time,total_y[21]*365*24*60*60/(0.032*1e12),'r' ,label = 'Outgassing')
        pylab.semilogx(total_time,(total_y[18]+total_y[19]+total_y[20]+total_y[21])*365*24*60*60/(0.032*1e12),'c--' ,label = 'Net')
        pylab.ylabel("O2 flux (Tmol/yr)")
        pylab.xlabel("Time (yrs)")
        pylab.xlim([1.0, np.max(total_time)])
        pylab.legend()
        
        pylab.figure()
        pylab.loglog(total_time,total_y[16],'r') #radius check
        pylab.figure()
        pylab.title('Fe conservation test')
        pylab.loglog(total_time,total_y[5]/(0.056+1.5*0.016),'b')
        pylab.loglog(total_time,total_y[6]/(0.056+0.016),'r')
        pylab.loglog(total_time,total_y[5]/(0.056+1.5*0.016)+total_y[6]/(0.056+0.016),'k')

        pylab.figure()
        pylab.loglog(total_time,total_y[22],'r') #radius check
        pylab.loglog(total_time,fO2_array,'b--')

        pylab.show()

    ### Return outputs         
    output_class = Model_outputs(total_time,total_y,FH2O_array,FCO2_array,MH2O_liq,MH2O_crystal,MCO2_liq,Pressre_H2O,CO2_Pressure_array,fO2_array,Mass_O_atm,Mass_O_dissolved,water_frac,Ocean_depth,Max_depth,Ocean_fraction)
    return output_class
