'''#!/home/magtest/test_env/bin/python'''

from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPSOCSet
from pymatgen import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator, CollinearMagneticStructureAnalyzer
from pymatgen.io.vasp.outputs import Vasprun, Chgcar, Oszicar, Outcar, Potcar
from pymatgen.command_line.bader_caller import bader_analysis_from_objects
import sys
import os
from shutil import copyfile
from subprocess import Popen
import datetime
from time import time, sleep
from ase.io import read, write
from ase.build import make_supercell, sort
import numpy as np
from sympy import Symbol, linsolve
from itertools import combinations
import math
from numba import jit, cuda
from pickle import load, dump


__author__ = "Arnab Kabiraj"
__copyright__ = "Copyright 2019, NSDRL, IISc Bengaluru"
__credits__ = ["Arnab Kabiraj", "Santanu Mahapatra"]

root_path = os.getcwd()
start_time_global = time()

xc = 'PBE_54'
mag_prec = 0.1
enum_prec = 0.001
max_neigh = 5
GPU_accel = False
padding = True
nsim = 4
kpar = 2
ncore = 1
symprec = 1e-8
d_thresh = 0.05
acc = 'default'
LDAUJ_povided = {}
LDAUU_povided = {}
LDAUL_povided = {}


with open('input') as f: 
    for line in f:
        row = line.split()
        if 'structure_file' in line:
            struct_file = row[-1]
        elif 'XC_functional' in line:
            xc = row[-1]
        elif 'VASP_command_std' in line:
            cmd = line[len('VASP_command_std =')+1:-1]
        elif 'VASP_command_ncl' in line:
            cmd_ncl = line[len('VASP_command_ncl =')+1:-1]
        elif 'mag_prec' in line:
            mag_prec = float(row[-1])
        elif 'enum_prec' in line:
            enum_prec = float(row[-1])
        elif 'max_neighbors' in line:
            max_neigh = int(row[-1])+1
        elif 'GPU_accel' in line:
            GPU_accel = row[-1]=='True'
        elif 'more_than_2_metal_layers' in line:
            padding = row[-1]=='True'
        elif 'NSIM' in line:
            nsim = int(row[-1])
        elif 'KPAR' in line:
            kpar = int(row[-1])
        elif 'NCORE' in line:
            ncore = int(row[-1])
        elif 'SYMPREC' in line:
            symprec= float(row[-1])
        elif 'LDAUJ' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUJ_povided[row[i]] = float(row[i+1])
        elif 'LDAUU' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUU_povided[row[i]] = float(row[i+1])
        elif 'LDAUL' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUL_povided[row[i]] = float(row[i+1])
        elif 'same_neighbor_thresh' in line:
            d_thresh = float(row[-1])
        elif 'accuracy' in line:
            acc = row[-1]

#print(LDAUU_povided)
# all functions

def replaceText(fileName,toFind,replaceWith):
    s = open(fileName).read()
    s = s.replace(toFind, replaceWith)
    f = open(fileName, 'w')
    f.write(s)
    f.close()


def writeLog(string):
    string = str(string)
    f = open(root_path+'/log','a+')
    time = datetime.datetime.now()
    f.write(str(time)+'    '+string+'\n')
    f.close()


def dist_neighbors(struct):
    struct_l = struct.copy()
    struct_l.make_supercell([20,20,1])
    distances = np.unique(np.sort(np.around(struct_l.distance_matrix[1],2)))[0:15]
    dr_max = 0.01
    for i in range(len(distances)):
        for j in range(len(distances)):
            dr = np.abs(distances[i]-distances[j])
            if distances[j]<distances[i] and dr<d_thresh:
                distances[i]=distances[j]
                if dr>dr_max:
                    dr_max = dr
    distances = np.unique(distances)
    msg = 'neighbor distances are: '+str(distances)+' ang'
    print(msg)
    writeLog(msg)
    msg = 'treating '+str(dr_max)+' ang separated atoms as same neighbors'
    print(msg)
    writeLog(msg)
    distances[0]=dr_max
    return distances

    
def Nfinder(struct_mag,site,d_N,dr):
    N = len(struct_mag)
    coord_site = struct_mag.cart_coords[site]
    Ns = struct_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        for j in range(N):
            if struct_mag[j].distance(Ns_wrapped[i])<0.01:
                candidates[i] = j
                break
    return candidates


@cuda.jit
def my_kernel(all_coords,coord_N,index):
    """
    Code for kernel.
    """
    pos = cuda.grid(1)
    if pos < all_coords.size:
        if math.sqrt((all_coords[pos]-coord_N[0])**2 + (all_coords[pos+1]-coord_N[1])**2 + (all_coords[pos+2]-coord_N[2])**2) < 0.01:
            index[0] = pos/3


def NfinderGPU(struc_mag,site, d_N, dr):
    coord_site = struc_mag.cart_coords[site]
    Ns = struc_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    #print(Ns)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        coord_N = np.array([Ns_wrapped[i].x,Ns_wrapped[i].y,Ns_wrapped[i].z],dtype='float32')
        index = np.array([-5])
        threadsperblock = 480
        blockspergrid = math.ceil(all_coords.shape[0] / threadsperblock)
        my_kernel[blockspergrid,threadsperblock](all_coords,coord_N,index)
        candidates[i]=index[0]
    return candidates


def find_max_len(lst): 
    maxList = max(lst, key = lambda i: len(i)) 
    maxLength = len(maxList)   
    return maxLength 


def make_homogenous(lst):
    msg = 'finding and padding neighbors'
    print(msg)
    writeLog(msg)
    max_len = find_max_len(lst)
    for i in range(len(lst)):
        if len(lst[i])<max_len:
            pad = [100000]*(max_len-len(lst[i]))
            lst[i] += pad
        print(str(i)+'p / '+str(len(lst)-1))


@jit(nopython=True)
def tFunc(spin_abs,spin_x,spin_y,spin_z,mags,magsqs,T,J2flag,J3flag,J4flag,J5flag):
    for t in range(trange):
        mag = 0
        
        for i in range(N):
            site = np.random.randint(0,N)
            N1s = N1list[site]
            N2s = N2list[site]
            N3s = N3list[site]
            N4s = N4list[site]
            N5s = N5list[site]            
            
            S_current = np.array([spin_x[site],spin_y[site],spin_z[site]])
            u, v = np.random.random(),np.random.random()
            phi = 2*np.pi*u
            theta = np.arccos(2*v-1)
            S_x = spin_abs[site]*np.sin(theta)*np.cos(phi)
            S_y = spin_abs[site]*np.sin(theta)*np.sin(phi)
            S_z = spin_abs[site]*np.cos(theta)
            S_after = np.array([S_x,S_y,S_z])
            E_current = 0
            E_after = 0
            
            for N1 in N1s:
                if N1!=100000 or N1!=-5:
                    S_N1 = np.array([spin_x[N1],spin_y[N1],spin_z[N1]])
                    E_current += -J1*np.dot(S_current,S_N1)
                    E_after += -J1*np.dot(S_after,S_N1)
            if J2flag:
                for N2 in N2s:
                    if N2!=100000 or N2!=-5:
                        S_N2 = np.array([spin_x[N2],spin_y[N2],spin_z[N2]])
                        E_current += -J2*np.dot(S_current,S_N2)
                        E_after += -J2*np.dot(S_after,S_N2)
            if J3flag: 
                for N3 in N3s:
                    if N3!= 100000 or N3!=-5:
                        S_N3 = np.array([spin_x[N3],spin_y[N3],spin_z[N3]])
                        E_current += -J3*np.dot(S_current,S_N3)
                        E_after += -J3*np.dot(S_after,S_N3)
            if J4flag: 
                for N4 in N4s:
                    if N4!= 100000 or N4!=-5:
                        S_N4 = np.array([spin_x[N4],spin_y[N4],spin_z[N4]])
                        E_current += -J4*np.dot(S_current,S_N4)
                        E_after += -J4*np.dot(S_after,S_N4)
            if J5flag: 
                for N5 in N5s:
                    if N5!= 100000 or N5!=-5:
                        S_N5 = np.array([spin_x[N5],spin_y[N5],spin_z[N5]])
                        E_current += -J5*np.dot(S_current,S_N5)
                        E_after += -J5*np.dot(S_after,S_N5)
                                
            E_current += k_x*np.square(S_current[0]) + k_y*np.square(S_current[1]) + k_z*np.square(S_current[2])
            E_after += k_x*np.square(S_x) + k_y*np.square(S_y) + k_z*np.square(S_z)
            del_E = E_after-E_current
                    
            if del_E < 0:
                spin_x[site],spin_y[site],spin_z[site] = S_x,S_y,S_z
            else:
                samp = np.random.random()
                if samp <= np.exp(-del_E/(kB*T)):
                    spin_x[site],spin_y[site],spin_z[site] = S_x,S_y,S_z
                        
                        
        if t>=threshold:
            mag_vec = 2*np.array([np.sum(spin_x),np.sum(spin_y),np.sum(spin_z)])
            mag = np.linalg.norm(mag_vec)
            mags[t-threshold]=np.abs(mag)
            magsqs[t-threshold]=np.square(mag)
            
    return np.mean(mags),np.mean(magsqs)


# main code

msg = '*'*150
print(msg)
writeLog(msg)
msg = '*** this code have been developed by Arnab Kabiraj at Nano-Scale Device Research Laboratory (NSDRL), IISc, Bengaluru, India ***\n'
msg += '*** for any queries please contact the authors at kabiraj@iisc.ac.in or santanu@iisc.ac.in ***'
print(msg)
writeLog(msg)
msg = '*'*150
print(msg)
writeLog(msg)
if acc == 'high':
    msg = '* command for high accuracy detected, the calculations could take significantly more time than ususal\n'
    msg += '* make sure the number of cores is an integer multiple of 4'
    print(msg)
    writeLog(msg)


cell = read(struct_file)
c = cell.get_cell_lengths_and_angles()[2]

for i in range(len(cell)):
    if cell[i].z > c*0.75:
        cell[i].z = cell[i].z - c

cell.center(12.75,2)
write('2D.xsf', sort(cell))

struct = Structure.from_file('2D.xsf')  
mag_enum = MagneticStructureEnumerator(struct,transformation_kwargs={'symm_prec':mag_prec,'enum_precision_parameter':enum_prec})
mag_structs = []

for s_mag in mag_enum.ordered_structures:
    if (s_mag.lattice.c < struct.lattice.c+3.0 and s_mag.lattice.c > struct.lattice.c-3.0):
        mag_structs.append(s_mag)
        
num_struct = len(mag_structs)
if num_struct == 1:
    msg = '*** only one config could be generated, exiting, try out a new material'
    print(msg)
    writeLog(msg)
    sys.exit()
msg = 'total '+str(num_struct)+' configs generated'
print(msg)
writeLog(msg)
sleep_time = 45

num_atoms = []
for struct in mag_structs:
    num_atoms.append(len(struct))
lcm_atoms = np.lcm.reduce(num_atoms)

num_valid_configs = 0

magnetic_list = [Element('Co'), Element('Cr'), Element('Fe'), Element('Mn'), Element('Mo'),
Element('Ni'), Element('V'), Element('W'), Element('Ce'), Element('Os'), Element('Sc'),
Element('Ti'), Element('Ag'), Element('Zr'), Element('Pd'), Element('Rh'), Element('Hf'),
Element('Nb'), Element('Y'), Element('Re'), Element('Cu'), Element('Ru'), Element('Pt'), Element('La')]

LDAUJ_dict = {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0,
'Nb': 0, 'Sc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Cu': 0, 'Y': 0, 'Os': 0, 'Ti': 0, 'Zr': 0, 'Re': 0, 'Hf': 0, 'Pt':0, 'La':0}
if LDAUJ_povided:
    LDAUJ_dict.update(LDAUJ_povided)

LDAUU_dict = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2,
'Nb': 1.45, 'Sc': 4.18, 'Ru': 4.29, 'Rh': 4.17, 'Pd': 2.96, 'Cu': 7.71, 'Y': 3.23, 'Os': 2.47, 'Ti': 5.89, 'Zr': 5.55,
'Re': 1.28, 'Hf': 4.77, 'Pt': 2.95, 'La':5.3}
if LDAUU_povided:
    LDAUU_dict.update(LDAUU_povided)

LDAUL_dict = {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2,
'Nb': 2, 'Sc': 2, 'Ru': 2, 'Rh': 2, 'Pd': 2, 'Cu': 2, 'Y': 2, 'Os': 2, 'Ti': 2, 'Zr': 2, 'Re': 2, 'Hf': 2, 'Pt':2, 'La':2}
if LDAUL_povided:
    LDAUL_dict.update(LDAUL_povided)

relx_dict = {'ISMEAR': 0, 'SIGMA': 0.01, 'ISIF': 4, 'EDIFF': 1E-4, 'POTIM': 0.3,
'EDIFFG': -0.01, 'SYMPREC': 1E-8, 'KPAR': kpar,  'NCORE': ncore, 'NSIM': nsim, 'LCHARG': False, 'ICHARG': 2,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'LWAVE': False,
'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4}

stat_dict = {'ISMEAR': -5, 'EDIFF': 1E-6, 'SYMPREC': 1E-8, 'KPAR': kpar,  'NCORE': ncore, 'NSIM': nsim, 'ICHARG': 2,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'NELM': 120, 'LVHAR': False,
'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4, 'LCHARG': True, 'LWAVE': False, 'LVTOT': False}

mae_dict = {'ISMEAR': -5, 'EDIFF': 1E-8, 'SYMPREC': 1E-8, 'KPAR': kpar,  'NCORE': ncore, 'NSIM': nsim, 'LORBMOM': True, 'LAECHG': False,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict,'NELMIN': 6, 'ICHARG': 1, 'NELM': 250, 'LVHAR': False,
'LDAUPRINT': 1, 'LDAUTYPE': 2, 'LASPH': True, 'LMAXMIX': 4, 'LCHARG': True, 'LWAVE': True, 'ISYM': -1, 'LVTOT': False}

if acc == 'high':
    relx_dict['ALGO'] = 'Normal'
    relx_dict['PREC'] = 'Accurate'
    relx_dict['EDIFF'] = 1E-5
    relx_dict['KPAR'] = 4
    stat_dict['KPAR'] = 4
    mae_dict['KPAR'] = 4


start_time_dft = time()
for i in range(num_struct):
    
    struct_current = mag_structs[i].get_sorted_structure()
    mag_tot = 0
    for j in range(len(struct_current)):
        element = struct_current[j].specie.element
        if element in magnetic_list:
            try:
                mag_tot += struct_current[j].specie.spin
            except Exception:
                msg = '** the structure '+str(i)+' has uneven spins, continuing without it'
                print(msg)
                writeLog(msg)
                mag_tot = 1000
                break

    if i>0 and np.abs(mag_tot) > 0.1:
        msg = '** the structure '+str(i)+' seems ferrimagnetic, continuing without it'
        print(msg)
        writeLog(msg)
        continue

    submission = 0
    p = 0
    relx_path = root_path+'/config_'+str(i)+'/relx'
    
    n = len(struct_current)
    spins = [0]*n
    for j in range(n):
        try:
            spins[j] = struct_current.species[j].spin
        except Exception:
            spins[j] = 0.0
    factor = float(lcm_atoms)/n
    
    if acc == 'high':
        relx = MPRelaxSet(struct_current,user_incar_settings=relx_dict,user_kpoints_settings={'reciprocal_density':300},force_gamma=True,user_potcar_functional=xc)
    else:
        relx = MPRelaxSet(struct_current,user_incar_settings=relx_dict,force_gamma=True,user_potcar_functional=xc)
    relx.write_input(relx_path)
    num_valid_configs += 1

    while 1:
        
        if submission>2:
            nsim_old = nsim
            nsim = max([4,nsim-4])
            replaceText(relx_path+'/INCAR','NSIM = '+str(nsim_old),'NSIM = '+str(nsim))
            msg = 'current NSIM is '+str(nsim_old)+', it will be changed to '+str(nsim)+' in this run'
            print(msg)
            writeLog(msg)
        if submission==5:
            replaceText(relx_path+'/INCAR','POTIM = 0.3','POTIM = 0.1')
            msg = 'reducing POTIM to 0.1'
            print(msg)
            writeLog(msg)
        elif submission==7:
            replaceText(relx_path+'/INCAR','POTIM = 0.1','POTIM = 0.05')
            msg = 'reducing POTIM to 0.05'
            print(msg)
            writeLog(msg)
        elif submission==9:
            replaceText(relx_path+'/INCAR','POTIM = 0.05','POTIM = 0.01')
            msg = 'reducing POTIM to 0.01'
            print(msg)
            writeLog(msg)
        elif submission==11:
            replaceText(relx_path+'/INCAR','SYMPREC = 1e-08','SYMPREC = 1e-05')
            msg = 'increasing SYMPREC'
            print(msg)
            writeLog(msg)
        elif submission==14:
            f = open(relx_path+'/INCAR','a+')
            f.write('\nISYM = 0')
            f.close()
            msg = 'lots of calculation have failed, turning off symmetry'
            print(msg)
            writeLog(msg)
        
        try:
            
            msg = 'checking vasp run status for config_'+str(i)+' relaxation'
            print(msg)
            writeLog(msg)
            try:
                temp_struct = Structure.from_file(relx_path+'/CONTCAR')
                copyfile(relx_path+'/CONTCAR',relx_path+'/CONTCAR.last')
                copyfile(relx_path+'/CONTCAR',relx_path+'/POSCAR')
                msg = 'copied CONTCAR to POSCAR'
                print(msg)
                writeLog(msg)
            except Exception:
                msg = 'no CONTCAR so far'
                print(msg)
                writeLog(msg)
            run = Vasprun(relx_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
            if run.converged:
                msg = 'relaxation finished'
                print(msg)
                writeLog(msg)
                break
            else:
                msg = 'relaxation have not converged, resubmitting'
                print(msg)
                writeLog(msg)
                raise ValueError
        
        except Exception as e:      

            print(e)
            writeLog(e)
            os.chdir(relx_path)
            p = (Popen(cmd, shell=True))
            os.chdir(root_path)
            submission += 1
            msg = 'submitted relaxation for config_'+str(i)+''
            print(msg)
            writeLog(msg)
            msg = 'this is submission #'+str(submission)+' for this config'
            print(msg)
            writeLog(msg)
            while p.poll() is None:
                msg = 'job running, waiting..'
                print(msg)
                sleep(sleep_time)

    path = root_path+'/config_'+str(i)+'/stat'   
    stat_struct = Structure.from_file(relx_path+'/CONTCAR',sort=True)
    stat_struct.add_spin_by_site(spins)
    if acc == 'high':
        stat = MPStaticSet(stat_struct,user_incar_settings=stat_dict,reciprocal_density=1000,force_gamma=True,user_potcar_functional=xc)
    else:
        stat = MPStaticSet(stat_struct,user_incar_settings=stat_dict,reciprocal_density=300,force_gamma=True,user_potcar_functional=xc)
    stat.write_input(path)
    
    p = 0
    submission = 0
    
    while 1:
        
        if submission>2:
            nsim_old = nsim
            nsim = max([4,nsim-4])
            replaceText(path+'/INCAR','NSIM = '+str(nsim_old),'NSIM = '+str(nsim))
            msg = 'current NSIM is '+str(nsim_old)+', it will be changed to '+str(nsim)+' in this run'
            print(msg)
            writeLog(msg)
            
        if submission==5:
            f = open(path+'/INCAR','a+')
            f.write('\nAMIX = 0.2\nBMIX = 0.0001\nAMIX_MAG = 0.8\nBMIX_MAG = 0.0001')
            f.close()
            msg = 'lots of calculation have failed, switching to Linear Mixing'
            print(msg)
            writeLog(msg)
            os.remove(path+'/WAVECAR')
            os.remove(path+'/CHGCAR')
        elif submission==7:
            replaceText(path+'/INCAR','SYMPREC = 1e-08','SYMPREC = 1e-05')
            msg = 'increasing SYMPREC'
            print(msg)
            writeLog(msg)
        elif submission==9:
            f = open(path+'/INCAR','a+')
            f.write('\nISYM = 0')
            f.close()
            msg = 'lots of calculation have failed, turning off symmetry'
            print(msg)
            writeLog(msg)
        
        try:
            
            msg = 'checking vasp run status for config_'+str(i)+' static run'
            print(msg)
            writeLog(msg)
            run = Vasprun(path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
            if run.converged_electronic:
                msg = 'static run finished'
                print(msg)
                writeLog(msg)
                energy = float(run.final_energy)
                energy = energy*factor
                if i==0:
                    E_FM = energy
                elif energy<E_FM:
                    msg = 'config_'+str(i)+' is more stable than FM'
                    print(msg)
                    writeLog(msg)
                    msg = '*** this material is NOT FM, exiting, try out a new material'
                    print(msg)
                    writeLog(msg)
                    sys.exit()
                osz = Oszicar(path+'/OSZICAR')
                config_mag = float(osz.ionic_steps[-1]['mag'])
                if i==0 and np.abs(config_mag)<1.0:
                    msg = '** too low magnetization ('+str(config_mag)+') for an FM config_'+str(i)+', check OUTCAR'
                    print(msg)
                    writeLog(msg)
                elif i>0 and np.abs(config_mag)>0.05:
                    msg = '** too large magnetization ('+str(config_mag)+') for an AFM config_'+str(i)+', check OUTCAR'
                    print(msg)
                    writeLog(msg)
                    
                try:
                    charge = Chgcar.from_file(path+'/CHGCAR')
                except Exception:
                    charge = Chgcar.from_file(path+'/CHGCAR.sp')
                if charge.is_spin_polarized:
                    try:
                        os.rename(path+'/CHGCAR',path+'/CHGCAR.sp')
                        msg = 'renamed CHGCAR to CHGCAR.sp'
                        print(msg)
                        writeLog(msg)
                    except Exception:
                        msg = 'CHGCAR.sp found'

                else:
                    msg = '*** no spin-polarized CHGCAR found for config_'+str(i)+' static run'
                    print(msg)
                    writeLog(msg)
                break
            else:
                msg = 'static run have not converged, resubmitting..'
                print(msg)
                writeLog(msg)
                raise ValueError
        
        except Exception as e:      
            
            print(e)
            writeLog(e)
            os.chdir(path)
            p = (Popen(cmd, shell=True))
            os.chdir(root_path)
            submission += 1
            msg = 'submitted static_run for config_'+str(i)+''
            print(msg)
            writeLog(msg)
            msg = 'this is submission #'+str(submission)+' for this config'
            print(msg)
            writeLog(msg)
            while p.poll() is None:
                msg = 'job running, waiting..'
                print(msg)
                sleep(sleep_time)
                
end_time_dft = time()
time_dft = np.around(end_time_dft - start_time_dft, 2)
msg = 'all relaxations and static runs have finished gracefully, proceeding to MAE calculations now'
print(msg)
writeLog(msg)
msg = 'DFT energy calculations of all possible configurations took total '+str(time_dft)+' s'
print(msg)
writeLog(msg)


submission = 0
p = 0
pre_path = root_path+'/config_0/stat'
path_coll = root_path+'/MAE/coll'

if acc == 'high':
    stat = MPStaticSet.from_prev_calc(pre_path,user_incar_settings=mae_dict,reciprocal_density=1000,force_gamma=True,user_potcar_functional=xc)
else:
    stat = MPStaticSet.from_prev_calc(pre_path,user_incar_settings=mae_dict,reciprocal_density=300,force_gamma=True,user_potcar_functional=xc)
stat.write_input(path_coll)

if os.path.isfile(path_coll+'/WAVECAR'):
    msg = 'collinear WAVECAR exists'
    print(msg)
    writeLog(msg)
if not os.path.isfile(path_coll+'/CHGCAR'):
    try:
        copyfile(pre_path+'/CHGCAR.sp',path_coll+'/CHGCAR')
    except:
        msg = 'no CHGCAR found in stat, continuing but it might fail'
        print(msg)
        writeLog(msg)
else:
    msg = 'collinear CHGCAR exists'
    print(msg)
    writeLog(msg)

start_time_mae = time()
while 1:
    
    if submission>2:
        nsim_old = nsim
        nsim = max([4,nsim-4])
        replaceText(path_coll+'/INCAR','NSIM = '+str(nsim_old),'NSIM = '+str(nsim))
        msg = 'current NSIM is '+str(nsim_old)+', it will be changed to '+str(nsim)+' in this run'
        print(msg)
        writeLog(msg)
    if submission==5:
        f = open(path_coll+'/INCAR','a+')
        f.write('\nAMIX = 0.2\nBMIX = 0.0001\nAMIX_MAG = 0.8\nBMIX_MAG = 0.0001')
        f.close()
        msg = 'lots of calculation have failed, switching to Linear Mixing'
        print(msg)
        writeLog(msg)
        os.remove(path_coll+'/WAVECAR')
        os.remove(path_coll+'/CHGCAR')
        try:
            copyfile(pre_path+'/CHGCAR',path_coll+'/CHGCAR')
        except:
            msg = 'no CHGCAR found in stat, continuing but it might fail'
            print(msg)
            writeLog(msg)
    
    try:
        
        msg = 'checking vasp run status for collinear'
        print(msg)
        writeLog(msg)
        run = Vasprun(path_coll+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        if run.converged_electronic:
            msg = 'collinear run finished'
            print(msg)
            writeLog(msg)
            break
        else:
            msg = 'collinear run have not converged, resubmitting..'
            print(msg)
            writeLog(msg)
            raise ValueError
    
    except:     
        
        os.chdir(path_coll)
        p = (Popen(cmd, shell=True))
        os.chdir(root_path)
        submission += 1
        msg = 'submitted collinear run'
        print(msg)
        writeLog(msg)
        msg = 'this is submission #'+str(submission)
        print(msg)
        writeLog(msg)
        while p.poll() is None:
            msg = 'job running, waiting..'
            print(msg)
            sleep(sleep_time)
            
saxes = [(1,0,0),(0,1,0),(1,1,0),(0,0,1)]

for axis in saxes:
    path = root_path+'/MAE/'+str(axis).replace(' ','')
    submission = 0
    p = 0
    if acc == 'high':
        soc = MPSOCSet.from_prev_calc(path_coll,saxis=axis,nbands_factor=2,reciprocal_density=1000,force_gamma=True,user_potcar_functional=xc)
    else:
        soc = MPSOCSet.from_prev_calc(path_coll,saxis=axis,nbands_factor=2,reciprocal_density=300,force_gamma=True,user_potcar_functional=xc)
    soc.write_input(path)
    replaceText(path+'/INCAR','LCHARG = True','LCHARG = False')
    replaceText(path+'/INCAR','LWAVE = True','LWAVE = False')
    if acc == 'high':
        replaceText(path+'/INCAR','LVTOT = False','KPAR = 4')
    else:
        replaceText(path+'/INCAR','LVTOT = False','KPAR = 2')

    try:
        copyfile(path_coll+'/WAVECAR',path+'/WAVECAR')
    except:
        msg = '*** no collinear WAVECAR found, exiting'
        print(msg)
        writeLog(msg)
        raise ValueError

    while 1:
        
        if submission>2:
            nsim_old = nsim
            nsim = max([4,nsim-4])
            replaceText(path+'/INCAR','NSIM = '+str(nsim_old),'NSIM = '+str(nsim))
            msg = 'current NSIM is '+str(nsim_old)+', it will be changed to '+str(nsim)+' in this run'
            print(msg)
            writeLog(msg)
        if submission==7:
            f = open(path+'/INCAR','a+')
            f.write('\nAMIX = 0.2\nBMIX = 0.0001\nAMIX_MAG = 0.8\nBMIX_MAG = 0.0001')
            f.close()
            msg = 'lots of calculation have failed, applying Linear Mixing'
            print(msg)
            writeLog(msg)
        
        try:
            
            msg = 'checking vasp run status for non-collinear '+str(axis)
            print(msg)
            writeLog(msg)
            run = Vasprun(path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
            if run.converged_electronic:
                msg = 'non-collinear run finished for '+str(axis)
                print(msg)
                writeLog(msg)
                try:
                    os.remove(path+'/CHGCAR')
                    msg = 'removed CHGCAR'
                    print(msg)
                    writeLog(msg)
                except:
                    msg = 'no CHGCAR found to remove'
                    print(msg)
                    writeLog(msg)
                try:
                    os.remove(path+'/WAVECAR')
                    msg = 'removed WAVECAR'
                    print(msg)
                    writeLog(msg)
                except:
                    msg = 'no WAVERCAR found to remove'
                    print(msg)
                    writeLog(msg)
                break
            else:
                msg = 'non-collinear run have not converged for '+str(axis)+', resubmitting..'
                print(msg)
                writeLog(msg)
                raise ValueError
        
        except:     
            
            os.chdir(path)
            p = (Popen(cmd_ncl, shell=True))
            os.chdir(root_path)
            submission += 1
            msg = 'submitted non-collinear run for '+str(axis)
            print(msg)
            writeLog(msg)
            msg = 'this is submission #'+str(submission)+' for this axis'
            print(msg)
            writeLog(msg)
            while p.poll() is None:
                msg = 'job running, waiting..'
                print(msg)
                sleep(sleep_time)
    
end_time_mae = time()
time_mae = np.around(end_time_mae - start_time_mae, 2)
msg='all MAE calculations finished, attempting to fit the Hamiltonian now'
print(msg)
writeLog(msg)
msg = 'the MAE calculations took '+str(time_mae)+' s'
print(msg)
writeLog(msg)


E0 = Symbol('E0')
J1 = Symbol('J1')
J2 = Symbol('J2')
J3 = Symbol('J3')
J4 = Symbol('J4')
J5 = Symbol('J5')

kB = np.double(8.6173303e-5)

num_neigh = min([max_neigh, num_valid_configs])
msg = 'total '+str(num_valid_configs)+' valid FM/AFM configs have been detected, including '+str(num_neigh)+' nearest-neighbors in the fitting'
print(msg)
writeLog(msg)
fitted = False

semifinal_list = []

for i in range(num_struct):
    
    path = root_path+'/config_'+str(i)+'/stat'
    msg = 'checking vasp run status for config_'+str(i)+' static run'
    print(msg)
    writeLog(msg)

    struct = mag_structs[i].get_sorted_structure()
    #print(struct)

    mag_tot = 0
    for j in range(len(struct)):
        element = struct[j].specie.element
        if element in magnetic_list:
            try:
                mag_tot += struct[j].specie.spin
            except Exception:
                msg = '** the structure '+str(i)+' has uneven spins, continuing without it'
                print(msg)
                writeLog(msg)
                mag_tot = 1000
                break

    if i>0 and np.abs(mag_tot) > 0.1:
        msg = '** the structure '+str(i)+' seems ferrimagnetic, continuing without it'
        print(msg)
        writeLog(msg)
        continue
    
    run = Vasprun(path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
    if not run.converged_electronic:
        msg = '*** static run have not converged for config_'+str(i)+', exiting'
        print(msg)
        writeLog(msg)
        raise ValueError
    else:
        msg = 'found converged static run'
        print(msg)
        writeLog(msg)
    energy = float(run.final_energy)

    factor = float(lcm_atoms)/len(struct)
    
    if factor!=int(factor):
        msg = '*** factor is float, '+str(factor)+', exiting'
        print(msg)
        writeLog(msg)
        raise ValueError
    
    energy = energy*factor
    if i==0:
        E_FM = energy
    elif energy<E_FM:
        msg = 'config_'+str(i)+' is more stable than FM'
        print(msg)
        writeLog(msg)
        msg = '*** this material is NOT FM'
        print(msg)
        writeLog(msg)
        
    semifinal_list.append((path,energy,struct))
        
semifinal_list = sorted(semifinal_list, key = lambda x : x[1])

while num_neigh>=2:

    if len(semifinal_list)>num_neigh:
        final_list = semifinal_list[0:num_neigh]
    else:
        final_list = semifinal_list[:]

    num_struct = len(final_list)
    eqn_set = [0]*num_struct

    for i in range(num_struct):
        
        path = final_list[i][0]
        energy = final_list[i][1] 
        struct = final_list[i][2]
        factor = float(lcm_atoms)/len(struct)
        config = path[-13:-5]

        if '_0' in config:
        
            if not os.path.exists(path+'/bader.dat'):
                chgcar = Chgcar.from_file(path+'/CHGCAR.sp')
                if not chgcar.is_spin_polarized:
                    msg = '** '+path+'/CHGCAR.sp is not spin-polarized, exiting'
                    print(msg)
                    writeLog(msg)
                    raise ValueError
                potcar = Potcar.from_file(path+'/POTCAR')
                aeccar0 = Chgcar.from_file(path+'/AECCAR0')
                aeccar2 = Chgcar.from_file(path+'/AECCAR2')
                msg = 'starting bader analysis for '+config
                print(msg)
                writeLog(msg)
                ba = bader_analysis_from_objects(chgcar=chgcar,potcar=potcar,aeccar0=aeccar0,aeccar2=aeccar2)
                msg = 'finished bader analysis successfully'
                print(msg)
                writeLog(msg)
                f = open(path+'/bader.dat','wb')
                dump(ba,f)
                f.close()
                magmom_FM = max(ba['magmom'])
                
            else:
                f = open(path+'/bader.dat','rb')
                ba = load(f)
                f.close()
                magmom_FM = max(ba['magmom'])
                msg = 'read magmoms from file'
                print(msg)
                writeLog(msg)


        osz = Oszicar(path+'/OSZICAR')
        out = Outcar(path+'/OUTCAR')
        config_mag = float(osz.ionic_steps[-1]['mag'])
        if i==0 and np.abs(config_mag)<1.0:
            msg = '** too low magnetization ('+str(config_mag)+') for an FM config_'+str(i)+', check OUTCAR'
            print(msg)
            writeLog(msg)
        elif i>0 and np.abs(config_mag)>0.05:
            msg = '** too large magnetization ('+str(config_mag)+') for an AFM config_'+str(i)+', check OUTCAR'
            print(msg)
            writeLog(msg)

        sites_mag = []
        magmoms_mag = []
        magmoms_out = []
        for j in range(len(struct)):
            element = struct[j].specie.element
            if element in magnetic_list:
                sign_magmom = np.sign(struct[j].specie.spin)
                magmom = sign_magmom*magmom_FM
                magmoms_mag.append(magmom)
                sites_mag.append(struct[j])
                magmoms_out.append(out.magnetization[j]['tot'])
        struct_mag = Structure.from_sites(sites_mag)
        struct_mag_out = Structure.from_sites(sites_mag)
        struct_mag.remove_spin()
        struct_mag.add_site_property('magmom',magmoms_mag)
        struct_mag_out.add_site_property('magmom',magmoms_out)

        N = len(struct_mag)
        msg = config+' with scaling factor '+str(factor)+' = '
        print(msg)
        writeLog(msg)
        print(struct_mag)
        writeLog(struct_mag)
        msg = 'same config with magmoms from OUTCAR is printed below, make sure this does not deviate too much from above'
        print(msg)
        writeLog(msg)
        print(struct_mag_out)
        writeLog(struct_mag_out)
        
        if i==0:
            S_FM = magmom_FM/2.0
            N_FM = float(len(struct_mag))
        
        ds = dist_neighbors(struct_mag)
        dr = ds[0]
        eqn = E0 - energy
        
        for j in range(N):
            site = j
            S_site = struct_mag.site_properties['magmom'][j]/2.0
            if num_struct==2:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = []
                N3s = []
                N4s = []
                N5s = []
            elif num_struct==3:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = []
                N4s = []
                N5s = []
            elif num_struct==4:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = Nfinder(struct_mag,site,ds[3],dr)
                N4s = []
                N5s = []
            elif num_struct==5:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = Nfinder(struct_mag,site,ds[3],dr)
                N4s = Nfinder(struct_mag,site,ds[4],dr)
                N5s = []
            elif num_struct==6:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = Nfinder(struct_mag,site,ds[3],dr)
                N4s = Nfinder(struct_mag,site,ds[4],dr)
                N5s = Nfinder(struct_mag,site,ds[5],dr)
            

            for N1 in N1s:
                S_N1 = struct_mag.site_properties['magmom'][N1]/2.0
                eqn += -0.5*J1*S_site*S_N1*factor
            if N2s:
                for N2 in N2s:
                    S_N2 = struct_mag.site_properties['magmom'][N2]/2.0
                    eqn += -0.5*J2*S_site*S_N2*factor
            if N3s:
                for N3 in N3s:
                    S_N3 = struct_mag.site_properties['magmom'][N3]/2.0
                    eqn += -0.5*J3*S_site*S_N3*factor
            if N4s:
                for N4 in N4s:
                    S_N4 = struct_mag.site_properties['magmom'][N4]/2.0
                    eqn += -0.5*J4*S_site*S_N4*factor
            if N5s:
                for N5 in N5s:
                    S_N5 = struct_mag.site_properties['magmom'][N5]/2.0
                    eqn += -0.5*J5*S_site*S_N5*factor
            
        eqn_set[i] = eqn

    msg = 'mu = '+str(magmom_FM)+' bohr magnetron/magnetic atom'
    print(msg)
    writeLog(msg)
            
    msg = 'eqns are:'
    print(msg)
    writeLog(msg)
    for eqn in eqn_set:
        msg = str(eqn)+' = 0'
        print(msg)
        writeLog(msg)

    if num_struct == 2:
        soln = linsolve(eqn_set, E0, J1)
    elif num_struct == 3:
        soln = linsolve(eqn_set, E0, J1, J2)
    elif num_struct == 4:
        soln = linsolve(eqn_set, E0, J1, J2, J3)
    elif num_struct == 5:
        soln = linsolve(eqn_set, E0, J1, J2, J3, J4)
    elif num_struct == 6:
        soln = linsolve(eqn_set, E0, J1, J2, J3, J4, J5)
    soln = list(soln)
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    print(soln)
    writeLog(soln) 

    if soln and np.max(np.abs(soln[0]))<1e3:
        fitted = True
        break
    else:
        num_neigh -= 1
        msg = 'looks like these set of equations are either not solvable or yielding unphysical values'
        print(msg)
        writeLog(msg)
        msg = 'reducing the number of included NNs to '+str(num_neigh)
        print(msg)
        writeLog(msg)

if not fitted:
    msg = '*** could not fit the Hamiltonian, exiting'
    print(msg)
    writeLog(msg)
    sys.exit()

if num_struct == 2:
    E0 = soln[0][0]
    J1 = soln[0][1]
    J2 = 0
    J3 = 0
    J4 = 0
    J5 = 0
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    msg = 'E0 = '+str(E0)+' eV'
    print(msg)
    writeLog(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds[1])+' ang and NN coordination = '+str(len(N1s))
    print(msg)
    writeLog(msg)

elif num_struct == 3:
    E0 = soln[0][0]
    J1 = soln[0][1]
    J2 = soln[0][2]
    J3 = 0
    J4 = 0
    J5 = 0
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    msg = 'E0 = '+str(E0)+' eV'
    print(msg)
    writeLog(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds[1])+' ang and NN coordination = '+str(len(N1s))
    print(msg)
    writeLog(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds[2])+' ang and NNN coordination = '+str(len(N2s))
    print(msg)
    writeLog(msg)
    
elif num_struct == 4:
    E0 = soln[0][0]
    J1 = soln[0][1]
    J2 = soln[0][2]
    J3 = soln[0][3]
    J4 = 0
    J5 = 0
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    msg = 'E0 = '+str(E0)+' eV'
    print(msg)
    writeLog(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds[1])+' ang and NN coordination = '+str(len(N1s))
    print(msg)
    writeLog(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds[2])+' ang and NNN coordination = '+str(len(N2s))
    print(msg)
    writeLog(msg)
    msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds[3])+' ang and NNNN coordination = '+str(len(N3s))
    print(msg)
    writeLog(msg)

elif num_struct == 5:
    E0 = soln[0][0]
    J1 = soln[0][1]
    J2 = soln[0][2]
    J3 = soln[0][3]
    J4 = soln[0][4]
    J5 = 0
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    msg = 'E0 = '+str(E0)+' eV'
    print(msg)
    writeLog(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds[1])+' ang and NN coordination = '+str(len(N1s))
    print(msg)
    writeLog(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds[2])+' ang and NNN coordination = '+str(len(N2s))
    print(msg)
    writeLog(msg)
    msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds[3])+' ang and NNNN coordination = '+str(len(N3s))
    print(msg)
    writeLog(msg)
    msg = 'J4 = '+str(J4*1e3)+' meV/link with d4 = '+str(ds[4])+' ang and NNNNN coordination = '+str(len(N4s))
    print(msg)
    writeLog(msg)

elif num_struct == 6:
    E0 = soln[0][0]
    J1 = soln[0][1]
    J2 = soln[0][2]
    J3 = soln[0][3]
    J4 = soln[0][4]
    J5 = soln[0][5]
    msg = 'the solutions are:'
    print(msg)
    writeLog(msg)
    msg = 'E0 = '+str(E0)+' eV'
    print(msg)
    writeLog(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds[1])+' ang and NN coordination = '+str(len(N1s))
    print(msg)
    writeLog(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds[2])+' ang and NNN coordination = '+str(len(N2s))
    print(msg)
    writeLog(msg)
    msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds[3])+' ang and NNNN coordination = '+str(len(N3s))
    print(msg)
    writeLog(msg)
    msg = 'J4 = '+str(J4*1e3)+' meV/link with d4 = '+str(ds[4])+' ang and NNNNN coordination = '+str(len(N4s))
    print(msg)
    writeLog(msg)
    msg = 'J5 = '+str(J5*1e3)+' meV/link with d5 = '+str(ds[5])+' ang and NNNNNN coordination = '+str(len(N5s))
    print(msg)
    writeLog(msg)

if ds[1]/ds[2] >= 0.8:
    msg = '** d1/d2 is greater than 0.8, consider adding the 2nd neighbor for accurate results'
    print(msg)
    writeLog(msg)
elif ds[1]/ds[3] >= 0.7:
    msg = '** d1/d3 is greater than 0.7, consider adding the 3rd neighbor for accurate results'
    print(msg)
    writeLog(msg)

saxes = [(1,0,0),(0,1,0),(1,1,0),(0,0,1)]
energies_ncl = [0]*len(saxes)

for i in range(len(saxes)):
    axis = saxes[i]
    path = root_path+'/MAE/'+str(axis).replace(' ','')
    msg = 'checking vasp run status for non-collinear '+str(axis)
    print(msg)
    writeLog(msg)
    run = Vasprun(path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
    energies_ncl[i] = float(run.final_energy)

EMA = saxes[np.argmin(energies_ncl)]
msg = 'the easy magnetization axis is '+str(EMA)
print(msg)
writeLog(msg)

E_100_001 = (energies_ncl[0] - energies_ncl[3])/N_FM
E_010_001 = (energies_ncl[1] - energies_ncl[3])/N_FM
E_110_001 = (energies_ncl[2] - energies_ncl[3])/N_FM
msg = 'magnetocrystalline anisotropic energies are:'
print(msg)
writeLog(msg)
msg = 'E[100]-E[001] = '+str(E_100_001*1e6)+' ueV/magnetic_atom'
print(msg)
writeLog(msg)
msg = 'E[010]-E[001] = '+str(E_010_001*1e6)+' ueV/magnetic_atom'
print(msg)
writeLog(msg)
msg = 'E[110]-E[001] = '+str(E_110_001*1e6)+' ueV/magnetic_atom'
print(msg)
writeLog(msg)

if np.around(E_100_001,9) == np.around(E_010_001,9) and np.around(E_100_001,9) == np.around(E_110_001,9):
    TC_XY = ((final_list[1][1]-final_list[0][1])/(N*factor))*(0.89/(8*kB))
    print(lcm_atoms)
    msg = 'this material seems to be an XY magnet, the calculated TC is '+str(TC_XY)+' k, not performing MC'
    print(msg)
    writeLog(msg)
    sys.exit()

if not os.path.exists(root_path+'/input_MC'):
    msg = 'no input_MC file detected, writing this'
    print(msg)
    writeLog(msg)

    energies_xyz = [energies_ncl[0], energies_ncl[1], energies_ncl[3]]
    k_x = (energies_ncl[0]-np.max(energies_xyz))/(np.square(S_FM)*N_FM)
    k_y = (energies_ncl[1]-np.max(energies_xyz))/(np.square(S_FM)*N_FM)
    k_z = (energies_ncl[3]-np.max(energies_xyz))/(np.square(S_FM)*N_FM)
    T_MF = (S_FM*(S_FM+1)/(3*kB))*(J1*len(N1s)) + (S_FM*(S_FM+1)/(3*kB))*(J2*len(N2s)) + (S_FM*(S_FM+1)/(3*kB))*(J3*len(N3s)) + (
    S_FM*(S_FM+1)/(3*kB))*(J4*len(N4s)) + (S_FM*(S_FM+1)/(3*kB))*(J5*len(N5s))

    f = open('input_MC','w+')
    f.write('directory = MC_Heisenberg\n')
    f.write('repeat = 50 50 1\n')
    f.write('restart = 0\n')
    f.write('J1 (eV/link) = '+str(J1)+'\n')
    f.write('J2 (eV/link) = '+str(J2)+'\n')
    f.write('J3 (eV/link) = '+str(J3)+'\n')
    f.write('J4 (eV/link) = '+str(J4)+'\n')
    f.write('J5 (eV/link) = '+str(J5)+'\n')
    f.write('EMA = '+str(EMA)+'\n')
    f.write('k_x (eV/atom) = '+str(k_x)+'\n')
    f.write('k_y (eV/atom) = '+str(k_y)+'\n')
    f.write('k_z (eV/atom) = '+str(k_z)+'\n')
    f.write('T_start (K) = 1e-6\n')
    f.write('T_end (K) = '+str(T_MF)+'\n')
    f.write('div_T = 25\n')
    f.write('mu (mu_B/atom) = '+str(S_FM*2)+'\n')
    f.write('MCS = 100000\n')
    f.write('thresh = 10000\n')
    f.close()

    msg = 'successfully written input_MC, now will try to run Monte-Carlo based on this'
    print(msg)
    writeLog(msg)
    msg = 'if you want to run the MC with some other settings, make the neccesarry changes in input_MC and stop and re-run this script'
    print(msg)
    writeLog(msg)
    sleep(3)

else:
    msg = 'existing input_MC detected, will try to run the MC based on this'
    print(msg)
    writeLog(msg)
    sleep(3)


with open('input_MC') as f: 
    for line in f:
        row = line.split()
        if 'directory' in line:
            path = root_path+'/'+row[-1]
        elif 'restart' in line:
            restart = int(row[-1])
        elif 'repeat' in line:
            rep_z = int(row[-1])
            rep_y = int(row[-2])
            rep_x = int(row[-3])
        elif 'J1' in line:
            J1 = np.double(row[-1])
        elif 'J2' in line:
            J2 = np.double(row[-1])
        elif 'J3' in line:
            J3 = np.double(row[-1])
        elif 'J4' in line:
            J4 = np.double(row[-1])
        elif 'J5' in line:
            J5 = np.double(row[-1])
        elif 'k_x' in line:
            k_x = np.double(row[-1])
        elif 'k_y' in line:
            k_y = np.double(row[-1])
        elif 'k_z' in line:
            k_z = np.double(row[-1])
        elif 'T_start' in line:
            Tstart = float(row[-1])
        elif 'T_end' in line:
            Trange = float(row[-1])
        elif 'div_T' in line:
            div_T = int(row[-1])
        elif 'mu' in line:
            mu = float(row[-1])
        elif 'MCS' in line:
            trange = int(row[-1])
        elif 'thresh' in line:
            threshold = int(row[-1])

if os.path.exists(path):
    new_name = path+str(time())
    os.rename(path,new_name)
    msg = 'found an old MC directory, renaming it to '+new_name
    print(msg)
    writeLog(msg)
os.makedirs(path)
        
repeat = [rep_x,rep_y,rep_z]
S = mu/2

struc = Structure.from_file('2D.xsf')
os.chdir(path)

if restart==0:
    analyzer = CollinearMagneticStructureAnalyzer(struc,overwrite_magmom_mode='replace_all_if_undefined')
    struc_mag = analyzer.get_structure_with_only_magnetic_atoms()
    struc_mag.make_supercell(repeat)
    N = len(struc_mag)
    dr_max = ds[0]
    d_N1 = ds[1]
    d_N2 = ds[2]
    d_N3 = ds[3]
    d_N4 = ds[4]
    d_N5 = ds[5]
    all_coords = [0]*N
    for i in range(N):
        all_coords[i] = [struc_mag[i].x,struc_mag[i].y,struc_mag[i].z]
    all_coords = np.array(all_coords,dtype='float32')
    all_coords = all_coords.flatten()
    N1list = [[1,2]]*N
    N2list = [[1,2]]*N
    N3list = [[1,2]]*N
    N4list = [[1,2]]*N
    N5list = [[1,2]]*N

    if GPU_accel:
        nf = NfinderGPU
        msg = 'neighbor mapping will try to use GPU acceleration'
        print(msg)
        writeLog(msg)
    else:
        nf = Nfinder
        msg = 'neighbor mapping will be sequentially done in CPU, can be quite slow'
        print(msg)
        writeLog(msg)
    start_time_map = time()
    for i in range(N):
        N1list[i] = nf(struc_mag,i,d_N1,dr_max)
        if J2!=0:
            N2list[i] = nf(struc_mag,i,d_N2,dr_max)
        if J3!=0:
            N3list[i] = nf(struc_mag,i,d_N3,dr_max)
        if J4!=0:
            N4list[i] = nf(struc_mag,i,d_N4,dr_max)
        if J5!=0:
            N5list[i] = nf(struc_mag,i,d_N5,dr_max)
        print(str(i)+' / '+str(N-1))

    if padding:
        msg = 'anticipating inhomogenous number of neighbors for some atoms, trying padding'
        print(msg)
        writeLog(msg)
        make_homogenous(N1list)
        make_homogenous(N2list)
        make_homogenous(N3list)
        make_homogenous(N4list)
        make_homogenous(N5list)

    end_time_map = time()
    time_map = np.around(end_time_map - start_time_map, 2)
    with open('N1list', 'wb') as f:
        dump(N1list, f)
    with open('N2list', 'wb') as f:
        dump(N2list, f)
    with open('N3list', 'wb') as f:
        dump(N3list, f)
    with open('N4list', 'wb') as f:
        dump(N4list, f)
    with open('N5list', 'wb') as f:
        dump(N5list, f)
    msg = 'neighbor mapping finished and dumped'
    print(msg)
    writeLog(msg)
    msg = 'the neighbor mapping process for a '+str(N)+' site lattice took '+str(time_map)+' s'
    print(msg)
    writeLog(msg)

else:
    with open('N1list', 'rb') as f:
        N1list = load(f)
    with open('N2list', 'rb') as f:
        N2list = load(f)
    with open('N3list', 'rb') as f:
        N3list = load(f)
    with open('N4list', 'rb') as f:
        N4list = load(f)
    with open('N5list', 'rb') as f:
        N5list = load(f)
    N = len(N1list)
    print('neighbor mapping successfully read')

N1list = np.array(N1list)
N2list = np.array(N2list)
N3list = np.array(N3list)
N4list = np.array(N4list)
N5list = np.array(N5list)

temp = N1list.flatten()
corrupt = np.count_nonzero(temp == -5)
msg = 'the amount of site corruption in NNs is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
print(msg)
writeLog(msg)
if J2!=0:
    temp = N2list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in NNNs is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    print(msg)
    writeLog(msg)
if J3!=0:
    temp = N3list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in NNNNs is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    print(msg)
    writeLog(msg)
if J4!=0:
    temp = N4list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in NNNNNs is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    print(msg)
    writeLog(msg)
if J5!=0:
    temp = N5list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in NNNNNNs is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    print(msg)
    writeLog(msg)

Ts = np.linspace(Tstart,Trange,div_T)
Ms = []
Xs = []

start_time_mc = time()
for T in Ts:
    spin_abs = S*np.ones(N)
    spin_x = np.zeros(N)
    spin_y = np.zeros(N)
    spin_z = S*np.ones(N)
    mags = np.zeros(trange-threshold)
    magsqs = np.zeros(trange-threshold)
    M,Msq=tFunc(spin_abs,spin_x,spin_y,spin_z,mags,magsqs,T,J2!=0,J3!=0,J4!=0,J5!=0)

    X = (Msq-np.square(M))/T
    Ms.append(M)
    Xs.append(X)
    print(str(T)+'    '+str(M)+'    '+str(X))
    f = open(str(struc.formula).replace(' ','').replace('1','')+'_'+str(int(np.floor(Tstart)))+'-'+str(int(np.floor(Trange)))+'.dat','a+')
    f.write(str(T)+'    '+str(M)+'    '+str(X)+'\n')
    f.close()

end_time_mc = time()
time_mc = np.around(end_time_mc - start_time_mc, 2)
msg = 'MC simulation have finished, analyse the output to determine the Curire temp.'
print(msg)
writeLog(msg)
msg = 'the MC simulation took '+str(time_mc)+' s'
print(msg)
writeLog(msg)

end_time_global = time()
time_global = np.around(end_time_global - start_time_global, 2)

msg = 'the whole end-to-end process took '+str(time_global)+' s'
print(msg)
writeLog(msg)

