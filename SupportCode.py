# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:15:25 2025

@author: msawe
"""


import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

data_lead_path="./data_e1"
PolLettDS_data_bin_path = "./PolLetDS_data_bin"

def vector_dot_product(a,b):
    
    f = np.vdot( a, b )
    
    return f

def fidelity_pure_states(a, b):
    return np.abs( vector_dot_product(a,b) ) ** 2.0

def renormalise_vector(a):
    return a/np.linalg.norm(a)    

def file_exists( file_name ):
    r = False
    
    path_to_file = Path( file_name )
    
    if path_to_file.is_file():
        r=True
    
    return r

def save_data_to_file( data_variable, file_name):
    np.save( file_name, data_variable )

def load_data_from_file( file_name):
    l = np.load( file_name )
    return l


def save_image_to_file( image_as_numpy_array, file_name ):
    
    plt.clf()
    
    plt.matshow( image_as_numpy_array.copy().reshape(64,64), 
                 cmap='gray', 
                 vmin=0, vmax=max(image_as_numpy_array) )
    
    #plt.xticks( range(0, 64, 2), rotation=90)    
    #plt.grid()
    
    plt.savefig( file_name, bbox_inches='tight')
    plt.close()

def save_image_to_file_v2( image_as_numpy_array, file_name ):
    
    plt.clf()
    
    fig, ax = plt.subplots( ) 
    ax.set_axis_off()
    ax.matshow( image_as_numpy_array.copy().reshape(64,64), 
                 cmap='gray', 
                 vmin=0, vmax=max(image_as_numpy_array) )
    
    
    #plt.xticks( range(0, 64, 2), rotation=90)    
    #plt.grid()
    
    plt.savefig( file_name, bbox_inches='tight' )
    plt.close()
    
def save_image_cuda_qstate_to_file( cuda_qstate,  file_name, number_of_qubits):
    cuda_qstate_as_numpy_array = np.zeros ( shape=2**number_of_qubits )
    for idx in range(2 ** number_of_qubits):
             ampl = cuda_qstate[idx]
             cuda_qstate_as_numpy_array[idx] = abs(ampl ** 2) 
    save_image_to_file(cuda_qstate_as_numpy_array, file_name)


def encode_image_to_qstate( _input_image ):
    
    v = _input_image.copy()
    v = v / np.linalg.norm(v)
    target_qstate = v.copy()
    
    return target_qstate


def create_probability_distribution_for_label( _label : int, number_of_qubits, labels_count ):
    
    size_of_vector = 2**number_of_qubits
    
    prob_distribution = np.zeros( shape = size_of_vector ) 
    
    i = int(_label)
       
    for _ in range(0, labels_count // 128):
        
        if i < size_of_vector:
            prob_distribution[ i ] = 1.0
        
        i = i + 128
             
    return encode_image_to_qstate(prob_distribution)

def compare_distro( ind_for_label, ind_for_label_from_QCNN_filtered):
    indices_compatibility = 0
    for v in ind_for_label:
        item_index = np.where(ind_for_label_from_QCNN_filtered == v)
        if item_index[0].size != 0:
            indices_compatibility = indices_compatibility + 1
            
    return indices_compatibility



def data_in_row_reorganize(b):
    a = b.copy()
    
    a[ 0] = b[0]
    a[ 1] = b[32]
    a[ 2] = b[16]
    a[ 3] = b[48]
    
    a[ 4] = b[8]
    a[ 5] = b[40]
    a[ 6] = b[24]
    a[ 7] = b[56]
    
    a[ 8] = b[4]
    a[ 9] = b[36]
    a[10] = b[20]
    a[11] = b[52]
    
    a[12] = b[12]
    a[13] = b[44]
    a[14] = b[28]
    a[15] = b[60]
    
    a[16] = b[2]
    a[17] = b[34]
    a[18] = b[18]
    a[19] = b[50]
    
    a[20] = b[10]
    a[21] = b[42]
    a[22] = b[26]
    a[23] = b[58] 
    
    a[24] = b[6]
    a[25] = b[38]
    a[26] = b[22]
    a[27] = b[54]
    
    a[28] = b[14]
    a[29] = b[46] 
    a[30] = b[30]
    a[31] = b[62]
    
    a[32] = b[1]
    a[33] = b[33]
    a[34] = b[17]
    a[35] = b[49]
    
    a[36] = b[9]
    a[37] = b[41]
    a[38] = b[25]
    a[39] = b[57]
    
    a[40] = b[5]
    a[41] = b[37]
    a[42] = b[21]
    a[43] = b[53]
    
    a[44] = b[13]
    a[45] = b[45]
    a[46] = b[29]
    a[47] = b[61]
    
    a[48] = b[3]
    a[49] = b[35]
    a[50] = b[19]
    a[51] = b[51]
    
    a[52] = b[11]
    a[53] = b[43]
    a[54] = b[27]
    a[55] = b[59]
    
    a[56] = b[7]
    a[57] = b[39]
    a[58] = b[23]
    a[59] = b[55]
    
    a[60] = b[15]
    a[61] = b[47]
    a[62] = b[31]
    a[63] = b[63]    
    
    return a

def data_reorganization(input_array, number_of_qubits):
    output_array = np.zeros( shape=2**number_of_qubits ).reshape(64,64)
    
    output_array[0, :] = data_in_row_reorganize( input_array[0, :] )
    output_array[1, :] = data_in_row_reorganize( input_array[32, :] )
    output_array[2, :] = data_in_row_reorganize( input_array[16, :] )
    output_array[3, :] = data_in_row_reorganize( input_array[48, :] )
    
    output_array[4, :] = data_in_row_reorganize( input_array[8, :] )
    output_array[5, :] = data_in_row_reorganize( input_array[40, :] )
    output_array[6, :] = data_in_row_reorganize( input_array[24, :] )
    output_array[7, :] = data_in_row_reorganize( input_array[56, :] )
    
    output_array[8, :] = data_in_row_reorganize( input_array[4, :] )
    output_array[9, :] = data_in_row_reorganize( input_array[36, :] )
    output_array[10, :] = data_in_row_reorganize( input_array[20, :] )
    output_array[11, :] = data_in_row_reorganize( input_array[52, :] ) 
    
    output_array[12, :] = data_in_row_reorganize( input_array[12, :] ) 
    output_array[13, :] = data_in_row_reorganize( input_array[44, :] ) 
    output_array[14, :] = data_in_row_reorganize( input_array[28, :] ) 
    output_array[15, :] = data_in_row_reorganize( input_array[60, :] ) 
    
    output_array[16, :] = data_in_row_reorganize( input_array[2, :] ) 
    output_array[17, :] = data_in_row_reorganize( input_array[34, :] )
    output_array[18, :] = data_in_row_reorganize( input_array[18, :] )
    output_array[19, :] = data_in_row_reorganize( input_array[50, :] )
    
    output_array[20, :] = data_in_row_reorganize( input_array[10, :] ) 
    output_array[21, :] = data_in_row_reorganize( input_array[42, :] ) 
    output_array[22, :] = data_in_row_reorganize( input_array[26, :] ) 
    output_array[23, :] = data_in_row_reorganize( input_array[58, :] ) 
    
    output_array[24, :] = data_in_row_reorganize( input_array[6, :] )
    output_array[25, :] = data_in_row_reorganize( input_array[38, :] )
    output_array[26, :] = data_in_row_reorganize( input_array[22, :] )
    output_array[27, :] = data_in_row_reorganize( input_array[54, :] )
    
    output_array[28, :] = data_in_row_reorganize( input_array[14, :] )
    output_array[29, :] = data_in_row_reorganize( input_array[46, :] )
    output_array[30, :] = data_in_row_reorganize( input_array[30, :] )
    output_array[31, :] = data_in_row_reorganize( input_array[62, :] )
    
    output_array[32, :] = data_in_row_reorganize( input_array[1, :] )
    output_array[33, :] = data_in_row_reorganize( input_array[33, :] )
    output_array[34, :] = data_in_row_reorganize( input_array[17, :] )
    output_array[35, :] = data_in_row_reorganize( input_array[49, :] )
    
    output_array[36, :] = data_in_row_reorganize( input_array[9, :] )
    output_array[37, :] = data_in_row_reorganize( input_array[41, :] )
    output_array[38, :] = data_in_row_reorganize( input_array[25, :] )
    output_array[39, :] = data_in_row_reorganize( input_array[57, :] )
    
    output_array[40, :] = data_in_row_reorganize( input_array[5, :] )
    output_array[41, :] = data_in_row_reorganize( input_array[37, :] )
    output_array[42, :] = data_in_row_reorganize( input_array[21, :] )
    output_array[43, :] = data_in_row_reorganize( input_array[53, :] )
    
    output_array[44, :] = data_in_row_reorganize( input_array[13, :] )
    output_array[45, :] = data_in_row_reorganize( input_array[45, :] )
    output_array[46, :] = data_in_row_reorganize( input_array[29, :] )
    output_array[47, :] = data_in_row_reorganize( input_array[61, :] )
    
    output_array[48, :] = data_in_row_reorganize( input_array[3, :] )
    output_array[49, :] = data_in_row_reorganize( input_array[35, :] )
    output_array[50, :] = data_in_row_reorganize( input_array[19, :] )
    output_array[51, :] = data_in_row_reorganize( input_array[51, :] )
    
    output_array[52, :] = data_in_row_reorganize( input_array[11, :] )
    output_array[53, :] = data_in_row_reorganize( input_array[43, :] )
    output_array[54, :] = data_in_row_reorganize( input_array[27, :] )
    output_array[55, :] = data_in_row_reorganize( input_array[59, :] )
    
    output_array[56, :] = data_in_row_reorganize( input_array[7, :] )
    output_array[57, :] = data_in_row_reorganize( input_array[39, :] )
    output_array[58, :] = data_in_row_reorganize( input_array[23, :] )
    output_array[59, :] = data_in_row_reorganize( input_array[55, :] )
    
    output_array[60, :] = data_in_row_reorganize( input_array[15, :] )
    output_array[61, :] = data_in_row_reorganize( input_array[47, :] )
    output_array[62, :] = data_in_row_reorganize( input_array[31, :] )
    output_array[63, :] = data_in_row_reorganize( input_array[63, :] )
    
    return output_array


def time_begin():
    return  time.perf_counter()

def time_end():
    return time.perf_counter()

def calculate_elapsed_time(a,b):
    return a-b
