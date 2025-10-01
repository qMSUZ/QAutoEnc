#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#

#/***************************************************************************
# *   Copyright (C) 2024 -- 2025   by Marek Sawerwain                       *
# *                                  <M.Sawerwain@gmail.com>                *
# *                                  <M.Sawerwain@issi.uz.zgora.pl>         *
# *                                                                         *
# *                                                                         *
# *   Part of the QAutoEnc:                                                *
# *         https://github.com/qMSUZ/QAutoEnc                               *
# *                                                                         *
# * Permission is hereby granted, free of charge, to any person obtaining   *
# * a copy of this software and associated documentation files              *
# * (the “Software”), to deal in the Software without restriction,          *
# * including without limitation the rights to use, copy, modify, merge,    *
# * publish, distribute, sublicense, and/or sell copies of the Software,    *
# * and to permit persons to whom the Software is furnished to do so,       *
# * subject to the following conditions:                                    *
# *                                                                         *
# * The above copyright notice and this permission notice shall be included *
# * in all copies or substantial portions of the Software.                  *
# *                                                                         *
# * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS *
# * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF              *
# * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  *
# * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY    *
# * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,    *
# * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH           *
# * THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.              *
# ***************************************************************************/



import PolLettDS as pld
#import entdetector as ed

import numpy as np
import cudaq
import timeit

from typing import List

import matplotlib.pyplot as plt
#import shelve

from skimage import img_as_float
from skimage.metrics import structural_similarity as structural_similarity_ssim
from skimage.metrics import mean_squared_error

from SupportCode import *
from enum import IntEnum


class QAE_TYPE(IntEnum):
    UNITARY_YXZ                     = 0x1000
    UNITARY_Y                       = 0x1001
    UNITARY_ZYZ                     = 0x1002
    LINEAR_ENTANGLEMENT             = 0x1100
    LINEAR_CYCLIC_ENTANGLEMENT      = 0x1101
    ENTANGLEMENT_EACH_WITH_EACH     = 0x1200
    ENTANGLEMENT_CZ_AND_LATEN_SPACE = 0x1300
    SWAPS                           = 0x1400
    QCNN                            = 0x1500


cost_list = [ ]
angles_matrix = None
max_iterations = None
latent_space = None
ridx_global=None

lrn_idxs=[14]
lrn_idxs_len = len(lrn_idxs)

circuit_variant = None

amplitudes_for_helper_kernel = None
#hamiltonian = None
#cost_values = None
target_qstate = None

input_qstate_00 = None
input_qstate_01 = None
input_qstate_02 = None
input_qstate_03 = None
input_qstate_04 = None

# input_qstate_05 = None
# input_qstate_06 = None
# input_qstate_07 = None
# input_qstate_08 = None
# input_qstate_09 = None

#input_qstates = None

layers = 3
number_of_params = None
number_of_qubits = 12

iteration_number = 0 
max_iterations = 2048

#gradient = cudaq.gradients.CentralDifference()

@cudaq.kernel
def qae_mqubits_test( state : cudaq.State, ridx : List[int] ):
   
    register = cudaq.qvector( input_qstate_00 )
       
    for idx in ridx:
        r = mx( register[idx] )
        if r == 1:
            x( register[idx] )

@cudaq.kernel
def qae_compression(state : cudaq.State, ridx : List[int]):

    register = cudaq.qvector( state )

    # compression, measure operation
    for idx in ridx:
        r = mx( register[idx] )
        if r == 1:
            x( register[idx] )

@cudaq.kernel
def qae_init_state():
    register = cudaq.qvector( input_qstate_00 )

@cudaq.kernel
def qae_init_state00():
    register = cudaq.qvector( input_qstate_00 )

@cudaq.kernel
def qae_init_state01():
    register = cudaq.qvector( input_qstate_01 )

@cudaq.kernel
def qae_init_state02():
    register = cudaq.qvector( input_qstate_02 )

@cudaq.kernel
def qae_init_state03():
    register = cudaq.qvector( input_qstate_03 )

@cudaq.kernel
def qae_init_state04():
    register = cudaq.qvector( input_qstate_04 )

@cudaq.kernel
def qae_variant_unitary_YXZ_encoder(state : cudaq.State, angles: List[float]):

    register = cudaq.qvector( state )

    # encoder part
    angle_idx=0
    for _ in range( layers ):
                              
        for idx in range( number_of_qubits ):
            ry( angles[angle_idx], register[idx] )
            angle_idx = angle_idx+1
    
            rx( angles[angle_idx], register[idx] )
            angle_idx = angle_idx+1   
    
            rz( angles[angle_idx], register[idx] )
            angle_idx = angle_idx+1

@cudaq.kernel
def qae_variant_unitary_YXZ_decoder(state : cudaq.State, angles: List[float]):           
    
        
    #cudaq.adjoint(qae_variant_unitary_YXZ_encoder, state, angles)
    
    register = cudaq.qvector( state )
    
    angle_idx = number_of_params - 1
    for _ in range( layers ):
        
        for idx in range( number_of_qubits-1, -1, -1 ):
    
            rz( -angles[angle_idx], register[idx] )
            angle_idx = angle_idx - 1      
    
            rx( -angles[angle_idx], register[idx] )
            angle_idx = angle_idx - 1      
    
            ry( -angles[angle_idx], register[idx] )
            angle_idx=angle_idx - 1
  
    # # end of qae_variant_unitary_YXZ kernel
  

@cudaq.kernel
def qae_variant_convolution_encoder(state : cudaq.State, angles: List[float]):
    register = cudaq.qvector( state )

    # encoder    

    angle_idx = 0
    noq = number_of_qubits

    for _ in range( layers ):
    
        # Begin of Layer
        # Convolutional layer 
        for idx in range( noq ):
            ry ( angles[ angle_idx ], register[ idx ] )
            angle_idx = angle_idx + 1
    
        for idx in range( noq ):
            rz ( angles[ angle_idx ], register[ idx ] )
            angle_idx = angle_idx + 1
    
        for idx in range( noq-1 ):
            rx.ctrl( angles[ angle_idx ], register[idx], register[idx+1] )
            angle_idx = angle_idx + 1
    
    
        # Pooling layer
        for idx in range( noq-1 ):
             rz.ctrl(angles[ angle_idx ], register[idx], register[idx+1])
             angle_idx = angle_idx + 1
            
             x(register[idx])
            
             rx.ctrl(angles[ angle_idx ], register[idx], register[idx+1])
             angle_idx = angle_idx + 1
        # End of Layer
        
        noq = noq - 1
    

@cudaq.kernel
def qae_variant_convolution_decoder(state : cudaq.State, angles: List[float]):   
    
    register = cudaq.qvector( state )  
    
     # decoder   
     
    angle_idx = number_of_params - 1
    noq = number_of_qubits-(layers-1)
    
    for _ in range( layers ):
    
        # Begin of Layer
        # Pooling layer
        for idx in range( noq-2, -1, -1 ):
            rx.ctrl(-angles[ angle_idx ], register[idx], register[idx+1])
            angle_idx = angle_idx - 1
            
            x(register[idx])
    
            rz.ctrl(-angles[ angle_idx ], register[idx], register[idx+1])
            angle_idx = angle_idx - 1
        
        # Convolutional layer 
        for idx in range( noq-2, -1, -1 ):
            rx.ctrl( -angles[ angle_idx ], register[ idx  ], register[ idx + 1] )
            angle_idx = angle_idx - 1
    
        for idx in range( noq-1, -1, -1 ):
            rz ( -angles[ angle_idx ], register[ idx ] )
            angle_idx = angle_idx - 1
    
        for idx in range( noq-1, -1, -1 ):
            ry ( -angles[ angle_idx ], register[ idx ] )
            angle_idx = angle_idx - 1
    
        # End of Layer
        
        noq = noq + 1
    
# change name
def compute_overlap_probability(initial_state: cudaq.State, evolved_state: cudaq.State):
    """Compute probability of the overlap with the initial state"""
    overlap = initial_state.overlap(evolved_state)
    return np.abs(overlap)**2

def get_number_of_params( autoencoder_type ):
    
    global number_of_qubits
    global latent_space
    global layers
      
    match autoencoder_type:
        case QAE_TYPE.UNITARY_YXZ:
            print("Variant: QAE_UNITARY_YXZ")
            return 3 * number_of_qubits * layers
                       
        case QAE_TYPE.QCNN:
            print("Variant: QAE_QCNN")
            # TODO correct
            noq = number_of_qubits
            rslt=0
            for _ in range(layers):
                rslt+=( 5 * (noq-1) ) + 2  
                noq=noq-1
            return rslt 
        
    return None
        


def save_parameters( params, file_name):
    pass

def cost_function_for_encoder_multi_train_imput( parameters ):
    pass

def cost_function_for_encoder( parameters ):

    global ridx_global    
    global iteration_number
    global cost_list
    global angles_matrix
    global circuit_variant
    
    expectation_value = 0.0
 
    # if circuit_variant == QAE_TYPE.UNITARY_YXZ:
    #     qstate = cudaq.get_state(qae_init_state)
    #     expectation_value = cudaq.observe( qae_variant_unitary_YXZ_encoder,
    #                                    hamiltonian,
    #                                    qstate, parameters ).expectation()

    # if circuit_variant == QAE_TYPE.QCNN:
    #     qstate = cudaq.get_state(qae_init_state)
    #     expectation_value = cudaq.observe( qae_variant_convolution_encoder,
    #                                    hamiltonian,
    #                                    qstate, parameters ).expectation()


    if circuit_variant == QAE_TYPE.UNITARY_YXZ:
          qstate = cudaq.get_state( qae_init_state )
          qstate = cudaq.get_state( qae_variant_unitary_YXZ_encoder, qstate, parameters )
          qstate = cudaq.get_state( qae_compression, qstate, ridx_global)
          qstate = cudaq.get_state( qae_variant_unitary_YXZ_decoder, qstate, parameters )
          expectation_value_tmp  = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state),
                                                            qstate)
          expectation_value = expectation_value + expectation_value_tmp


    if circuit_variant == QAE_TYPE.QCNN:
        
         qstate = cudaq.get_state( qae_init_state00 )
         qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, parameters )
         qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
         qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, parameters )
         expectation_value_tmp = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state00),
                                                           qstate)
         expectation_value = expectation_value + expectation_value_tmp
  
         qstate = cudaq.get_state( qae_init_state01 )
         qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, parameters )
         qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
         qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, parameters )
         expectation_value_tmp = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state01),
                                                           qstate)
         expectation_value = expectation_value + expectation_value_tmp
     
         qstate = cudaq.get_state( qae_init_state02 )
         qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, parameters )
         qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
         qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, parameters )
         expectation_value_tmp = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state02),
                                                           qstate)
         expectation_value = expectation_value + expectation_value_tmp
  
         qstate = cudaq.get_state( qae_init_state03 )
         qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, parameters )
         qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
         qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, parameters )
         expectation_value_tmp = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state03),
                                                           qstate)
         expectation_value = expectation_value + expectation_value_tmp

         qstate = cudaq.get_state( qae_init_state04 )
         qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, parameters )
         qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
         qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, parameters )
         expectation_value_tmp = 1 - compute_overlap_probability( cudaq.get_state(qae_init_state04),
                                                           qstate)
         expectation_value = expectation_value + expectation_value_tmp


    cost_list.append( expectation_value )    
    angles_matrix[ iteration_number ]=parameters
        
    print(f"i{iteration_number} / {max_iterations}: cost: {expectation_value}", end='\r')


    iteration_number = iteration_number + 1
    
    #return op_value
    return expectation_value



def pr_state_as_prob( r, num_of_qubits ):
    
    for idx in range(2 ** num_of_qubits):
        ampl = r[idx]
        if abs(ampl ** 2) != 0.0:
            vtmp=abs(ampl ** 2)
            print( f"{idx:012b}: {vtmp}" )



def training_for_encoder( encoder_type, sign, latent_space ):
    
    global ridx_global
    global circuit_variant
    global input_qstate_00
    global input_qstate_01
    global input_qstate_02
    global input_qstate_03
    global input_qstate_04
    
    #global hamiltonian
    global cost_list
    global number_of_params
    global angles_matrix
    
    circuit_variant = encoder_type
    
    
    loaded_data, loaded_labels, labels_count = pld.load_pol_lett_ds_from_files(
                                                    'pol_lett_ds.bin', 
                                                    'pol_lett_ds_labels.bin' )
    
    
    number_of_params = get_number_of_params( circuit_variant )

    ridx_global = latent_space
    lsf=str(np.array(ridx_global).flatten())[1:-1].replace(' ', '-')
    
    

    # get Ą
    #i=0
    #d=46 + i*80
    
    # # get k
    # i=0
    # d=23 + i*80
    
    # # get Ż
    # i=0
    # d=79 + i*80    

    # # get 4
    # i=0
    # d=4 + i*80

    # # get 9
    # i=0
    # d=9 + i*80
    
    # i=0, 4, 8, 16, 33
    
    # for state 00
    base_idx = pld.base_idx_order[sign]
    i = 0 
    d = base_idx + i * 80

    block_size = 64*64
    a = 64*64 * d
    
    v = loaded_data[a:a+block_size]
    lbl_v = loaded_labels[d]
    lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
    print( "Base index [00]:", pld.base_idx_order[sign])
    print( "Label as char or digit [00]:", pld.get_char_for_label(loaded_labels[d]) )
    
    v_qstate = encode_image_to_qstate( v )
    print( "Norm of v_qstate [00]:", np.linalg.norm(v_qstate) )
    print( "" )
    
    save_image_to_file(v_qstate, f"qae_input_image_bidx_{base_idx}_lspc_{lsf}_00.eps")
    
    input_qstate_00 = v_qstate.copy()
    input_qstate_00 = renormalise_vector(input_qstate_00)
    
    # for state 01
    base_idx = pld.base_idx_order[sign]
    i = 4 
    d = base_idx + i * 80
    
   
    block_size = 64*64
    a = 64*64 * d
    
    v = loaded_data[a:a+block_size]
    lbl_v = loaded_labels[d]
    lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
    print( "Base index [01]:", pld.base_idx_order[sign])
    print( "Label as char or digit [01]:", pld.get_char_for_label(loaded_labels[d]) )
    
    v_qstate = encode_image_to_qstate( v )
    print( "Norm of v_qstate [01]:", np.linalg.norm(v_qstate) )
    print( "" )
    
    save_image_to_file(v_qstate, f"qae_input_image_bidx_{base_idx}_lspc_{lsf}_01.eps")
    
    input_qstate_01 = v_qstate.copy()
    input_qstate_01 = renormalise_vector( input_qstate_01 )

    # for state 02
    base_idx = pld.base_idx_order[sign]
    i = 8 
    d = base_idx + i * 80
    
   
    block_size = 64*64
    a = 64*64 * d
    
    v = loaded_data[a:a+block_size]
    lbl_v = loaded_labels[d]
    lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
    print( "Base index [02]:", pld.base_idx_order[sign])
    print( "Label as char or digit [02]:", pld.get_char_for_label(loaded_labels[d]) )
    
    v_qstate = encode_image_to_qstate( v )
    print( "Norm of v_qstate [02]:", np.linalg.norm(v_qstate) )
    print( "" )
    
    save_image_to_file(v_qstate, f"qae_input_image_bidx_{base_idx}_lspc_{lsf}_02.eps")
    
    input_qstate_02 = v_qstate.copy()
    input_qstate_02 = renormalise_vector( input_qstate_02 )

    # for state 03
    base_idx = pld.base_idx_order[sign]
    i = 16 
    d = base_idx + i * 80
    
   
    block_size = 64*64
    a = 64*64 * d
    
    v = loaded_data[a:a+block_size]
    lbl_v = loaded_labels[d]
    lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
    print( "Base index [03]:", pld.base_idx_order[sign])
    print( "Label as char or digit [03]:", pld.get_char_for_label(loaded_labels[d]) )
    
    v_qstate = encode_image_to_qstate( v )
    print( "Norm of v_qstate [03]:", np.linalg.norm(v_qstate) )
    print( "" )
    
    save_image_to_file(v_qstate, f"qae_input_image_bidx_{base_idx}_lspc_{lsf}_03.eps")
    
    input_qstate_03 = v_qstate.copy()
    input_qstate_03 = renormalise_vector( input_qstate_03 )


    # for state 04
    base_idx = pld.base_idx_order[sign]
    i = 33 
    d = base_idx + i * 80
    
   
    block_size = 64*64
    a = 64*64 * d
    
    v = loaded_data[a:a+block_size]
    lbl_v = loaded_labels[d]
    lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
    print( "Base index [04]:", pld.base_idx_order[sign])
    print( "Label as char or digit [04]:", pld.get_char_for_label(loaded_labels[d]) )
    
    v_qstate = encode_image_to_qstate( v )
    print( "Norm of v_qstate [04]:", np.linalg.norm(v_qstate) )
    print( "" )
    
    save_image_to_file(v_qstate, f"qae_input_image_bidx_{base_idx}_lspc_{lsf}_04.eps")
    
    input_qstate_04 = v_qstate.copy()
    input_qstate_04 = renormalise_vector( input_qstate_04 )

    #
    #
    #
    
    _output_file_name = f"theta_parameters_bidx_{base_idx}"
    angles_matrix=np.ndarray(shape=(max_iterations,  number_of_params ))
    
    print(f"Numer of layers: {layers}")
    
    print( f"Number of angles: {number_of_params}" )
    angles_values = [0] * ( number_of_params )
    #angles_values = list( np.random.rand( number_of_params ) )

        
    
    #
    # example with seperated encoder, compression and decoder parts 
    #
    
    print(f"Measure qubits: ridx_global={ridx_global}")
    #hamiltonian = cudaq.spin.z( ridx_global[0] )
    #for i in range(1, len(ridx_global)):
    #    #print(ridx_global[i])
    #    hamiltonian += cudaq.spin.z( ridx_global[i] )
        
   
    print(f"Max iterations: {max_iterations}")
    print("Optimise: begin")
    
    start_time = time_begin()
    
    optimizer = cudaq.optimizers.COBYLA( )
    optimizer.initial_parameters = angles_values
    optimizer.max_iterations=max_iterations
    result = optimizer.optimize( dimensions = number_of_params, 
                                 function=cost_function_for_encoder )
    
    
    
    end_time = time_end()
    print( "\nOptimise: end." )
    print( "Elapsed time: = {val:.2f}s".format(
                              val=calculate_elapsed_time(end_time, start_time) ) )
    
    
    x_values = list(range(len(cost_list)))
    y_values = cost_list
    
    # # data dump
    # #print("x_values:", x_values)
    # #print("y_values:", y_values)
    
    plt.clf()
    plt.plot(x_values, y_values)
    
    plt.xlabel("Epochs")
    plt.ylabel("Value of cost function")
    plt.savefig( f"cost_values_bidx_{base_idx}_lspc_{lsf}.eps", bbox_inches='tight' )
    
    # save_data_to_file( x_values, "x_values.npy")
    # save_data_to_file( y_values, "y_values.npy")
    save_data_to_file( angles_matrix, f"{_output_file_name}_lspc_{lsf}_matrix.npy" )
    
    # image for lower cost
    
    best_cost_idx = np.argmin( y_values )
    print("Best cost idx:", best_cost_idx)
    print("        value:", y_values[best_cost_idx])
    #print(angles_matrix[best_cost_idx])
    save_data_to_file( angles_matrix[best_cost_idx], f"best_{_output_file_name}_lspc_{lsf}.npy" )
    
    #rslt_qstate = cudaq.get_state( qae_variant_unitary_YXZ, angles_matrix[best_cost_idx], ridx )
 
    if circuit_variant == QAE_TYPE.UNITARY_YXZ:
        qstate = cudaq.get_state( qae_init_state )
        qstate = cudaq.get_state( qae_variant_unitary_YXZ_encoder, qstate, angles_matrix[best_cost_idx] )
        qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
        qstate = cudaq.get_state( qae_variant_unitary_YXZ_decoder, qstate, angles_matrix[best_cost_idx] )
       
    if circuit_variant == QAE_TYPE.QCNN:    
        qstate = cudaq.get_state( qae_init_state )
        qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, angles_matrix[best_cost_idx] )
        qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
        qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, angles_matrix[best_cost_idx] )


    rslt_qstate = qstate

    rslt_qstate_as_numpy_array = np.zeros ( shape=2**number_of_qubits )
    for idx in range(2 ** number_of_qubits):
        ampl = rslt_qstate[idx]
        rslt_qstate_as_numpy_array[idx] = abs(ampl ** 2) 
    
    #print("Probability for angles_matrix[best_cost_idx]")
    #pr_state_as_prob(rslt_qstate, number_of_qubits)    
    
    rslt_qstate_as_numpy_array = data_reorganization( rslt_qstate_as_numpy_array.reshape(64,64).T, number_of_qubits ).flatten()
    save_image_to_file(rslt_qstate_as_numpy_array, f"rslt_best_cost_qstate_bidx_{base_idx}_lspc_{lsf}.eps")
    save_image_to_file_v2(rslt_qstate_as_numpy_array, f"rslt_best_cost_qstate_bidx_{base_idx}_lspc_{lsf}.png")
    

    #
    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
    #
    # calculate measure between input and reconstructed images
    # renormalised vectors of images
    #

    i_imag = input_qstate_00.copy()#.reshape(64,64)
    o_imag = rslt_qstate_as_numpy_array.copy()#.reshape(64,64)
    
    i_imag = i_imag / np.linalg.norm(i_imag)
    o_imag = o_imag / np.linalg.norm(o_imag)

    i_imag = i_imag.reshape(64,64)
    o_imag = o_imag.reshape(64,64)
    
    
    fid_value  = np.vdot( i_imag, o_imag )
    mse_value  = mean_squared_error( i_imag, o_imag )
    ssim_value = structural_similarity_ssim( i_imag, o_imag,
                                            data_range=o_imag.max() - o_imag.min() )
        
    print("\n")
    print(f"Fidelity = {fid_value:.05f}" )
    print(f"MSE      = {mse_value:.05f}" )
    print(f"SSIM     = {ssim_value:.05f}")

    # https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html


def perform_autoencoder( encoder_type, angles_file_name, base_idx, latent_space, verbose=0 ):
    global ridx_global
    global circuit_variant
    global input_qstate_00
    #global hamiltonian
    global cost_list
    global number_of_params
    global angles_matrix

    circuit_variant = encoder_type

    ridx_global = latent_space
    lsf=str(np.array(ridx_global).flatten())[1:-1].replace(' ', '-')

    print( f"PA: Loading angles form file: {angles_file_name}" )
    angles = load_data_from_file( angles_file_name )    
    
    
    loaded_data, loaded_labels, labels_count = pld.load_pol_lett_ds_from_files(
                                                    'pol_lett_ds.bin', 
                                                    'pol_lett_ds_labels.bin' )
        
    number_of_params = get_number_of_params( circuit_variant )
    
    fid_value_list  = [ ]
    mse_value_list  = [ ]
    ssim_value_list = [ ]


    for off_idx in range(0,52):
        
        print(f"PA: begin off_idx={off_idx}")
        d=base_idx + off_idx*80
           
        block_size = 64*64
        a = 64*64 * d
    
        
        v = loaded_data[a:a+block_size]
        lbl_v = loaded_labels[d]
        lbl_v_as_char =  pld.get_char_for_label( loaded_labels[d] )
        if verbose>0:
            print( "PA: Label as char or digit:", pld.get_char_for_label(loaded_labels[d]) )
    
        v_qstate = encode_image_to_qstate( v )
        if verbose>0:
            print( "PA: Norm of v_qstate:", np.linalg.norm(v_qstate) )
            print( "" )
    
            
        input_qstate_00 = v_qstate.copy()
        input_qstate_00 = renormalise_vector(input_qstate_00)
    
    
        if circuit_variant == QAE_TYPE.UNITARY_YXZ:
            qstate = cudaq.get_state( qae_init_state )
            qstate = cudaq.get_state( qae_variant_unitary_YXZ_encoder, qstate, angles )
            qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
            qstate = cudaq.get_state( qae_variant_unitary_YXZ_decoder, qstate, angles )
           
        if circuit_variant == QAE_TYPE.QCNN:    
            qstate = cudaq.get_state( qae_init_state )
            qstate = cudaq.get_state( qae_variant_convolution_encoder, qstate, angles )
            qstate = cudaq.get_state( qae_compression, qstate, ridx_global )
            qstate = cudaq.get_state( qae_variant_convolution_decoder, qstate, angles )
    
        rslt_qstate = qstate
    
        rslt_qstate_as_numpy_array = np.zeros ( shape=2**number_of_qubits )
        for idx in range(2 ** number_of_qubits):
            ampl = rslt_qstate[idx]
            rslt_qstate_as_numpy_array[idx] = abs(ampl ** 2) 
    
        rslt_qstate_as_numpy_array = data_reorganization( rslt_qstate_as_numpy_array.reshape(64,64).T, number_of_qubits ).flatten()
        #save_image_to_file(rslt_qstate_as_numpy_array, "rslt_PA_qstate.eps")
        #save_image_to_file_v2(rslt_qstate_as_numpy_array, "rslt_PA_qstate.png")
        
    
        #
        # https://scikit-image.org/docs/stable/auto_examples/transform/plot_ssim.html
        #
        # calculate measure between input and reconstructed images
        # renormalised vectors of images
        #
    
        i_imag = input_qstate_00.copy()#.reshape(64,64)
        o_imag = rslt_qstate_as_numpy_array.copy()#.reshape(64,64)
        
        i_imag = i_imag / np.linalg.norm(i_imag)
        o_imag = o_imag / np.linalg.norm(o_imag)
    
        i_imag = i_imag.reshape(64,64)
        o_imag = o_imag.reshape(64,64)
        
        
        fid_value  = np.vdot( i_imag, o_imag )
        mse_value  = mean_squared_error( i_imag, o_imag )
        ssim_value = structural_similarity_ssim( i_imag, o_imag,
                                                data_range=o_imag.max() - o_imag.min() )
            
        print(f"PA: Fidelity = {fid_value:.05f}" )
        print(f"PA: MSE      = {mse_value:.05f}" )
        print(f"PA: SSIM     = {ssim_value:.05f}")
        
        fid_value_list.append(fid_value)
        mse_value_list.append(mse_value)
        ssim_value_list.append(ssim_value)
        
        print(f"PA: end off_idx={off_idx}\n")

    print()
    print(f"PA: average of Fidelity", np.average(fid_value_list))
    print(f"PA: average of MSE", np.average(mse_value_list))
    print(f"PA: average of SSIM", np.average(ssim_value_list))

    save_data_to_file(fid_value_list, f"fid_value_list_bidx_{base_idx}_lspc_{lsf}.npy")
    save_data_to_file(mse_value_list, f"mse_value_list_bidx_{base_idx}_lspc_{lsf}.npy")
    save_data_to_file(ssim_value_list, f"ssim_value_list_bidx_{base_idx}_lspc_{lsf}.npy")

###
##    main code begin
###

cudaq.set_target("nvidia", option="mqpu")
target = cudaq.get_target()
num_qpus = target.num_qpus()

print(f"Running on target: {target.name}")
print(f"{cudaq.__version__}")  
print("Number of QPUs:", num_qpus)

#training_for_encoder( QAE_TYPE.UNITARY_YXZ, 'Ą' )
#training_for_encoder( QAE_TYPE.QCNN, 'Ą' )

# ć/14, Ę/52, k/23, L/59, 3/3, 7/7, Ź/78
#training_for_encoder( QAE_TYPE.QCNN, 'ć', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, 'Ę', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, 'k', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, 'L', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, '3', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, '7', [0,1] )
#training_for_encoder( QAE_TYPE.QCNN, 'Ź', [0,1] )

#training_for_encoder( QAE_TYPE.QCNN, 'ć', [0,6] ) 
#training_for_encoder( QAE_TYPE.QCNN, 'Ę', [0,6] )
#training_for_encoder( QAE_TYPE.QCNN, 'k', [0,6] )
#training_for_encoder( QAE_TYPE.QCNN, 'L', [0,6] )
#training_for_encoder( QAE_TYPE.QCNN, '3', [0,6] )
#training_for_encoder( QAE_TYPE.QCNN, '7', [0,6] )
#training_for_encoder( QAE_TYPE.QCNN, 'Ź', [0,6] )


# single image for learning set, Xeon W-2245/4090, ~98.24s, ~75.14s, ~86.86s;
#                                AMD 7950X/A6000,  ~29.57s
# five image for learning set,   Xeon W-2245/4090, ~272.82s
#

# 14, 52, 23, 59, 3, 7, 78
base_idx = 14
latent_space=[0,6]
lsf=str(np.array(latent_space).flatten())[1:-1].replace(' ', '-')
perform_autoencoder( QAE_TYPE.QCNN, f"best_theta_parameters/best_theta_parameters_bidx_{base_idx}_lspc_{lsf}.npy", base_idx, latent_space)
fid_value_list = load_data_from_file(f"fid_value_list_bidx_{base_idx}_lspc_{lsf}.npy")
mse_value_list = load_data_from_file(f"mse_value_list_bidx_{base_idx}_lspc_{lsf}.npy")
ssim_value_list = load_data_from_file(f"ssim_value_list_bidx_{base_idx}_lspc_{lsf}.npy")

print()
print(f"PA: Fidelity for idx=0: {fid_value_list[0]:.05f}" )
print(f"PA:      MSE for idx=0: {mse_value_list[0]:.05f}")
print(f"PA:     SSIM for idx=0: {ssim_value_list[0]:.05f}" )
print()
print(f"PA: average of Fidelity: {np.average(fid_value_list):.05f}" )
print(f"PA:      average of MSE: {np.average(mse_value_list):.05f}")
print(f"PA:     average of SSIM: {np.average(ssim_value_list):.05f}" )
print()
