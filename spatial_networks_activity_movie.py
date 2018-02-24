#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the NNGT project to generate and analyze
# neuronal networks and their activity.
# Copyright (C) 2015-2017  Tanguy Fardet
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

''' SpatialNetwork generation, complex shapes '''

import os
import time

import numpy as np
import matplotlib.pyplot as plt

import nngt
from nngt.geometry import Shape

# ~ nngt.use_backend("networkx")
#nngt.use_backend("graph-tool")

# nngt.set_config({"omp": 8, "palette": 'RdYlBu'})
nngt.set_config("multithreading", False)

nngt.seed(0)


''' Runtime options'''
plot_distribution = False # plot of the connectivity distribution
plot_graphs = False # graphic output of spatial network
simulate_activity =  True # whether to run or not a NEST simlation on the model


''' Neuron numbers'''
num_neurons = 500
fraction_excitatory=.8
num_excitatory=int(fraction_excitatory*num_neurons)
num_inhibitory=num_neurons-num_excitatory

''' Neurons  parameters '''
# adaptive spiking neurons
# base_params = {
#     'E_L': -60., 'V_th': -57., 'b': 20., 'tau_w': 100.,
#     'V_reset': -65., 't_ref': 2., 'g_L': 10., 'C_m': 250.
# }

# Parameters from "Recurrently connected and localized neuronal communities 
# initiate coordinated spontaneous activity in neuronal networks" Lonardoni et al PLOS 2016
Lonardoni_exitatory_params = {
    'a': 2.,'E_L': -70., 'V_th': -50., 'b': 60., 'tau_w': 300.,
    'V_reset': -58., 't_ref': 2., 'g_L': 12., 'C_m': 281., 'I_e': 300.
}

Lonardoni_inhibitory_params = {
    'a': 2.,'E_L': -70., 'V_th': -50., 'b': 0., 'tau_w': 30.,
    'V_reset': -58., 't_ref': 2., 'g_L': 10., 'C_m': 200., 'I_e': 300.
}

params1=Lonardoni_exitatory_params

# # oscillators
# params1, params2 = base_params.copy(), base_params.copy()
# params1.update({'E_L': -65., 'b': 40., 'I_e': 200., 'tau_w': 400.})
# # bursters
# params2.update({'b': 30., 'V_reset': -50., 'tau_w': 500.})

''' Create a circular shape to embed the neuros'''
shape = Shape.disk(radius=2500.)


####################################################################
def activity_simulation(net,pop,sim_duration=5000.):
    '''
    Execute activity simulation on the generated graph
    --------------------------------------------------
    Input :   net : nngt generated graph with model
            pop : neurons population
            sim_duration : time (s) duration of simulation

    Output :
            activity graphs
    '''

    if nngt.get_config('with_nest'):
        import nngt.simulation as ns
        import nest

        nest.ResetKernel()

        nest.SetKernelStatus({'local_num_threads': 14})

        gids = net.to_nest()

        #nngt.simulation.randomize_neural_states(net, {"w": ("uniform", 0, 200),
        #  "V_m": ("uniform", -70., -40.)})

        nngt.simulation.randomize_neural_states(net, {"w": ("normal", 50., 5.),
        "V_m": ("uniform", -80., -50.)})
        # add noise to the excitatory neurons
        excs = list(gids)
        #ns.set_noise(excs, 10., 2.)
        #ns.set_poisson_input(gids, rate=3500., syn_spec={"weight": 34.})

        target_rate    = 25.
        average_degree = net.edge_nb() / float(net.node_nb())
        syn_rate       = target_rate / average_degree
        print("syn_rate", syn_rate)
        print("avg_deg", average_degree)

        ns.set_minis(net, base_rate=syn_rate, weight_fraction=.4, gids=gids)
        vm, vs = ns.monitor_nodes([1], "voltmeter")
        recorders, records = ns.monitor_groups(pop.keys(), net)

        nest.Simulate(5000)

        # ~ if nngt.get_config('with_plot'):
            # ~ ns.plot_activity(vm, vs, network=net, show=False)
            # ~ ns.plot_activity(recorders, records, network=net, show=False, hist=True)

        anim = nngt.plot.AnimationNetwork(recorders, net, resolution=0.1, interval=20, decimate=-1)
        # ~ anim.save_movie("structured_bursting.mp4", fps=2, start=650.,
                        # ~ stop=1500, interval=20)
        # ~ anim.save_movie("structured_bursting.mp4", fps=5, num_frames=50)
        anim.save_movie("structured_bursting.mp4", fps=15, interval=10)

        plt.show()

########################################################################@


def with_obstacles(shape,params = {"height": 250., "width": 250.},filling_fraction = 0.4):
    '''
    Network generation and simulation with obsactles
    ------------------------------------------------
    Input :   shape : nngt defined spatial shape
              params : dictionary with obstacles specifications
              filling_fraction : float, fraction of the embedding shape filled with obstacles
  
    Output:   nngt network of neurons
    '''
  
    ''' Create obstacles within the shape'''
  
    shape.random_obstacles(filling_fraction, form="rectangle", params=params,
                        heights=30., etching=20.)
  
  
    ''' Create a Spatial network and seed neurons on top/bottom areas '''
  
    # neurons are reparted proportionaly to the area
    total_area  = shape.area
    bottom_area = np.sum([a.area for a in shape.default_areas.values()])
  
    density     = 300e-6
    num_bottom  = int(density * bottom_area)
  
    pop = nngt.NeuralPop(num_neurons,with_model=True)
  
    # # Instruction si on ne veut pas de modèle
    # pop.set_model(None)
    pop.create_group("bottom", num_bottom, neuron_model="aeif_psc_alpha",neuron_param= params1)
    # Create one group on each top area
    for name, top_area in shape.non_default_areas.items():
        num_top = int(density * top_area.area)
        group_name = "top_" + name
        if num_top:
            pop.create_group(group_name, num_top, neuron_model="aeif_psc_alpha", neuron_param=params1)
  
    # make the graph
    net = nngt.SpatialGraph(shape=shape, population=pop)
  
    # seed neurons
    bottom_pos     = shape.seed_neurons(num_bottom,
                                        on_area=shape.default_areas,
                                        soma_radius=15)
    bottom_neurons = np.array(net.new_node(num_bottom, positions=bottom_pos,
                                           groups="bottom"), dtype=int)

    top_pos = []
    top_neurons = []
    for name, top_area in shape.non_default_areas.items():
        group_name  = "top_" + name
        if group_name in pop:
            num_top     = pop[group_name].size
            top_pos_tmp = top_area.seed_neurons(num_top, soma_radius=15)
            top_pos.extend(top_pos_tmp)
            top_neurons.extend(net.new_node(num_top, positions=top_pos_tmp,
                                            groups=group_name))
    top_pos = np.array(top_pos)
    top_neurons = np.array(top_neurons, dtype=int)
  
  
    ''' Make the connectivity '''
  
    top_scale    = 200.
    bottom_scale = 100.
    mixed_scale  = 150.
  
    base_proba   = 3.
    p_up         = 0.6
    p_down       = 0.9
    p_other_up   = p_down**2
  
    # connect bottom area
    for name, area in shape.default_areas.items():
      # non_default_areas.items()  >>.default_areas.items()
        contained = area.contains_neurons(bottom_pos)
        neurons   = bottom_neurons[contained]
  
        nngt.generation.connect_nodes(net, bottom_neurons, bottom_neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
  
    # connect top areas
    for name, area in shape.non_default_areas.items():
        contained = area.contains_neurons(top_pos)
        neurons   = top_neurons[contained]
        other_top = [n for n in top_neurons if n not in neurons]
        print(name)
        print(neurons)
        if np.any(neurons):
            # connect intra-area
            nngt.generation.connect_nodes(net, neurons, neurons, "distance_rule",
                                          scale=top_scale, max_proba=base_proba)
            # connect between top areas (do it?)
            nngt.generation.connect_nodes(net, neurons, other_top, "distance_rule",
                                          scale=mixed_scale,
                                          max_proba=base_proba*p_other_up)
            # connect top to bottom
            nngt.generation.connect_nodes(net, neurons, bottom_neurons,
                                          "distance_rule", scale=mixed_scale,
                                          max_proba=base_proba*p_down)
            # connect bottom to top
            nngt.generation.connect_nodes(net, bottom_neurons, neurons,
                                          "distance_rule", scale=mixed_scale,
                                          max_proba=base_proba*p_up)
  
    # By default synapses are static in net
    # Here we set the synaptic weigth
    net.set_weights(80.0)
  
    if plot_distribution is True :
    
        ''' Check the degree distribution '''
    
        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='bayes', nodes=top_neurons, show=False)
        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='bayes', nodes=bottom_neurons, show=True)
    
    if plot_graphs is True :
    
        ''' Plot the resulting network and subnetworks '''
    
        restrict = [
            ("bottom", "bottom"), ("top", "bottom"), ("top", "top"), ("bottom", "top")
        ]
    
        for r_source, r_target in restrict:
            nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                                   restrict_sources=r_source,
                                   restrict_targets=r_target, show=False)
    
        fig, axis = plt.subplots()
        count = 0
        for r_source, r_target in restrict:
            show_env = (count == 0)
            nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                                   restrict_sources=r_source,
                                   restrict_targets=r_target,
                                   show_environment=show_env, axis=axis, show=False)
            count += 1
  
     
    # ~ nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5, show=True)
  
  
    # ------------------ #
    # Simulate with NEST #
    # ------------------ #
  
    if simulate_activity is True :
    
        print("Activity simulation")
    
        '''
        Send the network to NEST, monitor and simulate
        '''
        activity_simulation(net,pop)


###################################################################################
def no_obstacles(shape):
    '''
    Network generation and simulation without obsacles
  
    Input:  shape : nngt spatial shape object to embed the neurons
  
    Output: net : nngt network
    '''
  
  
    ''' Create a Spatial network and seed neurons on top/bottom areas '''
  
  
  
    # neurons are reparted proportionaly to the area
    total_area  = shape.area
  
    pop = nngt.NeuralPop(num_neurons,with_model=True)
  
    # # Instruction si on ne veut pas de modèle
    # pop.set_model(None)
    pop.create_group("excitatory", num_excitatory,neuron_model="aeif_psc_alpha",neuron_param= params1)
    pop.create_group("inhibitory", num_inhibitory,neuron_model="aeif_psc_alpha",neuron_param= params1)
  
  
    # make the graph
    net = nngt.SpatialGraph(shape=shape, population=pop)
  
    # seed neurons
    excitatory_pos     = shape.seed_neurons(num_excitatory,
                                        on_area=shape.default_areas,
                                        soma_radius=15)
    excitatory_neurons = np.array(net.new_node(num_excitatory, positions=excitatory_pos,
                                           groups="excitatory"), dtype=int)
    inhibitory_pos        = shape.seed_neurons(num_inhibitory, on_area=shape.non_default_areas,
                                        soma_radius=15)
    inhibitory_neurons    = np.array(net.new_node(num_inhibitory, positions=inhibitory_pos,
                                           groups="inhibitory",ntype=-1), dtype=int)
  
  
    ''' Make the connectivity '''
  
    #top_scale    = 200.
    bottom_scale = 100.
    #mixed_scale  = 150.
  
    base_proba   = 3.
    p_up         = 0.6
    p_down       = 0.9
    p_other_up   = p_down**2
  
    # connect bottom area
    for name, area in shape.default_areas.items():
      # non_default_areas.items()  >>.default_areas.items()
        contained = area.contains_neurons(excitatory_pos)
        neurons   = excitatory_neurons[contained]
  
        nngt.generation.connect_nodes(net, excitatory_neurons, excitatory_neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
  
        nngt.generation.connect_nodes(net, excitatory_neurons, inhibitory_neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
  
        nngt.generation.connect_nodes(net, inhibitory_neurons, inhibitory_neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
        nngt.generation.connect_nodes(net, inhibitory_neurons, excitatory_neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
  
  
    # By default synapses are static in net
    # Here we set the synaptic weigth
    net.set_weights(50.0)
  
    if plot_distribution is True :
    
        ''' Check the degree distribution '''
    
        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='bayes', nodes=inhibitory_neurons, show=False)
        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='bayes', nodes=excitatory_neurons, show=True)
  
    if plot_graphs is True :
    
        ''' Plot the resulting network and subnetworks '''
    
        restrict = [
            ("excitatory", "excitatory"), ("inhibitory", "excitatory"), ("inhibitory", "inhibitory"), ("excitatory", "inhibitory")
        ]
    
        for r_source, r_target in restrict:
            nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                                   restrict_sources=r_source,
                                   restrict_targets=r_target, show=False)
    
        fig, axis = plt.subplots()
        count = 0
        for r_source, r_target in restrict:
            show_env = (count == 0)
            nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                                   restrict_sources=r_source,
                                   restrict_targets=r_target,
                                   show_environment=show_env, axis=axis, show=False)
            count += 1
  
     
    # ~ nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5, show=True)
  
  
    # ------------------ #
    # Simulate with NEST #
    # ------------------ #
  
    if simulate_activity is True :
  
      print("Activity simulation")
  
      '''
      Send the network to NEST, monitor and simulate
      '''
      activity_simulation(net,pop)


with_obstacles(shape)
