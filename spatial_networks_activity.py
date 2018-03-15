#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# ==COPYRIGHT INFORMATION==
# This program strictly observes the tenets of fundamentalist Theravada Mahasi style Buddhism.
# Any use of this program in violation of these aforementioned tenets or in violation of the
# principles described in the Visuddhimagga Sutta is strictly prohibited and punishable by
# extensive Mahayana style practice. By being or not being mindful of the immediate present
# moment sensations involved in the use of this program, you confer your acceptance of these
# terms and conditions.
#
# Note that the observation of the tenets of fundamentalist Theravada Mahasi style Buddhism
# and the Visuddhimagga Sutta is optional as long as the terms and conditions of the GNU GPLv3+
# are upheld.
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

# nngt.use_backend("graph-tool")
nngt.use_backend("networkx")

# nngt.set_config({"omp": 8, "palette": 'RdYlBu'})
nngt.set_config("multithreading", False)

nngt.seed(0)

''' Runtime options'''
plot_distribution = False  # plot of the connectivity distribution
plot_graphs = True  # graphic output of spatial network
simulate_activity = False  # whether to run or not a NEST simlation on the model
sim_duration = 2000  # simulation duration in ms
plot_activity = True  # wheter to plot simulation activity
animation_movie = True  # wheter to generate activity on map movie
obstacles = True  # set to True for simulation with elevated obstacles

print("\n###################\n Runtime options\n------------------")
print("Plot of the spatial graph  : {0}".format(plot_graphs))
print("Run Nest simulation on the generated graph  : {0}"
      .format(simulate_activity))
print("Simulation duration (if True)  : {0}".format(sim_duration))
print("Plot simulation activity  : {0}".format(plot_activity))
print("Generate activity on map movie  : {0}".format(animation_movie))
print("Generate the graph with obstacles  : {0}".format(obstacles))

''' Neuron numbers'''
density = 300e-6  # neuron density
fraction_excitatory = .99
print("\n###################\n Neurons' population\n------------------")
print("Neurons' density : {0} neurons per mm^2".format(density*1e6))
print("Fraction of excitatory neurons : {0} %".format(fraction_excitatory))

#########################################################################
# Neurons  parameters '''

# Parameters from "Recurrently connected and localized neuronal communities
# initiate coordinated spontaneous activity in neuronal networks" Lonardoni et al PLOS 2016
Lonardoni_excitatory_params = {
    'a': 2., 'E_L': -70., 'V_th': -50., 'b': 60., 'tau_w': 300.,
    'V_reset': -58., 't_ref': 2., 'g_L': 12., 'C_m': 281., 'I_e': 300.
}

Lonardoni_inhibitory_params = {
    'a': 2., 'E_L': -70., 'V_th': -50., 'b': 0., 'tau_w': 30.,
    'V_reset': -58., 't_ref': 2., 'g_L': 10., 'C_m': 200., 'I_e': 300.
}

params1 = Lonardoni_excitatory_params
print "Neurons' parameters  : {0}".format(params1)

# Synpases
synaptic_weigth = 80. #set default synaptic weigth
print "Default synaptic weigth  : {0}".format(synaptic_weigth)

# Create a circular shape to embed the neuros
shape = Shape.disk(radius=2500.)
print "Seeding region: disk of radius {0} microns".format(2500.)


####################################################################
def activity_simulation(net, pop, sim_duration=2000):
    '''
    Execute activity simulation on the generated graph
    --------------------------------------------------
    Input :   net : nngt generated graph with model
            pop : neurons population
            sim_duration : time (s) duration of simulation

    Output :
            activity graphs
    '''
    if (nngt.get_config('with_nest')) and (simulate_activity is True):
        print("SIMULATE ACTIVITY")
        import nngt.simulation as ns
        import nest

        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': 14})
        nest.SetKernelStatus({'resolution': .5})

        gids = net.to_nest()

        # nngt.simulation.randomize_neural_states(net, {"w": ("uniform", 0, 200),
        #  "V_m": ("uniform", -70., -40.)})

        nngt.simulation.randomize_neural_states(net, {"w": ("normal", 50., 5.),
                                                "V_m": ("uniform", -80., -50.)}
                                                )
        # add noise to the excitatory neurons
        excs = list(gids)
        # ns.set_noise(excs, 10., 2.)
        # ns.set_poisson_input(gids, rate=3500., syn_spec={"weight": 34.})

        target_rate = 25.
        average_degree = net.edge_nb() / float(net.node_nb())
        syn_rate = target_rate / average_degree
        print("syn_rate", syn_rate)
        print("avg_deg", average_degree)

        ns.set_minis(net, base_rate=syn_rate, weight_fraction=.4, gids=gids)
        vm, vs = ns.monitor_nodes([1], "voltmeter")
        recorders, records = ns.monitor_groups(pop.keys(), net)

        nest.Simulate(sim_duration)

        if plot_activity is True:
            print("Plot raster")
            ns.plot_activity(vm, vs, network=net, show=False)
            ns.plot_activity(recorders, records, network=net, show=False,
                             hist=true)

        if animation_movie is True:
            print("Process animation")
            anim = nngt.plot.AnimationNetwork(recorders, net, resolution=0.1,
                                              interval=20, decimate=-1)
            # ~ anim.save_movie("structured_bursting.mp4", fps=2, start=650.,
            # ~ stop=1500, interval=20)
            # ~ anim.save_movie("structured_bursting.mp4", fps=5, num_frames=50)
            anim.save_movie("structured_bursting.mp4", fps=15, interval=10)

        plt.show()
#######################################################################
def output_graphs(net, set1, set2, restrict, set1_title="", set2_title=""):
    '''
    Graphs
    Input :
    - net : neuron population and network
    - set neurons 1, list of neurons ids
    - set neurons 2, list of neurons ids
    - restrict : types of graphs to map spatially
        Ex 1 : connectivities for elevated obstacles
        restrict = [
            ("bottom", "bottom"), ("top", "bottom"), ("top", "top"), ("bottom","top")
        ]

      output_graphs(top_neurons,bottom_neurons,[
            ("bottom", "bottom"), ("top", "bottom"), ("top", "top"), ("bottom", "top")
        ])
      set1_title : string, optional string for title of dirstributions of set 1
      set2_title : string, optional string for title of dirstributions of set 2

    Graphical output of :
    - degree distribution histograms
    - spatial connectivity maps
    '''
    if plot_distribution is True:
        # Check the degree distribution
        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='auto', nodes=set1, show=False,
            title=set1_title)
        nngt.plot.degree_distribution(
            net, ["in"], num_bins='auto', nodes=set1, show=False,
            title=set1_title)
        nngt.plot.degree_distribution(
            net, ["out"], num_bins='auto', nodes=set1, show=False,
            title=set1_title)

        nngt.plot.degree_distribution(
            net, ["in", "out"], num_bins='auto', nodes=set2, show=False,
            title=set2_title)
        nngt.plot.degree_distribution(
            net, ["in"], num_bins='auto', nodes=set2, show=False,
            title=set2_title)
        nngt.plot.degree_distribution(
            net, ["out"], num_bins='auto', nodes=set2, show=False,
            title=set2_title)

    if plot_graphs is True:
        ''' Plot the resulting network and subnetworks '''
        count = 1
        for r_source, r_target in restrict:
            # print (r_source, r_target)
            nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
                                   restrict_sources=r_source,
                                   restrict_targets=r_target, show=False,
                                   title="Map "+str(count))
            count += 1
        plt.show()
        # fig, axis = plt.subplots()
        # count = 0
        # for r_source, r_target in restrict:
        #     show_env = (count == 0)
        #     nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5,
        #                            restrict_sources=r_source,
        #                            restrict_targets=r_target,
        #                            show_environment=show_env, axis=axis, show=False)
        #     count += 1

        # ~ nngt.plot.draw_network(net, nsize=7.5, ecolor="groups", ealpha=0.5, show=True)

#########################################################################################
def with_obstacles(shape, params={"height": 250., "width": 250.}, filling_fraction=0.4):
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
    total_area = shape.area
    bottom_area = np.sum([a.area for a in shape.default_areas.values()])
    num_bottom = int(density * bottom_area)

    # compute number of neurons in each top obstacle according to its area
    num_top_list = []  # list for the number of neurons on each top obsatacle
    num_top_total = 0  # count tiotal number of neurons in top obstacles
    for name, top_area in shape.non_default_areas.items():
        num_top = int(density * top_area.area)
        num_top_list.append(num_top)
        num_top_total += num_top

    num_neurons = num_bottom+num_top_total # Total number of neurons

    print("Total number of neurons : {0}".format(num_neurons))
    print("Bottom neurons : {0}".format(num_bottom))
    print("Top neurons : {0}".format(num_top_total))

    pop = nngt.NeuralPop(num_neurons, with_model=True)

    # # Instruction si on ne veut pas de modèle
    # pop.set_model(None)
    pop.create_group("bottom", num_bottom, neuron_model="aeif_psc_alpha",
                     neuron_param=params1)

    # Create one group on each top area
    for num_top, (name, top_area) in zip(num_top_list, shape.non_default_areas.items()):
        # num_top = int(density * top_area.area)
        group_name = "top_" + name
        if num_top:
            pop.create_group(group_name, num_top, neuron_model="aeif_psc_alpha",
                             neuron_param=params1)

    # make the graph
    net = nngt.SpatialGraph(shape=shape, population=pop)

    # seed neurons
    bottom_pos = shape.seed_neurons(num_bottom,
                                    on_area=shape.default_areas,
                                    soma_radius=15)
    bottom_neurons = np.array(net.new_node(num_bottom, positions=bottom_pos,
                                           groups="bottom"), dtype=int)

    top_pos = []
    top_neurons = []
    top_groups_name_list = []
    for name, top_area in shape.non_default_areas.items():
        group_name = "top_" + name
        if group_name in pop:
            num_top = pop[group_name].size  # number of neurons in  top group
            # locate num_top neurons in the group area
            top_pos_tmp = top_area.seed_neurons(num_top, soma_radius=15)
            top_pos.extend(top_pos_tmp)
            top_neurons.extend(net.new_node(num_top, positions=top_pos_tmp,
                                            groups=group_name))
            top_groups_name_list.append(group_name)
    top_pos = np.array(top_pos)
    top_neurons = np.array(top_neurons, dtype=int)

    # Establishment of the connectivity
    # scales for the connectivity distance function.
    top_scale = 200.  # between top neurons
    bottom_scale = 100.  # between bottom neurons
    mixed_scale = 150.  # between top and bottom neurons both ways

    print "\n----------\n Connectivity"
    print "top neurons connectivity characteristic distance {0}".format(top_scale)
    print "bottom neurons connectivity characteristic distance {0}".format(bottom_scale)
    print "mixed neurons connectivity characteristic distance {0}".format(mixed_scale)

    # base connectivity probability
    base_proba = 3.
    p_up = 0.6
    p_down = 0.9
    p_other_up = p_down**2
    print "Connectivity basic probability {0}".format(base_proba)
    print "Up connexion probability on one shape {0}".format(p_up)
    print "Down connexion probability {0}".format(p_down)
    print "Between two different up shapes connexion probability {0}".format(p_other_up)

    # connect bottom area
    for name, area in shape.default_areas.items():
        contained = area.contains_neurons(bottom_pos)
        neurons = bottom_neurons[contained]

        # 2018 03 51 I think the use of "bottom_neurons" below is erroneous
        # it should be "neurons" as defined above, the neurons in the contained
        # area.
        # Indeed shape.default_areas.items() is a list with diferent disjoint
        # we want to prevent connections between non communicating areas
        # at the bottom bottom areas
        # nngt.generation.connect_nodes(net, bottom_neurons, bottom_neurons,
        #                               "distance_rule", scale=bottom_scale,
        #                               max_proba=base_proba)
        nngt.generation.connect_nodes(net, neurons, neurons,
                                      "distance_rule", scale=bottom_scale,
                                      max_proba=base_proba)
    # connect top areas
    print("Connect top areas")
    for name, area in shape.non_default_areas.items():
        contained = area.contains_neurons(top_pos)
        neurons = top_neurons[contained]
        other_top = [n for n in top_neurons if n not in neurons]
        print(name)
        # print(neurons)
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
    net.set_weights(synaptic_weigth)

    # Graphs output
    # Define the list of connectivity maps to be plotted
    # each tuple of the list contains a list with the groups names of
    # neurons for outgoing links and a list of the groups containing
    # the neurons towards which the ource neurons connect.
    #
    restrict = [(top_groups_name_list, top_groups_name_list),
                ("bottom", top_groups_name_list),
                (top_groups_name_list,"bottom"),
                ("bottom", "bottom")]
    output_graphs(net, top_neurons, bottom_neurons, restrict, "Top neurons",
                  "Bottom neurons")

    # Simulate with NEST
    activity_simulation(net, pop, sim_duration)


###############################################################################
def with_obstacles_EI(shape, params={"height": 250., "width": 250.},
                      filling_fraction=0.4):
    '''
    Network generation and simulation with obsactles and inhibitory neurons
    -----------------------------------------------------------------------
    Input :   shape : nngt defined spatial shape
              params : dictionary with obstacles specifications
              filling_fraction : float, fraction of the embedding shape filled 
                                 with obstacles

    Output:   nngt network of neurons
    '''

    ''' Create obstacles within the shape'''

    shape.random_obstacles(filling_fraction, form="rectangle", params=params,
                           heights=30., etching=20.)

    ''' Create a Spatial network and seed neurons on top/bottom areas '''

    # neurons are reparted proportionaly to the area
    total_area = shape.area
    bottom_area = np.sum([a.area for a in shape.default_areas.values()])

    num_bottom = int(density * bottom_area)
    num_bottom_E = int(fraction_excitatory*num_neurons)
    num_bottom_I = num_bottom-num_bottom_E

    # compute number of neurons in each top obstacle according to its area
    num_top_list = []  # list for the number of neurons on each top obsatacle
    # list for the number of excitatory of neurons on each top obsatacle
    num_top_list_E = []
    # list for the number of inhibitoryneurons on each top obsatacle
    num_top_list_I = []
    num_top_total = 0  # count total number of neurons in top obstacles
    num_top_E = 0  # count total number of neurons in top obstacles
    num_top_I = 0  # count total number of neurons in top obstacles

    for name, top_area in shape.non_default_areas.items():
        num_top = int(density * top_area.area)
        num_excitatory = int(fraction_excitatory*num_top)
        num_inhibitory = num_neurons-num_excitatory
        num_top_list.append(num_top)
        num_top_list_E.append(num_excitatory)
        num_top_list_I.append(num_inhibitory)
        num_top_total += num_top
        num_top_E += num_excitatory
        num_top_I += num_inhibitory

    num_neurons = num_bottom+num_top_total  # Total number of neurons
    num_neurons_E = num_bottom_E+num_top_E  # Total number excitatory neurons
    num_neurons_I = num_bottom_I+num_top_I  # Total number inhibitory neurons

    print("Total number of neurons : {0}".format(num_neurons))
    print("Bottom neurons : {0}".format(num_bottom))
    print("Top neurons : {0}".format(num_top_total))
    print("Total number of excitatory neurons : {0}".format(num_neurons_E))
    print("Bottom excitatory neurons : {0}".format(num_bottom_E))
    print("Top excitatory neurons : {0}".format(num_top_E))
    print("Total number of inhibitory neurons : {0}".format(num_neurons_I))
    print("Bottom inhibitory neurons : {0}".format(num_bottom_I))
    print("Top inhibitory neurons : {0}".format(num_top_I))

    pop = nngt.NeuralPop(num_neurons, with_model=True)

    # # Instruction si on ne veut pas de modèle
    # pop.set_model(None)
    # create groups for excitatory and inhibitory bottom neurons
    pop.create_group("bottom_E", num_bottom_E, neuron_model="aeif_psc_alpha",
                     neuron_param=params1)
    pop.create_group("bottom_I", num_bottom_I, neuron_model="aeif_psc_alpha",
                     neuron_param=params1)

    # Create groups at the top
    def create_top_groups(group_name_prefix, num_top_list, model, params):
        ''' Creation of neurons groups on the top of bastacles'''
        for num_top, (name, top_area) in,\
                zip(num_top_list, shape.non_default_areas.items()):
            # num_top = int(density * top_area.area)
            group_name = group_name_prefix + name
            if num_top:
                pop.create_group(group_name, num_top,
                                 neuron_model=model,
                                 neuron_param=params)

    # Create groups for excitatory  neurons on each top area
    create_top_groups("top_E_", num_top_list_E, "aeif_psc_alpha", params1)
    # Create groups for  inhibitory neurons on each top are
    create_top_groups("top_I_", num_top_list_I, "aeif_psc_alpha", params1)

    # make the graph
    net = nngt.SpatialGraph(shape=shape, population=pop)

    # seed neurons
    def seed_bottom_neurons(num_bottom, group_name):
        ''' Seed botoom neurons'''
        bottom_pos = shape.seed_neurons(num_bottom,
                                        on_area=shape.default_areas,
                                        soma_radius=15)
        bottom_neurons = np.array(net.new_node(num_bottom,
                                  positions=bottom_pos,
                                  groups=group_name), dtype=int)
        return bottom_pos, bottom_neurons

    bottom_pos_E, bottom_neurons_E = seed_bottom_neurons(num_bottom_E,
                                                         "bottom_E")
    bottom_pos_I, bottom_neurons_I = seed_bottom_neurons(num_bottom_I,
                                                         "bottom_I")

    def seed_top_neurons(group_name_prefix, top_pos, top_neurons,
                         top_groups_name_list):
        ''' Seed top neurons'''
        for name, top_area in shape.non_default_areas.items():
            group_name = group_name_prefix + name
            if group_name in pop:
                # number of neurons in the top group
                num_top = pop[group_name].size
                # locate num_top neurons in the group area
                top_pos_tmp = top_area.seed_neurons(num_top, soma_radius=15)
                top_pos.extend(top_pos_tmp)
                top_neurons.extend(net.new_node(num_top, positions=top_pos_tmp,
                                                groups=group_name))
                top_groups_name_list.append(group_name)
        top_pos = np.array(top_pos)
        top_neurons = np.array(top_neurons, dtype=int)

    top_pos_E = []
    top_neurons_E = []
    top_groups_name_list_E = []
    seed_top_neurons("top_E_", top_pos_E, top_neurons_E,
                     top_groups_name_list_E)
    top_pos_I = []
    top_neurons_I = []
    top_groups_name_list_I = []
    seed_top_neurons("top_I_", top_pos_I, top_neurons_I,
                     top_groups_name_list_I)

    # Establishment of the connectivity
    # scales for the connectivity distance function.
    top_scale = 200.  # between top neurons
    bottom_scale = 100.  # between bottom neurons
    mixed_scale = 150.  # between top and bottom neurons both ways

    print "\n----------\n Connectivity"
    print "top neurons connectivity characteristic distance {0}"
    .format(top_scale)
    print "bottom neurons connectivity characteristic distance {0}"
    .format(bottom_scale)
    print "mixed neurons connectivity characteristic distance {0}"
    .format(mixed_scale)

    # base connectivity probability
    base_proba = 3.
    p_up = 0.6  # modulation for connection bottom to top
    p_down = 0.9  # modulation for connection top to bottom
    p_other_up = p_down**2  # modulation connection top to top at the bottom
    print "Connectivity basic probability {0}".format(base_proba)
    print "Up connexion probability on one shape {0}".format(p_up)
    print "Down connexion probability {0}".format(p_down)
    print "Between two different up shapes connexion probability {0}"
    .format(p_other_up)

    # connect bottom area
    def connect_bottom(bottom_pos):
        ''' Connect bottom neurons'''

        for name, area in shape.default_areas.items():
            contained = area.contains_neurons(bottom_pos)
            neurons = bottom_neurons[contained]

            nngt.generation.connect_nodes(net, neurons, neurons,
                                          "distance_rule", scale=bottom_scale,
                                          max_proba=base_proba)
    connect_bottom(bottom_pos_E)

    # connect top areas
    print("Connect top areas")
    for name, area in shape.non_default_areas.items():
        contained = area.contains_neurons(top_pos)
        neurons = top_neurons[contained]
        other_top = [n for n in top_neurons if n not in neurons]
        print(name)
        # print(neurons)
        if np.any(neurons):
            # connect intra-area
            nngt.generation.connect_nodes(net, neurons, neurons,
                                          "distance_rule",
                                          scale=top_scale, max_proba=base_proba)
            # connect between top areas (do it?)
            nngt.generation.connect_nodes(net, neurons, other_top,
                                          "distance_rule",
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
    net.set_weights(synaptic_weigth)

    # Graphs output
    # Define the list of connectivity maps to be plotted
    # each tuple of the list contains a list with the groups names of
    # neurons for outgoing links and a list of the groups containing
    # the neurons towards which the ource neurons connect.
    #
    restrict = [(top_groups_name_list, top_groups_name_list),
                ("bottom", top_groups_name_list),
                (top_groups_name_list, "bottom"),
                ("bottom", "bottom")]
    output_graphs(net, top_neurons, bottom_neurons, restrict)

    # Simulate with NEST
    activity_simulation(net, pop, sim_duration)


###########################################################
def no_obstacles(shape):
    '''
    Network generation and simulation without obsacle
    Input:  shape : nngt spatial shape object to embed the neuron
    Output: net : nngt network

    # Create a Spatial network and seed neurons
    # Neurons are spread proportionaly to the area
    '''

    total_area = shape.area
    num_neurons = int(density * total_area)
    num_excitatory = int(fraction_excitatory*num_neurons)
    num_inhibitory = num_neurons-num_excitatory
    print("Total number of neurons : {}".format(num_neurons))
    print("Excitatory neurons : {}".format(num_excitatory))
    print("Inhibitory neurons : {}".format(num_inhibitory))

    pop = nngt.NeuralPop(num_neurons, with_model=True)
    # # Instruction si on ne veut pas de modèle
    # pop.set_model(None)
    pop.create_group("excitatory", num_excitatory,
                     neuron_model="aeif_psc_alpha",
                     neuron_param=params1)
    pop.create_group("inhibitory", num_inhibitory,
                     neuron_model="aeif_psc_alpha",
                     neuron_param=params1)
    # make the graph
    net = nngt.SpatialGraph(shape=shape, population=pop)
    # seed neurons
    excitatory_pos = shape.seed_neurons(num_excitatory,
                                        on_area=shape.default_areas,
                                        soma_radius=15)
    excitatory_neurons = np.array(net.new_node(num_excitatory,
                                  positions=excitatory_pos,
                                  groups="excitatory"), dtype=int)
    inhibitory_pos = shape.seed_neurons(num_inhibitory,
                                        on_area=shape.default_areas,
                                        soma_radius=15)
    inhibitory_neurons = np.array(net.new_node(num_inhibitory,
                                  positions=inhibitory_pos,
                                  groups="inhibitory",ntype=-1),
                                  dtype=int)

    # Establishment of the connectivity
    # Scale for the connectivity distance function.
    connectivity_scale = 100
    print("\n----------\n Connectivity")
    print("Connectivity characteristic distance {0}"
          .format(connectivity_scale))
    print("     (NB: the scale for bottom with obstacles is 100)")

    # base connectivity probability
    connectivity_proba = 3.
    print("Connectivity basic probability {0}".format(connectivity_proba))
    print("(Identical between any types of neurons)")
    # connect bottom area
    for name, area in shape.default_areas.items():
        # excitatory to excitatory
        nngt.generation.connect_nodes(net, excitatory_neurons,
                                      excitatory_neurons,
                                      "distance_rule",
                                      scale=connectivity_scale,
                                      max_proba=connectivity_proba)
        # excitatory to inhibitory
        nngt.generation.connect_nodes(net, excitatory_neurons,
                                      inhibitory_neurons,
                                      "distance_rule",
                                      scale=connectivity_scale,
                                      max_proba=connectivity_proba)
        # inhibitory to inhibitory
        nngt.generation.connect_nodes(net, inhibitory_neurons,
                                      inhibitory_neurons,
                                      "distance_rule",
                                      scale=connectivity_scale,
                                      max_proba=connectivity_proba)
        # inhobitory to excitatory
        nngt.generation.connect_nodes(net, inhibitory_neurons,
                                      excitatory_neurons,
                                      "distance_rule",
                                      scale=connectivity_scale,
                                      max_proba=connectivity_proba)

    # Here we set the synaptic weigth. By default synapses are static in net
    net.set_weights(synaptic_weigth)

    # Graphs output
    output_graphs(net, inhibitory_neurons, excitatory_neurons,
                  [("excitatory", "excitatory"), ("inhibitory", "excitatory"),
                   ("inhibitory", "inhibitory"), ("excitatory", "inhibitory")],
                  "inhibitory neurons", "excitatory neurons")
    # Simulate with NEST
    activity_simulation(net, pop)

###############################################################################
# Main


if obstacles is True:
    with_obstacles(shape)
else:
    no_obstacles(shape)
