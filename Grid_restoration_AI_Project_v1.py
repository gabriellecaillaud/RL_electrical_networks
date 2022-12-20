# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:55:21 2022

@author: I13980

This script will be launched from an Agent-based environment to perfom power flow (PF) calculation at each
time step. The model is a modified optimal power flow (OPF) as binary variables on the status of switches, 
connectivity of buses (nodes), and PF directions are introduced.
It takes as parameters the configuration of the electrical network and availability status of both 
nodes at some locations. Electrical quantities (active/reactive power, node voltage, and non-supplied 
power) as well as PF directions, node connectivity, and switch status; are returned.

PS: As ReconfigAction and RepairAction are not defined initially, you can launch an independent run by commenting
lines 40, 520-524 and adjust the indent of all other parts
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import networkx as nx
import csv
from collections import defaultdict
import cloudpickle
import os
from datetime import datetime
from pathlib import Path
import random as rd

class environement():

    def __init__(self):
        self.model = None
        self.graph = None
        self.next_state = None
        self.reward = None


    def reset(self):
        nodes_file = "Data_nodes_2.txt"
        nodes_ref = "./data/Data_nodes.txt"
        edges_file = "Data_edges_2.txt"
        edges_ref = "./data/Data_edges.txt"
        nb_default = 7
        try:
            os.remove(nodes_file)
        except:
            pass
        try:
            os.remove(edges_file)
        except:
            pass

        with open(nodes_file, mode='w') as nodes:
            with open(nodes_ref, mode = 'r') as ref_nodes:
                for line in ref_nodes.readlines():
                    nodes.write(line)

        L = [i for i in range(42)]
        Defaults = [0 for _ in range(42)]
        choice = rd.choices(L,k=nb_default)
        for k in choice:
            Defaults[k] = 1


        with open(edges_file, mode='w') as edges:
            with open(edges_ref, mode = 'r') as ref_edges:
                counter = 0
                for line in ref_edges.readlines():
                    if len(line)!=0:
                        if counter >0:
                            edges.write(line[:len(line)-2]+str(Defaults[counter-1])+"\n")
                        else:
                            edges.write(line)
                        counter +=1
        
        return self.step({},counter = 0)[0]


    # Set the folder "runs//" to contain all simulation resulted files
    def save_results(self,folder, model, g):
        with open(folder+'/model.pkl', mode='wb') as file:
            cloudpickle.dump(model, file)
            
        nx.write_gpickle(g, folder+'/graph.pkl') # save the resultant configuration  

        return folder


    def response(self,ReconfAction_fromRL,RepairAction_fromRL=None,counter = 1):
        # ***** small use case *****
        experiment = Path(__file__).stem
        folder = 'runs/'
        os.makedirs(folder, exist_ok=True)
        folder += datetime.now().strftime("%Y.%m.%d_%Hh%Mmn%Ss_") + experiment + '/'
        os.makedirs(folder, exist_ok=True)

        # load data from text files as csv readers
        nodes_file = "Data_nodes_2.txt"
        edges_file = "Data_edges_2.txt"
        csv_nodes = csv.DictReader(open(nodes_file),delimiter=";")              
        csv_edges = csv.DictReader(open(edges_file),delimiter=";")
        reader_nodes = list(csv_nodes)
        reader_edges = list(csv_edges)

        # construct an undirected graph from loaded data
        graph = nx.Graph()
        graph.add_nodes_from([(int(reader_nodes[i]['node']), reader_nodes[i]) for i in range(0,len(reader_nodes))])
        graph.add_edges_from([(int(reader_edges[i]['src']), int(reader_edges[i]['dst']), reader_edges[i]) for i in range(0,len(reader_edges))])

        # draw network without damages
        #plt.figure()
        #plt.box(False)
        #plt.title('T = '+str(0))
        # pos = nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
        '''
        pos = {node:np.array([float(graph.nodes[node]['pos_x']),float(graph.nodes[node]['pos_y'])]) for node in graph.nodes} # nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
        options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}
        options1 = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in graph.nodes() if graph.nodes[node]['HV_SS']=='1'], node_color="tab:red", **options)
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in graph.nodes() if graph.nodes[node]['MV_SS']=='1'], node_color="tab:blue", **options1)
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in graph}, font_family="times", font_size=8)
        # nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(graph,pos,edgelist=[edge for edge in graph.edges() if graph[edge[0]][edge[1]]['normal_open']=='1'],
                                edge_color="tab:orange", style='dashed')
        nx.draw_networkx_edges(graph,pos,edgelist=[edge for edge in graph.edges() if graph[edge[0]][edge[1]]['normal_open']=='0'], 
                                width=8, alpha=0.5, edge_color="tab:green")
        '''                        

        # draw network with damages
        #plt.figure()
        #plt.box(False)
        #plt.title('T = '+str(0))
        # pos = nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
        '''
        options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}
        options1 = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in graph.nodes() if graph.nodes[node]['HV_SS']=='1'], node_color="tab:red", **options)
        nx.draw_networkx_nodes(graph, pos, nodelist=[node for node in graph.nodes() if graph.nodes[node]['MV_SS']=='1'], node_color="tab:blue", **options1)
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in graph}, font_size=8)
        # nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(graph,pos,edgelist=[edge for edge in graph.edges() if graph[edge[0]][edge[1]]['normal_open']=='1'
                                and graph[edge[0]][edge[1]]['failed']=='0'], edge_color="tab:orange", style='dashed')
        nx.draw_networkx_edges(graph,pos,edgelist=[edge for edge in graph.edges() if graph[edge[0]][edge[1]]['normal_open']=='0' 
                                and graph[edge[0]][edge[1]]['failed']=='0'], width=8, alpha=0.5, edge_color="tab:green")
        nx.draw_networkx_edges(graph,pos,edgelist=[edge for edge in graph.edges() if graph[edge[0]][edge[1]]['failed']=='1'], 
                            width=8, alpha=0.5, edge_color="tab:red")
        '''
        #   Define the model
        model = pyo.ConcreteModel()

        #   Inputs
        T0 = 4
        model.T0_time_steps = pyo.Param(initialize=T0) # Restoration stage phases
            
        M = 1000 # Very large number
        experiment_data = {'Time steps': T0, 'Large Number M': M, 'Edges file': nodes_file, 'Nodes file': edges_file}
        with open(folder+'Experiment data.txt', 'w') as f:
            f.write(str(experiment_data))
            
        # ----------------------------- Sets ---------------------------------------------------

        # IMPORTANT: note the t here is different from the t used for different stages. Here as each stage has many phases:
        # initial phase (t=0, can be taken from previous stage), automatic isolation (t=1), remote isolation (t=2), 
        # and reconfiguration (t=3); we use t with a different meaning.
        model.T0_0 = pyo.Set(initialize=np.array([t for t in range(model.T0_time_steps.value)])) # Stage phases with t=0
        model.T0 = pyo.Set(initialize=np.array([t for t in range(1,model.T0_time_steps.value)])) # Stage phases without t=0
        # Stage phases without t=3, because reconfiguration actions will be received from the RL agent
        model.T = pyo.Set(initialize=np.array([t for t in range(1,model.T0_time_steps.value-1)])) 

        model.N = pyo.Set(initialize=list(graph.nodes())) # Power nodes in the network
        # Set of existing connections. Not all nodes connect to any node !
        adjacent = defaultdict(set)
        for (i, j) in graph.edges():
            adjacent[i].add(j)
            adjacent[j].add(i)
        model.NxN = pyo.Set(within=model.N*model.N, initialize={(i, j) for i in model.N for j in adjacent[i]})

        # In NxN both sides (i,j) and (j,i) of a line is present. In LxL, only one side.
        adjacent1 = defaultdict(set)
        for (i, j) in graph.edges():
            adjacent1[i].add(j)
        model.LxL = pyo.Set(within=model.N*model.N, initialize={(i, j) for i in model.N for j in adjacent1[i]})

        model.L = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if graph[edge[0]][edge[1]]['normal_open']=='0']) # Power lines in the network
        model.L_nor = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if graph[edge[0]][edge[1]]['normal_open']=='1']) # Initially open power lines in the network
        model.L_rc = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if graph[edge[0]][edge[1]]['switch']=='RC']) # Power lines with remote automatic switches
        model.L_cb = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if graph[edge[0]][edge[1]]['switch']=='CB']) # Circuit breakers
        model.L_r_rc = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if 'R' in graph[edge[0]][edge[1]]['switch']]) # Power lines with remote switches
        model.L_r = pyo.Set(within=model.NxN, 
        initialize=[edge for edge in model.NxN if graph[edge[0]][edge[1]]['switch']=='R']) # Power lines with remote and remote-automatic switches
        model.L_m = pyo.Set(within=model.NxN, initialize=model.NxN-model.L_rc-model.L_cb-model.L_r) # Power lines with manual switches
        model.S = pyo.Set(within=model.N, 
        initialize=[node for node in model.N if graph.nodes[node]['HV_SS']=='1']) # Power HV/MV substations

        # The commented part here can be used to assign a single index to each line 
        # v = list(model.L)
        # model.IdxL = pyo.Set(intialize={v.index(edge):edge for edge in v})
        # ----------------------------- Parameters ----------------------
        
        if RepairAction_fromRL != None:
            (rp1,rp2) = RepairAction_fromRL
            graph[rp1][rp2]['failed']='0'
            graph[rp2][rp1]['failed']='0'

        model.s = pyo.Param(model.N, within=pyo.Binary, initialize={node:1 if graph.nodes[node]['HV_SS']=='1' else 0 for node in model.N}) # 1 if a node is a HV/MV substation, 0 otherwise
        model.f_l = pyo.Param(model.NxN, within=pyo.Binary, initialize = {edge:1 if graph[edge[0]][edge[1]]['failed']=='1' else 0 for edge in model.NxN},mutable = True) # 1 if failure in line l, 0 otherwise
        model.p_d = pyo.Param(model.N, within=pyo.NonNegativeReals, initialize={node:float(graph.nodes[node]['P_d']) for node in model.N}) # Active power demand at each node
        model.q_d = pyo.Param(model.N, within=pyo.NonNegativeReals, initialize={node:float(graph.nodes[node]['Q_d']) for node in model.N}) # Reactive power demand at each node
        model.p_s = pyo.Param(model.N, within=pyo.NonNegativeReals, initialize={node:float(graph.nodes[node]['P_s']) for node in model.N}) # Active power supplied at each node
        model.q_s = pyo.Param(model.N, within=pyo.NonNegativeReals, initialize={node:float(graph.nodes[node]['Q_s']) for node in model.N}) # Reactive power supplied at each node
        model.r = pyo.Param(model.NxN, within=pyo.NonNegativeReals, initialize={edge:float(graph[edge[0]][edge[1]]['Z_r']) for edge in model.NxN}) # Resistance of a power line
        model.x = pyo.Param(model.NxN, within=pyo.NonNegativeReals, initialize={edge:float(graph[edge[0]][edge[1]]['Z_x']) for edge in model.NxN}) # Reactance of a power line
        model.P_max = pyo.Param(initialize=sum(model.p_d[i] for i in model.N-model.S)) # Total active power demand
        model.Q_max = pyo.Param(initialize=sum(model.q_d[i] for i in model.N-model.S)) # Total reactive power demand
        model.S_max = pyo.Param(initialize=model.P_max*model.Q_max) # Total apparent power demand
        model.v_max = pyo.Param(model.N, initialize=pow(20.1,2)) # Maximum node voltage in kV (squared for convenience)
        model.v_min = pyo.Param(model.N, initialize=pow(19.9,2)) # Minimum node voltage in kV (squared for convenience)
        model.l_cb = pyo.Param(model.NxN, within=pyo.Binary, 
        initialize={edge:1 if graph[edge[0]][edge[1]]['switch']=='CB' else 0 for edge in model.NxN}) # Parameter for the presence of a Circuit Breaker
        model.l_rc = pyo.Param(model.NxN, within=pyo.Binary, 
        initialize={edge:1 if graph[edge[0]][edge[1]]['switch']=='RC' else 0 for edge in model.NxN}) # Parameter for the presence of a remote-automatic switch 
        model.l_r = pyo.Param(model.NxN, within=pyo.Binary, 
        initialize={edge:1 if graph[edge[0]][edge[1]]['switch']=='R' else 0 for edge in model.NxN}) # Parameter for the presence of a remote switch

        # # It's at this point that data from RL agent can be taken as parameters to be used in this MILP model
        sw_fromRL = ReconfAction_fromRL # The received data should be organized as a dictionary, see how sw_init is defined below
        
        # example: sw_fromRL = {edge:0 if graph[edge[0]][edge[1]]['normal_open']=='1' else 1 for edge in model.NxN} 
        # rp_fromRL <- based on the information on this vector, the parameter model.f_l above can be modified by
        #              setting f_l[(i,j)] = 0 for the line (i,j) that was chosed to be repaired
        
        # ----------------------------- Variables ---------------------------------------------------
        model.p = pyo.Var(model.NxN,model.T0, within=pyo.NonNegativeReals) # The flow of active power over each line
        model.q = pyo.Var(model.NxN, model.T0, within=pyo.NonNegativeReals) # The flow of reactive power over each line
        model.p_ns = pyo.Var(model.N, model.T0, within=pyo.NonNegativeReals) # Non-supplied active power at each node
        model.q_ns = pyo.Var(model.N, model.T0, within=pyo.NonNegativeReals) # Non-supplied reactive power at each node
        model.v = pyo.Var(model.N, model.T0, within=pyo.NonNegativeReals) # Voltage level at each node
        # State of the switch at each remotely switchable line. Note that switch states at t=3 are not considered
        # as variables because these actions are received as parameters from the RL agent
        model.sw = pyo.Var(model.NxN, model.T0_0, within=pyo.Binary) 

        sw_init = {edge:0 if graph[edge[0]][edge[1]]['normal_open']=='1' else 1 for edge in model.NxN}

        model.X = pyo.Var(model.NxN, model.T0, within=pyo.Binary, initialize=0) # Direction of power flow at each line
        model.y_e = pyo.Var(model.N, model.T0, within=pyo.Binary, initialize=1) # Power node is connected
        model.a_e = pyo.Var(model.N, model.T0, within=pyo.Binary, initialize=1) # Power node is out of fault zone and available for connection

        model.z = pyo.Var(model.NxN, model.T0, within=pyo.Binary, initialize=0) # Used to linearize the the absolute value introduced with switching cost  

        model.alpha = pyo.Param(initialize=10)
        model.beta = pyo.Param(initialize=0.1)
        model.gamma = pyo.Param(initialize=0.1)
        model.C_sd = pyo.Param(initialize=0.5) # in $/KW
        model.C_sw = pyo.Param(initialize=0.1) # in $
        model.C_a = pyo.Param(initialize=1) # in $


        # ----------------------------- Objective ---------------------------------------------------
        model.obj = pyo.Objective(expr = model.alpha*sum(model.C_sd*model.p_ns[i,3] for i in model.N-model.S)
                                +model.alpha*sum(model.C_sd*model.p_ns[i,1] for i in model.N-model.S) 
                                +model.alpha*sum(model.C_a*(1-model.a_e[i,2]) for i in model.N-model.S)
                                +model.beta*sum(model.C_sw*model.z[i,j,t] for (i,j) in model.LxL for t in range(1,T0))
                                , sense=pyo.minimize)

        # ----------------------------- Constraints ---------------------------------------------------

        # Constraints used to limit the number of switching operations
        model.c_obj = pyo.Constraint(model.LxL, model.T, rule=lambda model, i, j, t:
                                model.sw[i,j,t-1]-model.sw[i,j,t] 
                                <= model.z[i,j,t]
                                )
        model.c_obj1 = pyo.Constraint(model.LxL, model.T, rule=lambda model, i, j, t:
                                model.sw[i,j,t]-model.sw[i,j,t-1] 
                                <= model.z[i,j,t]
                                )

        # C0 constraints are used for initialization
        model.c0 = pyo.Constraint(model.LxL, model.T0, rule=lambda model, i, j, t:
                                model.sw[i,j,t] 
                                == model.sw[j,i,t]
                                )


        model.c0_ = pyo.Constraint(model.LxL, rule=lambda model, i, j:
                                model.sw[i,j,0] == sw_init[(i,j)]
                                )
                            

        # If suitable values of sw_fromRL are set above, activate this constraint by uncommenting
        if ReconfAction_fromRL != {}:
            model.c0_0 = pyo.Constraint(model.L_r_rc, rule=lambda model, i, j:
                                        model.sw[i,j,3] == sw_fromRL[(i,j)]
                                        )
        else:
            model.c0_0 = pyo.Constraint(model.L_r_rc, rule=lambda model, i, j:
                                        model.sw[i,j,3] == model.sw[i,j,2]
                                        )


        #ICI#

        model.c0_1 = pyo.Constraint(model.LxL, rule=lambda model, i, j:
                                model.p[i,j,1] == model.p[i,j,2]
                                )
        model.c0_2 = pyo.Constraint(model.LxL, rule=lambda model, i, j:
                                model.q[i,j,1] == model.q[i,j,2]
                                )
        model.c0_3 = pyo.Constraint(model.N, rule=lambda model, i:
                                model.p_ns[i,1] == model.p_ns[i,2]
                                )
        model.c0_4 = pyo.Constraint(model.N, rule=lambda model, i:
                                model.q_ns[i,1] == model.q_ns[i,2]
                                )
        model.c0_5 = pyo.Constraint(model.N, rule=lambda model, i:
                                model.v[i,1] == model.v[i,2]
                                )
        model.c0_6 = pyo.Constraint(model.N, rule=lambda model, i:
                                model.y_e[i,1] == model.y_e[i,2]
                                )
        model.c0_7 = pyo.Constraint(model.NxN, rule=lambda model, i, j:
                                model.X[i,j,1] == model.X[i,j,2]
                                )
        model.c0_8 = pyo.Constraint(model.N, rule=lambda model, i:
                                model.a_e[i,2]
                                == model.a_e[i,3]
                                )   

        # M and R switches are not operated during automatic response
        model.c1 = pyo.Constraint(model.NxN-model.L_cb-model.L_rc, rule=lambda model, i, j: 
                                model.sw[i,j,1]
                                == model.sw[i,j,0]
                                )
        # Only CB and RC switches are automatically opened as first response -- putting i or j is important in underground
        model.c1_0 = pyo.Constraint(model.NxN-model.L_r-model.L_m, rule=lambda model, i, j:
                                model.sw[i,j,1]
                                <= model.sw[i,j,0]*model.a_e[i,1]
                                )
        # CB, RC, and M switches are not operated for isolation
        model.c1_1 = pyo.Constraint(model.NxN-model.L_r, rule=lambda model, i, j: 
                                model.sw[i,j,2]
                                == model.sw[i,j,1]
                                )
        # Only R switches are used for isolation
        model.c1_2 = pyo.Constraint(model.L_r, rule=lambda model, i, j:
                                model.sw[i,j,1]-(2-model.a_e[i,2]-model.a_e[j,2])-model.f_l[i,j]
                                <= model.sw[i,j,2]
                                ) 
        # R switches are only opened during isolation
        model.c1_3 = pyo.Constraint(model.L_r, rule=lambda model, i, j: 
                                model.sw[i,j,2]
                                <= model.sw[i,j,1]
                                )
        # R and RC switches can be opened during fast reconfiguration
        model.c1_4 = pyo.Constraint(model.L_r_rc, rule=lambda model, i, j:
                                model.sw[i,j,2]-(2-model.a_e[i,3]-model.a_e[j,3])
                                <= model.sw[i,j,3]
                                )
        # R and RC switches can be closed during fast reconfiguration
        model.c1_5 = pyo.Constraint(model.L_r_rc, rule=lambda model, i, j: 
                                model.sw[i,j,3] 
                                <= model.sw[i,j,2]+model.a_e[i,3]
                                )
        # CBs are not opened during fast reconfiguration
        model.c1_6 = pyo.Constraint(model.L_cb, rule=lambda model, i, j: 
                                model.sw[i,j,2]
                                <= model.sw[i,j,3]
                                )
        # CBs can be closed during fast reconfiguration
        model.c1_7 = pyo.Constraint(model.L_cb, rule=lambda model, i, j: 
                                model.sw[i,j,3] 
                                <= model.sw[i,j,2]+model.a_e[i,3]
                                )
        # M switches are not operated during fast reconfiguration
        model.c1_8 = pyo.Constraint(model.L_m, rule=lambda model, i, j: 
                                model.sw[i,j,3]
                                == model.sw[i,j,2]
                                )  

        # Power flows only in one direction (or doesn't flow) when a line is closed    
        model.c6 = pyo.Constraint(model.LxL, model.T0, rule=lambda model, i, j, t:
                                model.X[i,j,t]+model.X[j,i,t] 
                                <= model.sw[i,j,t]
                                )
        model.c6_0 = pyo.Constraint(model.LxL, rule=lambda model, i, j:
                                model.X[i,j,3]+model.X[j,i,3]-(2-model.a_e[i,3]-model.a_e[j,3])
                                <= model.sw[i,j,3]
                                )
        model.c6_1 = pyo.Constraint(model.LxL, rule=lambda model, i, j:
                                model.sw[i,j,3]-(2-model.a_e[i,3]-model.a_e[j,3])
                                <= model.X[i,j,3]+model.X[j,i,3]
                                )

        # If a line is faulted, both connected nodes are in the fault zone from the start to the end of service restoration
        model.c7 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                model.a_e[j,t]
                                <= (1-model.f_l[i,j]*model.sw[i,j,0]+model.s[j])
                                )

        # Fault propagation, and rules of reconnection
        model.c8 = pyo.Constraint(model.NxN, rule=lambda model, i, j:
                                model.a_e[j,1]+model.sw[i,j,0]*(1-model.l_cb[i,j])*(1-model.l_rc[i,j])-1
                                <= model.a_e[i,1]
                                )
        model.c8_0 = pyo.Constraint(model.NxN, rule=lambda model, i, j:
                                model.a_e[j,2]+model.sw[i,j,2]-1
                                <= model.a_e[i,2]
                                )    
        model.c8_1 = pyo.Constraint(model.NxN, rule=lambda model, i, j:
                                model.a_e[j,3]+model.sw[i,j,3]-1
                                <= model.a_e[i,3]
                                )   
        model.c9 = pyo.Constraint(model.N, model.T0, rule=lambda model, j, t:
                                model.y_e[j,t]
                                == sum(model.X[i,j,t] for i in graph.neighbors(j))
                                )
        model.c9_0 = pyo.Constraint(model.N, model.T0, rule=lambda model, j, t:
                                sum(model.X[j,i,t] for i in graph.neighbors(j))
                                <= (sum(model.X[i,j,t] for i in graph.neighbors(j))+model.s[j])*M
                                )
        model.c10 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                sum(model.X[i,j,t] for i in graph.neighbors(j))
                                <= model.a_e[j,t]-model.s[j]
                                )

        # Active power flow balance equation
        model.c11 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                sum(model.p[i,k,t] for k in graph.neighbors(i))+model.p_d[i]
                                <= sum(model.p[j,i,t] for j in graph.neighbors(i)) + model.p_ns[i,t]+model.p_s[i]
                                )
        # Reactive power flow balance equation
        model.c12 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                sum(model.q[i,k,t] for k in graph.neighbors(i))+model.q_d[i]
                                <= sum(model.q[j,i,t] for j in graph.neighbors(i)) + model.q_ns[i,t]+model.q_s[i]
                                )
        # Equations relating the voltage difference to active/reactive power and line resitance/reactance
        model.c13 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                model.v[i,t]-model.v[j,t]-2*(model.r[i,j]*model.p[i,j,t]+model.x[i,j]*model.q[i,j,t])/1000
                                <= (1-model.X[i,j,t])*M
                                )
        model.c14 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                -(1-model.X[i,j,t])*M
                                <= model.v[i,t]-model.v[j,t]-2*(model.r[i,j]*model.p[i,j,t]+model.x[i,j]*model.q[i,j,t])/1000
                                )  

        # Line active power capacity limit
        model.c15 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                model.p[i,j,t] 
                                <= model.S_max*model.X[i,j,t]
                                )   
        # Line reactive power capacity limit
        model.c16 = pyo.Constraint(model.NxN, model.T0, rule=lambda model, i, j, t:
                                model.q[i,j,t] 
                                <= model.S_max*model.X[i,j,t]
                                )  
        # Equations of node voltage limits
        model.c17 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                model.v[i,t]
                                <= model.v_max[i]
                                )
        model.c17_ = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                model.v_min[i]
                                <= model.v[i,t]
                                )
        # Shed power p_ns or non-supplied active power   
        model.c18 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                (1-model.y_e[i,t])*model.p_d[i]
                                <= model.p_ns[i,t]
                                )
        model.c18_0 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                model.p_ns[i,t]
                                <= model.p_d[i]
                                )

        # Shed power q_ns or non-supplied reactive power  
        model.c19 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                (1-model.y_e[i,t])*model.q_d[i]
                                <= model.q_ns[i,t]
                                )
        model.c19_0 = pyo.Constraint(model.N, model.T0, rule=lambda model, i, t:
                                model.q_ns[i,t]
                                <= model.q_d[i]
                                )
            
        # ------------------------- Solve the problem -------------------------   
        opt = SolverFactory("cplex")
        results = opt.solve(model)
        #sends results to stdout
        #results.write()
        #print("\nDisplaying Solution\n" + '-'*60)
        # model.p_ns.display() -> search for pyomo for more commands to see results
        # model.p_ns.pprint()

        # ------------------------- Plot restoration steps -------------------------
        try:
            for step in model.T0:
                g = nx.Graph()
                for index in model.X:
                    if index[2]==step:
                        g.add_edges_from([(index[0], index[1], graph.get_edge_data(index[0],index[1]))])
                        g.add_nodes_from([(index[0], graph.nodes[index[0]]), (index[1], graph.nodes[index[1]])])
                        if model.X[index[0],index[1],step].value>0.5 or model.X[index[1],index[0],step].value>0.5: # IMPORTANT: use sw and not X in this condition
                            g[index[0]][index[1]]['failed'] = '0'
                            g[index[0]][index[1]]['normal_open'] = '0'
                        else:

                            if graph[index[0]][index[1]]['failed'] == '1' : # failed lines remain failed during this stage
                                g[index[0]][index[1]]['failed'] = '1'
                                g[index[0]][index[1]]['normal_open'] = '0'
                        
                            elif model.sw[index[0],index[1],step].value<0.5: 
                                # Opened CB and RC switches at t=1, and R switches at t=2
                                g[index[0]][index[1]]['failed'] = '0'
                                g[index[0]][index[1]]['normal_open'] = '1'
                            else: # healthy, but not powered lines
                                g[index[0]][index[1]]['failed'] = '1'
                                g[index[0]][index[1]]['normal_open'] = '1'
                        
                    
            # draw nodes and edges nodes
            #plt.figure()
            #plt.title('T = '+str(step))
            pos = {node:np.array([float(graph.nodes[node]['pos_x']),float(graph.nodes[node]['pos_y'])]) for node in graph.nodes} # nx.spring_layout(graph, seed=3113794652)  # positions for all nodes
            # pos = nx.spring_layout(graph, fixed=graph.nodes(), pos=pos, seed=3113794652)  # positions for all nodes
            options = {"edgecolors": "tab:gray", "node_size": 400, "alpha": 0.9}
            options1 = {"edgecolors": "tab:gray", "node_size": 200, "alpha": 0.9}
            '''nx.draw_networkx_nodes(g, pos, nodelist=[node for node in g.nodes() if g.nodes[node]['HV_SS']=='1'], node_color="tab:red", **options)
            nx.draw_networkx_nodes(g, pos, nodelist=[node for node in g.nodes() if g.nodes[node]['MV_SS']=='1'], node_color="tab:blue", **options1)
            nx.draw_networkx_labels(g, pos, labels={n: n for n in graph}, font_size=8)
            # nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='0' 
                                    and g[edge[0]][edge[1]]['normal_open']=='0'], width=8, alpha=0.5, edge_color="tab:green")
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='1' 
                                    and g[edge[0]][edge[1]]['normal_open']=='0'], width=8, alpha=0.5, edge_color="tab:red")
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='0' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and g[edge[0]][edge[1]]['switch']=='R'], edge_color="tab:orange", style='dashed')
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='0' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and (g[edge[0]][edge[1]]['switch']=='RC' or g[edge[0]][edge[1]]['switch']=='CB')
                                    and model.sw[edge[0],edge[1],step].value>0.5], width=8, alpha=0.5, edge_color="tab:pink")
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='0' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and (g[edge[0]][edge[1]]['switch']=='RC' or g[edge[0]][edge[1]]['switch']=='CB')
                                    and model.sw[edge[0],edge[1],step].value<0.5], edge_color="tab:pink", style='dashed')
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='1' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and g[edge[0]][edge[1]]['switch']=='R'], width=8, alpha=0.5, edge_color="tab:orange")
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='1' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and g[edge[0]][edge[1]]['switch']=='M' and model.sw[edge[0],edge[1],step].value>0.5], 
                                    width=8, alpha=0.5, edge_color="tab:brown")
            nx.draw_networkx_edges(g,pos,edgelist=[edge for edge in g.edges() if g[edge[0]][edge[1]]['failed']=='1' 
                                    and g[edge[0]][edge[1]]['normal_open']=='1' and g[edge[0]][edge[1]]['switch']=='M' and model.sw[edge[0],edge[1],step].value<0.5],
                                    edge_color="tab:brown", style='dashed')
            '''    
            #plt.box(False)
            #plt.savefig(folder+'plot_T_'+str(step)+'.pdf')
            #plt.savefig(folder+'plot_T_'+str(step)+'.svg')
            #plt.show()

        except:
            self.next_state = None
            self.reward = -len(graph.nodes)
            self.graph = g
            self.model = model
            return None


        for node,(x,y) in pos.items(): # save node positions as labels (x,y)
            g.nodes[node]['pos_x'] = float(x)
            g.nodes[node]['pos_y'] = float(y)

        # ------------------------- Plot performance evolution -------------------------
        # p_ns = [sum(model.p_ns[n,t].value for n in model.N) for t in model.T0]
        p_ns = [100-sum(model.p_ns[n,t].value for n in model.N)*100/sum(model.p_d[n] for n in model.N) for t in model.T0]
        
        #if counter==128:
        #    # red dashes
        #    plt.plot(model.T0_0, [100, p_ns[0], p_ns[1], p_ns[2]], 'ro')
        #    csfont = {'fontname':'Times New Roman'}
        #    plt.xlabel('Service Restoration', fontsize=14, **csfont)
        #    plt.ylabel('Supplied Power ( % )', fontsize=14, **csfont)
        #    plt.ylim(ymin=0, ymax=110)
        #    plt.xticks(np.arange(0, T0, step=1), fontsize=12, **csfont)
        #    plt.yticks(np.arange(0, 110, step=10), fontsize=12, **csfont)
        #    plt.savefig(folder+'Restoration curve.pdf')
        #    plt.savefig(folder+'Restoration curve.svg')
        #    plt.show()

        # ------------------------- Save results ---------------------------------
        self.save_results(folder, model, g) # Save results of the experiment in a directory

        # Save the resultant model and graph as the latest experiment output of this script
        with open(experiment+'_model.pkl', mode='wb') as file:
            cloudpickle.dump(model, file) # save the resultant model

        nx.write_gpickle(g, experiment+'_graph.pkl') # save the resultant configuration 

        # next_state <- remaining damaged lines + active/reactive power, node voltage, non-supplied load
        # reward <- if the problem gives a solution, assign the difference between resulted p_ns and previous p_ns
        #           if the problem is infeasible, assign a negative reward

        ind = list(model.f_l)
        val = list(model.f_l[:,:].value)
        result_fl = [ j for i,j in zip(ind, val)]


        ind = list(model.p)
        val = list(model.p[:,:,:].value)
        result_p = [ j for i,j in zip(ind, val) if i[2] == 3]

        result = result_fl + result_p

        ind = list(model.q)
        val = list(model.q[:,:,:].value)
        result_q = [ j for i,j in zip(ind, val) if i[2] == 3]


        result += result_q

        ind = list(model.v)
        val = list(model.v[:,:].value)
        result_v = [ j for i,j in zip(ind, val) if i[1] == 3]


        result += result_v

        ind = list(model.p_ns)
        val = list(model.p_ns[:,:].value)
        result_p_ns = [ j for i,j in zip(ind, val) if i[1] == 3]


        self.next_state = result + result_p_ns
        self.reward = p_ns[2] - p_ns[1]
        self.graph = g
        self.model = model
    
    def step(self, action, counter):
        self.response(action, counter = counter)
        return self.next_state, self.reward, True, [self.graph,self.model]
    
    def close(self):
        self.model = None
        self.graph = None
        self.next_state = None
        self.reward = None
        return None


if __name__ == "__main__":
    # ReconfigAction and RepairAction are to be taken from an RL agent
    #model, graph, next_state, reward = response(ReconfigAction, RepairAction)
    reconf = {(16, 29): False, (29, 16): False, (8, 9): False, (9, 8): False, (22, 35): False, (35, 22): False, (20, 22): False, (22, 20): False, (14, 33): False, (33, 14): False, (24, 10): False, (10, 24): False, (33, 31): False, (31, 33): False, (11, 9): True, (9, 11): True, (13, 12): True, (12, 13): True, (25, 11): True, (11, 25): True, (18, 5): False, (5, 18): False, (24, 23): True, (23, 24): True, (20, 19): False, (19, 20): False, (21, 36): False, (36, 21): False, (20, 21): True, (21, 20): True, (31, 30): True, (30, 31): True, (32, 13): False, (13, 32): False, (27, 26): True, (26, 27): True, (5, 4): True, (4, 5): True, (23, 19): False, (19, 23): False, (7, 27): False, (27, 7): False}
    repair = (12,13)
    envir = environement()
    envir.step(reconf,0)
    print(len(envir.next_state))

    #if envir.next_state == None:
    #    print("reset")
    #    envir.reset()