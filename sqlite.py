

#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import sqlite3 as lite
import sys
import cPickle
import glob
import os
#import pdb
#pdb.set_trace()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

def insertDB(dir, specs, model_id):
    
    db = model_id + '.db'
    
    folderlist = glob.glob(dir + '/jobman*')
    opt_model_path = ''
    table = []
    for folder in folderlist:
        state_path = folder + '/state.pkl'

        opt_model_path = folder + '/model_' + model_id + '_optimum.pkl'
        
        
        if os.path.exists(state_path) and os.path.exists(opt_model_path): 
            
            with open(state_path, 'rb') as f:
                state = cPickle.load(f)
            with open(opt_model_path, 'rb') as g:
                model = cPickle.load(g)
                
            #test_error = model.monitor.channels['test_softmax1_misclass'].val_record
            valid_error = model.monitor.channels['valid_softmax2_misclass'].val_record
            #best_test_error = np.sort(test_error)[0]
            best_valid_error = np.min(valid_error)

             
	    #print state.dataset
            print state.dataset, folder, 'best valid error', np.asscalar(best_valid_error)
            #print 'best test error', np.asscalar(best_test_error)
            #print folder
	    #print state.batch_size.__class__
            #print 'EXIST', folder
            #import pdb
            #pdb.set_trace()
#             if state.dataset == 'mnist':
#             for spec in specs:
#                 table.append(getattr(state, spec))
#                 
                
#                 
#                 print state.dataset, 'EXISTS'
#                 print state.config_id
#                 print "best test error", np.asscalar(best_valid_error)
#                 
#                 
                
#                 table.append((folder,
#                 
#                             state.dataset,
#                             np.asscalar(best_test_error),
#                             state.batch_size, 
#                             state.learning_rate, 
#                             state.init_momentum, 
#                             state.ext_array.moment_adj.final_momentum,
#                             state.layers.hidden1.max_col_norm,
#                             state.layers.hidden1.weight_decay,
#                             state.layers.hidden1.dim,
#                             state.layers.hidden2.max_col_norm,
#                             state.layers.hidden2.weight_decay,
#                             state.layers.hidden2.dim
#                             ))

       # else:
        #    print (state_path +' or ' + opt_model_path,
         #   'does not exist')
    
#     
#     conn = lite.connect(db)
# 
#     with conn:
#         cur = conn.cursor()
#         cur.execute("DROP TABLE IF EXISTS stats")
#         cur.execute("CREATE TABLE stats(folder TEXT PRIMARY KEY, \
# 					                    dataset TEXT, \
#                                         test_error REAL, \
#                                         batch_size INT, \
#                                         learning_rate REAL, \
#                                         init_momentum REAL, \
#                                         final_momentum REAL, \
#                                         h1_max_col_norm REAL, \
#                                         h1_weight_decay REAL, \
#                                         h1_dim, \
#                                         h2_max_col_norm REAL, \
#                                         h2_weight_decay REAL, \
#                                         h2_dim)")
#         cur.executemany("INSERT INTO stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", table)
#         conn.commit()
        
        #cur.execute(".header on")
        #cur.execute(".mode column")
        #cur.execute("SELECT * FROM stats ORDER BY learning_rate")
        #print cur.fetchall()
        #cur.execute("SELECT min(h1_dim) FROM (SELECT *  FROM stats WHERE valid_error<0.5)")
        
        
        
    
#     cars = (
#         (1, 'Audi', 52642),
#         (2, 'Mercedes', 57127),
#         (3, 'Skoda', 9000),
#         (4, 'Volvo', 29000),
#         (5, 'Bentley', 350000),
#         (6, 'Hummer', 41400),
#         (7, 'Volkswagen', 21600))
#     
#     
#     con = lite.connect('test.db')
#     
#     with con:
#         
#         cur = con.cursor()    
#         
#         cur.execute("DROP TABLE IF EXISTS Cars")
#         cur.execute("CREATE TABLE Cars(Id INT, Name TEXT, Price INT)")
#         cur.executemany("INSERT INTO Cars VALUES(?, ?, ?)", cars)

def queryDB(specs, model_id):
    
    db = model_id + '.db'
    
    conn = lite.connect(db)
    #import pdb
    #pdb.set_trace()
    print np.__version__
    with conn:
        cur = conn.cursor()
        

        misclass = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        min_dict = {}
        max_dict = {}
        for var in specs:
            min_dict[var] = []
            max_dict[var] = []        
            for valid_err in misclass:
                cur.execute("SELECT min(%s) FROM (SELECT * FROM stats WHERE valid_error<?)" % var, (valid_err,))
                min_dict[var].append(cur.fetchone()[0])
                cur.execute("SELECT max(%s) FROM (SELECT * FROM stats WHERE valid_error<?)" % var, (valid_err,))
                max_dict[var].append(cur.fetchone()[0])
        
        fig = plt.figure(figsize=(10,80))
        
#         plot_arg = 920
        
        plots = {}
#         left = 10
#         bottom = 110
#         width = 100
#         height = 100
        gs = gridspec.GridSpec(20, 1)
        i = 0
        for param in specs:
#             plot_arg += 1
            

            
            plots[param] = fig.add_subplot(gs[i,0])
            plots[param].plot(misclass, min_dict[param])
            plots[param].plot(misclass, max_dict[param])
            plots[param].set_title(param)
            
            #bottom += 120
            i+=1
                
        
        plt.savefig("params.pdf")
        plt.close()
            



if __name__ == '__main__':
    dir = './mlp'
    #db = 'config_mlp_mnist_noisy.db'

#     specs = ["batch_size", "learning_rate", "init_momentum", 
# 		"final_momentum", "h1_max_col_norm", "h1_weight_decay",
# 		"h1_dim", "h2_max_col_norm", "h2_weight_decay", "h2_dim"]

    specs = ["batch_size", 
	    "learning_rate", 
	    "init_momentum", 
	    "ext_array.moment_adj.final_momentum",
	    "layers.hidden1.max_col_norm",
	    "layers.hidden1.weight_decay",
	    "layers.hidden1.dim",
	    "layers.hidden2.max_col_norm",
	    "layers.hidden2.weight_decay",
	    "layers.hidden2.dim"]


    #model_id = 'Clean200-200Cifar200epochPreproc'
    #model_id = 'Clean100cifar200epochPreproc'
    model_id = 'Noisy200-2kCifar200epochPreproc'
    insertDB(dir, specs, model_id)
#     queryDB(specs, model_id)
