#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('Wan_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    
    np.random.seed(0)


    if args.cluster_num:

        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state

        mus = np.random.standard_normal((args.cluster_num,2))
        # initiate sigmas
        if not args.tied:
            # k * 2 * 2 
            sigmas = 4*np.array([np.eye(2) for i in range(args.cluster_num)])
        # tied variances 
        else:
            sigmas = 4*np.array(np.identity(2)) # 2*2
    
        initials[:] = 1.0 / args.cluster_num ## initiate pi - k * 1  sum = 1
        # initiate transition matrix k * k 
        transitions = np.random.dirichlet(np.ones(args.cluster_num),size=args.cluster_num).T
        
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = (initials, transitions, mus, sigmas)
    # raise NotImplementedError #remove when model initialization is implemented
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log

    alphas = np.zeros((len(data),args.cluster_num))

    log_likelihood = 0.0
    initials, transitions, mus, sigmas = model 
    
    if not args.tied: 
        emissions = np.array([multivariate_normal(mus[k],sigmas[k]).pdf(data) for k in range(args.cluster_num)]).T
    else:
        emissions = np.array([multivariate_normal(mus[k],sigmas).pdf(data) for k in range(args.cluster_num)]).T  
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
   
    # base case for alpha 
    for t in range(len(data)):
        # initialize pi--initials 
        if t == 0: # pi[k]
            for k in range(args.cluster_num):
                alphas[0,k] = initials[k] * emissions[t,k]  

            log_likelihood += np.log(np.sum(alphas[0]))

            alphas[0] /= np.sum(alphas[0])   
    # recursion 


        elif t > 0:
            for k in range(args.cluster_num):
              
                alphas[t,k] = np.sum(alphas[t-1, :] * transitions[:, k]) * emissions[t,k]         

            # sum alpha 
            sum_alpha = np.sum(alphas[t])

            # calculate likelihood 
            log_likelihood += np.log(sum_alpha)

            # normalize data 
            alphas[t] /= sum_alpha   
 
    # raise NotImplementedError
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    initials, transitions, mus, sigmas = model 


    if not args.tied: 
        emissions = np.array([multivariate_normal(mus[k],sigmas[k]).pdf(data) for k in range(args.cluster_num)]).T
    else:
        emissions = np.array([multivariate_normal(mus[k],sigmas).pdf(data) for k in range(args.cluster_num)]).T  
    
    for t in range(len(data)-1,-1,-1):
        # initiate the base case 
        if t == len(data)-1:
            for k in range(args.cluster_num):   
                betas[t,k] = 1

            betas[t] /= np.sum(betas[t])

        # recursive for beta 
        else:
            for k in range(args.cluster_num):

                betas[t,k]= np.sum(betas[t+1,:] * transitions[k,:]* emissions[t+1,:])     

            sum_betas= np.sum(betas[t])
            betas[t] /= sum_betas        

    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    # raise NotImplementedError
    
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    initials, transitions, mus, sigmas = model 

    for r in range(args.iterations):
        # print(r)

        if not args.tied: 
            emissions = np.array([multivariate_normal(mus[k],sigmas[k]).pdf(train_xs) for k in range(args.cluster_num)]).T
        else:
            emissions = np.array([multivariate_normal(mus[k],sigmas).pdf(train_xs) for k in range(args.cluster_num)]).T 

        alphas, _ = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)

        # print(model[])

        # E-step 
        gammas = np.zeros((train_xs.shape[0], args.cluster_num))
        ksi = np.zeros((len(train_xs), args.cluster_num, args.cluster_num))
        
        for t in range(len(train_xs)):
            for k in range(args.cluster_num):
                gammas[t, k] = alphas[t, k] * betas[t, k] 
            
            gammas[t, :] /= np.sum(gammas[t, :]) 

            if t != 0:
                for k in range(args.cluster_num): # state from 
                    for j in range(args.cluster_num): # state to 
                        ksi[t, k, j] = alphas[t-1, k] * transitions[k, j]* betas[t, j]* emissions[t,j]

                ksi[t] /= np.sum(ksi[t])

        # M-step 
        # update initials 

        # for k in range(args.cluster_num):
        initials = gammas[0]
        
        # update transitions 
        for j in range(args.cluster_num): # state to 
            for k in range(args.cluster_num): # state from 
                transitions[j,k] = np.sum(ksi[:,j,k]) / np.sum(gammas[:,j])
        # transitions = np.sum(ksi, axis=0)/np.sum(gammas,axis=0).reshape(-1,1)

        sigmas_tied = []
        # update mus
        for k in range(args.cluster_num): 
            mus[k] = np.dot(gammas[:,k], train_xs) / np.sum(gammas[:,k])

        # update sigmas 
            if not args.tied: 

                sigmas[k] = np.dot(gammas[:,k] * (train_xs - mus[k]).T, train_xs- mus[k])/np.sum(gammas[:,k])

            else:

                sigmas_tied.append(np.dot(gammas[:,k] * (train_xs - mus[k]).T, train_xs- mus[k]))

        if args.tied: 
                
            sigmas  = np.sum(sigmas_tied,axis = 0)/len(train_xs)            
        # print(sigmas)

        model = (initials, transitions, mus, sigmas)

    # raise NotImplementedError #remove when model training is implemented
    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    _, log_likelihood = forward(model,data,args)
    ll = log_likelihood / data.shape[0]
    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model[0]
    transitions = model[1]
    mus = model[2]
    sigmas = model[3]
    # raise NotImplementedError #remove when parameter extraction is implemented
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
