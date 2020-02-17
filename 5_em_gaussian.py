#!/usr/bin/env python3
import numpy as np


if not __file__.endswith('Wan_em_gaussian.py'):
   print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
   exit(1)

DATA_PATH = "./points.dat" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    # read data 
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    
    np.random.seed(0)
    
    clusters = []

    if args.cluster_num:
        # lambdas = probability for each group 
        lambdas = np.random.dirichlet(np.ones(args.cluster_num),size=1).T
        # assign cluster_num to lambdas 
        args.cluster_num = len(lambdas)

        # initiate mu for each k cluster k*2 
        mus = np.random.standard_normal((args.cluster_num,2))
        # initiatie the covariance matrix K*2*2
        # full covariance matrix 

        if not args.tied:
            #sigmas = np.random.standard_normal((args.cluster_num,2,2))
            sigmas = np.array([np.eye(2)*3 for i in range(args.cluster_num)])
        # tied variances
        else:
            sigmas = np.array(np.identity(2))*3
        #TODO: randomly initialize clusters (lambdas, mus, and sigmas)
        
        # raise NotImplementedError #remove when random initialization is implemented

    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    #TODO: do whatever you want to pack the lambdas, mus, and sigmas into the model variable (just a tuple, or a class, etc.)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    model = (lambdas,mus,sigmas)
    # raise NotImplementedErrr #remove when model initialization is implemented
    return model

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # recall the model 
    lambdas, mus, sigmas = model 
     # probability_of_xn_given_mu_and_sigma 
     # Use train data only 
    # dev_likelihood = []
    # train_likelihood = []
    # E step 

    for j in range(args.iterations):

        ksi_prime = [ ]

        for k in range(args.cluster_num):
            # probability_of_xn_given_mu_and_sigma * lambdas
            # lambdas[k] = 1* K vector  
            ksi_prime.append (lambdas[k] * multivariate_normal(mus[k],sigmas[k]).pdf(train_xs))

        ksi = ksi_prime /np.sum(ksi_prime,axis = 0)

        sigmas_tied = []

        # ksi = tau / np.sum(tau,axis = 0)
    # M step 

        for k in range(args.cluster_num):
                # maximize mu 
            mus[k] = np.dot(ksi[k], train_xs) / np.sum(ksi[k])
            
             # maximize lamda
            lambdas[k] = np.sum(ksi[k])/ train_xs.shape[0]
            # miximize sigmas 
            # ksi = (900,1) ksi[k] =1*1 train_xs = (900,2) mu[k] = (2,1)
            if not args.tied: 

                sigmas[k] = np.dot(ksi[k] * (train_xs - mus[k]).T, train_xs- mus[k])/np.sum(ksi[k])

            else:
                sigmas_tied.append(np.dot(ksi[k] * (train_xs - mus[k]).T, train_xs- mus[k]))/np.sum(ksi[k])

        if args.tied: 
            
            sigmas  = np.sum(sigmas_tied,axis = 0)/len(train_xs)
               
    return (lambdas,mus,sigmas) 
    # else: 
   
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    # raise NotImplementedError #remove when model training is implemented
def average_log_likelihood(model, data, args):
    from math import log
    from scipy.stats import multivariate_normal
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    lambdas, mus, sigmas = model 
    ll = 0.0
    ksi = []

    for k in range(args.cluster_num):

        if not args.tied:

            ksi.append(lambdas[k] * multivariate_normal(mus[k],sigmas[k]).pdf(data))

        else:

            ksi.append(lambdas[k] * multivariate_normal(mus[k],sigmas).pdf(data))

    ll = np.sum(np.log(np.sum(ksi,axis = 0))) / data.shape[0] 

    # raise NotImplementedError #remove when average log likelihood calculation is implemented
    return ll

def extract_parameters(model):
    #TODO: extract lambdas, mus, and sigmas from the model and return them (same type and shape as in init_model)
    lambdas = model[0]
    mus = model [1]
    sigmas = model[2]
    # raise NotImplementedError #remove when parameter extraction is implemented
    return lambdas, mus, sigmas
def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
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
    print('train model')
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs,args)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs,args)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
