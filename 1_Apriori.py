
# coding: utf-8

# In[12]:


# Import the package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

adult=pd.read_csv('/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/adult.data.txt',header=None)
test=pd.read_csv('/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/adult.test.txt',header=None)


# In[13]:


# Preview the dataset
adult.head(5)
# Check the values of the dataset 
adult.values
#Assign the column names 
adult.columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income']
test.columns=['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','Income']
adult.head(10)
# Preview the data
adult.describe()
# Check the datatypes 
adult.dtypes


# In[14]:


adult.describe()


# In[15]:


# Check the count for the unique value in each column 
for i in adult.columns:
    print(i,(len(set(adult[i]))))


# # Data Processing

# In[16]:


def data_processing(dataset):
# Drop one irrelevant variable 
    dataset.drop(labels='fnlwgt',axis=1,inplace=True)
# Discreteize continuous variables: age, education_num, capital_gain, capital_loss, hoursperweek
    bin_age=[10,30,50,70,90]
    dataset['age']=pd.cut(dataset['age'],bins=bin_age,labels=["Youth", "Middle-age","older","senior"]) # binning age
    bin_edu_num=[1,5,10,15,20]
    dataset['education_num']=pd.cut(dataset['education_num'],bins=bin_edu_num,labels=["1-5", "6-10","11-15","16-20"])# binning education_num
    dataset['capital_gain']=pd.cut(dataset['capital_gain'],2,labels=["low", "high"])# binning capital_gain 
    dataset['capital_loss']=pd.cut(dataset['capital_loss'],2,labels=["low", "high"])# binning capital_loss 
    dataset['hours_per_week']=pd.cut(dataset['hours_per_week'],4,labels=["part-time", "full-time","extra","workholic"])# binning hours_per_week
    
    return dataset 


# In[17]:


data_processing(adult)


# In[18]:


data_processing(test)


# In[19]:


# Save data to csv
adult.to_csv('/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/adult_modified.csv',index=False)


# In[20]:


# Save data to csv
test.to_csv('/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/test_modified.csv',index=False)


# In[21]:


# Create 1-item dataset in the transaction that include all unique items 
I1 =(set(np.array(adult).reshape((-1,)))) # Reshape the dimension of matrix to one set 
I1 = [{i} for i in I1] # Use a set to restore each individual item 


# In[22]:


# Create teh scanner to traverse and count each item 
def item_scanner(D,Ik,minsupport): # D is the dataset, Ik is the candidate dataset, minsupport is a vlue
    scanset={} # build a empty set
    for t2 in D.values: #traverse each transaction in dataset, each item in each traction to check if the item is a subset of transaction
        for ci in Ik:
            if ci.issubset(set(t2)):
                if not tuple(ci) in scanset: # if not, add the item into the scanset
                    scanset[tuple(ci)]=1
                else:
                    scanset[tuple(ci)]+=1 # if yes, then increase the candidate item count 
                    
    item_num=float(adult.shape[0])
    target_list=[]  # build a list to store the candidate item 
    supportdic={} # store the frequent items with a dictionary
    for key in scanset:  # find the frequent iterm that meets the min_support 
        support = scanset[key]/item_num
        if support>=minsupport:
            target_list.insert(0,key) # insert the frequent item into the top of the dictionary
        supportdic[key]=support       # record the value of support
    return target_list, supportdic


# In[23]:


# Based on I1, create the apriori algorithm to continue scan the k items
def apriori(dataset, minsupport=0.23):
    I1 = [{i} for i in set(np.array(dataset).reshape(-1))]
    L1, supportdic = item_scanner(dataset,I1,minsupport)
    L=[L1]
    k=2 # start with the item K=2 to traverse the transactions
    while (len(L[k-2])>0):
        Ik = apriori_generator(L[k-2],k)
        Lk, supK=item_scanner(dataset,Ik,minsupport)
        supportdic.update(supK)
        L.append(Lk)
        k+=1
        print(k)
    return L, supportdic     


# In[24]:


# The apriori generator to generate the candidate K itemsets
def apriori_generator(Lk, k):
    result_list=[]
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)): # to reduce the traversal cost, merge two sets to the candidate set 
            lst1=list(Lk[i])[:k-2]  # If two lists have the same (0:K-2)item, then merge two sets
            [str(i) for i in lst1].sort()
            lst2=list(Lk[j])[:k-2]
            [str(i) for i in lst2].sort()
            if lst1==lst2:
                result_list.append(set(Lk[i]) | set(Lk[j])) # Merge two sets with the unique number
    return result_list


# # Testing the Algorithm

# In[25]:


print(I1)


# In[26]:


len(I1)


# In[27]:


get_ipython().run_cell_magic('time', '', '# Time the apriori algorithm for adult dataset \nLK, SK= apriori(adult)')


# In[28]:


print(LK)


# In[29]:


# Check the frequent itemsets
print([len(l) for l in LK])


# In[30]:


# Check the sum of the frequent itemsests
sum([len(l) for l in LK])


# In[31]:


# Test the dataset for train dataset 
I11 =(set(np.array(test).reshape((-1,)))) # Reshape the dimension of matrix to one set 
I11 = [{i} for i in I11] # Use a set to restore each individual item 
print(len(I11))


# In[32]:


print(I11)


# In[33]:


get_ipython().run_cell_magic('time', '', '# Time the apriori algorithm for test dataset \nLK2, SK2= apriori(test)')


# In[34]:


print(LK2)


# In[38]:


# Check the frequent itemsets
print([len(l) for l in LK2])


# In[39]:


# Check the sum of the frequent itemsests
sum([len(l) for l in LK2])


# # Apriori Algorithm Optimization
# The improved algorithm are refered from the following paper:
# 
# 
# Basically, the improvement of algorithm can be described as follows:
# //Generate items, items support, their transaction ID
# (1) L1 = find_frequent_1_itemsets (T);
# (2) For(k=2;Lk-1 ≠Φ;k++){
# //Generate the Ck from the LK-1
# (3) Ck = candidates generated from Lk-1;
# //get the item Iw with minimum support in Ck using L1,(1≤w≤k). (4) x = Get _item_min_sup(Ck, L1);
# // get the target transaction IDs that contain item x.
# (5) Tgt = get_Transaction_ID(x);
# (6) For each transaction t in Tgt Do
# (7) Increment the count of all items in Ck that are found in Tgt; (8) Lk= items in Ck ≥ min_support;
# (9) End;
# (10) }
# 

# In[ ]:


def transaction_scanner(dataset, Ik, minsupport):
    scanset={}
    dict_item_support = {}  # store the item and the support count 
    dict_item_transactions = {} # store the item and corresponding transctions 
    transaction_num = dataset.shape[0] # tranction_num
    for key in scanset:
        minsupport = scanset[key]/transaction_num # min_support
   # traverse to get the key and values for two dictionaries 
    for item in Ik:
        dict_item_support[tuple(item)] = dict_item_support.get(tuple(item), 0)
        dict_item_transactions[tuple(item)] = []
        for index, tid in enumerate(dataset.values):
            if set(item).issubset(set(tid)):
                dict_item_support[tuple(item)] += 1
                dict_item_transactions[tuple(item)].append(str(index))
   # Find the dandidate that meets the minsupport
    dict_item_support_can = {i:j for i,j in dict_item_support.items() if j > minsupport}
   # traverse to get the key and values for transactions 
    dict_item_transactions_can = {}
    for select_item in dict_item_support_can:
        if select_item in dict_item_transactions:
            dict_item_transactions_can[select_item]=dict_item_transactions[select_item]
    return dict_item_support_can, dict_item_transactions_can

def item_generator(dict_item_support, dict_item_transactions, k, dataset, minsupport):
    Len = len(dict_item_support)
    dict_item_support=list(dict_item_support)
    newList =[]
    new_tran_list = []          
    for i in range(Len):
        for j in range(i+1,Len):
            L1 = list(dict_item_support[i])[:k-2]
            L2 = list(dict_item_support[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                newList.append(set(dict_item_support[i])|set(dict_item_support[j]))
                new_tran_list.append(set(dict_item_transactions[dict_item_support[i]])&set(dict_item_transactions[dict_item_support[j]]))
    dict_item_support_New = {}
    dict_item_transactions_New = {}
    key_length = [len(i) for i in new_tran_list]
    miniS = dataset.shape[0]*minsupport
    for i in range(len(newList)):
        if key_length[i] > miniS:
            dict_item_support_New[tuple(newList[i])] = key_length[i]
            dict_item_transactions_New[tuple(newList[i])] = new_tran_list[i]

    return dict_item_support_New, dict_item_transactions_New

def Apriori(dataset, minsupport):
    C = []
    L1, L2 = [], []
    k = 2
    C1 = (np.array(dataset).reshape(-1))
    C1 = [{i} for i in list(set(C1))]
    L11, L12 = transaction_scanner(dataset, C1, minsupport)
    L1.append(L11)
    L2.append(L12)
    while len(L1[k-2])>0:
        temp_L1, temp_L2 = item_generator(L1[k-2], L2[k-2], k, dataset, minsupport)
    # temp_L1, temp_L2 = scanD(dataset, temp_C, miniSupport)
        k+=1
        L1.append(temp_L1)
        L2.append(temp_L2)
        print('K:' ,k)
    return L1, L2


# In[36]:


get_ipython().run_cell_magic('time', '', 'L_full, C_full = Apriori(adult, 0.23)')


# In[37]:


sum([len(l) for l in L_full])


# In[41]:


get_ipython().run_cell_magic('time', '', 'L_full2, C_full2 = Apriori(test, 0.23)')


# In[42]:


sum([len(l) for l in L_full2])

