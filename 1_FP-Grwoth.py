
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[29]:


adult = pd.read_csv("/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/adult_modified.csv")
test = pd.read_csv("/Users/ellabella/Desktop/Course 2018 Fall/2018 Fall DSC440 Data Mining/Project/mini-project/test_modified.csv")
test.head(5)


# In[39]:


class TreeNode:
    def __init__(self, name, count, parentNode): # construct a tree node
        self.name = name
        self.count = count
        self.parent = parentNode
        self.children = {}
        self.nodeLink = None

    def inc(self, num): # increase the count during traversing 
        self.count += num

    def disp(self, ind = 1):  # a helper function for testing 
        print('  '*ind, self.name, '  ', self.count)
        for children in self.children.values():
            children.disp(ind+1)

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        key = frozenset(trans)
        if key in retDict:
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict

def createTree(dataSet, minSup):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k]) # delete all the items that are less than min_support
    freqItemSet = set(headerTable.keys()) # all fruequent itemsets
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] # element: [count, node]

    retTree = TreeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
    # dataSet：[element, count]
        localD = {}
        for item in tranSet:
            if item in freqItemSet: # filter all the frequent itemset
                localD[item] = headerTable[item][0] # element : count
        if len(localD) > 0:
            # sort the item form the order
            orderedItem = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            # order the 
            updateTree(orderedItem, retTree, headerTable, count)
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = TreeNode(items[0], count, inTree)
        if headerTable[items[0]][1]==None: # point to the item first time occur 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items)> 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath):  
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, myHeaderTab):
    treeNode = myHeaderTab[basePat][1]
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count 
        treeNode = treeNode.nodeLink
    return condPats

# recursively find freqitems
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # sort items ascending order
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:  
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        #print ('finalFrequent Item: ',newFreqSet)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable)
        #print ('condPattBases :',basePat, condPattBases)

        myCondTree, myHead = createTree(condPattBases, minSup)
        #print ('head from conditional tree: ', myHead)
        if myHead != None: 
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)     


# ## Testing the Data

# In[41]:


get_ipython().run_cell_magic('time', '', 'test_data = np.array(test).tolist() # Convert the dataframe to list \ninitSet = createInitSet(test_data) # Construct a dataset of trasactions\nminSup=len(initSet)*0.23 # minsupport\nmyFPtree, myHeaderTab = createTree(initSet, minSup)\n\nfreqItems = []\nmineTree(myFPtree, myHeaderTab, minSup, set([]), freqItems)\nfor x in freqItems:\n    print(x)')


# In[43]:


len(freqItems)


# In[48]:


get_ipython().run_cell_magic('time', '', 'adult_data = np.array(adult).tolist() # Convert the dataframe to list \ninitSet = createInitSet(adult_data) # Construct a dataset of trasactions\nminSup=len(initSet)*0.23 # minsupport\nmyFPtree, myHeaderTab = createTree(initSet, minSup)\n\nfreqItems_2 = []\nmineTree(myFPtree, myHeaderTab, minSup, set([]), freqItems_2)\nfor x in freqItems_2:\n    print(x)')


# In[49]:


len(freqItems_2)

