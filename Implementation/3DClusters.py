# -*- coding: utf-8 -*-

import numpy as np
from math import cos,sin
from random import randrange,random
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import DBSCAN, SpectralClustering, AgglomerativeClustering, KMeans, Birch
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score
from random import randrange
from ioFunctions import *

np.set_printoptions(precision=4)

def KmeansC(points, k):
    clustering = KMeans(k,init='random').fit(points)
    return clustering.labels_

def SpectralC(points, k):
    clustering = SpectralClustering(k).fit(points)
    return clustering.labels_

def AgglomC(points, k):
    clustering = AgglomerativeClustering(k).fit(points)
    return clustering.labels_

def BirchC(points, k):
    clustering = Birch(0.015,n_clusters = k).fit(points)
    return clustering.labels_

def DBSCANf(points, k):
    eps0 = 10**(-5)
    eps1 = 1
    for i in range(100):
        clustering = DBSCAN((eps0+eps1)/2).fit(points)
        k0=max(clustering.labels_)
        if k0 > k :
            eps0=(eps0+eps1)/2
        if k0 < k or k0 == -1:
            eps1=(eps0+eps1)/2
        if k0 == k:
            break
    print("DBSCAN chose epsilon of",(eps0+eps1)/2,"and got k=",max(clustering.labels_) )        
    return clustering.labels_

def RandomC(points, k):
    return [randrange(k+1) for i in range(len(points))]

def comparison():
    # [ 200 hand, 51 glasses, 124 octopus, 257 bird, 353 bearing]
    inputs = ["hand","glasses","octopus","bird","bearing"]
    inputsIds = [200, 51, 124, 257, 353]
    
    globalResults = []
    maxVMeasure = []
    minVMeasure = []
    avgVMeasure = []
    maxVMeasureT = []
    minVMeasureT = []
    avgVMeasureT = []
    for i in range(len(inputs)):
        name = inputs[i]
        print("\n"+name+" model:")
        inId = str(inputsIds[i])
        readseg = open("seg/"+inId+"/"+inId+"_1.seg", "r")
        groundT = read_seg(readseg)
        k = max(groundT)+1
        
        points = np.fromfile("points/"+name+"_points.dat")
        transformedP = np.fromfile("points/"+name+"_transformed.dat")
        points = points.reshape((int(points.shape[0]/3),3))
        transformedP = transformedP.reshape((int(transformedP.shape[0]/k),k))
        
        results = []
        resultsNames = []
        maxVMeasure.append(-np.inf)
        minVMeasure.append(np.inf)
        avgVMeasure.append(0)
        maxVMeasureT.append(-np.inf)
        minVMeasureT.append(np.inf)
        avgVMeasureT.append(0)
        
        VMeasures = []
        RandId = []
        fList = [KmeansC, AgglomC, BirchC]
        for func in fList:
            labels = func(points,k)
            scores = evaluateC(labels,groundT)
            if scores[-2]<minVMeasure[-1]:
                minVMeasure[-1] = scores[-2]
            if scores[-2]>maxVMeasure[-1]:
                maxVMeasure[-1] = scores[-2]   
            avgVMeasure[-1]+=scores[-2]/len(fList)
            VMeasures.append(scores[-2])
            RandId.append(scores[-1])
            results.append(scores)
            resultsNames.append(func.__name__[:-1])
            labels = func(transformedP,k)
            scores = evaluateC(labels,groundT)
            if scores[-2]<minVMeasureT[-1]:
                minVMeasureT[-1] = scores[-2]
            if scores[-2]>maxVMeasureT[-1]:
                maxVMeasureT[-1] = scores[-2]
            avgVMeasureT[-1]+=scores[-2]/len(fList)
            VMeasures.append(scores[-2])
            RandId.append(scores[-1])
            results.append(scores)
            resultsNames.append(func.__name__[:-1]+" (T)")
            
        line = "\t\t"
        for i in ["Homog.","Compl.","V_meas.","Rand Id."]:
            line += i+"\t"
        print("\n"+line)
            
        for i in range(len(results)):
            line = resultsNames[i]
            if len(line)<8:
                line += "\t"
            for j in range(len(results[i])):
                s = str(results[i][j])
                if len(s)>4:
                    s = s[:4]
                line += "\t" + s
            print(line)
        globalResults.append(results)
        
        colors = ['#9baed9','#9baed9','#e3a624','#e3a624','#8ecea8','#8ecea8']
        
        x = []
        y = np.array(VMeasures)
        x_pos = []
        for i in range(len(fList)):
            x.append(fList[i].__name__[:-1])
            x.append("(T)")
            x_pos.append(3*i)
            x_pos.append(3*i+1)
            
        barList = plt.bar(x_pos,y,width=0.9)
        for i in range(0,len(barList)):
            barList[i].set_color(colors[i])
        plt.xticks(x_pos, x) 
        plt.xlabel('Algorithms before and after transformation (T)')
        plt.ylabel('V-Measure')
        plt.title("V-Measure results on "+name+" model")
        plt.savefig("graph/VMeasure"+name+".jpg".format(i))
        plt.show()
        
        y = np.array(VMeasures)
        barList = plt.bar(x_pos,y,width=0.9)
        for i in range(0,len(barList)):
            barList[i].set_color(colors[i])
        plt.xticks(x_pos, x) 
        plt.xlabel('Algorithms before and after transformation (T)')
        plt.ylabel('Rand Index')
        plt.title("Rand Index results on "+name+" model")
        plt.savefig("graph/RandIndex"+name+".jpg".format(i))
        plt.show()
        
    avgV = []
    xp = []
    menStd = []
    x_pos = []
    
    for i in inputs:
        xp.append(str(i))
        xp.append("(T)")
    
    for i in range(len(avgVMeasure)):
        avgV.append(avgVMeasure[i])
        avgV.append(avgVMeasureT[i])
        menStd.append([avgVMeasure[i]-minVMeasure[i],maxVMeasure[i]-avgVMeasure[i]])
        menStd.append([avgVMeasureT[i]-minVMeasureT[i],maxVMeasureT[i]-avgVMeasureT[i]])
        x_pos.append(i*3)
        x_pos.append(i*3+1)
        
    x = xp
    y = np.array(avgV)
    menStd = np.array(menStd)
    menStd=menStd.transpose()
    barList = plt.bar(x_pos,y,width=0.9,yerr=menStd)
    for i in range(0,len(barList)):
        if i%2:
            barList[i].set_color("#3d5a80")
        else:
            barList[i].set_color("#ee6c4d")
            
    plt.xticks(x_pos, x) 
    plt.xlabel('3D Models')
    plt.ylabel('Average/Minimum/Maximum Clustering Accurracy')
    plt.title("Clustering accurracy on models with different algorithms")
    plt.savefig("graph/ClusteringResults.jpg".format(i))
    plt.show()
        
    
def visualizeC(name,transform=True,showGT=False):
    func = KmeansC
    
    
    inputs = ["hand","glasses","octopus","bird","bearing"]
    inputsIds = [200, 51, 124, 257, 353]
    idM = str(inputsIds[inputs.index(name)])
    readseg = open("seg/"+idM+"/"+idM+"_1.seg", "r")
    groundT = read_seg(readseg)
    k = max(groundT)+1
    
    if not(showGT):
        points = np.fromfile("points/"+name+"_points.dat")
        points = points.reshape((int(points.shape[0]/3),3))
        if transform:
            transformedP = np.fromfile("points/"+name+"_transformed.dat")
            transformedP = transformedP.reshape((int(transformedP.shape[0]/k),k))
            points = transformedP
        labels = func(points,k)
    else:
        labels = groundT
    outClusters(labels,name)
            
def evaluateC(labels,groundT):
    assert(len(labels)==len(groundT))
    scores = []
    scores.append(homogeneity_score(groundT,labels)) 
    scores.append(completeness_score(groundT,labels))
    scores.append(v_measure_score(groundT,labels))
    scores.append(adjusted_rand_score(groundT,labels))
    return scores