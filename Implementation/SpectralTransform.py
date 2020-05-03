# -*- coding: utf-8 -*-

from scipy.sparse.csgraph import dijkstra
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from time import time
from ioFunctions import *

delta = 0.05 #Weight parameter, see associated paper
nu = 0.8 #Angular weight amplitude

class Triangle:
    def __init__(self,id0,id1,id2):
        self.id0 = id0
        self.id1 = id1
        self.id2 = id2
    
    def p0(self):
        return verts[self.id0]
    def p1(self):
        return verts[self.id1]
    def p2(self):
        return verts[self.id2]
    
    def n(self):
        v = np.cross(self.p1()-self.p0(),self.p2()-self.p1())
        return v/np.linalg.norm(v)
    
    def isAdjacent(self,otherTri):
        otherId = [otherTri.id0,otherTri.id1,otherTri.id2]
        counter = 0
        if self.id0 in otherId:
            counter += 1
        if self.id1 in otherId:
            counter += 1
        if self.id2 in otherId:
            counter += 1
        if counter == 2:
            return True
        return False
        
    def geoDist(self,otherTri): #Geodesic distance, for adjacent triangles
        assert(self.isAdjacent(otherTri))
        center0 = (self.p0()+self.p1()+self.p2())/3
        center1 = (otherTri.p0()+otherTri.p1()+otherTri.p2())/3
        if not(otherTri.id0 in [self.id0,self.id1,self.id2]):
            commonEdge = (otherTri.id1,otherTri.id2)
            outlier1 = 0
        elif not(otherTri.id1 in [self.id0,self.id1,self.id2]):
            commonEdge = (otherTri.id0,otherTri.id2)
            outlier1 = 1
        elif not(otherTri.id2 in [self.id0,self.id1,self.id2]):
            commonEdge = (otherTri.id0,otherTri.id1)
            outlier1 = 2
        if not(self.id0 in commonEdge):
            outlier0 = 0
        elif not(self.id1 in commonEdge):
            outlier0 = 1
        elif not(self.id2 in commonEdge):
            outlier0 = 2
        
        u,v,w = self.baryCoord(center0) 
        proj0 = self.p0()*u*(outlier0 != 0) + self.p1()*v*(outlier0 != 1) +self.p2()*w*(outlier0 != 2)
        u,v,w = otherTri.baryCoord(center1) 
        proj1 = otherTri.p0()*u*(outlier1 != 0) + otherTri.p1()*v*(outlier1 != 1) +otherTri.p2()*w*(outlier1 != 2)
        middle = (proj1+proj0)/2
        return np.linalg.norm( center0-middle )+ np.linalg.norm( center1-middle )
    
    def angDist(self,otherTri):
        assert(self.isAdjacent(otherTri))
        n0 = self.n()
        n1 = otherTri.n()
        angle = np.dot(n0,n1)/(np.linalg.norm(n0)*np.linalg.norm(n1))
        return nu*(1-angle)
    
    def arcWeight(self,otherTri,avgGeo,avgDist):
        assert(self.isAdjacent(otherTri))
        geoD = delta*self.geoDist(otherTri)/avgGeo
        angD = (1-delta)*self.angDist(otherTri)/avgDist
        return geoD + angD
    
    def baryCoord(self,vertex):
        a = self.p0()
        b = self.p1()
        c = self.p2()
        v0 = b - a
        v1 = c - a
        v2 = vertex - a
        d00 = np.dot(v0, v0);
        d01 = np.dot(v0, v1);
        d11 = np.dot(v1, v1);
        d20 = np.dot(v2, v0);
        d21 = np.dot(v2, v1);
        denom = d00 * d11 - d01 * d01;
        v = (d11 * d20 - d01 * d21) / denom;
        w = (d00 * d21 - d01 * d20) / denom;
        u = 1.0 - v - w;
        return u,v,w
    
class DualGraph:
    def __init__(self,faces):
        self.triangles = [Triangle(i[0],i[1],i[2]) for i in faces] 
        self.avgGeo,self.avgAng = self.computeAvgDist()
        self.weightList = self.computeWeights()        
        self.distMat = self.computeDist()
        self.sigma = self.computeSigma()
        self.affinityMat = self.computeAffMat()
    
    def computeWeights(self):
        wList = []
        for i in range(len(self.triangles)):
            for j in range(len(self.triangles)):
                if self.triangles[i].isAdjacent(self.triangles[j]):
                    wList.append((i,j,self.triangles[i].arcWeight(self.triangles[j],self.avgGeo,self.avgAng)))
        return wList
    
    def computeAvgDist(self):
        avgAng = 0
        avgGeo = 0
        n = 0
        for i in range(len(self.triangles)):
            for j in range(i+1,len(self.triangles)):
                if not(self.triangles[i].isAdjacent(self.triangles[j])):
                    continue
                n += 1 
                avgAng += self.triangles[i].angDist(self.triangles[j])
                avgGeo += self.triangles[i].geoDist(self.triangles[j])
        avgAng = avgAng/n
        avgGeo = avgGeo/n
        return avgAng,avgGeo
    
    def computeDist(self):
        print("Computing distances...")
        n = len(self.triangles)
        array = np.zeros((n,n))
        for i in range(len(self.weightList)):
            a,b,w = self.weightList[i]
            array[a][b] = w
        return dijkstra(array, directed=False)
    
    def computeSigma(self):
        n = len(self.triangles)
        sig = np.sum(self.distMat,axis=(0,1))
        return sig/(n*n)
    
    def computeAffMat(self):
        n = len(self.triangles)
        out = np.zeros((n,n))
        out = np.exp( -self.distMat/( 50*2*self.sigma**2 ))
        return out


def spectralTransform(k): #Compute transformed points, see associated paper
    
    dg = DualGraph(faces) 
    W = dg.affinityMat
    n,m = W.shape
    
    print("Data transformation...")

    DL= []
    for i in range(n):
        DL.append(np.sum(W[i]))
    DL = np.array(DL)
        
    N = np.zeros((n,m))
    N = W/np.outer(DL,DL)
    u,d,vh = np.linalg.svd(N)
    
    V = vh[:k]
    V = np.transpose(np.array(V))
    
    Vp = V.copy()
    for i in range(n):
        Vp[i] = Vp[i]/np.linalg.norm(Vp[i])
        
    return Vp, W, dg.distMat

inputs = ["hand","glasses","octopus","bird","bearing"]
inputsIds = [200, 51, 124, 257, 353]

for i in range(len(inputs)):
    name = inputs[i]
    inId = str(inputsIds[i])
    print("Computing transformed points for mesh",name,"...")
    ref= time()
    readfo = open("off/"+name+".off", "r")
    verts,faces = read_off(readfo)
    print("Model",name,"has",len(faces),"triangles.")
    
    filePoints = open("points/"+name+"_points.dat", mode='wb')
    fileTransfo = open("points/"+name+"_transformed.dat", mode='wb')
    
    readseg = open("seg/"+inId+"/"+inId+"_1.seg", "r")
    groundT = read_seg(readseg)
    k = max(groundT)+1
    
    points = []
    for tri in faces:
        mid = (np.array(verts[tri[0]]) + np.array(verts[tri[1]]) + np.array(verts[tri[2]]))/3
        points.append( mid )
    transformedP, affMat, distMat = spectralTransform(k)
    points = np.array(points)
    points.tofile(filePoints)
    transformedP.tofile(fileTransfo)
    
    readfo.close()
    filePoints.close()
    fileTransfo.close()
    readseg.close()
    
    distMat = distMat.astype(int)
    n,m = distMat.shape
    unique,counts = np.unique(distMat,return_counts=True)
    counts[1:] = gaussian_filter1d(counts[1:], sigma=2)
    plt.plot(unique[1:],counts[1:]/(n*m))
    plt.xlabel('Value in Distance Matrix')
    plt.ylabel('Frequency')
    plt.title("Value frequency in Distance matrix, "+name+" model ")
    plt.savefig("graph/"+name+"DistMat.jpg".format(i))
    plt.show()
    
    affMat = affMat*40000
    affMat = affMat.astype(int)
    n,m = affMat.shape
    unique,counts = np.unique(affMat,return_counts=True)
    counts[:-1] = gaussian_filter1d(counts[:-1], sigma=2)    
    plt.plot(unique[:-1]/40000,counts[:-1]/(n*m)) #Frequency of 0 is not interesting
    plt.xlabel('Value in Affinity Matrix')
    plt.ylabel('Frequency')
    plt.title("Value frequency in Affinity matrix, "+name+" model ")
    plt.savefig("graph/"+name+"AffMat.jpg".format(i))
    plt.show()
    
    assocMat = transformedP @ np.transpose(transformedP)
    assocMat = abs(assocMat)
    assocMat = assocMat*1000
    assocMat = assocMat.astype(int)
    n,m = assocMat.shape
    unique,counts = np.unique(assocMat,return_counts=True)
    counts[:-1] = gaussian_filter1d(counts[:-1], sigma=2)
    plt.plot(unique[:-1]/1000,counts[:-1]/(n*m))
    plt.xlabel('Absolute value in Association Matrix')
    plt.ylabel('Frequency')
    plt.title("Value frequency in Association matrix, "+name+" model ")
    plt.savefig("graph/"+name+"AssocMat.jpg".format(i))
    plt.show()
    
    print("Transform took",time()-ref,"seconds.")






























#