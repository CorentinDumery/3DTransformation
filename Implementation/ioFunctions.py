# -*- coding: utf-8 -*-

from stl import mesh
import numpy as np
import os, shutil

def read_off(file): #Read off file into verts, faces format
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [np.array([float(s) for s in file.readline().strip().split(' ')]) for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def read_seg(file): #Read segmentation file
    labels = []
    s = file.readline()
    while len(s)>0 and ord(s[0]) >= 48 and ord(s[0])<= 57:
        labels.append(int(s))
        s = file.readline()
    return labels
    
def clean_out(): #empty out folder
    folder = 'out'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def sendOut(outData,name="out"): #Output STL mesh
    data = np.zeros(len(outData), dtype=mesh.Mesh.dtype)
    out = mesh.Mesh(data, remove_empty_areas=False)
    out.vectors[:len(outData)] = outData
    out.save("out/"+name+".stl")
    
def outClusters(outvec,name,dx=0,dy=0,dz=0,clean=True): #Output STL meshes from clusters
    readfo = open("off/"+name+".off", "r")
    verts,faces = read_off(readfo)
    if clean:
        clean_out()
    stlList = []
    for i in range(max(outvec)+1):
        stlList.append([])
    for i in range(len(faces)):
        stlList[outvec[i]].append([verts[faces[i][0]],verts[faces[i][1]],verts[faces[i][2]]])
    
    for s in range(len(stlList)):
        if len(stlList[s])>0:
            sendOut(offsetStl(stlList[s],dx,dy,dz),str(s)+name)

def offsetStl(stl,x,y,z): #offset STL points by constant value
    res = []
    for t in stl:
        newt = []
        for l in t:
            newt.append([l[0]+x,l[1]+y,l[2]+z])
        res.append(newt)
    return res