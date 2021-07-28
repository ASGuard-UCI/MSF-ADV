from __future__ import division
import numpy as np

def processPC(PC,min_h,max_h,min_v,max_v):

    PC_w = PC
    PC_w = PC_w[PC_w[:,5] > min_v]
    PC_w = PC_w[PC_w[:,5] < max_v]
    PC_w = PC_w[PC_w[:,6] > min_h]
    PC_w = PC_w[PC_w[:,6] < max_h]

    aset = set([tuple(x) for x in PC_w])
    bset = set([tuple(x) for x in PC])

    PC_wo = np.array([x for x in bset - aset])

    return PC_w, PC_wo

def injectCube(PC,leng,x,y,z):


    PC = appendSpherical_np(PC)

    X_og = np.linspace(x,x+leng,leng*400)
    Y_og = np.linspace(y,y+leng,leng*400)
    Z_og = np.linspace(z,z+leng,leng*400)

    X, Y = np.meshgrid(X_og, Y_og)
    cube_1 = np.array([X.flatten(), Y.flatten(), np.ones_like(X.flatten())*np.min(Z_og), np.ones_like(X.flatten())*144./255.]).transpose()
    cube_2 = np.array([X.flatten(), Y.flatten(), np.ones_like(X.flatten())*np.max(Z_og), np.ones_like(X.flatten())*144./255.]).transpose()

    X, Z = np.meshgrid(X_og, Z_og)
    cube_3 = np.array([X.flatten(), np.ones_like(X.flatten())*np.min(Y_og), Z.flatten(), np.ones_like(X.flatten())*144./255.]).transpose()
    cube_4 = np.array([X.flatten(), np.ones_like(X.flatten())*np.max(Y_og), Z.flatten(), np.ones_like(X.flatten())*144./255.]).transpose()

    Y, Z = np.meshgrid(Y_og, Z_og)
    cube_5 = np.array([np.ones_like(Y.flatten())*np.min(X_og), Y.flatten(), Z.flatten(), np.ones_like(Y.flatten())*144./255.]).transpose()
    cube_6 = np.array([np.ones_like(Y.flatten())*np.max(X_og), Y.flatten(), Z.flatten(), np.ones_like(Y.flatten())*144./255.]).transpose()

    cube = np.concatenate((cube_1, cube_2, cube_3, cube_4, cube_5, cube_6), axis=0)

    cube = appendSpherical_np(cube)

    h_granularity = 0.2 / 180 * np.pi
    v_granularity = 26.8 / 63 / 180 * np.pi


    min_h = np.min(cube[:,6])
    max_h = np.max(cube[:,6])
    min_v = np.min(cube[:,5])
    max_v = np.max(cube[:,5])
    

    PC_w,PC_wo = processPC(PC,min_h,max_h,min_v,max_v)

    totalsub = np.concatenate((PC_w,cube), axis=0)

    h_grids = round((max_h - min_h) / h_granularity)
    v_grids = round((max_v - min_v) / v_granularity)

    cube_project = []

    for h in range(int(h_grids)):
        for v in range(int(v_grids)):
            subset = totalsub
            subset = subset[subset[:,5] > min_v + v * v_granularity]
            subset = subset[subset[:,5] < min_v + (v + 1) * v_granularity]
            subset = subset[subset[:,6] > min_h + h * h_granularity]
            subset = subset[subset[:,6] < min_h + (h + 1) * h_granularity]
            if subset.size != 0:
                cube_project.append(subset[np.argmin(subset[:,4])])


    cube_project = np.array(cube_project)
    total = np.concatenate((PC_wo,cube_project), axis=0)

    return total

def injectCylinder(PC,h,r,x,y,z):


    PC = appendSpherical_np(PC)

    C_og = np.linspace(0,2*np.pi,400)
    Z_og = np.linspace(z,z+h,400)
    R_og = np.linspace(0,r,400)

    C, Z = np.meshgrid(C_og, Z_og)
    surface_1 = np.array([np.ones_like(C.flatten())*r, C.flatten(), Z.flatten(), np.ones_like(C.flatten())*144./255.]).transpose()
    
    C, R = np.meshgrid(C_og, R_og)
    surface_2 = np.array([R.flatten(), C.flatten(), np.ones_like(R.flatten())*np.min(Z_og), np.ones_like(C.flatten())*144./255.]).transpose()
    surface_3 = np.array([R.flatten(), C.flatten(), np.ones_like(R.flatten())*np.max(Z_og), np.ones_like(C.flatten())*144./255.]).transpose()

    
    cylinder = np.concatenate((surface_1, surface_2, surface_3), axis=0)
    cylinder = replaceCoord_np(cylinder)
    cylinder[:,0] += x
    cylinder[:,1] += y

    cylinder = appendSpherical_np(cylinder)

    h_granularity = 0.2 / 180 * np.pi
    v_granularity = 26.8 / 63 / 180 * np.pi


    min_h = np.min(cylinder[:,6])
    max_h = np.max(cylinder[:,6])
    min_v = np.min(cylinder[:,5])
    max_v = np.max(cylinder[:,5])
    

    PC_w,PC_wo = processPC(PC,min_h,max_h,min_v,max_v)

    totalsub = np.concatenate((PC_w,cylinder), axis=0)

    h_grids = round((max_h - min_h) / h_granularity)
    v_grids = round((max_v - min_v) / v_granularity)

    cylinder_project = []

    for h in range(int(h_grids)):
        for v in range(int(v_grids)):
            subset = totalsub
            subset = subset[subset[:,5] > min_v + v * v_granularity]
            subset = subset[subset[:,5] < min_v + (v + 1) * v_granularity]
            subset = subset[subset[:,6] > min_h + h * h_granularity]
            subset = subset[subset[:,6] < min_h + (h + 1) * h_granularity]
            if subset.size != 0:
                cylinder_project.append(subset[np.argmin(subset[:,4])])


    cylinder_project = np.array(cylinder_project)
    total = np.concatenate((PC_wo,cylinder_project), axis=0)

    return total


def injectPyramid(PC,h,r1,r2,x,y,z):


    PC = appendSpherical_np(PC)

    C_og = np.linspace(0,2*np.pi,400)
    Z_og = np.linspace(z,z+h,400)
    R1_og = np.linspace(0,r1,400)
    R2_og = np.linspace(0,r2,400)
    R3_og = np.linspace(r1,r2,400)

    C, Z = np.meshgrid(C_og, Z_og)
    _,R3 = np.meshgrid(C_og, R3_og) 
    surface_1 = np.array([R3.flatten(), C.flatten(), Z.flatten(), np.ones_like(C.flatten())*144./255.]).transpose()
    
    C, R1 = np.meshgrid(C_og, R1_og)
    surface_2 = np.array([R1.flatten(), C.flatten(), np.ones_like(R1.flatten())*np.min(Z_og), np.ones_like(C.flatten())*144./255.]).transpose()
    
    C, R2 = np.meshgrid(C_og, R2_og)
    surface_3 = np.array([R2.flatten(), C.flatten(), np.ones_like(R2.flatten())*np.max(Z_og), np.ones_like(C.flatten())*144./255.]).transpose()

    
    pyramid = np.concatenate((surface_1, surface_2, surface_3), axis=0)
    pyramid = replaceCoord_np(pyramid)
    pyramid[:,0] += x
    pyramid[:,1] += y

    pyramid = appendSpherical_np(pyramid)

    h_granularity = 0.2 / 180 * np.pi
    v_granularity = 26.8 / 63 / 180 * np.pi


    min_h = np.min(pyramid[:,6])
    max_h = np.max(pyramid[:,6])
    min_v = np.min(pyramid[:,5])
    max_v = np.max(pyramid[:,5])
    

    PC_w,PC_wo = processPC(PC,min_h,max_h,min_v,max_v)

    totalsub = np.concatenate((PC_w,pyramid), axis=0)

    h_grids = round((max_h - min_h) / h_granularity)
    v_grids = round((max_v - min_v) / v_granularity)

    pyramid_project = []

    for h in range(int(h_grids)):
        for v in range(int(v_grids)):
            subset = totalsub
            subset = subset[subset[:,5] > min_v + v * v_granularity]
            subset = subset[subset[:,5] < min_v + (v + 1) * v_granularity]
            subset = subset[subset[:,6] > min_h + h * h_granularity]
            subset = subset[subset[:,6] < min_h + (h + 1) * h_granularity]
            if subset.size != 0:
                pyramid_project.append(subset[np.argmin(subset[:,4])])


    pyramid_project = np.array(pyramid_project)
    total = np.concatenate((PC_wo,pyramid_project), axis=0)

    return total

def appendSpherical_np(xyzi):

    xyz = xyzi[:,:-1]
    ptsnew = np.hstack((xyzi, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,4] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,5] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,6] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def replaceCoord_np(rpzi):

    ptsnew = np.zeros(rpzi.shape)
    
    ptsnew[:,0] = rpzi[:,0] * np.cos(rpzi[:,1])
    ptsnew[:,1] = rpzi[:,0] * np.sin(rpzi[:,1])
    ptsnew[:,2] = rpzi[:,2]
    ptsnew[:,3] = rpzi[:,3]

    return ptsnew

