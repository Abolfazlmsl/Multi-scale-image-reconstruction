import numpy as np
from PIL import Image
import glob
import random
from CCF import CCFCalculator

def getImages(path, load_path, image_format, cubes, phases, firstPoint,
              connectedRegions, neighbors):
    
    try:
        loaded_startpoint = np.load(load_path+"loaded_startpoint.npy", allow_pickle=True)
        loaded_startpoint = np.ndarray.tolist(loaded_startpoint)
        loaded_points = np.load(load_path+"loaded_points.npy", allow_pickle=True)
        loaded_points = np.ndarray.tolist(loaded_points)
        loaded_temImage = np.load(load_path+"loaded_temImage.npy", allow_pickle=True)
        loaded_temImage = np.ndarray.tolist(loaded_temImage)
        loaded_overlap = np.load(load_path+"loaded_overlap.npy", allow_pickle=True)
        loaded_overlap = np.ndarray.tolist(loaded_overlap)
        loaded_olvalues = np.load(load_path+"loaded_olvalues.npy", allow_pickle=True)
        loaded_olvalues = np.ndarray.tolist(loaded_olvalues)
    except:
        loaded_startpoint = []
        loaded_points = []
        loaded_temImage = []
        loaded_overlap = []
        loaded_olvalues = []
        
    loadRegionNums = len(loaded_startpoint)
    
    # reading imag pathes
    imgs_path = glob.glob(path+image_format)
    imgs_path.sort()
    
    I = Image.open(imgs_path[0])
    I = I.convert("L")
    width, height = np.shape(I)
    depth = len(imgs_path)
    
    # loading images
    segmented_SEM = np.zeros(shape=[width,height,depth])
    count = 0
    for i in imgs_path:
        I = Image.open(i)
        I = I.convert("L")
        segmented_SEM[:,:,count] = I
        count += 1
        
    images = []
    [images.append([]) for i in range(len(cubes))]
    
    overlap = False
    for ff in range(len(connectedRegions)):
        print("Connected regions ", str(ff), ": ", str(len(connectedRegions[ff])))
    print(len(cubes))
    for i in range(0,len(connectedRegions)):
        connectedRegion = connectedRegions[i]
        #======================================================
        # Default values for the first unresolved template
        counter = 0
        ccfwidthFirstPoint, ccfwidthSecondPoint,ccfheightFirstPoint, ccfheightSecondPoint, ccfdepthFirstPoint, ccfdepthSecondPoint = [0,0,0,0,0,0]
        OverlapWidth, OverlapHeight, OverlapDepth = [0,0,0]
        temImage = []
        #======================================================
        regionNums = 0
        if (i<loadRegionNums):
            regionNums = len(loaded_startpoint[i])
        print(str(i)," Region: ", str(len(connectedRegion)))
        for region in connectedRegion:
            if counter >= regionNums:
                if counter == 0:
                    startPoint = [random.randint(0, width-cubes[region[0]-1][0]-1),random.randint(0, height-cubes[region[0]-1][1]-1),
                                  random.randint(0, depth-cubes[region[0]-1][2]-1)]
                    overlap = False
                else:
                    OverlapWidth = len(np.intersect1d(np.arange(firstPoint[region[0]-1][0], firstPoint[region[0]-1][0]+cubes[region[0]-1][0]),
                                       np.arange(firstPoint[region[1]-1][0], firstPoint[region[1]-1][0]+cubes[region[1]-1][0])))
                    
                    OverlapHeight = len(np.intersect1d(np.arange(firstPoint[region[0]-1][1], firstPoint[region[0]-1][1]+cubes[region[0]-1][1]),
                                        np.arange(firstPoint[region[1]-1][1], firstPoint[region[1]-1][1]+cubes[region[1]-1][1])))
                    
                    OverlapDepth = len(np.intersect1d(np.arange(firstPoint[region[0]-1][2], firstPoint[region[0]-1][2]+cubes[region[0]-1][2]),
                                        np.arange(firstPoint[region[1]-1][2], firstPoint[region[1]-1][2]+cubes[region[1]-1][2])))
                    
                    
                    overlapList = [OverlapWidth, OverlapHeight, OverlapDepth]
                    direction = np.where(overlapList==np.max(overlapList))
                    
                    cropImage = images[region[1]-1] #Edit
                    # Lx, Ly, Lz = np.shape(cropImage)
                    w, h, d = np.shape(cropImage)
                    #Edit
                    # shiftX, shiftY, shiftZ = np.array(firstPoint[region[1]-1]) - np.array(firstPoint[region[0]-1])
                    Lx, Ly, Lz = cubes[region[0]-1]
                    
                    if direction == 0:
                        #Edit
                        if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                            if (OverlapHeight == 0):
                                if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                            else:
                                if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                        
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True

                        else:
                            if (OverlapHeight == 0):
                                if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                       
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                  
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                           
                                    
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                 
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False
                                    
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                          
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                    
         
                            else:
                                if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                          
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                          
                                        else:
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                        
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                            
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False
                                        
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                           
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                          
                                    
                            
                    elif (direction == 1):
                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                            if (OverlapWidth == 0):
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                     
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                         
                                        else:
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                        
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                          
                                        else:
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                          
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                         
                                        else:
                                            temImage = cropImage[:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                            
                            else:
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                          
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                          
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                     
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                        
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                          
                                        else:
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                          
                           
                        else:
                            if (OverlapWidth == 0):
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                      
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                         
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                          
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                        
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                       
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                       
                            else:
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                         
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                        
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                         
                                else:
                                    if (OverlapDepth == 0):
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz-lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz
                                            ccfdepthSecondPoint = depth-lz-lz
                                            overlap = False 
                                         
                                    else:
                                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                      
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                         
                    
                    else:
                        if (firstPoint[region[0]-1][2] >= firstPoint[region[1]-1][2]):
                            if (OverlapWidth == 0):
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                       
                                        else:
                                            temImage = cropImage[:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False 
                                           
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                          
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                        
                                else:
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                          
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                          
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                      
                            else:
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False 
                                          
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                          
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                         
                                else:
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:OverlapWidth, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = False 
                                         
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                           
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, d-OverlapDepth:d]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = 0
                                            ccfdepthSecondPoint = depth-lz-Lz
                                            overlap = True
                                            
                        else:
                            if (OverlapWidth == 0):
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False 
                                            
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                            
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                          
                                else:
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False 
                                           
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:w, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                         
                                        else:
                                            temImage = cropImage[:w, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx
                                            ccfwidthSecondPoint = width-lx-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                           
                            else:
                                if (firstPoint[region[0]-1][0] >= firstPoint[region[1]-1][0]):
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[w-OverlapWidth:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                           
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False 
                                            
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[w-OverlapWidth:w, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                         
                                        else:
                                            temImage = cropImage[w-OverlapWidth:w, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = 0
                                            ccfwidthSecondPoint = width-lx-Lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                            
                                else:
                                    if (OverlapHeight == 0):
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:OverlapWidth, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False
                                        
                                        else:
                                            temImage = cropImage[:OverlapWidth, :h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly
                                            ccfheightSecondPoint = height-ly-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = False 
                                          
                                    else:
                                        if (firstPoint[region[0]-1][1] >= firstPoint[region[1]-1][1]):
                                            temImage = cropImage[:OverlapWidth, h-OverlapHeight:h, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = 0
                                            ccfheightSecondPoint = height-ly-Ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                            
                                        else:
                                            temImage = cropImage[:OverlapWidth, :OverlapHeight, :OverlapDepth]
                                            lx, ly, lz = np.shape(temImage)
                                            ccfwidthFirstPoint = Lx-lx
                                            ccfwidthSecondPoint = width-lx
                                            ccfheightFirstPoint = Ly-ly
                                            ccfheightSecondPoint = height-ly
                                            ccfdepthFirstPoint = Lz-lz
                                            ccfdepthSecondPoint = depth-lz
                                            overlap = True
                                            
                    
                    if (2*cubes[region[0]-1][0]<ccfwidthSecondPoint-ccfwidthFirstPoint and 2*cubes[region[0]-1][1]<ccfheightSecondPoint-ccfheightFirstPoint \
                        and 2*cubes[region[0]-1][2]<ccfdepthSecondPoint-ccfdepthFirstPoint):
                        startRandomHR = [random.randint(ccfwidthFirstPoint, ccfwidthSecondPoint-2*cubes[region[0]-1][0]),
                                         random.randint(ccfheightFirstPoint, ccfheightSecondPoint-2*cubes[region[0]-1][1]),
                                         random.randint(ccfdepthFirstPoint, ccfdepthSecondPoint-2*cubes[region[0]-1][2])]
                        cropHR = segmented_SEM[startRandomHR[0]:startRandomHR[0]+2*cubes[region[0]-1][0],
                                               startRandomHR[1]:startRandomHR[1]+2*cubes[region[0]-1][1],
                                               startRandomHR[2]:startRandomHR[2]+2*cubes[region[0]-1][2]]

                        startPointHR = CCFCalculator(cropHR, cropImage, temImage, 0, 
                                                     2*cubes[region[0]-1][0]-lx, 0, 2*cubes[region[0]-1][1]-ly,
                                                     0, 2*cubes[region[0]-1][2]-lz).compute_ccf()
                        
                        startPoint = np.array(startRandomHR).T + np.array(startPointHR).T

                        
                    else:
                        startPoint = ccf(segmented_SEM, cropImage, temImage, ccfwidthFirstPoint, 
                                         ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint,
                                         ccfdepthFirstPoint, ccfdepthSecondPoint)
                        startPoint = np.array(startPoint)
                        
                    startPoint -= np.array([ccfwidthFirstPoint, ccfheightFirstPoint, ccfdepthFirstPoint]).T
                    negativeIndices = np.where(startPoint<0)
                    startPoint[negativeIndices] = 0
                    startPoint = list(startPoint)
                        
                if counter == 0:
                    loaded_startpoint.append([startPoint])
                    loaded_points.append([[ccfwidthFirstPoint, ccfwidthSecondPoint,
                                           ccfheightFirstPoint, ccfheightSecondPoint,
                                           ccfdepthFirstPoint, ccfdepthSecondPoint]])
                    loaded_temImage.append([temImage])
                    loaded_overlap.append([overlap])
                    loaded_olvalues.append([[OverlapWidth, OverlapHeight, OverlapDepth]])
                else:
                    loaded_startpoint[i].append(startPoint)
                    loaded_points[i].append([ccfwidthFirstPoint, ccfwidthSecondPoint,
                                             ccfheightFirstPoint, ccfheightSecondPoint,
                                             ccfdepthFirstPoint, ccfdepthSecondPoint])
                    loaded_temImage[i].append(temImage)
                    loaded_overlap[i].append(overlap)
                    loaded_olvalues[i].append([OverlapWidth, OverlapHeight, OverlapDepth])
                    
                np.save(load_path + 'loaded_startpoint.npy', loaded_startpoint) 
                np.save(load_path + 'loaded_points.npy', loaded_points) 
                np.save(load_path + 'loaded_temImage.npy', loaded_temImage)
                np.save(load_path + 'loaded_overlap.npy', loaded_overlap)
                np.save(load_path + 'loaded_olvalues.npy', loaded_olvalues)
                
            else:
                startPoint = loaded_startpoint[i][counter]
                ccfwidthFirstPoint, ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint, ccfdepthFirstPoint, ccfdepthSecondPoint = loaded_points[i][counter]
                temImage = loaded_temImage[i][counter]
                overlap = loaded_overlap[i][counter]
                OverlapWidth, OverlapHeight, OverlapDepth = loaded_olvalues[i][counter]
                
            extractedImage = segmented_SEM[startPoint[0]:startPoint[0]+cubes[region[0]-1][0],
                                           startPoint[1]:startPoint[1]+cubes[region[0]-1][1],
                                           startPoint[2]:startPoint[2]+cubes[region[0]-1][2]]
            boolCropImage = np.full(np.shape(extractedImage), fill_value=False)

            boolCropImage[phases[region[0]-1]] = True
            # print("-----------------------------------------")
            
            extractedImage[np.where(~boolCropImage)] = 0
            if (overlap):
                extractedImage[ccfwidthFirstPoint:ccfwidthFirstPoint+OverlapWidth,
                               ccfheightFirstPoint:ccfheightFirstPoint+OverlapHeight,
                               ccfdepthFirstPoint:ccfdepthFirstPoint+OverlapDepth] = temImage
            images[region[0]-1] = np.array(extractedImage, dtype=np.uint8)
            counter += 1
            print(counter)
        
    return images