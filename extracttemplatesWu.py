import numpy as np
from PIL import Image
import glob
from CCF import CCFCalculator

def getImages(lowResImage, path, load_path, image_format, cubes, phases, firstPoint,
              connectedRegions, neighbors):
    
    try:
        loaded_startpoint = np.load(load_path+"loaded_startpoint_Wu.npy", allow_pickle=True)
        loaded_startpoint = np.ndarray.tolist(loaded_startpoint)
        loaded_points = np.load(load_path+"loaded_points_Wu.npy", allow_pickle=True)
        loaded_points = np.ndarray.tolist(loaded_points)
        loaded_temImage = np.load(load_path+"loaded_temImage_Wu.npy", allow_pickle=True)
        loaded_temImage = np.ndarray.tolist(loaded_temImage)
    except:
        loaded_startpoint = []
        loaded_points = []
        loaded_temImage = []
        
    loadRegionNums = len(loaded_startpoint)
    
    gray_values = np.sort(np.unique(lowResImage))
    UNRESOLVED = gray_values[1]
    GRAIN = gray_values[2]

    unresolvedPores = np.where(lowResImage==UNRESOLVED)
    lowResImage[unresolvedPores] = GRAIN
    
    ###################################################################
    # Compute unresolved regions and clusters properties
    clusterCubes = []
    clusterPhases = []
    firstPoints = []
    for i in range(len(connectedRegions)):
        uniqueRegions = np.unique(connectedRegions[i])
        regionPhase = [[],[],[]]
        [[[regionPhase[kk].append(index+firstPoint[region-1][kk]) for index in phases[region-1][kk]] for kk in range(len(regionPhase))] for region in uniqueRegions]

        firstPixel = [np.min(regionPhase[0]), np.min(regionPhase[1]), np.min(regionPhase[2])]
        firstPoints.append(firstPixel)
        cubeWidth = np.max(regionPhase[0]) - np.min(regionPhase[0]) + 1
        cubeHeight = np.max(regionPhase[1]) - np.min(regionPhase[1]) + 1
        cubeDepth = np.max(regionPhase[2]) - np.min(regionPhase[2]) + 1
        clusterCubes.append([cubeWidth, cubeHeight, cubeDepth])
        for k in range(len(regionPhase)):
            regionPhase[k][:] -= firstPixel[k]
        clusterPhases.append(regionPhase)
    
    ###################################################################

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
    [images.append([]) for i in range(len(clusterCubes))]

    print("Connected regions: ", str(len(connectedRegions)))
    for i in range(0,len(connectedRegions)):
        #======================================================
        # Default values for the first unresolved template
        ccfwidthFirstPoint, ccfwidthSecondPoint,ccfheightFirstPoint, ccfheightSecondPoint, ccfdepthFirstPoint, ccfdepthSecondPoint = [0,0,0,0,0,0]
        temImage = []
        #======================================================
        if (i+1>loadRegionNums):    
            cropImage = lowResImage[firstPoints[i][0]:firstPoints[i][0]+clusterCubes[i][0],
                                    firstPoints[i][1]:firstPoints[i][1]+clusterCubes[i][1],
                                    firstPoints[i][2]:firstPoints[i][2]+clusterCubes[i][2]]
            
            w, h, d = np.shape(cropImage)
            Lx, Ly, Lz = clusterCubes[i]
            
            if (firstPoints[i][0]-int(w/5) >= 0):
                temImage = lowResImage[firstPoints[i][0]-int(w/5):firstPoints[i][0], firstPoints[i][1]:firstPoints[i][1]+int(h/5), firstPoints[i][2]:firstPoints[i][2]+int(d/5)]
            else:
                temImage = lowResImage[firstPoints[i][0]:firstPoints[i][0]+int(w/5), firstPoints[i][1]:firstPoints[i][1]+int(h/5), firstPoints[i][2]:firstPoints[i][2]+int(d/5)]
            lx, ly, lz = np.shape(temImage)
            ccfwidthFirstPoint = 0
            ccfwidthSecondPoint = width-lx-Lx
            ccfheightFirstPoint = 0
            ccfheightSecondPoint = height-ly-Ly
            ccfdepthFirstPoint = 0
            ccfdepthSecondPoint = depth-lz-Lz
                   
            secondPoints = np.array([ccfwidthSecondPoint, ccfheightSecondPoint, ccfdepthSecondPoint])
            if (np.count_nonzero(secondPoints.T < 0) > 0):
                startPoint = [0,0,0]
            else:
                startPoint = CCFCalculator(segmented_SEM, cropImage, temImage, ccfwidthFirstPoint, 
                                  ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint,
                                  ccfdepthFirstPoint, ccfdepthSecondPoint).compute_ccf()
        
            loaded_startpoint.append(startPoint)
            loaded_points.append([ccfwidthFirstPoint, ccfwidthSecondPoint,
                                  ccfheightFirstPoint, ccfheightSecondPoint,
                                  ccfdepthFirstPoint, ccfdepthSecondPoint])
            loaded_temImage.append(temImage)

            np.save(load_path + 'loaded_startpoint_Wu.npy', loaded_startpoint) 
            np.save(load_path + 'loaded_points_Wu.npy', loaded_points) 
            np.save(load_path + 'loaded_temImage_Wu.npy', loaded_temImage)
            
        else:
            startPoint = loaded_startpoint[i]
            ccfwidthFirstPoint, ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint, ccfdepthFirstPoint, ccfdepthSecondPoint = loaded_points[i]
            temImage = loaded_temImage[i]
            
        #Edit
        extractedImage = segmented_SEM[startPoint[0]:startPoint[0]+clusterCubes[i][0],
                                        startPoint[1]:startPoint[1]+clusterCubes[i][1],
                                        startPoint[2]:startPoint[2]+clusterCubes[i][2]]
        boolCropImage = np.full(np.shape(extractedImage), fill_value=False)
        boolCropImage[clusterPhases[i]] = True
        # print("-----------------------------------------")
        
        extractedImage[np.where(~boolCropImage)] = 0
        images[i] = np.array(extractedImage, dtype=np.uint8)
        print(i+1)

    return clusterCubes, firstPoints, clusterPhases