import numpy as np
from PIL import Image
import glob
import random
from bestMatchTemplate import bestMatchTemFirstPixel

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
                    # Select random template for the first unresolved template
                    startPoint = [random.randint(0, width-cubes[region[0]-1][0]-1),random.randint(0, height-cubes[region[0]-1][1]-1),
                                  random.randint(0, depth-cubes[region[0]-1][2]-1)]
                    overlap = False
                else:
                    startPoint, overlap = bestMatchTemFirstPixel(segmented_SEM, firstPoint, region, cubes, images, width, height, depth)
                
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