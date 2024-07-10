import numpy as np
from CCF import CCFCalculator

def bestMatchTemFirstPixel(segmented_SEM, firstPoint, region, cubes, images, width, height, depth):
        OverlapWidth = len(np.intersect1d(np.arange(firstPoint[region[0]-1][0], firstPoint[region[0]-1][0]+cubes[region[0]-1][0]),
                           np.arange(firstPoint[region[1]-1][0], firstPoint[region[1]-1][0]+cubes[region[1]-1][0])))
        
        OverlapHeight = len(np.intersect1d(np.arange(firstPoint[region[0]-1][1], firstPoint[region[0]-1][1]+cubes[region[0]-1][1]),
                            np.arange(firstPoint[region[1]-1][1], firstPoint[region[1]-1][1]+cubes[region[1]-1][1])))
        
        OverlapDepth = len(np.intersect1d(np.arange(firstPoint[region[0]-1][2], firstPoint[region[0]-1][2]+cubes[region[0]-1][2]),
                            np.arange(firstPoint[region[1]-1][2], firstPoint[region[1]-1][2]+cubes[region[1]-1][2])))
        
        
        overlapList = [OverlapWidth, OverlapHeight, OverlapDepth]
        direction = np.where(overlapList==np.max(overlapList))
        
        cropImage = images[region[1]-1] #Edit
        w, h, d = np.shape(cropImage)
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
                                
            startPoint = CCFCalculator(segmented_SEM, cropImage, temImage, ccfwidthFirstPoint, 
                             ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint,
                             ccfdepthFirstPoint, ccfdepthSecondPoint).compute_ccf()
            startPoint = np.array(startPoint)
            
        startPoint -= np.array([ccfwidthFirstPoint, ccfheightFirstPoint, ccfdepthFirstPoint]).T
        negativeIndices = np.where(startPoint<0)
        startPoint[negativeIndices] = 0
        startPoint = list(startPoint)
        
        return startPoint, overlap
      