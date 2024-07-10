import numpy as np
from CCF import CCFCalculator

def bestMatchTemFirstPixel(firstPoint, region, width, height, depth, Lx, Ly, Lz, OverlapWidth, OverlapHeight, OverlapDepth, segmented_SEM, cropImage):
    if firstPoint[region[0] - 1][2] >= firstPoint[region[1] - 1][2]:
        if OverlapWidth == 0:
            if firstPoint[region[0] - 1][0] >= firstPoint[region[1] - 1][0]:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, :height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:width, :height, depth - OverlapDepth:depth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, height - OverlapHeight:height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:width, :OverlapHeight, depth - OverlapDepth:depth]
            else:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, :height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:width, :height, depth - OverlapDepth:depth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, height - OverlapHeight:height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:width, :OverlapHeight, depth - OverlapDepth:depth]
        else:
            if firstPoint[region[0] - 1][0] >= firstPoint[region[1] - 1][0]:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[width - OverlapWidth:width, :height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[width - OverlapWidth:width, :height, depth - OverlapDepth:depth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[width - OverlapWidth:width, height - OverlapHeight:height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[width - OverlapWidth:width, :OverlapHeight, depth - OverlapDepth:depth]
            else:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:OverlapWidth, :height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:OverlapWidth, :height, depth - OverlapDepth:depth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:OverlapWidth, height - OverlapHeight:height, depth - OverlapDepth:depth]
                    else:
                        temImage = cropImage[:OverlapWidth, :OverlapHeight, depth - OverlapDepth:depth]
    else:
        if OverlapWidth == 0:
            if firstPoint[region[0] - 1][0] >= firstPoint[region[1] - 1][0]:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, :height, :OverlapDepth]
                    else:
                        temImage = cropImage[:width, :height, :OverlapDepth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, height - OverlapHeight:height, :OverlapDepth]
                    else:
                        temImage = cropImage[:width, :OverlapHeight, :OverlapDepth]
            else:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, :height, :OverlapDepth]
                    else:
                        temImage = cropImage[:width, :height, :OverlapDepth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:width, height - OverlapHeight:height, :OverlapDepth]
                    else:
                        temImage = cropImage[:width, :OverlapHeight, :OverlapDepth]
        else:
            if firstPoint[region[0] - 1][0] >= firstPoint[region[1] - 1][0]:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[width - OverlapWidth:width, :height, :OverlapDepth]
                    else:
                        temImage = cropImage[width - OverlapWidth:width, :height, :OverlapDepth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[width - OverlapWidth:width, height - OverlapHeight:height, :OverlapDepth]
                    else:
                        temImage = cropImage[width - OverlapWidth:width, :OverlapHeight, :OverlapDepth]
            else:
                if OverlapHeight == 0:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:OverlapWidth, :height, :OverlapDepth]
                    else:
                        temImage = cropImage[:OverlapWidth, :height, :OverlapDepth]
                else:
                    if firstPoint[region[0] - 1][1] >= firstPoint[region[1] - 1][1]:
                        temImage = cropImage[:OverlapWidth, height - OverlapHeight:height, :OverlapDepth]
                    else:
                        temImage = cropImage[:OverlapWidth, :OverlapHeight, :OverlapDepth]

    lx, ly, lz = np.shape(temImage)
    ccfwidthFirstPoint = 0 if lx == width else Lx
    ccfwidthSecondPoint = width - lx - (2 * Lx if lx != width else 0)
    ccfheightFirstPoint = 0 if ly == height else Ly
    ccfheightSecondPoint = height - ly - (2 * Ly if ly != height else 0)
    ccfdepthFirstPoint = 0 if lz == depth else Lz - lz
    ccfdepthSecondPoint = depth - lz - (Lz if lz != depth else 0)
    overlap = True if lz != depth or ly != height or lx != width else False

    startPoint = CCFCalculator(segmented_SEM, cropImage, temImage, ccfwidthFirstPoint, 
                               ccfwidthSecondPoint, ccfheightFirstPoint, ccfheightSecondPoint,
                               ccfdepthFirstPoint, ccfdepthSecondPoint).compute_ccf()
    startPoint = np.array(startPoint)
    
    startPoint -= np.array([ccfwidthFirstPoint, ccfheightFirstPoint, ccfdepthFirstPoint]).T

    return startPoint, overlap