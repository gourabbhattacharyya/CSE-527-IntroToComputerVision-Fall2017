#!/usr/bin/env python
'''
    ===============================================================================
    Interactive Image Segmentation using Maxflow GraphCut algorithm.
    
    USAGE:
    python main_bonus.py astronaut.png 
    
    README FIRST:
    Initally the input window will open. Once you draw both foreground and background lines, another output window pops up with result.
    Remember, if you draw either background or forground only, you will not get output window display.
    
    Use the below keys to draw forground and background lines. Press '0' and start dragging or clicking for background line(blue)
    and in a similar way, press '1' and start dragging or clicking for foreground line(red) in the areas you want.
    
    Then again press 'n' for updating the output. You can keep drawing additional foreground or background lines and press 'n' to see real-time updates.
    Anytime you plan to reset, press 'r' button. The output window will vanish and you can begin from start with a fresh frame of astronaut.png.
    To exit(anytime), press 'esc' key.
    
    Key '0' - To draw background lines(BLUE)
    Key '1' - To draw foreground lines(RED)
    
    Key 'n' - To update the segmentation
    Key 'r' - To reset the setup
    Key 'esc' - To close the program
    ===============================================================================
    '''

# Python 2/3 compatibility
from __future__ import print_function
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay
from skimage import img_as_ubyte
np.set_printoptions(threshold=np.inf)
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=502, compactness=18.5)
    segments_ids = np.unique(segments)
#    # show the output of SLIC
#    fig = plt.figure("Superpixels -- 500 segments")
#    ax = fig.add_subplot(1, 1, 1)
#    ax.imshow(mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments))
#    plt.axis("off")
#    plt.show()

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])
    
    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [24, 24] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])
    
    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)
    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:]==1])
    bg_segments = np.unique(superpixels[marking[:,:]==0])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)
    
    hist_comp_alg = cv2.HISTCMP_KL_DIV
    
    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                       20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))


    g.maxflow()
    return g.get_grid_segments(nodes)




BLUE = [255,0,0]        # BG
RED = [0,0,255]         # FG

DRAW_BG = {'color' : BLUE, 'val' : 0}
DRAW_FG = {'color' : RED, 'val' : 1}

# setting up flags
drawing = False         # flag for drawing curves
value = 0               # drawing val initialized
thickness = 3           # brush thickness
fgbg=False
out = False


def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
    # draw touchup curves
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img,(x,y),thickness,value['color'],-1)
        cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)
        
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)


if __name__ == '__main__':
    # print documentation
    print(" Instructions: \n")
    print(__doc__)
    # Loading images
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img2 = img.copy()                               # a copy of original image
    output = np.zeros(img.shape,np.uint8)           # output image to be shown
    mask = np.zeros(img.shape[:2],dtype = np.uint8)+255 # mask initialized
    #pre-calculation for graphcut
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img2)
    norm_hists = normalize_histograms(color_hists)
    
    # input and output windows
    
    cv2.namedWindow('input')
    cv2.setMouseCallback('input',onmouse)
    cv2.moveWindow('input',img.shape[1]+10,90)
    video_in = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('bonus.avi',video_in, 30.0, img.shape[:2])
    
    while(1):
        bar = np.zeros((img.shape[0],5,3),np.uint8)
        if fgbg:
            cv2.namedWindow('output')
            cv2.imshow('output',output)
        
        cv2.imshow('input',img)
        k = cv2.waitKey(1)
        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print(" mark background regions in blue \n")
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print(" mark foreground regions in red \n")
            value = DRAW_FG
        elif k == ord('r'): # reset everything
                print("resetting \n")
                drawing = False
                value = 0
                img = img2.copy()
                mask = np.zeros(img.shape[:2],dtype = np.uint8)+255 # mask initialized
                output = np.zeros(img.shape,np.uint8)
                if fgbg:
                    cv2.destroyWindow('output')
                fgbg=False
        
                
        elif k == ord('n'):
            # segment the image
            print("updating segment \n")
            fg_segments, bg_segments = find_superpixels_under_marking(mask, superpixels)
            h_fg = np.sum(color_hists[fg_segments],axis=0)
            h_bg = np.sum(color_hists[bg_segments],axis=0)
            if h_fg.sum()!=0.0 and h_bg.sum()!=0.0:
                fgbg = True
            fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
            bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

            fgbg_hists = {}
            fgbg_hists[0] = fg_cumulative_hist
            fgbg_hists[1] = bg_cumulative_hist
            
            fgbg_superpixels = {}
            fgbg_superpixels[0] = fg_segments
            fgbg_superpixels[1] = bg_segments
            
            graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
            output = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
            output = np.uint8(output * 255)


            #output_name = sys.argv[2] + "mask_bonus.png"
            #cv2.imwrite(output_name, output);

    cv2.destroyAllWindows()
