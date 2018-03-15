
import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys


# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================



def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    global camera_points
    global projector_points
    global rgb
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    pattern1 = cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR)
    
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
    scan_bits = np.zeros((h,w), dtype=np.uint16)
    

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        #patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
        
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        # TODO: populate scan_bits by putting the bit_code according to on_mask
        scan_bits+= on_mask*bit_code

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    
    img  = np.zeros((h,w,3), dtype=np.uint8)
    camera_points = []
    projector_points = []
    rgb = []

    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
            
            (p_x, p_y) = binary_codes_ids_codebook[scan_bits[y,x]]
            
            if p_x >= 1279 or p_y >= 797: # filter
                continue
                    
            projector_points.append((p_x, p_y))
            camera_points.append((x/2.0, y/2.0))
            rgb.append((pattern1[y,x,2],pattern1[y,x,1],pattern1[y,x,0]))
            img[y,x,0]=255.0*p_x/1280.0
            img[y,x,1]=255.0*p_y/800.0
            img[y,x,2]=0


#    # now that we have 2D-2D correspondances, we can triangulate 3D points!
#    # load the prepared stereo calibration between projector and camera


    output_name = sys.argv[1] + "correspondence.jpg"
    # save the correspondence.jpg image
    cv2.imwrite(output_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #    plt.imshow(img)
    #    plt.show()
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
        camera_points = np.array(camera_points, dtype=np.float32)
        
        num_pts = camera_points.size / 2
        camera_points.shape = (num_pts, 1, 2)

        projector_points = np.array(projector_points, dtype=np.float32)
        num_pts = projector_points.size / 2
        projector_points.shape = (num_pts, 1, 2)

        camera_normalized_pts = cv2.undistortPoints(camera_points, camera_K, camera_d)
        projector_normalized_pts = cv2.undistortPoints(projector_points, projector_K, projector_d)
        
        P1 = np.eye(3,4)
        P2 = np.zeros((3,4))
        P2[:,:3] = projector_R
        P2[:,3:] = projector_t

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
        pts = cv2.triangulatePoints(P1, P2, camera_normalized_pts, projector_normalized_pts)
        newrgb = []
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
        pts = pts.T
        points_3d = cv2.convertPointsFromHomogeneous(pts)
        indices = np.argwhere((points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400))
        for j in indices:
            newrgb.append(rgb[j[0]])
        rgb = newrgb
        points_3d =  points_3d[(points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)]
        points_3d = np.expand_dims(points_3d, axis=1)
        
#    # TODO: name the resulted 3D points as "points_3d"

	return points_3d

def write_3d_points_color(points_3d, rgb):
    
    # ===== to render color to 3D point cloud: Bonus =====
    
    print("write output point color cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name,"w") as f:
        i = 0
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n"%(p[0,0],p[0,1],p[0,2],rgb[i][0],rgb[i][1],rgb[i][2]))
            i=i+1

def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
	
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))
    return points_3d, camera_points, projector_points

if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====
    
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_3d_points_color(points_3d, rgb)
	
