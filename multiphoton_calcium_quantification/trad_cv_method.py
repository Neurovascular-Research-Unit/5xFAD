import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
import cv2 as cv
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.animation as animation
from skimage import exposure
from tqdm import tqdm
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from ipywidgets.widgets import Dropdown
from matplotlib import patches
import matplotlib as mpl
import skimage

def decomb(vid, le_shift, ml_shift, mr_shift, re_shift):

    """
    Each frame in each video is divided into four vertical sections of equal length, and each of those sections is decombed independently

    Start with all zeros just to see what it looks like unprocessed
    It's not perfect; edges may still be a little interlaced
    Just try a few values to find a good balance

    Parameters:
        le_shift (int): left section 
        ml_shift (int): middle left section
        mr_shift (int): middle right section
        re_shift (int): right section 

    Returns:
        array: Decombed video as a numpy array corresponding to the original multistack tiff file
    """
    
    video = vid.copy()
    q1 = len(video[0,0]) // 4 # left section cutoff
    q2 = len(video[0,0]) // 2 # middle
    q3 = q1*3 # right section cutoff
    video_ml = video[:,:,q1:q2] # middle left section
    video_mr = video[:,:,q2:q3] # middle right section
    video_l = video[:,:,:q1] # left section
    video_r = video[:,:,q3:] # right section

    video_ml = shift_scale(video_ml, ml_shift)
    video_mr = shift_scale_r(video_mr, mr_shift)
    video_l = shift_scale(video_l, le_shift)
    video_r = shift_scale_r(video_r, re_shift)
    video = np.concatenate((video_l, video_ml, video_mr, video_r), axis=2)
        
    return video

def shift_scale(vid, shift):

    if shift == 0:
        return vid

    half = vid[:,0::2,:]
    
    if shift>0:
        half_crop = half[:,:,shift:len(half[0,0,:])]
    else:
        half_crop = half[:,:,:len(half[0,0,:])+shift]
        
    half_scaled = np.zeros_like(half)
    (width, height) = len(half[0,0]), len(half[0])
    
    for i in range(len(half_crop)):
        half_scaled[i,:,:] = cv.resize(half_crop[i], (width,height), interpolation=cv.INTER_LANCZOS4)
    vid[:,0::2,:] = half_scaled

    return vid

def shift_scale_r(vid, shift):

    if shift == 0:
        return vid

    half = vid[:,1::2,:]
    
    if shift>0:
        half_crop = half[:,:,:len(half[0,0,:])-shift]
    else:
        half_crop = half[:,:,-shift:len(half[0,0,:])]
        
    half_scaled = np.zeros_like(half)
    (width, height) = len(half[0,0]), len(half[0])
    
    for i in range(len(half_crop)):
        half_scaled[i,:,:] = cv.resize(half_crop[i], (width,height), interpolation=cv.INTER_LANCZOS4)
    vid[:,1::2,:] = half_scaled

    return vid


def correct_motion(frames, return_offsets=False, precalculated_offsets=None):

    """
    Corrects for XY motion via rigid registration by aligning frames with the first frame. Is known to fail if large Z-motion bumps are present in the video

    Parameters:
        frames (array): numpy array corresponding to video you would like to motion correct
        return_offsets (bool): Default False. Whether or not you want to also return the list of offsets for all registered frames in format [x_offsets, y_offsets]
                                Useful when rigid registration fails on green channel but works on red channel. You would return the offsets when correcting 
                                on red channel, and then plug the offsets into precalculated_offsets.
        precalculated_offsets (list): Default None. List where 0th index = list of x offsets and 1st index = list of y offsets. Returned by return_offsets.
                                        Used when you want to input offsets from red channel to correct green channel because green channel does not work.
    
    Returns:
        tuple: a tuple containing:
            - registered_frames (array): numpy array corresponding to video after rigid registration is applied
            - x_crop_bounds, y_crop bounds (int): integers corresponding to maximum pixel values shifted during rigid registration
            - [x_offsetList, y_offsetList] (list): Returned if return_offsets = True. x_offsetList and y_offsetList are lists containing x and y offset amounts to align each frame
    """
    
    # Convert frames to float for accurate transformation
    #frames = frames.astype(np.uint8)
    # Initialize the list of registered frames
    registered_frames = [frames[0]]  # Use the first frame as the reference

    y_offsetList=[]
    x_offsetList=[]
    
    scalingFactor = np.iinfo(frames.dtype).max

    if precalculated_offsets == None:
        for i in tqdm(range(1, len(frames))):
            
            # Compute phase cross-correlation to estimate translation
            (y_offset,x_offset),_,_ = phase_cross_correlation(registered_frames[0], frames[i])
            y_offsetList.append(int(y_offset))
            x_offsetList.append(int(x_offset))
            # Create an affine transformation (translation only)
            transform = AffineTransform(translation=(-x_offset,-y_offset)) # shift by opposite of offset to motion correct
            # Apply the transformation
            corrected_frame = warp(frames[i], transform, mode='constant', cval=0)
            registered_frames.append(corrected_frame*scalingFactor)
    
    else:
        x_offsetList = precalculated_offsets[0]
        y_offsetList = precalculated_offsets[1]

        for i in tqdm(range(1, len(frames))):

            x_offset = x_offsetList[i-1]
            y_offset = y_offsetList[i-1]
            # Create an affine transformation (translation only)
            transform = AffineTransform(translation=(-x_offset,-y_offset)) # shift by opposite of offset to motion correct
            # Apply the transformation
            corrected_frame = warp(frames[i], transform, mode='constant', cval=0)
            registered_frames.append(corrected_frame*scalingFactor)

    x_crop_bounds = (int(np.min(x_offsetList)), int(np.max(x_offsetList)))
    y_crop_bounds = (int(np.min(y_offsetList)), int(np.max(y_offsetList)))

    if return_offsets == True:
        return np.array(registered_frames), x_crop_bounds, y_crop_bounds, [x_offsetList, y_offsetList]

    return np.array(registered_frames), x_crop_bounds, y_crop_bounds


def denoiseVideo(video, avg_size=3, step_size=3, ksize=3, sigma=1):

    """
    3 frame rolling average and gaussian blur to denoise video

    Parameters:
        video (array): numpy array corresponding to video you would like to denoise
        avg_size (int): determines over how many frames to frame average by (default = 3 as described in paper)
        step_size (int): just keep this as 3
        ksize (int): kernel size for gaussian filter (default = 3)
        sigma (int): standard deviation for gaussian kernel (idk what this actually should be but i've just been using 1 ...)
    
    Returns:
        array: numpy array corresponding to denoisd video
    """

    averaged_vid = np.zeros((len(video)-2, len(video[0,:]), len(video[0,0,:])))

    for frame in range(1,len(video)-avg_size+2):
        avg_frame = np.mean(video[frame-1:frame+2], axis=0)
        averaged_vid[frame-1] = avg_frame
    
    video = averaged_vid

    #Gaussian blur 3x3 pixels
    blurred_vid = np.empty(( len(video) ,512,512))
    
    for frame in range(len(video)):
        blur = cv.GaussianBlur(video[frame], (ksize,ksize),sigma)
        blurred_vid[frame] = list(blur)
        
    return blurred_vid


def SD(video, span):
    
    """
    Calculates standard deviation (SD) of pixel intensity values over a span of a certain number frames over the video, with a step size of 1.

    Parameters:
        video (array): Should be numpy array corresponding to the denoised video (returned by denoiseVideo function)
        span (int): Number of frames over wich SD is calculated. I used 30. Paper used between 27-31
    
    Returns:
        array: Numpy array corresponding to SD of each pixel's intensity value over each span across the video
    """

    sd_video = np.empty((len(video) // (span), len(video[0]), len(video[0,0])))

    sdvidframe = 0
    for frame in range(0, len(video)- (span-1), span):
        sd_array = np.std(video[frame:frame+span], axis=0)
        sd_array[sd_array == 0] = np.nan
        sd_video[sdvidframe] = sd_array
        sdvidframe+=1
        
    return sd_video


def idB(video):
    """
    Performs imaging decible (idB) calculations. idB formula is as follows:
        idB = log_10(SD_n / SD_n-1) * 20, where SD_n is the current span's SD.
    
    Parameters:
        video (array): numpy array corresonding to video of SD of pixel values (returned by SD function)

    Returns:
        array: numpy array corresponding to idB over the SD spans
    """
    
    idB_array = np.zeros(( len(video)-1 , len(video[0]), len(video[0,0])))
    
    for timespan in range(1,len(video)):
        idB = np.log10((video[timespan] / video[timespan-1])) * 20
        idB[np.isnan(video[timespan-1]) | np.isnan(video[timespan])] = 0
        idB_array[timespan-1] = idB
        
    return idB_array



def multiThreshold(video, thresh, x_crop_bounds, y_crop_bounds):
    """
    From idB output, creates a binary array where |idB| <= thresh = 0 and |idB| > thresh = 1.
    Then maxed over all frames to return a 2d array corresponding to pixels where an active site occured
    at any time during the video.

    Parameters:
        video (array): numpy array corresponding to idB function output
        thresh (float): threshold; idB values > thresh and < -thresh are considered to be indicative of an active site
        x_crop_bounds, y_crop_bounds (int): used to crop edges of frame based on rigid registration shifts

    Returns:
        array: numpy array corresponding to putative locations of active sites. Will need to have small noise particles removed. 
    """
    
    thresh_array = np.zeros(video.shape)
    thresh_array[(video>thresh) | (video<-thresh)] = 1
    thresh_array = np.max(thresh_array, axis=0)

    # crops bounds based on shifting from XY correction
    if y_crop_bounds[0]<0:
        thresh_array[:y_crop_bounds[1]] = 0
        thresh_array[len(thresh_array[0])+y_crop_bounds[0]:] = 0
    else:
        thresh_array[:y_crop_bounds[1]] = 0

    if x_crop_bounds[0]<0:
        thresh_array[:,:x_crop_bounds[1]] = 0
        thresh_array[:,len(thresh_array[0])+x_crop_bounds[0]:] = 0
    else:
        thresh_array[:,:x_crop_bounds[1]] = 0

    return thresh_array


def particleFilter(thresh_array, particle_size=25):

    """
    Filters small noise particles from the thresholded idB array. Also returns info corresponding to final determined active sites.

    Parameters:
        thresh_array (array): numpy array of idB after thresholding (the array returned by multiThreshold)
        particle_size (int): particles of area less than particle_size will be considered as noise and eliminated. Default = 25.
    
    Returns:
        tuple: a tuple containing:
            - filtered_image (array): numpy array corresponding to thresh_array after particle filtering
            - activeSites (list of int): list containing integers corresponding to indices of active sites greater than particle filter limit
            - labels (array): numpy array where each unique particle from thresh_array has it's own integer ID
            - statsList (dict): dictionary with keys being particle indices of active sites (from activeSites) and values being area of the active site in pixels
            - boundingCoords (list of list): list containing lists of ints: [x0, x1, y0, y1], corresopnding to bounding coordinates of each active site
            going from top to down, left to right order with respect to the plot
    """

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats((np.uint8(thresh_array)*255), 4, cv.CV_32S)

    # Filter small particles
    min_area = particle_size  # Set minimum particle size
    filtered_image = np.zeros_like(thresh_array)

    activeSites = []
    statsList = []
    boundingCoords = []
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv.CC_STAT_AREA]
        if area > min_area:
            filtered_image[labels == i] = 255
            activeSites.append(i)
            statsList.append(stats[i, cv.CC_STAT_AREA])
            boundingCoords.append([stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_LEFT]+stats[i, cv.CC_STAT_WIDTH],
                                   stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_TOP]+stats[i, cv.CC_STAT_HEIGHT]])

    statsList = dict(zip(activeSites, statsList))

    return filtered_image, activeSites, labels, statsList, boundingCoords


def getActiveSiteMap(video, span=30, thresh=3, particle_size=25):
    
    """
    Executes rigid registration, denoising, SD, idB, thresholding, and particle filtering in one function.
    """

    xyCorrectedVideo, x_crop_bounds, y_crop_bounds = correct_motion(video)
    denoised_video = denoiseVideo(xyCorrectedVideo)
    sd_video = SD(denoised_video, span)
    idB_array = idB(sd_video)
    thresh_array = multiThreshold(idB_array, thresh, x_crop_bounds, y_crop_bounds)
    filtered_image, activeSites, labels, statsList, boundingCoords = particleFilter(thresh_array, particle_size)

    return denoised_video, filtered_image, activeSites, labels, statsList, boundingCoords


def detect_orientation_angle(binary_image):
    """
    Calculates angle of an active site with respect to horizontal plane. Angle is used to rotate the active site in rotate_image function.

    Parameters:
        binary_image (array): numpy array corresponding to cropped active site map (filtered_image from particleFilter) based on active site's bounding box coordinates
    
    Returns:
        float: angle to rotate image
    """
    
    # Find contours in the binary image
    try:
        _, contours, _ = cv.findContours(binary_image.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv.findContours(binary_image.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")
    
    # Find the largest contour (assuming it's the object)
    largest_contour = max(contours, key=cv.contourArea)

    # vx and vy are vectors in the fitted line's direction
    vx,vy,_,_ = cv.fitLine(largest_contour, cv.DIST_L2,0,0.01,0.01)
    
    return np.arctan( (vy) / (vx) )[0] * 180/np.pi # convert vector to angle by which to rotate the active site so it is flat


def rotate_image(image, angle):
    """
    Rotates bounding box containing an object such that it is parallel to the horizontal axis lengthwise.

    Parameters:
        image (array): numpy array corresponding to frame in video cropped based on active site's bounding box coordinates
        angle (float): angle by which you will rotate image (returned from detect_orientation_angle)
    
    Returns:
        array: numpy array corresponding to rotated image with active site parallele to horizontal axis lengthwise
    """
    
    # Get the image dimensions
    (h, w) = image.shape[:2]
    # Compute the center of the image
    center = (w // 2, h // 2)
    
    # Generate the rotation matrix
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate the bounding box of the new image
    # Ensure the entire image is within the new bounding box
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to take into account the translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation and return the result
    rotated_image = cv.warpAffine(image, M, (new_w, new_h))
    
    return rotated_image
    


def getRibbon(video, activeSiteBounds, activeSiteMap, correct_drift=False):

    """
    Creates ribbon for a single active site. A ribbon is the building block for a spatiotemporal map. 
    Basically, a (t x h x w) array, corresponding to the video of a rotated and cropped active site, where t = frames, h = height of active site, and w = width of active site,
    is compressed to a (t x 1 x w) array, where pixel values are averaged across its thickness (height).

    Parameters:
        video (array): numpy array corresponding to denoised video
        activeSiteBounds (list of int): the top, bottom, left, and right side bounds for an active site;
        accessed by indexing desired active site from boundingCoords (returned by particleFilter)
        activeSiteMap (array): numpy array corresponding to map with active site masks
    
    Returns:
        array: a single ribbon for an active site, the building block of an ST Map
    """
    
    top, bottom, left, right = activeSiteBounds[2], activeSiteBounds[3], activeSiteBounds[0], activeSiteBounds[1]
    boundedVid = video[:,top:bottom,left:right] # crops video to isolate active site on video
    
    activeSiteBound = activeSiteMap[top:bottom,left:right] # isolates coordinates for active site on the map with active site masks

    angle = detect_orientation_angle(activeSiteBound) # calculates angle by which to rotate active site so it is flat (long side facing down)

    rotatedSite = rotate_image(activeSiteBound, angle) # rotate active site by angle

    # calculates top and bottom cutoff for rotated active site
    starting_bound, ending_bound = np.where(rotatedSite>0)[0][0], np.where(rotatedSite>0)[0][-1]+1 
    rotatedShape = rotatedSite.shape
    # rotatedVid is array for frames of active site after rotation and cropping
    rotatedVid = np.zeros((len(video), ending_bound-starting_bound, rotatedShape[1]))

    # initiate array for ribbon, mean intensity is mean of pixel values down each column of pixels
    meanIntensity = np.zeros((len(video), rotatedShape[1]))
    
    for frame in range(len(video)):

        # rotates cropped frame of video with active site, and crops top and bottom to isolate active site
        rotatedFrame = rotate_image(boundedVid[frame], angle)[starting_bound:ending_bound]

        # updates frame from rotatedVid with rotated frame
        # rotatedVid can be used if you want to look at individual pixel intensities in the active site
        rotatedVid[frame] = rotatedFrame
        # updates row in meanIntensity with mean pixel values over each column in active site
        meanIntensity[frame] = (np.mean(rotatedVid[frame], axis=0))

    # meanIntensity is an array corresponding to the ribbon for an active site
    # each row in meanIntensity represents a frame from the video

    ribbon_drift_corrected = np.zeros(meanIntensity.shape)

    if correct_drift:
        for col_index in range(len(meanIntensity[0])):
            col = meanIntensity[:, col_index]
            
            coeffs = np.polyfit([frame for frame in range(len(col))], col, 3)
            fitted_pts = np.polyval(coeffs[:], [frame for frame in range(len(col))])
            col = col - fitted_pts + coeffs[-1]

            ribbon_drift_corrected[:, col_index] = col
        return ribbon_drift_corrected

    return meanIntensity



def buildSTMap(video, boundingCoords, activeSiteMap, mapType, **kwargs):

    """
    Builds a spatiotemporal (ST) Map.

    Parameters:
        video (array): numpy array corresponding to the denoised video
        boundingCoords (list of list): list of coordinates of active sites (returned by particle_filter function)
        activeSiteMap (array): numpy array corresponding to map of active sites (should be filtered_image returned by particle_filter function)
        mapType (str): 'raw' or 'zscr'. 'raw' outputs a black and white ST Map with raw pixel intensities. 
        'zscr' outputs ST Map showing active sites and is the basis of quantitative analysis from the Sci Adv paper.

        **kwargs: Additional keyword arguments. **MANDATORY FOR Z-SCORE ST MAP!!! (you do not need these for the raw ST map, though)**
            - 'cutoff' (float): z-score cutoff to distinguish noise from active site. I used 5, but paper used 2.5. See documentation for z-score calculation method.
            - 'percentile' (float): percent of pixel intensity values used to calculate baseline/quiescent average, default = 0.35. 
            z-score is calculated with respect to the mean and SD of the pixel intensity values below this percentile
            - 'particle size' (int): remove small noise specks, default = 13
    
    Returns:
        array: the ST Map
    """
    correct_drift = kwargs.get('correct_drift')

    if mapType == 'raw':
        # following builds the ST map for raw data, not the z score ST Map
        ribbons = getRibbon(video, boundingCoords[0], activeSiteMap, correct_drift=correct_drift) # initiate array for ST map with first ribbon
        for i in range(1, len(boundingCoords)):
            ribbons = np.append(ribbons, getRibbon(video, boundingCoords[i], activeSiteMap, correct_drift=correct_drift), axis=1) # append rest of the ribbons to array
            
        STMap = ribbons
    
    elif mapType == 'zscr':

        cutoff = kwargs.get('cutoff')  # paper uses cutoff of 2.5 but I used 5
        percentile = kwargs.get('percentile') # default is 35, but this is pretty arbitrary, just an estimation of the baseline 
        particle_size = kwargs.get('particle_size') # default is 13
        correct_drift = kwargs.get('correct_drift')

        if kwargs.get('artifact_cutoff') != None:
            artifact_cutoff = kwargs.get('artifact_cutoff')
        else:
            artifact_cutoff = 2

        # initialize first z-score ribbon
        zscrrib = zscoreConversion(getRibbon(video, boundingCoords[0], activeSiteMap, correct_drift=correct_drift), percentile, cutoff, artifact_cutoff=artifact_cutoff)

        for i in range(1, len(boundingCoords)): # build remaining z-score ribbons
            zscrrib = np.append(zscrrib, zscoreConversion(getRibbon(video, boundingCoords[i], activeSiteMap, correct_drift=correct_drift), percentile, cutoff, artifact_cutoff=artifact_cutoff), axis=1)

        # binary map of pixels above and below z-score threshold to use for particle size filtering
        zscores_binary = (~np.isnan(zscrrib)).astype(np.int8) 
        # returns binary filtered image where particles below threshold are removed
        zscores_f = particleFilterSTMap(zscores_binary, particle_size)
        # removes pixels on z-score STmap which corresponded to pixels that were below particle size threshold and removed
        zscrrib[zscores_f == 0] = np.nan 
        
        #zscrrib -= cutoff # subtract cutoff from z-scores (i.e. if cutoff is 5, 5 is subtracted from all z-scores)

        STMap = zscrrib
           
    return STMap
    

def zscoreConversion(ribbon, percentile=.35, cutoff=2.5, artifact_cutoff=2):

    """
    Converts pixel intensity values to a z-score with respect to the mean and SD of the lowest n percentile of pixel intensity values.

    Parameters:
        percentile (float): z-score is calculated with respect to the mean and SD of the pixel intensity values below this percentile
        cutoff (int): z-score cutoff to distinguish noise from active site. I used 5, but paper used 2.5. See documentation for z-score calculation method.
    
    Returns:
        array: numpy array of values corresponding to 'z-scores' for each column in a ribbon
    """
            
    zscores = np.zeros_like(ribbon)

    for i in range(len(ribbon[0])):
        col = ribbon[:,i]

        # Following if statement: I noticed that on one of the columns when doing the 600 frames unprocessed video, 
        # the 35 percentile intensity is 0, so the SD of that range is 0. Therefore, I encounter divide by zero errors if i proceed normally,
        # so I just change all zscores in that column to be just 0. Will lose some signal but best trade off I can think of for now
        # since it happens just rarely anyways. I am still not 100% sure if the ST map should be based on the processed video and 
        # 3-frame-averaged video of 200 frames, or the original 600 frame video
        SDminima = np.std(col[col < np.quantile(col, q=percentile)]) 

        quiescent_region = col[col < (np.mean(col[col < np.quantile(col, q=percentile)]) + 10*SDminima)]
        #quiescent_region = col[col < np.quantile(col, q=percentile)]
        SDq = np.std(quiescent_region)
        avgq = np.mean(quiescent_region)

        if SDq == 0:
            zscores[:,i] = 0

        elif np.max(col) <= artifact_cutoff: # possibly to get rid of artifact
            zscores[:,i] = 0

        else:
            zscores[:,i] = (col - avgq) / SDq


        """
        Not sure if cutoff is a cutoff for the raw intensity values,
        calculated as 2.5(or whatever number) * SD of 35 percentile + mean of 35 percentile (line of code below),
        or if the cutoff is just z-scores below the cutoff value
        if the former, replace zscr_col[zscr_col < cutoff] with zscr_col[col < cutoff]...
        ...I have been doing the latter with a cutoff of 5 though...
        """
        #cutoff = np.std(col[col < np.quantile(col, q=percentile)]) * 2.5 + np.mean(col[col < np.quantile(col, q=percentile)])
        
        zscr_col = zscores[:,i]
        zscr_col[zscr_col < cutoff] = np.nan
    
        zscores[:,i] = zscr_col
        
    return zscores



def particleFilterSTMap(thresh, particle_size):

    """
    This is just to remove specks from the STMap, not to remove specks from the threshold active site map
    
    Parameters:
        thresh (array): numpy array of the zscr ST Map before particle filtering
        particle_size (int): size of particles below which are considered noise specks (default = 13, from video Amreen sent, not in publication)
    
    Returns:
        array: filtered ST Map
    """

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats((np.uint8(thresh)*255), cv.CV_32S, connectivity=4)

    # Filter small particles
    min_area = particle_size  # Set minimum particle size
    filtered_image = np.zeros_like(thresh)
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv.CC_STAT_AREA]
        if area > min_area:
            filtered_image[labels == i] = 1
            
    return filtered_image
    

def play_video(video, adjust_contrast):

    """
    Displays a scrollable "video" that allows you to easily scroll through frames in a Jupyter notbeook.
    Glitchy when scrolling through frames if running Jupyter notebook in Safari.
    Works fine in VSCode. I did not test any other browsers. 

    Parameters:
        video (array): numpy array corresponding to the video or series of frames you want to display
        adjust_contrast (bool): whether or not you want to adjust contrast of the video
    
    Returns:
        None
    """
    if adjust_contrast:
        contrast_enhanced_video = np.zeros_like(video)
        
        for frame in tqdm(range(len(video))):
            p2, p98 = np.percentile(video[frame], (2, 98))
            rescaled_frame = exposure.rescale_intensity(video[frame], in_range=(p2, p98))
            contrast_enhanced_video[frame] = rescaled_frame
    
        video = contrast_enhanced_video

    # Create a function to update the displayed frame
    def update_frame(frame):
        plt.figure(figsize=(6, 6))
        plt.imshow(video[frame], cmap='gray')
        plt.axis('off')
        plt.show()
    
        # Create a slider widget to select the frame
    frame_slider = widgets.IntSlider(value=0, min=0, max=len(video)-1, step=1, description='Frame:', layout=widgets.Layout(width='500px'))
    
    # Create buttons for frame navigation
    next_button = widgets.Button(description="+1")
    prev_button = widgets.Button(description="-1")
    next10_button = widgets.Button(description="+10")
    prev10_button = widgets.Button(description="-10")

    # Define button click handlers
    def on_next_button_clicked(b):
        frame_slider.value +=1
    def on_prev_button_clicked(b):
        frame_slider.value -=1
    def on_next10_button_clicked(b):
        frame_slider.value +=10
    def on_prev10_button_clicked(b):
        frame_slider.value -=10
    
    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)
    next10_button.on_click(on_next10_button_clicked)
    prev10_button.on_click(on_prev10_button_clicked)

    # Use the interactive function to link the slider with the update function
    interactive_plot = widgets.interactive(update_frame, frame=frame_slider)
    display(widgets.VBox([widgets.HBox([prev10_button, prev_button, next_button, next10_button])]))

    display(interactive_plot)


def interactive_decomber(video):

    """
    An interactive decombing tool that implements decomb_video in a more user-friendly interface.

    **When you first run the cell it does not display a plot; you have to first change any number in order to show the plot. This is just how it is coded.
    It is not the most ideal but I couldn't find a way to display an initial plot and then delete it when you update the plot.

    Parameters:
        video (array): numpy array corresponding to the video you want to decomb

    Returns:
        None
    """
    def on_value_change(change):
        with output:
            clear_output(wait=True)
            shift_params = [num.value for num in number_displays]
            le, ml, mr, re = shift_params
            decombed_video = decomb(video, le, ml, mr, re)
            figure = plt.figure(figsize=(15, 15))
            plt.imshow(np.max(decombed_video, axis=0))

            for i in range(3):
                plt.axvline(x=(len(video[0, 0]) // 4) * (i + 1), color='magenta') # vertical bars that delineate section boundaries

            plt.show()
            plt.close()
            
    output = widgets.Output()

    # Define the labels and create IntText widgets
    fields = ["Left", "Middle Left", "Middle Right", "Right"]
    vboxes = [widgets.VBox([widgets.Label(value=f'{field}:'), widgets.IntText(layout=widgets.Layout(width='100px'))]) for field in fields]

    number_displays = [box.children[1] for box in vboxes]

    for num in number_displays:
        num.observe(on_value_change, names='value')  # Attach the callback function

    # Create the container widgets for layout
    fields_row = widgets.HBox(vboxes)

    # Display the row of fields and the output widget
    display(fields_row, output)

def activeSiteViewer(denoised_video, boundingCoords, filtered_image):
    """
    Allows user to visualize active site and corresponding ribbon over time. Plots three graphs:
        1) active site ribbon, 2) active site cropped based on bounding coordinates, and 3) entire FOV with bounding box of active site
    User can click through +/- 1 and 10 buttons or use the frame slider to view active site and ribbon over time at different frames. 
    The back and forward buttons, and dropdown menu cycle through different active sites.

    Parameters:
        BoundingCoords (list of list): list of bounding box coordinates
        denoised_video (array): numpy array of denoised video
        filtered (array): numpy array corresponding to active site masks (used for getRibbon function)
    Returns:   
        None
    """
    dropdown = Dropdown(
        options=[(f"Box {i}: {bbox}", bbox) for i, bbox in enumerate(boundingCoords)],
        description='Select Bounding Box:',
        disabled=False,
    )


    frame_slider = widgets.IntSlider(value=0, min=0, max=len(denoised_video)-1, step=1, description='Frame:', layout=widgets.Layout(width='500px'))

    # Create back and forward buttons
    back_button = widgets.Button(description='Prev Site')
    forward_button = widgets.Button(description='Next Site')

    def find_index(tuples_list, x):
        for index, (text, bbox) in enumerate(tuples_list):
            if bbox == x:
                return index

    # Function to go back
    def go_back(b):
        current_index = find_index(dropdown.options, dropdown.value)
        new_index = max(0, current_index - 1)  # Ensure it doesn't go below 0
        dropdown.value = dropdown.options[new_index][1]

    # Function to go forward
    def go_forward(b):
        current_index = find_index(dropdown.options, dropdown.value)
        new_index = min(len(dropdown.options) - 1, current_index + 1)  # Ensure it doesn't exceed the list length
        dropdown.value = dropdown.options[new_index][1]

    # Attach button click events
    back_button.on_click(go_back)
    forward_button.on_click(go_forward)

    maxxed = np.max(denoised_video, axis=0)

    ribbon_zscr=zscoreConversion(getRibbon(denoised_video, boundingCoords[0], filtered_image), .35, 2.5)

    def update_ribbon(site):
        bbox=dropdown.value
        global ribbon_zscr
        ribbon_zscr = zscoreConversion(getRibbon(denoised_video, bbox, filtered_image), .35, 2.5)
        zscores_filetered = particleFilterSTMap(ribbon_zscr, 13)
        ribbon_zscr[zscores_filetered == 0] = np.nan
        
    def update_frame(frame, site):
        fig,ax=plt.subplots(1,3,figsize=(10,7))
        bbox=dropdown.value
        video = denoised_video[:,bbox[2]:bbox[3],bbox[0]:bbox[1]]
        
        ribn=zscoreConversion(getRibbon(denoised_video, bbox, filtered_image), .35, 2.5)
        ribn[frame:]=np.nan
        ax[0].imshow(ribn, cmap='plasma', vmin=5,vmax=75)

        ax[1].imshow(video[frame], cmap='gray', vmin=5,vmax=250)
        ax[2].imshow(maxxed)
        rect = patches.Rectangle((bbox[0]-1, bbox[2]-1), bbox[1]-bbox[0]+1, bbox[3]-bbox[2]+1, linewidth=1, edgecolor='magenta', facecolor='none')

        ax[2].add_patch(rect) 
        plt.tight_layout()
        plt.show()
        plt.close()

    # Create buttons for frame navigation
    next_button = widgets.Button(description="+1")
    prev_button = widgets.Button(description="-1")
    next10_button = widgets.Button(description="+10")
    prev10_button = widgets.Button(description="-10")

    # Define button click handlers
    def on_next_button_clicked(b):
        frame_slider.value +=1
    def on_prev_button_clicked(b):
        frame_slider.value -=1
    def on_next10_button_clicked(b):
        frame_slider.value +=10
    def on_prev10_button_clicked(b):
        frame_slider.value -=10

    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)
    next10_button.on_click(on_next10_button_clicked)
    prev10_button.on_click(on_prev10_button_clicked)

    interactive_plot = widgets.interactive(update_frame, frame=frame_slider, site=dropdown)
    dropdown.observe(update_ribbon, names='value')
    display(widgets.VBox([widgets.HBox([prev10_button, prev_button, next_button, next10_button])]))
    display(widgets.HBox([back_button, forward_button]))
    display(interactive_plot)
    display(widgets.VBox([widgets.HBox([prev10_button, prev_button, next_button, next10_button])]))




