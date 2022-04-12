from cgi import print_directory
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import glob


# def Difference (left_fitx, right_fitx, width):
#     mean = (right_fitx[0][0][0] - left_fitx[0][0][0])/2
#     diff = (width/2 - mean)*0.0002645833
#     if (diff > 0) :
#         direction = "right"
#     else :
#         direction = "left"
#     return diff , direction

def Difference (left_fitx, right_fitx, width):
    xm_per_pix = 3.7 / 720
    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    mpts = pts_mean[-1][-1][-2].astype(int)
    pixelDeviation =width / 2 - abs(mpts)
    deviation = pixelDeviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"
    return deviation,direction

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # <-- This line altered for grayscale.
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
    
def addText(img, deviation,devDirection):

    # Add the radius and center position to the image
    font = cv2.FONT_HERSHEY_TRIPLEX

    # Deviation
    deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
    cv2.putText(img, deviation_text, (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 2, cv2.LINE_AA)

    return img

def show_image(image, title="title", cmap_type="gray"):
    plt.imshow(image, cmap_type)
    plt.title(title)
    # plt.axis("off")

# prespective
def perspectiveTransform(srcPoints, dstPoints):
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    Minv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
    return M, Minv

def warpPerspective(img, imgSize, M):
    return cv2.warpPerspective(img, M, imgSize, cv2.INTER_LINEAR)


cap = cv2.VideoCapture('test_video/challenge.mp4')
img_array = []
while cap.isOpened():
    ret, image = cap.read()

    if image is None:
        break

    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width / 2,400),
        (width, height),
    ]

    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    gray_image = hls_image[:,:,2]

    ############################################canny##############################################
    cannyed_image = cv2.Canny(gray_image,100,200, L2gradient = True)
    (T, threshInv) = cv2.threshold(cannyed_image, 190, 255,cv2.THRESH_BINARY)
    #############################################################################################

    cropped_image = region_of_interest(
        threshInv,
        np.array([region_of_interest_vertices], np.int32)
    )

    ysize = cropped_image.shape[0]
    xsize = cropped_image.shape[1]

    # undist = undistort(img, mtx, dist)

    src = np.float32([
        (696,455),    
        (587,455), 
        (235,700),  
        (1075,700)
    ])

    dst = np.float32([
        (xsize - 350, 0),
        (350, 0),
        (350, ysize),
        (xsize - 350, ysize)
    ])

    M, Minv = perspectiveTransform(src, dst)

    size = cropped_image.shape[1::-1]
    warped_image = warpPerspective(cropped_image.astype(np.float32), size, M)

    out_img = np.dstack((warped_image, warped_image, warped_image))*255
    # out_img = warped_image.copy()

    nwindows = 9
    # Set height of windows
    window_height = int(warped_image.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_base = 350
    rightx_base = 950
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # out_image = np.zeros_like(warped_image)
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    ###############################################################################################################
    window_img = np.zeros_like(out_img)
    ploty = np.linspace(0, warped_image.shape[0]-1, warped_image.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    ltx = np.trunc(left_fitx)
    rtx = np.trunc(right_fitx)




    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin/4, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin*5,
                                ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin*5, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin/4, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    Lwindow1 = np.array([np.transpose(np.vstack([left_fitx-margin/4, ploty]))])
    Lwindow2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin/4,
                                ploty])))])
    left_pts = np.hstack((Lwindow1, Lwindow2))
    Rwindow1 = np.array([np.transpose(np.vstack([right_fitx-margin/4, ploty]))])
    Rwindow2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin/4, ploty])))])
    right_pts = np.hstack((Rwindow1, Rwindow2))
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([left_pts]), (0, 0, 255))
    cv2.fillPoly(window_img, np.int_([right_pts]), (0, 0, 255))

    result = cv2.addWeighted(out_img, 0.5, window_img, 0.3, 0)
    M_res, Minv_res = perspectiveTransform( dst,src)
    re_image = warpPerspective(result.astype(np.float32), size, M_res)

    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2RGB)
    cannyed_image = cv2.cvtColor(cannyed_image, cv2.COLOR_GRAY2RGB)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
    threshInv = cv2.cvtColor(threshInv, cv2.COLOR_GRAY2RGB)

    deviation,direction = Difference(left_fitx, right_fitx, width)
    image = addText(image , deviation,direction)

    # Draw the result on the base image
    image = image.astype('uint8') + re_image.astype('uint8')

    # stack images
    scale = 0.5
    
    image = cv2.resize(image.astype('uint8'), (0,0), None, scale, scale)
    warped_image = cv2.resize(warped_image.astype('uint8'), (0,0), None, scale, scale)
    re_image = cv2.resize(re_image.astype('uint8') ,(0,0), None, scale, scale)
    cannyed_image = cv2.resize(cannyed_image.astype('uint8') ,(0,0), None, scale, scale)
    cropped_image = cv2.resize(cropped_image.astype('uint8') ,(0,0), None, scale, scale)
    threshInv = cv2.resize(threshInv.astype('uint8') ,(0,0), None, scale, scale)
    
    hor1 = np.hstack((image, re_image))
    hor2 = np.hstack((warped_image, cropped_image))
    vir = np.vstack((hor1, hor2))

    cv2.imshow('frame', vir)

    img_array.append(vir)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Save video
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
out = cv2.VideoWriter('out.mp4', fourcc, 25, (width, height), isColor=True)

for img in img_array:
    out.write(img)

out.release()