import numpy as np
import cv2 as cv
import os

import preprocessor as p
from CharacterRecognizer import interpreter
from MyNN import myNN

def sample_row(img, row, percent_threshold):
    w, _ = img.size

    row_val = []

    # take the intensity values of each pixel in the row
    for x in range(w):
        row_val.append(img.getpixel((x, row)))
    
    # if the average intensity is less than the threshold, leave blank
    if np.mean(row_val) < percent_threshold:
        return []
    
    return row_val

def highlight_areas(pil_img, cv_img, center_row_index, intensity_threshold, detail):
    # extract height from image
    _, height = pil_img.size
    
    # sample the center row
    row_mid = sample_row(pil_img, center_row_index, intensity_threshold)
    rects = find_rects(center_row_index, row_mid, detail)

    # initialize rows to sample
    row_down = row_mid
    row_up = row_mid

    i = 1
    # loop *up to* the top/bottom part of the image
    while i < height / (2 * detail):
        # while the row above has a sufficient amount of non-black pixels
        if len(row_down) > 0:
            row_down = sample_row(pil_img, center_row_index + detail * i, intensity_threshold)
            rects_down = find_rects(center_row_index + detail * i, row_down, detail)
        # while the row below has a sufficient amount of non-black pixels
        if len(row_up) > 0:
            row_up = sample_row(pil_img, center_row_index - detail * i, intensity_threshold)
            rects_up = find_rects(center_row_index - detail * i, row_up, detail)
        i += 1

        # draw found rects on the image
        draw_rects(cv_img, rects_down, (0, 0, 255))
        draw_rects(cv_img, rects_up, (0, 0, 255))

        # merge the new-found rects with the existing ones
        rects = merge_rects(rects, rects_down)
        rects = merge_rects(rects, rects_up)

    return rects

def merge_rects(r1, r2):
    rects = r1

    if len(r1) == 0:
        return r2
    if len(r2) == 0:
        return r1

    for i in range(len(r1)):
        for j in r2:
            # if rects i and j overlap, then join them

            # if one rect's top left corner [0][0] is greater than the other rect's bottom right corner [1][1], they do not intersect
            # or if one rect's bottom right corner [1][1] is less than the other rect's top left corner [0][0], they do not intersect
            if (rects[i][0][0] >= j[1][0] or j[0][0] >= rects[i][1][0]) or (rects[i][1][1] < j[0][1] or j[1][1] < rects[i][0][1]):
                continue

            # if neither of those cases occur, then the rects overlap
            
            # find the top-leftmost point
            leftmost_coordinate = min(rects[i][0][0], j[0][0])
            upmost_coordinate = min(rects[i][0][1], j[0][1])
            # and the bottom-rightmost point
            rightmost_coordinate = max(rects[i][1][0], j[1][0])
            downmost_coordinate = max(rects[i][1][1], j[1][1])
            
            # create a new rect with the extreme points
            new_rect = ((leftmost_coordinate, upmost_coordinate), (rightmost_coordinate, downmost_coordinate))
            # store this new rect
            rects[i] = new_rect
    return rects

def find_rects(row_idx, pixels, width = 5):
    rects = []

    i = 1
    while i < len(pixels):
        # when it crosses from a black to a white pixel
        if pixels[i - 1] == 0 and pixels[i] == 255:
            # start the rectangle here
            start = i
            # loop from the start point
            for j in range(i, len(pixels)):
                # when it crosses from a white to a black pixel
                if pixels[j - 1] == 255 and pixels[j] == 0:
                    # mark the last pixel as the end
                    end = j - 1
                    # only write if the rectangle is more than 0.5% of the total width, and it is not at the edge of the image
                    if end - start > len(pixels) * 0.005 and j != len(pixels) - 1:
                        # store the detected rect
                        rects.append(((start, row_idx - width), (end, row_idx + width)))
                    # continue at the pixel after the black pixel
                    i = j
                    break
        i += 1

    return rects

def cull_abnormal_rects(rects, percent_threshold_length, percent_threshold_height):
    new_rects = []

    # calculate the average dimensions of all rects
    avg_length = 0
    avg_height = 0
    for r in rects:
        avg_length += r[1][0] - r[0][0]
        avg_height += r[1][1] - r[0][1]
    avg_length /= len(rects)
    avg_height /= len(rects)

    for r in rects:
        # if the dimensions of r are outside the given threshold, remove it from the list

        # find the dimensions of this rectangle
        length = r[1][0] - r[0][0]
        height = r[1][1] - r[0][1]

        # calculate how far off these lengths are from the averages
        l_error = abs(length - avg_length) / avg_length * 100
        h_error = abs(height - avg_height) / avg_height * 100

        # if they are too far off average, do not keep them
        if l_error <= percent_threshold_length and h_error <= percent_threshold_height:
            new_rects.append(r)
    
    return merge_rects(new_rects, new_rects)

def remove_duplicate_rects(rects, threshold):
    rects = merge_rects(rects, rects)
    new_rects = []
    for r in rects:
        valid = True
        for c in new_rects:
            # check if the corners of 2 rects are within some range of each other
            if r[0][0] - c[0][0] <= threshold and r[1][0] - c[1][0] <= threshold:
                valid = False
                break
        if valid:
            new_rects.append(r)
    return new_rects

def draw_rects(cv_img, rects, color):
    for i in rects:
        cv.rectangle(cv_img, i[0], i[1], color, 2)

def export_rects(img, rects, folder):
    neuralNetwork = interpreter()
    characters = ""
    for i in range(len(rects)):
        section = img.crop((rects[i][0][0],rects[i][0][1],rects[i][1][0],rects[i][1][1]))

        # w, h = section.size
        # l = max(w, h)
        # section = section.resize((l,l))

        for j in range(3):
            section = p.clear(section, 4, 4)

        if not os.path.isdir(f"{folder}"):
            os.mkdir(f"{folder}")
        section.save(f"{folder}\\{i}.png")

        characters += neuralNetwork.predicter(section)
    return characters