from PIL import Image, ImageFilter, ImageEnhance
import cv2 as cv
import os

import preprocessor as process
import raycast as ray

def driver(filepath):
    # opening the image
    img = Image.open(f"{filepath}")

    # reset the counter for operations
    global steps
    steps = 1

    # create output folder
    directory = filepath.split('.')[0]
    directory = directory.replace("inputs", "outputs")
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    if not os.path.isdir(f"{directory}"):
        os.mkdir(f"{directory}")

    # store image dimensions
    width, height = img.size

    # ensure the image has adequate size
    while (width, height) < (800, 600):
        new_size = (width * 2, height * 2)
        img = img.resize(new_size)
        width, height = img.size
    while (width, height) > (1600, 900):
        new_size = (int(width / 2), int(height / 2))
        img = img.resize(new_size)
        width, height = img.size
    save(img, "resized", directory)

    # convert the image to grayscale
    img = img.convert("L")
    save(img, "grayscale", directory)

    # increase image contrast to make areas clearer
    img = ImageEnhance.Contrast(img).enhance(2)
    save(img, "enhanced", directory)

    # calculate the average intensity of the middle row
    center = ray.sample_row(ImageEnhance.Contrast(img).enhance(5), int(height/2), 0)
    avg_value = int(sum(center)/len(center))

    # if the background is bright (i.e. the text is dark), then invert the image 
    # this ensures that the text will be white on a black background
    if avg_value > 120:
        # take the image negative so the letters are in white and the background is in black
        img = process.invert(img)
        save(img, "negative", directory)

    # blur image to reduce grain
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    save(img, "blur", directory)

    # convert edge map to binary black and white
    img = process.binarize(img, 170)
    save(img, "binary", directory)
    
    # clean noise by culling lone white pixels
    img = process.clear(img, 8, 8)
    save(img, "cleaned", directory)

    # load instances of the cleaned file for modifying
    cv_img = cv.imread(f"{directory}\\{steps - 1}_cleaned.png")
    cv_img2 = cv.imread(f"{directory}\\{steps - 1}_cleaned.png")
    cv_img3 = cv.imread(f"{directory}\\{steps - 1}_cleaned.png")

    # find bounds of areas with high intensity
    rects = ray.highlight_areas(img, cv_img, int(height / 2), 0.1 * 255, 3)
    rects = ray.remove_duplicate_rects(rects, width * 0.01)

    cv.imwrite(f"{directory}\\{steps}_lined.png", cv_img)
    steps += 1

    # draw boxes around large points of interest
    ray.draw_rects(cv_img2, rects, (0, 0, 255))

    cv.imwrite(f"{directory}\\{steps}_bounds.png", cv_img2)
    steps += 1

    # remove rectangles that are not similar to the others
    rects = ray.cull_abnormal_rects(rects, 80, 40)
    ray.draw_rects(cv_img3, rects, (0, 0, 255))

    cv.imwrite(f"{directory}\\{steps}_culled.png", cv_img3)
    steps += 1

    ray.export_rects(img, rects, f"{directory}\\chars")

    print(f"DONE: {filepath}")

steps = 1
# save a file with a generated name
def save(img, filename, folder = ""):
    global steps
    if len(folder) == 0:
        img.save(f"{steps}_{filename}.png")
    else:
        img.save(f"{folder}\\{steps}_{filename}.png")
    steps += 1

# run with all inputs
# for f in os.listdir("inputs"):
#     driver(f"inputs\\{f}")

# run with 1 input
f = os.listdir("inputs")[19]
driver(f"inputs\\{f}")