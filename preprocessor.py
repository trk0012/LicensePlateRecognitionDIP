from PIL import Image

# erode image by checking the intersection with either the 4 or 8-neighborhood
def clear(img, n, neighborhood = 4):
    w, h = img.size

    new_img = Image.new(mode="L", size=(w, h))

    for x in range(w):
        for y in range(h):
            # ignore black pixels
            if img.getpixel((x, y)) == 0:
                continue

            # get 4-neighborhood
            neighbors = find_neighbors(img, x, y, neighborhood)

            # if there are fewer than n white neighbors, clear the pixel
            num_white_neighbors = 0
            for i in neighbors:
                if i == 255:
                    num_white_neighbors += 1
            
            # erode if the intersection is too small
            if num_white_neighbors < n:
                new_img.putpixel((x, y), 0)
            else:
                new_img.putpixel((x, y), 255)
    
    return new_img

def invert(img):
    w, h = img.size

    # create a new grayscale image to work with
    new_img = Image.new(mode="L", size=(w, h))

    for x in range(w):
        for y in range(h):
            # invert the pixel by subtracting its intensity from the maximum
            new_img.putpixel((x, y), 255 - img.getpixel((x, y)))
    
    return new_img

def find_neighbors(img, x, y, neighborhood = 4):
    try:
        # get the adjacent pixels in the 4 cardinal directions
        if neighborhood == 4:
            return [img.getpixel((x+1, y)), img.getpixel((x-1, y)), img.getpixel((x, y+1)), img.getpixel((x, y-1))]
        # get the 4-neighborhood as well as the pixels diagonally adjacent
        elif neighborhood == 8:
            return [img.getpixel((x+1, y+1)), img.getpixel((x-1, y+1)), img.getpixel((x+1, y-1)), img.getpixel((x-1, y-1))] + find_neighbors(img, x, y, 4)
    except:
        return []

def binarize(img, threshold):
    w, h = img.size

    for x in range(w):
        for y in range(h):
            if img.getpixel((x,y)) >= threshold:
                # set the pixel to white
                img.putpixel((x,y), 255)
            else:
                # set the pixel to black
                img.putpixel((x,y), 0)
    
    return img

def avg_intensity(img):
    w, h = img.size

    total = 0
    for x in range(w):
        for y in range(h):
            total += img.getpixel((x, y))
    return total / w * h