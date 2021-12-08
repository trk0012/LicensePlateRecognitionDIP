import torch
import torchvision.transforms as transf
from MyNN import myNN
import PIL
from PIL import Image
from PIL import ImageOps as iop

class interpreter:
    def __init__(self) -> None:
        self.character_corelations = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
                                11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K', 
                                21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R', 28:'S', 29:'T', 30:'U',
                                31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z', 36:'a' , 37:'6', 38:'d', 39:'e', 40:'f',
                                41:'9', 42:'h', 43:'n', 44:'q', 45:'r', 46:'t'}
        self.model = torch.load('nnmodel.pth')
        self.model.eval()


    def predicter(self, original_image):
        imgsize = original_image.size
        padImg = iop.pad(original_image, [imgsize[1], imgsize[1]]) #pads the image to make it a square
        scale = imgsize[1]/28.0
        scaledImg = iop.scale(padImg, 1/scale) #scales down the image to make it the same size as the images of the dataset we worked with
        #Now to make it recognizable for the ML model, the image needs to be rotated 90 degrees counter clockwise and flipped horizontally 
        rotatedImg = scaledImg.rotate(90)
        flippedImg = iop.flip(rotatedImg)
        transformer = transf.ToTensor()
        tensorImg = transformer(flippedImg)
        result = self.model(tensorImg)
        __, predicted = torch.max(result, 1)
        output = self.character_corelations[predicted.item()] #using the dictionary defined above, determines what the output of the model is as a character
        return output