import torch

#Images is a list of tensors
#Targets is a list of dictionaries, with image_id, boxes: tensor[labels], and labels: tensor[N_boxes, 4]
def partition_images(images:list,targets:list,sizeX:int=512, sizeY:int=512) -> tuple[list,list]:
    for image,target in zip(images,targets):



def recombine_images(images, targets):