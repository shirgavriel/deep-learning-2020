# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:28:24 2020

@author: Hila Daniel and Shir Gavriel
"""
import os
import glob
import shutil 
from IPython.display import display, Image

def input_path_and_library():
    #This function inputs the path and library name.
    path = input("Please enter the path: ") #the path name
    library_name = input("Please enter the library name: ") #the library name
    if path == "":
        path = r"C:\Users\שיר גבריאל\Desktop" #defult
    os.chdir(path) #changing the current path
    if not os.path.exists(library_name):
        os.mkdir(library_name) #makes the library in the path
        print ("Library", library_name ,"created")
    path = os.path.abspath(library_name) #updates path with the library
    os.chdir(path)
  #  if not os.path.exists("train") and not os.path.exists("test"):
    os.mkdir("train") #makes the library train
    os.mkdir("test") #makes the library test
       

def images_to_library(t_path, images_path):
    """
    This function gets the train path or the test path and the images path.
    If t_path represents train path, than the function transfers 70% of the images
    from images labrary to the train labrary.
    If t_path represents test path, than the function transfers 100% of the images
    from images labrary to the train labrary.
    """   
    images_list = glob.iglob(os.path.join(images_path, "*.jpg")) #contains the files
    num_of_images = len(os.listdir(images_path)) #the num of files from the images library   
    os.chdir(t_path)
    # cheaks if t_path is for test or train
    if r"\test" in t_path: 
        for jpgfile in images_list: #copy all of the images to test
            shutil.copy(jpgfile, t_path)
    else:
        counter = 0 #counts the number of times the loop repeat 
        for jpgfile in images_list: #copy 70% of the images to train
            if (counter < 0.7*num_of_images):
                shutil.copy(jpgfile, t_path)
                counter = counter + 1
            
def print_files_name(t_path):
    """  
    The function gets test path or train path.
    It's prints the files' names in the labrary and showes them on the plot.
    """
    print(" The images' names: ")
    images_list = os.listdir(t_path) #contains the files
    images_path = glob.iglob(os.path.join(t_path, "*.jpg")) #contains the paths of the files 
    os.chdir(t_path)
    for jpgfile in images_list: 
        print(jpgfile) #print the images' names
    for jpgpath in images_path:
        display(Image(filename=jpgpath)) #display the images in Plots
   

       

def main():    
    input_path_and_library()
    t_path = r"C:\Users\שיר גבריאל\Desktop\hila123\test" #contains the relevant path
    images_path = r"C:\Users\שיר גבריאל\Desktop\imagesfordina" #contains the path of the images library
    images_to_library(t_path,images_path)
    print_files_name(t_path)
        

if __name__ == "__main__": 
    main()    
    