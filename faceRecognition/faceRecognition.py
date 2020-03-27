import functions as fn
import sys
import random

random.seed(42)
indicator = 0

'''
Set variables
'''

images_per_class        = 30
number_person_testing   = 1140
epsilon                 = 0.01
height                  = 12                    
width                   = 10                    
threshold               = 0.1
vertical                = 4                     # Has to be an even number
horizontal              = 2                     # Has to be an even number
directory               = "CroppedYale/*"



'''
Studies
'''
test_accuracy_normal    = False
test_cut_image          = True
individual_test         = False

subject_test,num_im     = 19,8



if test_cut_image:
   A, number_classes, remaining_images = fn.getmatrixes(dir = directory,images_per_class = images_per_class,height = height, width = width,vertical=vertical,horizontal=horizontal)
else:
   A, number_classes, remaining_images = fn.getmatrix(dir = directory,images_per_class = images_per_class,height = height, width = width)

image_for_testing       = remaining_images[subject_test][num_im] # <- In the dataset


'''
TEST FOR AN INDIVIDUAL IMAGE (cut)
'''

if test_cut_image:
   indicator = 1

   print("**********************************************************")
   print("*       Start test for an individual image (cut)...      *")   
   print("**********************************************************\n")

   class_image = fn.classifyCutImages(image_for_testing,width,height,vertical,horizontal,number_classes,images_per_class,A,epsilon,threshold,plot=False)

   if class_image == -1:
      print("Image is not a subject in the dataset")
   else:
      print("Image was classified as subject ",class_image)
   print("\n\n\n")

'''
TEST FOR AN INDIVIDUAL IMAGE
'''

if individual_test:
   indicator = 1
   print("**********************************************************")
   print("*         Start test for an individual image...          *")   
   print("**********************************************************\n")
   class_image = fn.classify(image_for_testing,width,height,
                             number_classes,images_per_class,
                             A,epsilon,threshold,plotting=True)
   
   if class_image == -1:
      print("Image is not a subject in the dataset")
   else:
      print("Image was classified as subject ",class_image)
   print("\n\n\n")


'''
TEST ACCURACY WITH NORMAL IMAGES
'''

if test_accuracy_normal:
   indicator = 1
   print("**********************************************************")
   print("*      Start testing accuracy with normal images...      *")
   print("**********************************************************\n")
   i = 0
   counter = 0
   
   testing_each_class = int(number_person_testing/number_classes)

   for id_number in range(number_classes):
      
      testing_images = random.sample(remaining_images[id_number],k=testing_each_class)

      for image in testing_images:
         
         class_image = fn.classify(image,width,height,number_classes,images_per_class,A,epsilon,threshold,False)
         
         if class_image == id_number:
            #print("Image was correctly classified")
            i = i+1
            
         else:
            #print("Image was not correctly classified")
            pass
               
         progress = counter*100/number_person_testing

         if progress == 0:
            print("Initializing...")
         elif progress % 10 == 0:
            print("Overall progress ",progress,"%")
            
         counter = counter + 1
      
   print("\nPercentage of accuracy:", float(i*100/number_person_testing),"%")
   
if indicator==0:
   print("\n\n\nYou have not selected any of the studies")

