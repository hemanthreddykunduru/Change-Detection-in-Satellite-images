let take the 2 images img_1 img_2

**Stage: detection (YOLO)**

img_1⇒Detect model⇒ get the objects with the coordinates and store in PostgreSQL let be detected_1

img_2⇒Detect model⇒ get the objects with the coordinates and store in PostgreSQL let be detected_2

**Stage: Change detection (PIXEL SUBSTRACTION if not we will change)**

Change detection we will get where the changes are happened

and save with the coordinates of the change happened

**Stage: Change Detection with Vision Models**

we will make the slices of the img_1 and img_2 and take one slice from each big image

img_1 and img_2 + detected_ 1 and detected_2 + change detected information (compare if there is change is detected portion is not there just skip that slice)

this will give the best understanding to the Vision Model we are explaining if there is some change detection we are stressing the Vision Model that at this location the change is happened just observe more focus this will give 

**Stage: Final Output**

We will get the perfectly what changes are happened

Disadvantage:

- Resource is wasted i.e. more computation but worth it
- and it is impossible to find the what object is changed if the detection model finds the orientation of the object is placed then we have one more feature we need maximum of the features not all features. all featured are impossible but we need maximum
- if there is more no of images in the timeline then we have to store the info what changes are happened in the 2 images because if we give all the things at a time LLM will forget and give the misinformation







================================
Asper now im not using the pixel substraction method to find the chnage detection because it getting clumsy clumsy