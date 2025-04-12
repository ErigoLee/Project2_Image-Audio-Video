## Project2_Image-Audio-Video
I developed a script that creates animated videos using the Stable Diffusion model. 
The process involves generating images by gradually interpolating between two distinct text prompts, allowing for a smooth transition of visual themes and styles. 
This method highlights the model’s capability to blend concepts and produce visually coherent frames across time.

Specifically, the script generates two separate videos, 
each representing a different artistic style: Studio Ghibli and The Simpsons. For each style, two prompts are defined to represent the starting and ending scenes. 
The script then computes intermediate embeddings and generates a sequence of images that transition from the first prompt to the second.

Finally, the images are compiled into a video using imageio, and the output is saved in MP4 format. 
The result is two short animations, each capturing the essence of its respective style while demonstrating the creative potential of prompt interpolation in generative AI.


Two videos:
1) Ghibil Version
https://github.com/user-attachments/assets/af44354d-5133-4aaf-8db5-235eea690569

2) Simpson Version
https://github.com/user-attachments/assets/f3a2bd0c-bdab-4744-813a-c7e17a94bc16

