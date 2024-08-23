# Super Resolution GAN-Based Image Upscaling for Sandstone Micro-CT Imaging: 
<p align="center">
<img src="https://d12oja0ew7x0i8.cloudfront.net/images/Article_Images/ImageForArticle_20456_16221026898502608.jpg">
</p>
<p>Low-resolution data is a major problem, it acts as a major hurdle for tasks that we can perform with such datasets such as image classification and object detection. This problem is very much evident in geological research, acquisition of high-resolution data is generally limited because of hardware limitations of systems. To address this issue, I have developed a Super Resolution Generative Adversarial Network (SRGAN) model and trained it effectively. This sophisticated deep learning model excels at upscaling 64x64 pixel images to a higher and detailed resolution of 256x256 pixels, significantly enhancing the quality and utility of the data for geological analysis, and overcoming the limitations of hardware constraints.</p>
<h2>Libraries Used</h2>
<ul>
  <li>Tensorflow</li>
  <li>Numpy</li>
  <li>Pandas </li>
  <li>Matplotlib</li>
  <li>Seaborn</li>
  <li>Sklearn</li>
</ul>
<h2>Dataset Visualization</h2>
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/data_2.png" width="550" height="600">
</p>
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/Data2.png" width="550" height="500">
</p>
<h2>What is Super Resolution GAN?</h2>
<p>Super-resolution GAN applies a deep network in combination with an adversary network to produce higher-resolution images. During the training, A high-resolution image (HR) is downsampled to a low-resolution image (LR). A GAN generator upsamples LR images to super-resolution images (SR). We use a discriminator to distinguish the HR images and backpropagate the GAN loss to train the discriminator and the generator.</p>
<p>It uses a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes the solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, the authors use a content loss motivated by perceptual similarity instead of similarity in pixel space.</p>
<p align="center">
<img src="https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png" width="900" height="600">
</p>
<h2>Model Details</h2>
<p>The model was rigorously trained on a comprehensive dataset of sandstone micro-CT images for 30 epochs. The dataset included downscaled images of sandstone with dimensions of (64,64,3) and their corresponding upscaled versions with dimensions of (256,256,3). To guide the model's learning, a combination of binary cross-entropy and mean squared error loss functions was employed. Additionally, the Adam optimizer was utilized to efficiently update the model's parameters, ensuring convergence towards optimal performance.
</p>
<h2>Model Architecture</h2>   
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/SRGAN.png">
</p>
<h2>Model Training & Testing</h2>   
<p>The generator loss was 13.67 discriminator loss for fake images was 0.46 and for real images was 0.98.</p>
<h4>Generator Loss:</h4>   
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/Generator%20Loss.png" width="1000" height="350">
</p>
<h4>Discriminator Loss:</h4>  
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/Discriminator%20Loss.png" width="1000" height="350">
</p>
<h2>Model Prediction</h2>
<p align="center">
<img src="https://github.com/NavinBondade/Micro-CT-Sandstone-Image-Super-Resolution-With-SRGAN/blob/main/Graphs/Output.png" width="800" height="1200">
</p>
<h2>Conclusion</h2>  
<p>In this project, I have implemented and trained the SRGAN deep learning model for performing sandtone image upscaling from the resolution of 64x64 to 256x256.</p>  


