### 1.0 Introduction
The purpose of the project is to generate human faces that do not exist in the training set. Four models will be compared: An autoencoder (AE), a variational autoencoder (VAE), a Deep Convolutional Generative Adversarial Network (DC-GAN), and a Variational Autoencoder Generative Adversarial Network (VAE-GAN). All models will be trained on a processed version of the publicly available LFW face dataset.

Our motivation is to learn about different face generation architectures since there are many applications related to face generation, which includes police sketching and data augmentation. We are also interested in assessing the effects of GANs on face generation. We would like to determine if GANs create sharper reconstructions and whether a VAE-GAN has more control over a generation due to the restrictions on its latent space [1].

Considering the complexity of the task, machine learning is an appropriate tool for it; face generation is a task that does not lend itself to rules-based systems. Unlike traditional algorithms, machine learning algorithms can learn from large amounts of input data and create new data based on the structure of existing data [2].

### 2.0 Illustration
The four models will be trained on the cleaned data (Figure 1). Quantitative assessment will be conducted by comparing MSE on the training set. However, the DC-GAN does not perform reconstruction so it will be assessed based on the quality of the generated faces. Ease of manipulating embeddings to generate new images will also be considered.

<img align="center" src=/images/image15.png>
<p align="center">
Figure 1. Overall structure
</p>

### 3.0 Background & Related Work
Our project goal is to make an easy to control and effective face generating VAE. A recent state of the art model for face generation is Nvidia’s StyleGan [3]; however, such a model requires computing power beyond the reach of an undergraduate student. The GAN structure can be mimicked however by adding a discriminator as a loss function for out VAE. Discriminator networks are used to increase the photo-realism of VAE output [4]; this is intuitive because a loss such as MSE will push a model to generate blurry outputs that have a low expected pixel-wise loss. Principal Component Analysis will be used to simplify the face generation process. It has already been used in a non-academic setting to create easy to modify generated faces [5].

### 4.0 Data Processing
Data processing is relatively straight forward for this project as the online resources are abundant. We are going to use LFWcrop Face Dataset (Figure 2) for this project. One advantage of this dataset is all the images are preprocessed. All backgrounds are cropped off and all the images are of the same size, which saved us a lot of data processing efforts. Another advantage is that the image size is small, which will largely reduce the training complexity.


Figure 2. Example images from the dataset

To increase the number of images, the following data augmentation techniques will be applied to the images in each epoch. As a result, we will have a slightly different training set in each epoch.  
A horizontal flipping with a probability of 50%.
A random color jittering using torchvision.transforms.ColorJitter [6]. This function randomly changes the brightness, contrast, and saturation of an image.
An image normalization such that all pixel intensities are between -1 and 1.

The original dataset has 13,233 images in total. After applying the data augmentation techniques, we will obtain (13,233 * the number of epochs) images. Below are samples of the training images (Figure 3):


Figure 3. Training images

### 5.0 Architecture
#### 5.1 VAE
The Variational Autoencoder has 4 convolutional and 4 convolutional transpose layers. Batch normalization [7] is added at all convolutional layers, and the activation function has been changed to leaky ReLU [8] to aid training. The latent space is modeled as a standard Gaussian distribution. The similarity between the learned representation and a standard Gaussian is controlled by penalizing Kullback-Leibler Divergence (KLD) between the two [9]. The weight of the KLD loss relative to the pixel-wise loss is treated as a hyperparameter. New samples can be generated by drawing form a standard Gaussian in the latent space.

Figure 4. VAE

#### 5.2 DC-GAN
The DC-GAN consists of a generator and a discriminator. The generator is made up of four convolutional transpose layers and each layer is followed by a batch normalization and ReLU layer. Since the images’ pixel intensities in the dataset are normalized to -1 to 1 during data processing, a tanh layer is attached at the end of the generator to remain consistent with the range. Similarly, the discriminator has five convolutional layers and each layer is followed by a leaky ReLU and batch normalization layer. A sigmoid activation will be applied at the end to produce a probability.

Figure 5. Generator of DC-GAN

Figure 6. Discriminator of DC-GAN


#### 5.3 VAE-GAN
The same VAE architecture described above (section 5.1) is used for reconstruction. A 5 layer convolutional, batch-normed, leaky ReLU network with sigmoid output is used as the discriminator. Binary Cross-Entropy loss is used as the objective function for GAN training since the labels are binary values. The training cycle alternates training the discriminator and the generator. During discriminator training, the loss is calculated on equally sized batches of original images, their reconstructions, and standard Gaussian noise reconstructed from the decoder of the VAE. The generator is then trained with the discriminator loss on these same three groups; VAE MSE reconstruction loss and KLD loss are then backpropagated to further incentivize accurate reconstruction and a Gaussian distributed latent space.


Figure 7. VAE-GAN


### 6.0 Baseline Model
Our baseline model is a convolutional autoencoder. We will train the autoencoder and test if random noise in the latent space yields a coherent reconstruction. Without constraints on the latent space, the training data is not likely to be normally or uniformly distributed throughout the embedding space, which leads to poor reconstruction. The depth of the autoencoder will be restricted to three since the AE serves only as a baseline.

Figure 8. Baseline Model


### 7.0 Quantitative and Qualitative Results
The performance will be measured quantitatively by comparing the MSE loss on the training data. Generated images will be assessed qualitatively. The lowest MSE is reported below for each model. A DC-GAN has no comparable loss since it has no reconstruction element; a VAE-GAN will also be compared in the future.


Table 1. Training loss of all models

#### 7.1AutoEncoder
The loss of AE drops rapidly at the first few epochs and then settles at an MSE off 0.0077. Since no constraints are placed on the latent space of the AE, the reconstruction of latent space noise yields nothing of interest.
   	 
Figure 9. Training loss of AE
 
#### 7.2 VAE
Two VAE models performed well; one with a KLD weighting of 0.1 and the other with a KLD weighting of 0.01. While the model with lower KLD weighting achieved a better MSE loss its generative capabilities were inferior to the other model. This is because the latent space had a looser constraint and its final distribution differed substantially from a standard Gaussian - decoding noise yielded poor results. In both models, KLD loss stays incredibly stable leading us to believe that the distribution is optimized early and later epochs minimize reconstruction loss.

Figure 12. Training loss of VAE with 0.1 KLD loss 

Figure 13. Training loss of VAE with 0.01 KLD loss

The qualitative examination of reconstructed faces confirms that placing more weight on constraining the latent distribution causes reconstruction quality to suffer. Below are faces generated by reconstructing Gaussian noise from both models’ latent spaces. Rows are ordered by standard deviation; beginning at 0.25 and increasing to 2.0 in 0.25 increments. The first row shows faces considered more prototypical by the VAE; as the standard deviation increases more interesting faces emerge.


Figure 14. Generated images from VAE with KLD 0.1 loss

Figure 15. Generated images from VAE with KLD 0.01 loss

#### 7.3 DC-GAN
The DC-GAN generated distorted faces, yet much sharper than what the auto-encoders produced. Without control of the sampling space as in probabilistic models, it is difficult to generate a specific type of face.
	

Figure 18. Generated images from GAN

#### 7.4 VAE-GAN
Despite stable training and trying batch-normalization, discriminator pre-training, varying levels of dropout, various activation functions, and architectures, the VAE-GAN did not perform as desired. Depending on the relative weighting between the KLD and GAN loss the model would perform as one of the two and never quite gain the benefits from both. We believe further experimentation with VQ-VAE architecture or dense convolutions may help this issue. It comes to our relief that as of 2018 DeepMind researchers themselves claim “Currently, VAE -GANs do not deliver on their promise to stabilize GAN training or improve VAEs”[10].


### 8.0 Discussion
The AE has limited control over face generation since it is not able to generate faces through random noise in the embedding space. Perturbations of a trained image’s encoding allow modification of a specific image, however, this is not a sufficient amount of generative freedom.
Our VAE permits controlled generation and achieves lower MSE reconstruction loss. A major drawback of the VAE is the blurriness of its outputs. We believe this model is a large improvement over the regular AE.

The blurred output is solved by an Adversarial network loss function. DC-GAN creates sharper outputs yet has very little control over what image is generated since there are no constraints on the input space; noise is used to generate images in this model. The VAE-GAN failed to improve solve these issues, a balance could not be struck between generative control image quality. We believe this issue could be solved by a larger training set and a convolutional architecture that uses residual connections.

Principal Component Analysis is used to restructure the latent space such that the most salient features can be identified. To avoid further information loss we do not use PCA as a dimensionality reduction tool but leave the same number of principal components as latent dimensions; the gain lies in the ordering of these dimensions by variance. By modifying these principle components we can modify faces or generate new ones with ease and significant control.

Figure 19. Face generation GUI

### 9.0 Ethical Considerations

As social media accounts for an ever-increasing amount of news consumption in our society, more safeguards will need to be placed on the verification of shared information. Generated images and videos can ease the spread of disinformation; the success of Cambridge Analytica on Facebook is but one warning of the dangers of disinformation spread in loosely regulated social networks. GAN training provides an insufficient solution by training a discriminator to detect false images/video/text. Solving this issue extends far beyond the scope of this course, however, being aware of it may help us make ethical choices in the future.

Data privacy is yet another issue; there is no feasible way to guarantee that each image of a large dataset had the consent of the subject. Furthermore, for deployed applications there currently exists no exact method for “deleting” the data of an individual in the training set should they request it, besides the unappealing solution of retraining.



### 10.0 References
[1] Medium. (2019). Generate Anime Character with Variational Auto-encoder. [online] Available at: https://medium.com/@wuga/generate-anime-character-with-variational-auto-encoder-81e3134d1439 [Accessed 29 Jun. 2019].

[2] YouTube. (2019). Variational Autoencoders - EXPLAINED!. [online] Available at: https://www.youtube.com/watch?v=fcvYpzHmhvA [Accessed 29 Jun. 2019].

[3] T. Karras, S. Laine, and T. Aila, “A Style-Based Generator Architecture for Generative Adversarial Networks,” CVPR 2019, Mar. 2019.

[4] S. H. Khan, M. Hayat, and N. Barnes, “Adversarial Training of Variational Auto-encoders for High Fidelity Image Generation,” arXiv.org, 27-Apr-2018. [Online]. Available: https://arxiv.org/abs/1804.10323. [Accessed: 30-Jun-2019].

[5] CodeParade, “Computer Generates Human Faces,” YouTube, 02-Oct-2017. [Online]. Available: https://www.youtube.com/watch?v=4VAkrUNLKSo. [Accessed: 30-Jun-2019].

[6]Pytorch.org. (2019). torchvision.transforms — PyTorch master documentation. [online] Available at: https://pytorch.org/docs/stable/torchvision/transforms.html [Accessed 24 Jul. 2019]. 

[7] S. Ioffe, C. Szegedy. 2015. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. In Proc. of International Conference on Machine Learning. 448–456.  [Accessed: 30-Jun-2019]

[8] B. Xu, N.Wang, T. Chen, M. Li. 2015. Empirical Evaluation of Rectified Activations in Convolutional Network. [online] https://arxiv.org/abs/1505.00853 [Accessed: 30-Jun-2019]

[9] D. Kingma, M. Welling 2013. Auto-Encoding Variational Bayes. [online] https://arxiv.org/abs/1312.6114 [Accessed: 30-Jun-2019]

[10]Efrosgans.eecs.berkeley.edu. (2019). [online] Available at: http://efrosgans.eecs.berkeley.edu/CVPR18_slides/VAE_GANS_by_Rosca.pdf [Accessed 14 Aug. 2019].

