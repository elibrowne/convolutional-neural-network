# convolutional-neural-network
Work with CNNs as part of Advanced Honors Computer Science

### Notes on filenames and how to run the program

This project has become a bit chaotic organization-wise, so here are some tips for navigating it. 
 - The difference between `cnn.py` and `cnn_for_ssh.py` started as a small difference—`cnn_for_ssh.py` contained a few extra lines of code that would prevent it from using up too many resources on a virtual desktop—but as time has gone on and I've used the virtual desktop to train more, `cnn_for_ssh.py` has gained more features, like data augmentation, which allow for it to do more complex training that I wouldn't want to run on my laptop. 
 - The program `webcam.py` is set up for testing the neural network against real world images. I usually go in front of it and mess with my mask. It isn't super advanced right now, and it uses an XML-sourced Haar cascade classifier from the internet to find faces before sending them through the network. It's pretty rough right now—I haven't spent a lot of time on this part of the project and only use it for quick testing—but I would hope to improve it a bit more, as sometimes it isn't able to even detect a person wearing a mask as a person.  
 - Data from prior runs is collected in the folder `weights-from-runs/`, which is generally sorted by date. I've tried to organize it as well as I could, but some iterations of the network required much more debugging than just one day (see the gap before February 23!). I took down notes on the type of network and its efficacy for some entries, which can be found below.
 - The dataset `masks/` is the original dataset, sourced from Kaggle. The dataset `masks-expanded/` has added 1,000 images from another computer-generated database to each class. 

### Different training runs and observations on their efficacy

**128x128x3 → 40x40x16 (ReLU) → 20x20x16 → Batch Normalization → 18x18x24 (ReLU) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all ReLU)**
<br>jan25-overnight • loss: 0.0320 - accuracy: 0.9477 - val_loss: 0.0385 - val_accuracy: 0.9310 (250 epochs)

**128x128x3 → 40x40x16 (Leaky ReLU) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all Leaky ReLU)**
<br>jan26-1 • loss: 0.0255 - accuracy: 0.9607 - val_loss: 0.0293 - val_accuracy: 0.9499 (211 epochs)

I wore a black mask to test this network—I believe that it does better with lighter colors.
 - In bright lighting, this neural network typically guesses that someone wearing a mask is wearing it incorrectly, regardless of how well it is worn. I could occasionally get it to say that I was wearing a mask correctly, but it didn’t seem correlated to how well the mask was actually being worn.
 - In dim lighting, both correctly worn and incorrectly worn masks are registered as correctly worn.
 - In both circumstances, this neural network was not good at determining when someone was not wearing a mask—it almost always categorized them as wearing one. 
I also tested it briefly with Sasha (thank you Sasha!), who was wearing a lighter maroon mask. 
 - Inside, it seemed decent—we didn’t test it with mask off, as to not violate the school’s policies, but it discerned between an incorrect and a correct mask somewhat reasonably (with uncertainty, which is expected).
 - Outside, it returned a blue frame the entire time, which means that the program encountered an error in resizing the data. I don’t understand why this was happening. (I also got it to return a blue frame a few times when outside.)

**128x128x3 → 40x40x16 (Leaky ReLU) → dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (activation = Leaky ReLU) → dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all Leaky ReLU)**
<br>jan27-1 • loss: 0.0104 - accuracy: 0.9852 - val_loss: 0.0615 - val_accuracy: 0.8719 (1000 epochs)

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all Leaky ReLU)**
<br>feb8-1 • loss: 0.0106 - accuracy: 0.9835 - val_loss: 0.0248 - val_accuracy: 0.9540 (1000 epochs)

This was really good at classifying images from the dataset (95% validation accuracy and very good training accuracy!). It also seemed a bit better at differentiating between proper mask wearing and improper mask wearing, although lighting differences seemed to play a big role in whether or not a mask was worn correctly or wrong.

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all Leaky ReLU)**
<br>feb23-1 • loss: 0.0047 - accuracy: 0.9950 - val_loss: 0.0245 - val_accuracy: 0.9507 (2000 epochs)
<br>feb23-2 • loss: 0.0013 - accuracy: 0.9989 - val_loss: 0.0199 - val_accuracy: 0.9605 (5000 epochs)

I trained this model using an expanded dataset, which included thousands of computer-generated images of faces wearing masks, wearing masks incorrectly, or not wearing any masks. These photos are higher quality than those of the original dataset, and they’re more zoomed out, which adds some depth to the dataset. My biggest frustration with the additional photos is that every mask is blue—while this will help a lot with recognizing blue masks, I’m not sure that impact that it will have on masks that aren’t blue. 

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 3, all Leaky ReLU)**
<br>feb28-1 • loss: 4.3012e-04 - accuracy: 0.9996 - val_loss: 0.0215 - val_accuracy: 0.9577 (6000 epochs)

This model utilized data augmentation to work on making accuracy higher for different light conditions and color schemes. I used `layers.RandomContrast(0.2)` and `tf.image.random_brightness(x, 0.3)` to change the sizes, and then concatenated the edited images with the original dataset. It took much longer to run. Overall, I am not actually seeing much improvement on my own face from this dataset. It is much more frequent/happy in classifying something as a poorly-worn mask (yellow) on both blue and black masks. On blue masks, it seems a bit more accurate in reaching a well-worn mask (green), but it's harder to get it to register a well-wron mask. On bare faces, these networks seem to all default to a poorly-worn mask, and not a no mask at all situation. 

This has posed a new idea for me: setting up two neural networks, one of which classifies a correct mask from an incorrect/nonexistent one before the other determines whether the wrong mask is incorrect or nonexistent. I think that this might help the network identify important ideas (like noses being visible, etc.) before moving to the question of if a mask is present at all. 

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 2, all Leaky ReLU)**
<br>mar2-1 • loss: 3.9479e-04 - accuracy: 0.9998 - val_loss: 0.0272 - val_accuracy: 0.9675 (5000 epochs)

I've decided to try to create a network with two classes (a correctly worn mask and one that isn't correct, either from being worn wrong or not having a mask present), which I think may make it easier for the neural network. I trained this network on an awkwardly-sized dataset (more incorrect images than correct ones), which may explain a bias towards incorrect masks. In my personal experience, this network was more accurate with blue masks than with black ones. 

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 2, all Leaky ReLU)**
<br>mar11-1 • loss: 5.3010e-04 - accuracy: 0.9996 - val_loss: 0.0401 - val_accuracy: 0.9474

This network actually worked pretty well. When looking at it straight on, assuming the classes are `without_passing_mask` for `[0]` and `with_passing_mask` for `[1]`, it identifies blue masks quite well and other colored masks at a pretty solid level. It does seem a bit skewed when rotated, which may have to do with the data augmentation used. 

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 2, all Leaky ReLU)**
<br>mar22-1 • loss: 4.1634e-04 - accuracy: 0.9999 - val_loss: 0.0307 - val_accuracy: 0.9606

I strengthened the data augmentation further, changing colors, lighting, and more to try to improve accuracy. To me, my mask good/not good network works on blue masks in good lighting, but I'd like to see that extended to less ideal circumstances. 

**128x128x3 → 40x40x16 (Leaky ReLU) → Dropout (0.2) → 20x20x16 → Batch Normalization → 18x18x24 (Leaky ReLU) → Dropout (0.2) → 9x9x24 → Batch Normalization → Dense (1024, 256, 64, 2, all Leaky ReLU)**
<br>mar23-1 • loss: 1.6973e-04 - accuracy: 0.9999 - val_loss: 0.0036 - val_accuracy: 0.9958 (4000 epochs)

This net classifies between images where a mask is worn poorly and images where a mask is not present. It's used in tandem with an above classifier which determines if a mask is worn well or if there is no well worn mask in the image.

### Observations on the network 

*Note: this is old; I've left it here because addressing these challenges has guided some of my work since.* I've tried out a few images of myself and some friends (with permission, of course!) in different trial runs of the convolutional neural network. What I've found is that head-on images, like `eli_w_mask.jpg`, work much better than images from the side, like `jasper_w_mask.jpg`. Scrolling through the dataset, this seems somewhat understandable: the images in the dataset are all very much zoomed in, head-on, and distorted, but not really to the side as seen in some of the sample images. (On the other hand, it makes me less worried about distorting images!) In implementation, being able to extract pictures of the face that are mainly head-on would be very useful.