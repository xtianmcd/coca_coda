# coca_coda Inference Application
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/xtianmcd/coca_coda/master?filepath=https%3A%2F%2Fgithub.com%2Fxtianmcd%2Fcoca_coda%2Fblob%2Fmaster%2Fcoca_coda_inference.ipynb)

I chose to use a deep learning model for this project for several reasons: 

        1. The nature of the task at hand, i.e., real-time object detection, lends itself well to deep convolutional neural networks.

        2. Transfer learning allows us to borrow from complex and expensive models and use their pre-trained weights on new data; plus, I hadn't implemented transfer learning before.

        3. I had just given a tutorial on 'An Introduction to Data science' and finished a class project, both of which used classical machine learning models, so I wanted to change it up a bit. :)

    \item
    Data Augmentation
    \begin{itemize}
    \item
        Segmentation: Performed by hand on the 50 provided images of coke bottles via Labelbox.
    \item
        scripts/data\_aug.py  
        \begin{itemize}
            \item
            Resizing (for multi-resolution)
            \item
            Channel Inversion (for edge device color discrepancies)
            \item 
            Translation - takes into account the location of the object to keep it intact
            \item
            Rotation - 90,180,270 degrees
            \item
            Yields 24,000 images
            \item 
            Keeps track of segmentation coordinates for coke bottle objects 
            \item
            No further processing was done to the images (e.g., thresholding for edge detection) aside form normalizing the values in [0,1].
        \end{itemize}
    \item
        autoskiplabel/index.html
        Utilizes Labelbox' API to create dummy segmentations for the augmented dataset and exports the dataset as JSON file; I subsequently wrote the coordinates into the JSON file
    \item
        scripts/json\_to\_coco.py
        Converted the JSON file to the format used in the Coco data set (see Faster R-CNN Transfer Learning Attempt section below); also generated bounding box corner coordinates from the segmentation coordinates. 
    \end{itemize}
    \item
    \textbf{Faster R-CNN Transfer Learning Attempt.}
        Initially, my goal was to use a pre-trained Faster R-CNN model to generate the bounding boxes. I successfully re-trained the final layers of the Faster R-CNN model with ResNet50 as the encoder (pre-trained on the Coco dataset, obtained from TensorFlow's Model Zoo), obtaining 86\% testing accuracy. 
        However, when trying to restore the graph for inference, I encountered an issue that seems to be caused by an active bug in the code, given Issue \#5003 in Tensorflow/models on GitHub (I added my error outputs and observations to the ticket). 
        Given the time it was taking to debug this problem and the time invested to get to this point, I decided to pivot to another method. 
        I saved the commands used to train the model and restore the graph so that if the bug is fixed (or I figure out what was going wrong) I can retry on this or another dataset.
    \item
    \textbf{MobileNetV2 Classification.}
        My next idea was to use a pre-trained model in Keras, given its ease of use. 
        Retraining the final layers of the MobileNetV2 model (pre-trained on ImageNet), I reached a validation accuracy of 98.8\% and 0.07 loss in just 3 training epochs (see Figure 1). Testing accuracy was around 70\%. 
        
        \begin{figure}[!h]
            \centering
            \includegraphics[width=0.8\linewidth]{training.png}
            \caption{Train and Validation Accuracy and Loss During Training. Higher Validation Accuracy may indicate the augmented images are too similar.}
         \end{figure}
        
        The models available in Keras restrict the problem to a simple classification task, so I additionally trained a custom Haar-like Feature-based Cascade classifier using the OpenCV CLI on the data. The training of the cascade parameters terminated early without error, which indicates it was already outperforming a preset threshold; however, the bounding box prediction falls short of perfect, possibly due to the size constraint passed to the box. 
        Combining these two techniques yields a classifier and a bounding box generator for the coke bottle identification problem. 
    
    \item 
    \textbf{Computing Note}. Our departments' GPU's were either in use for thesis work or out of operation, so I resorted to using my CPU for all training processing. Given the size of the augmented dataset and the complexity of the models, this made for quite slow progression of model building. 
    
 \end{enumerate}
 
\section{Inference App}
    \item
    Flask App
        This was my first exposure to Flask, html, JavaScript and app dev in general. I experimented with several Flask boilerplate templates, and spent considerable time exploring the file structure and contents therein. Customizing the template, adding buttons and running locally was a no-brainer. 
        However, I ran into issues when trying to get the buttons and functions to actually perform operations. I went through numerous tutorials and tried attacking the issue from the python, html, and JavaScript files, but with no avail. 
    
        Given the fortunate fact that I have some time before starting my employment, I think the best method will be to go through some tutorials start-to-finish and work may way up. Thankfully, I enjoyed these initial attempts and look forward to improving my skills. 
        The app as-is can be found at 
        https://github.com/xtianmcd/coca\_coda\_app; run the server locally with the command `python manage.py runserver` from the root directory. You can log in with the dummy account email@email.com, password = pa\$\$w0rd
        
         \begin{figure}[!h]
            \centering
            \includegraphics[width=0.8\linewidth]{coca_coda_app.png}
            \caption{Screenshot of the Home page once logged in.}
         \end{figure}
    
    \item
        In order to still provide a usable product, I have made available a launchable Jupyter notebook that will run inference on a test set or new files/video, including from the device's webcam. This can be launched from https://github.com/xtianmcd/coca\_coda. I was unable to test the webcam detection with an actual coke bottle, but it should perform about as well as the test-set and file-upload modes. 
