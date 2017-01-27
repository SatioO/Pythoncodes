Image Classification Algorithms

- Image Classifcation
- Challenges
- semantic gap - the inherent disconnect between how humans and computers interptet images
- object/image variations an image classification system should be to handle and tolerate
   - Viewpoint variation
   - Scale variation
   - Deformation
   - occlusions
   - illumination
   - background clutter
   - intra-class variation

Types of learning
- Supervised learning
- Unsupervised learning
- Semi-supervised learning

Image Classification pipeline
##Dataset of Images ---> Training/Testing Split ---> Feature Extraction ---> Training classifier ---> Evaluate Classifier
- Structuring your initial dataset.
- Splitting the dataset into two (optionally three) parts.
- Extracting features.
- Training your classification model.
- Evaluating your classifier.

Algorithms
- KNN
- Logistic
- SVM
- Trees

{
"image_dataset" :"datasets/ct101",

"kp_detector": "GFTT",
"descriptor":"RootSIFT",
"features_path":"output/ct101/features.hdf5",

"vocab_path":"output/ct101/vocab.cpickle",
"bovw_path":"output/ct101/bovw.hdf5",
"vocab_sizes":[32,64,128,256,512,1024,2048,4096],
"sample_size":0.1,
"num_passes":3,

"classifier_path":"output/ct101/model.cpickle",
"accuracies_path":"output/ct101/accuracies.cpickle",
"train_size":0.75,

}

##Tips
- weather to use global features or local features
- Document the following things
   - Which features did i use?
   - What Machine learning Algorithm did i use ?
   - What were my results?
   - How might I imporve them in the future?
