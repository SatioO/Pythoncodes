
Object detection is hard because of substantial variation in
- viewpoint
- scale
- deformation
- occlusion
- illumination
- background clutter
- intra-class variation


- Template Matching
- Multiscale Template Matching
- Custom Object detection framework
        - Scan images at all scales and locations
        - Extract features over each sliding window location
        - Using linear svm to classify features extracted from each window
        - Apply non-maxima suppression to obtain final bounding boxes

Object detection using
 - keypoints
 - local invariant descriptors
 - bag of visual words models
 - Histogram of Oriented Gradients
 - Deformable parts model
 - Exemplar models
