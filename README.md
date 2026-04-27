## Data Augmentation & Balancing

We applied data augmentation only on the training dataset to improve generalization and reduce overfitting.

The augmentation techniques included:

* Random horizontal flipping
* Rotation (±15°)
* Zoom (0.15)
* Rescaling (1./255)

The dataset was loaded using TensorFlow `image_dataset_from_directory` with a batch size of 16.

Prefetching was used to optimize performance.

Class imbalance was handled using class weights computed from the training data.

Validation and test datasets were not augmented to ensure fair evaluation.
