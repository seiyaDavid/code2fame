"""


# Training Epochs vs. Number of Optimization Trials

These are two different concepts in your synthetic data generation application:

## Training Epochs

**Training Epochs** refers to how many times the model (CTGAN, TVAE, or CopulaGAN) will iterate through the entire dataset during training.

- **What it does**: Each epoch represents one complete pass through all your data
- **Effect on model**: More epochs generally allow the model to learn the data patterns better
- **Trade-off**: More epochs = better learning but longer training time and potential overfitting
- **Typical values**: 100-300 epochs for synthetic data generation
- **When to increase**: When your synthetic data doesn't capture the patterns in your original data
- **When to decrease**: When training takes too long or when the model is memorizing the original data

## Number of Optimization Trials

**Number of Optimization Trials** refers to how many different combinations of hyperparameters Optuna will test when you enable the "Optimize Model Parameters" option.

- **What it does**: Each trial tests a different set of hyperparameters (epochs, batch size, network architecture, etc.)
- **Effect on model**: More trials help find better hyperparameter combinations
- **Trade-off**: More trials = better parameters but much longer total processing time
- **Typical values**: 10-30 trials is usually sufficient
- **When to increase**: When you want to find the absolute best parameters and have time to wait
- **When to decrease**: When you need results quickly and are willing to accept "good enough" parameters

## Key Differences

1. **Purpose**:
   - Epochs: How long to train a single model
   - Trials: How many different model configurations to test

2. **Time Impact**:
   - 10 epochs = 10× through your data
   - 10 trials = 10 different models trained (each with their own epochs)

3. **What's Being Changed**:
   - Epochs: Just the weights in the neural network
   - Trials: The entire model architecture and training parameters

When you enable optimization, the epochs parameter becomes one of the hyperparameters that Optuna will tune, so the final model might use a different number of epochs than what you initially specified.



These additions will provide a more comprehensive privacy assessment of the 
synthetic data, including advanced metrics that are commonly used in 
privacy-preserving data synthesis. The membership attack simulation tests how 
well an adversary could determine if a record was part of the original dataset, 
the differential privacy estimate provides an approximation of the privacy leakage, 
and the k-anonymity distance measures how well the synthetic data preserves anonymity.

Understanding K-Anonymity Distance of 0.56
A K-Anonymity Distance of 0.56 indicates a moderate to strong level of privacy protection in your synthetic data. Let me explain what this metric means:
What is K-Anonymity Distance?
K-Anonymity Distance measures how far synthetic data points are from their k-nearest neighbors in the original dataset, after standardization (scaling the data to have zero mean and unit variance).
In your implementation:
We use k=5 (or less if there's not enough data)
We measure the Euclidean distance in the standardized feature space
The value 0.56 represents the average minimum distance to the k-th neighbor
Interpreting the Value of 0.56
Scale: The distances are measured in standardized space, where a distance of 1.0 roughly corresponds to one standard deviation in the original data
Privacy Implication: A value of 0.56 means that, on average, synthetic records are more than half a standard deviation away from their 5th nearest neighbor in the original data
Comparison Point:
Values below 0.1 would indicate very low privacy protection
Values between 0.1-0.5 indicate moderate protection
Values above 0.5 (like yours) suggest strong privacy protection
What This Means for Your Data
With a K-Anonymity Distance of 0.56:
Good Privacy Protection: Your synthetic data points are sufficiently distant from clusters of original data points
Low Re-identification Risk: It would be difficult to link synthetic records back to specific original records
Good Generalization: The model has learned patterns rather than memorizing specific data points
This suggests your synthetic data generation process is working well from a privacy perspective - it's creating data that maintains statistical properties while keeping a safe distance from the original records.
"""
