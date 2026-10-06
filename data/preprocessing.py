import numpy as np

#Applies Z-score standardization exclusively on training partitions
class ClimateNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    #Calculates the average and the standard deviation from the 12 variables
    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)  # Per-feature mean across the time axis
        self.std = np.std(data, axis=0)    # Per-feature std across the time axis
        self.std[self.std==0.0] = 1.0 #Prevents division by zero if a feature has constant values

    #Standardizes the dataset using pre computed training mean and standard deviaton. Calculate the normanlized value =raw data-average/speed
    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("ClimateNormalizer must be fitted before transform.")
        return (data - self.mean) / (self.std + 1e-8)

    #As the model finishes prediction the model is scaled down to normalize the numbers and these numbers is translated back into real-world units
    def inverse_transform(self,data:np.ndarray)-> np.ndarray:
        return (data*self.std) + self.mean


