import numpy as np
import matplotlib.pyplot as plt

#Mean Squared Error Loss Function

def mse_loss(y_true,y_pred):
    return np.mean(np.power(y_true-y_pred,2))

#Binary Cross Entropy Loss Function

def bce_loss(y_true,y_pred):
    y_pred = np.clip(y_pred,1e-15,1-1e-15)
    return -np.mean(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))


#Example Data

y_true = np.array([1,0,1,1])
y_pred = np.array([0.9,0.2,0.8,0.7])

#Calculate Loss

mse = mse_loss(y_true,y_pred)
bce = bce_loss(y_true,y_pred)

print(f"Mean Squared Error Loss: {mse:.4f}")
print(f"Binary Cross Entropy Loss: {bce:.4f}")

#Derivative of MSE loss
def mse_gradient(y_true,y_pred):
    return 2*(y_pred-y_true)/len(y_true)

#Derivative of BCE loss

def bce_gradient(y_true,y_pred):
    y_pred = np.clip(y_pred,1e-15,1-1e-15)
    return (y_pred-y_true)/(y_pred*(1-y_pred))

#Calculate Gradient

grad_mse = mse_gradient(y_true,y_pred)
grad_bce = bce_gradient(y_true,y_pred)

print(f"Mean Squared Error Gradient: {grad_mse}")
print(f"Binary Cross Entropy Gradient: {grad_bce}")


#Define Predictions and true labels

predictions = np.linspace(0,1,100)
true_label = 1

#Compute losses

mse_losses = [(true_label - p)**2 for p in predictions]
bse_losses = [-true_label*np.log(max(p,1e-15)) - (1-true_label)*np.log(max(1-p,1e-15)) for p in predictions] 

#Plot Losses
plt.figure(figsize=(10,5))
plt.plot(predictions,mse_losses,label="Mean Squared Error")
plt.plot(predictions,bse_losses,label="Binary Cross Entropy")
plt.title("Loss Function Comparison")
plt.xlabel("Predictions")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()


