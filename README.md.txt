# Neural Network from Scratch (NumPy)

## Project Overview
This project demonstrates the implementation of a **basic feedforward neural network** from scratch using **only NumPy**. The network is trained to recognize **letters A, B, and C** represented as 5×6 binary pixel images. No external machine learning libraries (TensorFlow, PyTorch, etc.) were used, giving hands-on experience with the core concepts of neural networks.

---

## Approach

1. **Data Representation**  
   - Each letter (A, B, C) is represented as a **5×6 grid** of binary pixels.
   - Flattened into a **1D array of 30 elements** for input to the network.
   - Labels are one-hot encoded:
     ```
     A → [1, 0, 0]
     B → [0, 1, 0]
     C → [0, 0, 1]
     ```

2. **Network Architecture**  
   - **Input layer:** 30 neurons (pixels)  
   - **Hidden layer:** 10 neurons, sigmoid activation  
   - **Output layer:** 3 neurons, sigmoid activation  
   - **Loss function:** Mean Squared Error (MSE)  
   - **Optimizer:** Gradient Descent using Backpropagation

3. **Training Process**  
   - Perform **forward pass**: calculate activations for hidden and output layers.  
   - Compute **loss** between predicted output and actual labels.  
   - Perform **backpropagation** to update weights and biases.  
   - Repeat for multiple **epochs** until loss converges.  

---

## Analysis Process

- **Forward Pass:** Computes network output using current weights.  
- **Backpropagation:** Computes gradients using the derivative of sigmoid activation and updates weights.  
- **Accuracy Tracking:** Accuracy calculated by comparing predicted labels with true labels.  
- **Visualization:**  
  - Letters visualized using `matplotlib.pyplot.imshow()`.  
  - Training loss plotted over epochs.

Example: Training loss plot  

![Training Loss](images/training_loss.png)  

Predicted Letter Example:  

![Predicted Letter A](images/prediction_A.png)  

---

## Key Findings

- The network **successfully learned** the patterns for letters A, B, and C.  
- With only three training samples, the network achieved **100% accuracy** on the training set.  
- Visualization confirmed that the network correctly identifies the input letters.  
- This project demonstrates the **fundamentals of feedforward networks, sigmoid activation, backpropagation, and gradient descent**.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/NeuralNetworkFromScratch.git
cd NeuralNetworkFromScratch
