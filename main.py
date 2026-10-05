# Import the os module for interacting with the operating system
import os
# Import zipfile to compress files into zip archives
import zipfile
# Import NumPy for array operations and linear algebra calculations
import numpy as np
# Import Matplotlib's pyplot module for rendering visual plots
import matplotlib.pyplot as plt

# =========================================================================
# 1. Core Mathematical Helpers & Matrix Operations
# =========================================================================
# Define the Rectified Linear Unit activation function
def ReLU(Z):
    # Return 0 for negative values, and pass positive values through as-is
    return np.maximum(0, Z)

# Define the derivative of the ReLU function for backpropagation
def ReLU_deriv(Z):
    # Return True (1) for positive elements and False (0) otherwise
    return Z > 0

# Define the Softmax function to convert raw scores into probabilities
def softmax(Z):
    # Subtract max value per column for numerical stability, then exponentiate
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    # Divide each exponentiated value by the sum along the column to normalize
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Define a function to convert class indices into one-hot encoded vectors
def one_hot(Y, num_classes=10):
    # Create a matrix of zeros matching sample count and class count
    one_hot_Y = np.zeros((Y.size, num_classes))
    # Set index 1 at the position corresponding to the label for each row
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Transpose matrix so classes are along rows and samples along columns
    return one_hot_Y.T

# Define the combined Softmax Cross-Entropy Loss and gradient computation
def softmax_cross_entropy_loss(logits, y_indices):
    # Get total number of samples from the batch size
    N = logits.shape[0]
    # Subtract max for numerical stability before computing exponents
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    # Normalize exponent values into probabilities across classes
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Compute average cross-entropy loss over all batch samples
    loss = -np.sum(np.log(probs[np.arange(N), y_indices] + 1e-12)) / N

    # Copy probabilities matrix to construct gradient tensor
    dlogits = probs.copy()
    # Subtract 1 from correct class position according to cross-entropy derivative
    dlogits[np.arange(N), y_indices] -= 1
    # Average gradient across the batch dimension
    dlogits /= N

    # Return total scalar loss, probabilities matrix, and gradient tensor
    return loss, probs, dlogits


# =========================================================================
# 2. Vectorized im2col / col2im Operations
# =========================================================================
# Helper function to generate lookup index grids for fast 2D convolutions
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # Unpack dimensions: Batch Size (N), Channels (C), Height (H), Width (W)
    N, C, H, W = x_shape
    # Calculate output feature map height based on spatial dimensions
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    # Calculate output feature map width based on spatial dimensions
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    # Create row index pattern for filter kernel
    i0 = np.repeat(np.arange(field_height), field_width)
    # Tile row indices across input channels
    i0 = np.tile(i0, C)
    # Scale vertical spatial shifts by stride
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    # Create column index pattern for filter kernel
    j0 = np.tile(np.arange(field_width), field_height * C)
    # Scale horizontal spatial shifts by stride
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    # Broadcast row indices to match kernel positions across patches
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    # Broadcast column indices to match kernel positions across patches
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    # Repeat channel indices across total receptive field area
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    # Return calculated channel, row, and column index matrices
    return k.astype(int), i.astype(int), j.astype(int)


# Rearrange image patches into matrix columns for fast matrix multiplication
def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    # Set padding size shortcut variable
    p = padding
    # Apply zero-padding around the border of input feature maps
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    # Retrieve index matrices for receptive field extraction
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    # Slice receptive field patches using calculated index arrays
    cols = x_padded[:, k, i, j]
    # Extract channel size of input array
    C = x.shape[1]
    # Reshape extracted patches into 2D matrix columns
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    # Return vectorized columns
    return cols


# Convert column gradients back into image gradient tensors
def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    # Unpack dimensions from input array shape tuple
    N, C, H, W = x_shape
    # Add zero-padding dimensions to target image height and width
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    # Allocate zero array matching padded input shape
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    # Get index matrices matching padded image dimensions
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    # Reshape column vector back into structured patch blocks
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    # Transpose batch dimension to fit padded tensor indices
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    # Accumulate patch gradients back into padded input tensor positions
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    # Return padded gradient directly if padding was zero
    if padding == 0:
        return x_padded
    # Crop out applied padding borders to return original image dimensions
    return x_padded[:, :, padding:-padding, padding:-padding]


# =========================================================================
# 3. Modular Deep Architecture Layers
# =========================================================================
# Define 2D Convolutional Layer class
class Conv2D:
    # Initialize layer with channel counts, kernel dimensions, stride, and padding
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # Store input channel count
        self.in_channels = in_channels
        # Store output channel count (number of filters)
        self.out_channels = out_channels
        # Store square kernel size dimension
        self.kernel_size = kernel_size
        # Store convolution stride
        self.stride = stride
        # Store convolution padding size
        self.padding = padding

        # Initialize filter weights with He Normal scaling
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        # Initialize bias vector with zeros
        self.b = np.zeros((out_channels, 1))

        # Declare gradient variables for weights and biases
        self.dW, self.db = None, None
        # Initialize Adam momentum vectors with zeros
        self.v_dW, self.s_dW = np.zeros_like(self.W), np.zeros_like(self.W)
        # Initialize Adam velocity vectors with zeros
        self.v_db, self.s_db = np.zeros_like(self.b), np.zeros_like(self.b)

    # Perform forward pass convolution
    def forward(self, X):
        # Unpack input tensor dimensions
        N, C, H, W = X.shape
        # Calculate spatial output height
        out_h = int((H + 2 * self.padding - self.kernel_size) / self.stride + 1)
        # Calculate spatial output width
        out_w = int((W + 2 * self.padding - self.kernel_size) / self.stride + 1)

        # Convert input images into 2D column matrix
        X_col = im2col_indices(X, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)
        # Flatten filter weights into 2D row matrix
        W_row = self.W.reshape(self.out_channels, -1)

        # Perform matrix multiplication and add bias vector
        out = W_row.dot(X_col) + self.b
        # Reshape matrix output back into 4D tensor format
        out = out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)

        # Store input tensors in cache for gradient calculation
        self.cache = (X, X_col, W_row)
        # Return output feature maps
        return out

    # Perform backward pass to calculate layer gradients
    def backward(self, dout):
        # Retrieve cached forward pass variables
        X, X_col, W_row = self.cache
        # Extract batch size from input shape
        N = X.shape[0]

        # Reshape upstream gradient to align with matrix operations
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        # Calculate weight gradients via matrix dot product
        self.dW = dout_reshaped.dot(X_col.T).reshape(self.W.shape) / N
        # Calculate bias gradients by summing across spatial dimensions
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True) / N

        # Propagate loss gradient down to layer input columns
        dX_col = W_row.T.dot(dout_reshaped)
        # Convert column gradients back into input tensor spatial dimensions
        dX = col2im_indices(dX_col, X.shape, self.kernel_size, self.kernel_size, padding=self.padding, stride=self.stride)

        # Return gradient with respect to layer input
        return dX


# Define Max Pooling 2D Layer class
class MaxPool2D:
    # Initialize pooling layer with pool size and stride
    def __init__(self, pool_size=2, stride=2):
        # Store pooling window dimension
        self.pool_size = pool_size
        # Store step stride
        self.stride = stride

    # Perform forward max pooling pass
    def forward(self, X):
        # Unpack input feature map dimensions
        N, C, H, W = X.shape
        # Compute pooled output spatial height
        out_h = int((H - self.pool_size) / self.stride + 1)
        # Compute pooled output spatial width
        out_w = int((W - self.pool_size) / self.stride + 1)

        # Flatten channels into batch dimension for pooling
        X_reshaped = X.reshape(N * C, 1, H, W)
        # Extract patch columns using im2col helper
        X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)

        # Identify index of maximum value per patch column
        max_idx = np.argmax(X_col, axis=0)
        # Extract maximum values using calculated indices
        out = X_col[max_idx, np.arange(max_idx.size)]
        # Reshape max values into 4D output tensor
        out = out.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        # Cache variables needed for backpropagation
        self.cache = (X, X_col, max_idx)
        # Return pooled output tensor
        return out

    # Perform backward pass for max pooling
    def backward(self, dout):
        # Unpack cached variables
        X, X_col, max_idx = self.cache
        # Get shape dimensions of input feature map
        N, C, H, W = X.shape

        # Create zero array matching spatial column shape
        dX_col = np.zeros_like(X_col)
        # Flatten upstream gradient tensor into 1D array
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()

        # Route gradients exclusively to original max index positions
        dX_col[max_idx, np.arange(max_idx.size)] = dout_flat
        # Reconstruct spatial image gradient tensor from column data
        dX = col2im_indices(dX_col, (N * C, 1, H, W), self.pool_size, self.pool_size, padding=0, stride=self.stride)
        
        # Reshape gradient tensor back to full input batch shape
        return dX.reshape(N, C, H, W)


# Define Flatten layer to reshape 4D tensors into 2D matrices
class Flatten:
    # Reshape input tensor to (batch_size, features)
    def forward(self, X):
        # Save original 4D tensor shape for reconstruction
        self.orig_shape = X.shape
        # Extract batch size
        N = X.shape[0]
        # Reshape features to 2D
        return X.reshape(N, -1)

    # Reshape gradient tensor back to original 4D tensor shape
    def backward(self, dout):
        # Restore saved shape
        return dout.reshape(self.orig_shape)


# Define Fully Connected (Dense) Layer class
class DenseLayer:
    # Initialize weights and biases for dense connections
    def __init__(self, in_features, out_features):
        # Initialize weight matrix using He Normal scaling
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        # Initialize bias vector to zeros
        self.b = np.zeros((out_features, 1))

        # Declare gradient storage
        self.dW, self.db = None, None
        # Initialize Adam momentum arrays
        self.v_dW, self.s_dW = np.zeros_like(self.W), np.zeros_like(self.W)
        # Initialize Adam velocity arrays
        self.v_db, self.s_db = np.zeros_like(self.b), np.zeros_like(self.b)

    # Compute forward dot product pass
    def forward(self, X):
        # Save input matrix for backpropagation
        self.X = X
        # Perform linear transform: X * W^T + b
        return X.dot(self.W.T) + self.b.T

    # Compute layer parameter gradients during backward pass
    def backward(self, dout):
        # Extract batch size N
        N = dout.shape[0]
        # Compute average weight gradient matrix
        self.dW = dout.T.dot(self.X) / N
        # Compute average bias gradient vector
        self.db = np.sum(dout.T, axis=1, keepdims=True) / N
        # Propagate loss gradient back to upstream layer inputs
        return dout.dot(self.W)


# Define Activation Layer for ReLU
class ReLULayer:
    # Pass inputs through ReLU function during forward pass
    def forward(self, X):
        # Cache input tensor
        self.X = X
        # Apply element-wise max thresholding
        return np.maximum(0, X)

    # Zero out gradients corresponding to negative forward inputs
    def backward(self, dout):
        # Multiply gradient by derivative mask
        return dout * (self.X > 0)


# =========================================================================
# 4. Neural Network Assembly & Main Pipeline
# =========================================================================
# Define sequential CNN architecture class
class NumPyCNN:
    # Construct layers in sequential order
    def __init__(self, in_channels=1, num_classes=3):
        # Conv layer: 1 input channel -> 8 output filters
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        # Activation for conv layer
        self.relu1 = ReLULayer()
        # Downsample spatial dimension by 2x
        self.pool1 = MaxPool2D(pool_size=2, stride=2)
        
        # Flatten tensor for dense layers
        self.flatten = Flatten()
        # Dense layer: 392 features -> 32 hidden units
        self.dense1 = DenseLayer(in_features=392, out_features=32)
        # Activation for hidden dense layer
        self.relu2 = ReLULayer()
        # Output classification dense layer
        self.dense2 = DenseLayer(in_features=32, out_features=num_classes)

        # Group all network layers into an ordered list
        self.layers = [self.conv1, self.relu1, self.pool1, self.flatten, self.dense1, self.relu2, self.dense2]
        # Track total optimization step counter for Adam
        self.t = 0

    # Sequential forward pass through all layers
    def forward(self, X):
        # Set initial input data
        out = X
        # Pass data through each layer sequentially
        for layer in self.layers:
            out = layer.forward(out)
        # Return final raw output scores (logits)
        return out

    # Sequential backward pass through layers in reverse order
    def backward(self, dloss):
        # Set starting loss gradient
        dout = dloss
        # Iterate backward through layers
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    # Update trainable parameters using the Adam optimizer algorithm
    def update_adam(self, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        # Increment time step counter
        self.t += 1
        # Loop over layers with trainable parameters
        for layer in [self.conv1, self.dense1, self.dense2]:
            # Loop over weight and bias parameter sets
            for param, grad, v, s in [
                (layer.W, layer.dW, layer.v_dW, layer.s_dW),
                (layer.b, layer.db, layer.v_db, layer.s_db)
            ]:
                # Update first moment (momentum) estimate
                v[...] = beta1 * v + (1 - beta1) * grad
                # Update second moment (uncentered variance) estimate
                s[...] = beta2 * s + (1 - beta2) * (grad ** 2)
                # Compute bias-corrected first moment estimate
                v_corr = v / (1 - beta1 ** self.t)
                # Compute bias-corrected second moment estimate
                s_corr = s / (1 - beta2 ** self.t)
                # Update parameter values in-place
                param -= lr * v_corr / (np.sqrt(s_corr) + eps)


# Generate synthetic visual dataset (Horizontal, Vertical, and Diagonal lines)
def generate_synthetic_data(num_samples=360):
    # Allocate Gaussian noise input images tensor
    X = np.random.randn(num_samples, 1, 14, 14) * 0.1
    # Generate random ground-truth class labels from 0 to 2
    y = np.random.randint(0, 3, size=num_samples)

    # Add geometric pixel patterns based on class label
    for i in range(num_samples):
        # Label 0: Draw horizontal line
        if y[i] == 0:
            X[i, 0, 5:9, :] += 1.0   # Horizontal bar
        # Label 1: Draw vertical line
        elif y[i] == 1:
            X[i, 0, :, 5:9] += 1.0   # Vertical bar
        # Label 2: Draw diagonal line
        elif y[i] == 2:
            np.fill_diagonal(X[i, 0], X[i, 0].diagonal() + 1.0)  # Diagonal line

    # Return dataset image array and label vector
    return X, y

def visualize_feature_maps(cnn, X_test, y_test, num_samples=3):
    class_names = {0: "Horizontal Bar", 1: "Vertical Bar", 2: "Diagonal Line"}
    
    unique_classes = np.unique(y_test)
    sample_indices = []
    for c in unique_classes:
        idx = np.where(y_test == c)[0][0]
        sample_indices.append(idx)
        if len(sample_indices) == num_samples:
            break

    X_samples = X_test[sample_indices]
    
    conv1_out = cnn.conv1.forward(X_samples)
    relu1_out = cnn.relu1.forward(conv1_out)

    num_filters = cnn.conv1.out_channels
    
    fig, axes = plt.subplots(
        nrows=len(sample_indices), 
        ncols=num_filters + 1, 
        figsize=(15, 2.5 * len(sample_indices))
    )

    for row_idx, sample_idx in enumerate(sample_indices):
        orig_img = X_test[sample_idx, 0]
        label = y_test[sample_idx]
        
        ax_orig = axes[row_idx, 0] if len(sample_indices) > 1 else axes[0]
        ax_orig.imshow(orig_img, cmap='gray')
        ax_orig.set_title(f"Input: {class_names.get(label, label)}", fontsize=8, fontweight='bold')
        ax_orig.axis('off')
        
        for f in range(num_filters):
            fmap = relu1_out[row_idx, f]
            fmap_norm = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            
            ax_fmap = axes[row_idx, f + 1] if len(sample_indices) > 1 else axes[f + 1]
            ax_fmap.imshow(fmap_norm, cmap='viridis')
            ax_fmap.set_title(f"Filter {f + 1}", fontsize=8)
            ax_fmap.axis('off')

    plt.suptitle("Conv2D Internal Feature Map Activations (Post-ReLU)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()

# Main execution entry point
if __name__ == "__main__":
    # Print status message
    print("--- Starting LightEngine AI Framework Pipeline ---")
    # Fix random seed for reproducible results
    np.random.seed(42)

    # Generate synthetic dataset samples
    X_data, y_data = generate_synthetic_data(num_samples=360)
    # Split into training set features and labels
    X_train, y_train = X_data[:300], y_data[:300]
    # Split into test set features and labels
    X_test, y_test = X_data[300:], y_data[300:]

    # Instantiate model instance
    cnn = NumPyCNN(in_channels=1, num_classes=3)
    # Set epoch count
    epochs = 10
    # Set batch size
    batch_size = 32

    # Loop over training epochs
    for epoch in range(1, epochs + 1):
        # Shuffle dataset order per epoch
        perm = np.random.permutation(len(X_train))
        # Reorder training feature data
        X_shuffled = X_train[perm]
        # Reorder training label data
        y_shuffled = y_train[perm]
        # Compute batch iterations count
        num_batches = len(X_train) // batch_size
        # Track epoch loss accumulator
        epoch_loss = 0.0

        # Loop over training mini-batches
        for b in range(num_batches):
            # Extract mini-batch feature inputs
            bx = X_shuffled[b * batch_size : (b + 1) * batch_size]
            # Extract mini-batch target labels
            by = y_shuffled[b * batch_size : (b + 1) * batch_size]

            # Compute predictions forward pass
            logits = cnn.forward(bx)
            # Compute cross-entropy loss and starting gradients
            loss, _, dlogits = softmax_cross_entropy_loss(logits, by)
            # Add batch loss to epoch total
            epoch_loss += loss

            # Perform error backpropagation
            cnn.backward(dlogits)
            # Update layer weights via Adam optimizer
            cnn.update_adam(lr=0.003)

        # Compute average training loss
        avg_loss = epoch_loss / num_batches
        # Evaluate model predictions on unseen test dataset
        test_logits = cnn.forward(X_test)
        # Compute test set accuracy percentage
        acc = np.mean(np.argmax(test_logits, axis=1) == y_test)
        # Print epoch metrics to stdout
        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Test Accuracy: {acc * 100:.2f}%")

    # Export learned layer parameter arrays to numpy archive file
    np.savez(
        'cnn_model_weights.npz',
        conv1_W=cnn.conv1.W, conv1_b=cnn.conv1.b,
        dense1_W=cnn.dense1.W, dense1_b=cnn.dense1.b,
        dense2_W=cnn.dense2.W, dense2_b=cnn.dense2.b
    )
    # Print export completion confirmation message
    print("\nModel weights successfully saved to 'cnn_model_weights.npz'.")
    visualize_feature_maps(cnn,X_test,y_test, num_samples = 3)
