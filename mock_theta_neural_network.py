import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional
import time


class MockThetaFunction:
    def __init__(self, q: complex, max_n: int = 50):
        """
        Initialize the mock theta function with a complex q and number of terms.
        :param q: complex base of the q-series (|q| < 1 for convergence)
        :param max_n: number of terms in the series
        """
        self.q = q
        self.max_n = max_n
    
    def f(self) -> complex:
        """
        Ramanujan's third-order mock theta function f(q)
        :return: Approximate value of f(q)
        """
        q = self.q
        result = 1.0  # starting with n = 0 term
        
        for n in range(1, self.max_n + 1):
            numerator = q**(n**2)
            denominator = np.prod([(1 + q**k)**2 for k in range(1, n + 1)])
            result += numerator / denominator
        
        return result
    
    def compute_series_terms(self) -> list:
        """
        Compute individual terms of the mock theta series
        :return: List of series terms
        """
        q = self.q
        terms = [1.0]  # n = 0 term
        
        for n in range(1, self.max_n + 1):
            numerator = q**(n**2)
            denominator = np.prod([(1 + q**k)**2 for k in range(1, n + 1)])
            terms.append(numerator / denominator)
        
        return terms


class NeuralNetwork(MockThetaFunction, nn.Module):
    """
    Neural network that inherits from MockThetaFunction, demonstrating polymorphism
    by using mock theta function properties as network features or initialization
    """
    
    def __init__(self, q: complex, max_n: int = 50, input_size: int = 10, 
                 hidden_size: int = 64, output_size: int = 1):
        # Initialize both parent classes
        MockThetaFunction.__init__(self, q, max_n)
        nn.Module.__init__(self)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights using mock theta function properties
        self._initialize_weights_with_theta()
    
    def _initialize_weights_with_theta(self):
        """
        Initialize network weights using properties derived from the mock theta function
        """
        # Compute mock theta value for weight initialization
        theta_value = self.f()
        
        # Use the real part of the theta function for initialization scaling
        scale_factor = abs(theta_value.real) if theta_value.real != 0 else 0.1
        
        # Initialize weights with theta-based scaling
        with torch.no_grad():
            for layer in [self.fc1, self.fc2, self.fc3]:
                nn.init.xavier_uniform_(layer.weight, gain=scale_factor)
                nn.init.constant_(layer.bias, theta_value.real * 0.01)
    
    def forward(self, x):
        """
        Forward pass of the neural network
        :param x: Input tensor
        :return: Output tensor
        """
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        
        return x
    
    def f(self) -> complex:
        """
        Override the parent's f() method to demonstrate polymorphism
        Can add neural network specific modifications to the theta function
        """
        # Call parent's method
        base_result = super().f()
        
        # Add neural network specific modification (example)
        # This could incorporate learned parameters or network state
        modification = 1.0 + (self.hidden_size / 1000.0)  # Simple example
        
        return base_result * modification
    
    def get_theta_features(self) -> torch.Tensor:
        """
        Extract features from the mock theta function for use in the network
        :return: Tensor of theta-based features
        """
        # Get series terms
        terms = self.compute_series_terms()
        
        # Convert to real-valued features (take real and imaginary parts)
        features = []
        for term in terms[:self.input_size//2]:  # Limit to input size
            if isinstance(term, complex):
                features.extend([term.real, term.imag])
            else:
                features.extend([float(term), 0.0])
        
        # Pad or truncate to match input size
        while len(features) < self.input_size:
            features.append(0.0)
        features = features[:self.input_size]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def theta_informed_prediction(self, x):
        """
        Make predictions incorporating both input data and theta function features
        :param x: Input tensor
        :return: Prediction tensor
        """
        # Get theta-based features
        theta_features = self.get_theta_features().unsqueeze(0)  # Add batch dimension
        
        # Combine with input (simple concatenation approach)
        if x.size(1) == self.input_size:
            # Weighted combination of input and theta features
            combined_input = 0.7 * x + 0.3 * theta_features
        else:
            combined_input = x
        
        return self.forward(combined_input)
    
    def train_epochs(self, train_loader: DataLoader, criterion, optimizer, 
                    num_epochs: int = 100, val_loader: Optional[DataLoader] = None,
                    print_every: int = 10, early_stopping_patience: int = None) -> dict:
        """
        Train the neural network for a specified number of epochs
        
        :param train_loader: DataLoader for training data
        :param criterion: Loss function
        :param optimizer: Optimizer for training
        :param num_epochs: Number of epochs to train
        :param val_loader: Optional DataLoader for validation data
        :param print_every: Print training progress every N epochs
        :param early_stopping_patience: Stop training if validation loss doesn't improve for N epochs
        :return: Dictionary containing training history
        """
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'epoch_times': [],
            'theta_values': []  # Track how theta function evolves during training
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Initial theta function value: {self.f()}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                output = self(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                if self.output_size == 1:  # Regression
                    train_total += target.size(0)
                else:  # Classification
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
            
            # Calculate average training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total if train_total > 0 else 0.0
            
            training_history['train_losses'].append(avg_train_loss)
            training_history['train_accuracies'].append(train_accuracy)
            training_history['theta_values'].append(complex(self.f()))
            
            # Validation phase
            val_loss = 0.0
            val_accuracy = 0.0
            if val_loader:
                val_loss, val_accuracy = self._validate(val_loader, criterion)
                training_history['val_losses'].append(val_loss)
                training_history['val_accuracies'].append(val_accuracy)
                
                # Early stopping check
                if early_stopping_patience:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping at epoch {epoch + 1} due to no improvement in validation loss")
                        break
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            training_history['epoch_times'].append(epoch_time)
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1:4d}/{num_epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Train Acc: {train_accuracy:.4f} | "
                      + (f"Val Loss: {val_loss:.6f} | Val Acc: {val_accuracy:.4f} | " if val_loader else "") +
                      f"Time: {epoch_time:.2f}s | "
                      f"Theta: {self.f():.4f}")
        
        print("-" * 60)
        print(f"Training completed! Final theta function value: {self.f()}")
        
        return training_history
    
    def _validate(self, val_loader: DataLoader, criterion) -> Tuple[float, float]:
        """
        Validate the model on validation data
        
        :param val_loader: DataLoader for validation data
        :param criterion: Loss function
        :return: Tuple of (validation_loss, validation_accuracy)
        """
        self.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                
                if self.output_size == 1:  # Regression
                    val_total += target.size(0)
                else:  # Classification
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        return avg_val_loss, val_accuracy
    
    def plot_training_history(self, history: dict, save_path: Optional[str] = None):
        """
        Plot training history including loss, accuracy, and theta function evolution
        
        :param history: Training history dictionary from train_epochs
        :param save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Plot training and validation loss
        axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss')
        if history['val_losses']:
            axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training and validation accuracy
        axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
        if history['val_accuracies']:
            axes[0, 1].plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot theta function evolution (real part)
        theta_real = [theta.real for theta in history['theta_values']]
        axes[1, 0].plot(epochs, theta_real, 'g-', label='Theta Function (Real)')
        axes[1, 0].set_title('Mock Theta Function Evolution (Real Part)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Theta Value (Real)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot epoch times
        axes[1, 1].plot(epochs, history['epoch_times'], 'm-', label='Epoch Time')
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str, training_history: Optional[dict] = None):
        """
        Save the trained model and optionally the training history
        
        :param filepath: Path to save the model
        :param training_history: Optional training history to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'q_value': complex(self.q),  # Ensure it's serializable
            'max_n': int(self.max_n),
            'input_size': int(self.input_size),
            'hidden_size': int(self.hidden_size),
            'output_size': int(self.output_size),
            'theta_value': complex(self.f()),
        }
        
        if training_history:
            # Convert complex values to serializable format
            serializable_history = training_history.copy()
            if 'theta_values' in serializable_history:
                serializable_history['theta_values'] = [complex(val) for val in serializable_history['theta_values']]
            checkpoint['training_history'] = serializable_history
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a saved model
        
        :param filepath: Path to the saved model
        :return: Loaded NeuralNetwork instance
        """
        try:
            # First try with weights_only=True (secure)
            checkpoint = torch.load(filepath, weights_only=True)
        except Exception:
            # Fall back to weights_only=False if needed (less secure but compatible)
            try:
                checkpoint = torch.load(filepath, weights_only=False)
            except Exception as e:
                print(f"Failed to load model: {e}")
                raise
        
        # Create new instance
        model = cls(
            q=checkpoint['q_value'],
            max_n=checkpoint['max_n'],
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size']
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint.get('training_history', None)


# Example usage and demonstration
def demonstrate_polymorphism():
    """
    Demonstrate polymorphism between MockThetaFunction and NeuralNetwork
    """
    # Create instances
    q = 0.5 + 0.1j  # Complex q value with |q| < 1
    
    # Original mock theta function
    theta_func = MockThetaFunction(q, max_n=20)
    
    # Neural network inheriting from mock theta function
    neural_net = NeuralNetwork(q, max_n=20, input_size=10, hidden_size=32, output_size=1)
    
    print("=== Demonstrating Polymorphism ===")
    
    # Both objects can call f() method, but with different behaviors
    print(f"MockThetaFunction f(): {theta_func.f()}")
    print(f"NeuralNetwork f(): {neural_net.f()}")
    
    # Neural network has additional capabilities
    print(f"\nNeural Network Architecture:")
    print(f"Input size: {neural_net.input_size}")
    print(f"Hidden size: {neural_net.hidden_size}")
    print(f"Output size: {neural_net.output_size}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 10)  # Batch size 1, input size 10
    output = neural_net(dummy_input)
    print(f"\nNetwork output shape: {output.shape}")
    print(f"Network output: {output.item():.4f}")
    
    # Test theta-informed prediction
    theta_output = neural_net.theta_informed_prediction(dummy_input)
    print(f"Theta-informed output: {theta_output.item():.4f}")


def create_sample_data(n_samples: int = 1000, input_size: int = 10, noise_level: float = 0.1):
    """
    Create sample regression data for testing the neural network
    
    :param n_samples: Number of samples to generate
    :param input_size: Size of input features
    :param noise_level: Amount of noise to add to targets
    :return: Tuple of (X, y) tensors
    """
    # Generate random input data
    X = torch.randn(n_samples, input_size)
    
    # Create a simple target function (sum of squares with some weights)
    weights = torch.randn(input_size) * 0.5
    y = torch.sum(X * weights, dim=1, keepdim=True) + noise_level * torch.randn(n_samples, 1)
    
    return X, y

def create_sample_classification_data(n_samples: int = 1000, input_size: int = 10, n_classes: int = 3):
    """
    Create sample classification data for testing the neural network
    
    :param n_samples: Number of samples to generate
    :param input_size: Size of input features
    :param n_classes: Number of classes
    :return: Tuple of (X, y) tensors
    """
    # Generate random input data
    X = torch.randn(n_samples, input_size)
    
    # Create class labels based on simple rules
    feature_sum = torch.sum(X[:, :3], dim=1)  # Use first 3 features
    y = torch.zeros(n_samples, dtype=torch.long)
    
    # Assign classes based on feature sum
    y[feature_sum < -1] = 0
    y[(feature_sum >= -1) & (feature_sum < 1)] = 1
    y[feature_sum >= 1] = 2
    
    return X, y

def demonstrate_epoch_training():
    """
    Demonstrate epoch-based training with both regression and classification examples
    """
    print("\n" + "="*80)
    print("DEMONSTRATING EPOCH-BASED TRAINING")
    print("="*80)
    
    # Example 1: Regression task
    print("\n1. REGRESSION EXAMPLE")
    print("-" * 40)
    
    # Create regression data
    X_train, y_train = create_sample_data(n_samples=800, input_size=10)
    X_val, y_val = create_sample_data(n_samples=200, input_size=10)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create neural network for regression
    q = 0.5 + 0.1j
    model = NeuralNetwork(q, max_n=20, input_size=10, hidden_size=32, output_size=1)
    
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    history = model.train_epochs(
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        val_loader=val_loader,
        print_every=10,
        early_stopping_patience=10
    )
    
    # Save the model
    model.save_model("theta_regression_model.pth", history)
    
    # Example 2: Classification task
    print("\n\n2. CLASSIFICATION EXAMPLE")
    print("-" * 40)
    
    # Create classification data
    X_train_cls, y_train_cls = create_sample_classification_data(n_samples=800, input_size=10, n_classes=3)
    X_val_cls, y_val_cls = create_sample_classification_data(n_samples=200, input_size=10, n_classes=3)
    
    # Create data loaders
    train_dataset_cls = TensorDataset(X_train_cls, y_train_cls)
    val_dataset_cls = TensorDataset(X_val_cls, y_val_cls)
    train_loader_cls = DataLoader(train_dataset_cls, batch_size=32, shuffle=True)
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=32, shuffle=False)
    
    # Create neural network for classification
    model_cls = NeuralNetwork(q, max_n=20, input_size=10, hidden_size=64, output_size=3)
    
    # Setup training components
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = optim.Adam(model_cls.parameters(), lr=0.001)
    
    # Train the classification model
    history_cls = model_cls.train_epochs(
        train_loader=train_loader_cls,
        criterion=criterion_cls,
        optimizer=optimizer_cls,
        num_epochs=30,
        val_loader=val_loader_cls,
        print_every=5,
        early_stopping_patience=8
    )
    
    # Save the classification model
    model_cls.save_model("theta_classification_model.pth", history_cls)
    
    print("\n\n3. TESTING MODEL LOADING")
    print("-" * 40)
    
    # Demonstrate model loading
    loaded_model, loaded_history = NeuralNetwork.load_model("theta_regression_model.pth")
    print(f"Loaded model theta value: {loaded_model.f()}")
    print(f"Loaded model architecture: {loaded_model.input_size}x{loaded_model.hidden_size}x{loaded_model.output_size}")
    
    # Test loaded model
    test_input = torch.randn(1, 10)
    original_output = model(test_input)
    loaded_output = loaded_model(test_input)
    
    print(f"Original model output: {original_output.item():.6f}")
    print(f"Loaded model output: {loaded_output.item():.6f}")
    print(f"Outputs match: {torch.allclose(original_output, loaded_output)}")
    
    return history, history_cls

def demonstrate_theta_informed_training():
    """
    Demonstrate training with theta-informed predictions
    """
    print("\n\n4. THETA-INFORMED TRAINING EXAMPLE")
    print("-" * 40)
    
    # Create data
    X_train, y_train = create_sample_data(n_samples=400, input_size=10)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Create model
    q = 0.3 + 0.2j  # Different q value
    model = NeuralNetwork(q, max_n=15, input_size=10, hidden_size=24, output_size=1)
    
    # Custom training loop using theta-informed predictions
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    print(f"Training with theta-informed predictions...")
    print(f"Initial theta value: {model.f()}")
    
    model.train()
    for epoch in range(20):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Use theta-informed prediction instead of regular forward pass
            output = model.theta_informed_prediction(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch [{epoch + 1:2d}/20] | Loss: {avg_loss:.6f} | Theta: {model.f():.4f}")
    
    print(f"Final theta value: {model.f()}")


# Mandelbrot Set Functions
def mandelbrot_iteration(c: complex, max_iter: int = 100) -> int:
    """
    Calculate the number of iterations for a complex number c in the Mandelbrot set
    
    :param c: Complex number to test
    :param max_iter: Maximum number of iterations
    :return: Number of iterations before divergence (or max_iter if convergent)
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def generate_mandelbrot_set(width: int = 800, height: int = 600, 
                           xmin: float = -2.5, xmax: float = 1.0,
                           ymin: float = -1.25, ymax: float = 1.25,
                           max_iter: int = 100) -> np.ndarray:
    """
    Generate the Mandelbrot set as a 2D array
    
    :param width: Width of the output image
    :param height: Height of the output image
    :param xmin, xmax: Real axis bounds
    :param ymin, ymax: Imaginary axis bounds
    :param max_iter: Maximum iterations for convergence test
    :return: 2D numpy array with iteration counts
    """
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    
    # Create complex plane
    C = X + 1j * Y
    
    # Initialize result array
    mandelbrot = np.zeros((height, width))
    
    # Calculate Mandelbrot set
    for i in range(height):
        for j in range(width):
            mandelbrot[i, j] = mandelbrot_iteration(C[i, j], max_iter)
    
    return mandelbrot

def plot_mandelbrot_set(mandelbrot_data: np.ndarray, 
                       xmin: float = -2.5, xmax: float = 1.0,
                       ymin: float = -1.25, ymax: float = 1.25,
                       title: str = "Mandelbrot Set",
                       save_path: Optional[str] = None,
                       colormap: str = 'hot'):
    """
    Plot the Mandelbrot set using matplotlib
    
    :param mandelbrot_data: 2D array from generate_mandelbrot_set
    :param xmin, xmax: Real axis bounds for display
    :param ymin, ymax: Imaginary axis bounds for display
    :param title: Plot title
    :param save_path: Optional path to save the plot
    :param colormap: Matplotlib colormap name
    """
    plt.figure(figsize=(12, 9))
    plt.imshow(mandelbrot_data, extent=[xmin, xmax, ymin, ymax], 
               cmap=colormap, origin='lower', interpolation='bilinear')
    plt.colorbar(label='Iterations to divergence')
    plt.title(title, fontsize=16)
    plt.xlabel('Real axis', fontsize=12)
    plt.ylabel('Imaginary axis', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mandelbrot plot saved to: {save_path}")
    
    plt.show()

def create_mandelbrot_dataset(n_samples: int = 1000, 
                             xmin: float = -2.5, xmax: float = 1.0,
                             ymin: float = -1.25, ymax: float = 1.25,
                             max_iter: int = 100,
                             feature_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a dataset from Mandelbrot set for neural network training
    
    :param n_samples: Number of samples to generate
    :param xmin, xmax: Real axis bounds
    :param ymin, ymax: Imaginary axis bounds
    :param max_iter: Maximum iterations for Mandelbrot calculation
    :param feature_size: Size of feature vector for each sample
    :return: Tuple of (features, labels) tensors
    """
    # Generate random complex numbers within the bounds
    real_parts = np.random.uniform(xmin, xmax, n_samples)
    imag_parts = np.random.uniform(ymin, ymax, n_samples)
    complex_points = real_parts + 1j * imag_parts
    
    # Calculate Mandelbrot iterations for each point
    iterations = np.array([mandelbrot_iteration(c, max_iter) for c in complex_points])
    
    # Create features: [real, imag, |c|, arg(c), real^2, imag^2, re*im, iterations_normalized]
    features = np.zeros((n_samples, feature_size))
    
    for i, c in enumerate(complex_points):
        features[i, 0] = c.real
        features[i, 1] = c.imag
        features[i, 2] = abs(c)
        features[i, 3] = np.angle(c)
        features[i, 4] = c.real ** 2
        features[i, 5] = c.imag ** 2
        features[i, 6] = c.real * c.imag
        features[i, 7] = iterations[i] / max_iter  # Normalized iterations
    
    # Create labels: binary classification (in set vs not in set)
    # Points that reach max_iter are considered "in the set"
    labels = (iterations == max_iter).astype(np.float32)
    
    # Convert to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Add dimension for regression
    
    return X, y

def demonstrate_mandelbrot_training():
    """
    Demonstrate neural network training on Mandelbrot set data
    """
    print("\n" + "="*80)
    print("MANDELBROT SET NEURAL NETWORK TRAINING")
    print("="*80)
    
    # Generate and visualize Mandelbrot set
    print("\n1. GENERATING MANDELBROT SET")
    print("-" * 40)
    
    mandelbrot_data = generate_mandelbrot_set(width=400, height=300, max_iter=80)
    plot_mandelbrot_set(mandelbrot_data, title="Mandelbrot Set - Training Data Source",
                       save_path="mandelbrot_set.png")
    
    # Create training and validation datasets
    print("\n2. CREATING MANDELBROT DATASET")
    print("-" * 40)
    
    X_train, y_train = create_mandelbrot_dataset(n_samples=2000, max_iter=80, feature_size=8)
    X_val, y_val = create_mandelbrot_dataset(n_samples=500, max_iter=80, feature_size=8)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Points in Mandelbrot set (training): {y_train.sum().item():.0f}/{len(y_train)}")
    print(f"Points in Mandelbrot set (validation): {y_val.sum().item():.0f}/{len(y_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create neural network with Mandelbrot-inspired theta function
    print("\n3. TRAINING NEURAL NETWORK ON MANDELBROT DATA")
    print("-" * 40)
    
    # Use a complex q value inspired by the Mandelbrot set
    q = -0.7 + 0.3j  # A point near the boundary of the Mandelbrot set
    model = NeuralNetwork(q, max_n=25, input_size=8, hidden_size=64, output_size=1)
    
    print(f"Using theta function with q = {q}")
    print(f"Initial theta value: {model.f()}")
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    history = model.train_epochs(
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        val_loader=val_loader,
        print_every=10,
        early_stopping_patience=15
    )
    
    # Save the trained model
    model.save_model("mandelbrot_neural_network.pth", history)
    
    # Test the model on some specific points
    print("\n4. TESTING TRAINED MODEL")
    print("-" * 40)
    
    test_points = [
        0 + 0j,        # Origin (in set)
        -1 + 0j,       # Real axis (in set)
        0.3 + 0.3j,    # Should be outside
        -0.7 + 0.3j,   # Near boundary
        -0.5 + 0.5j,   # Should be outside
    ]
    
    model.eval()
    with torch.no_grad():
        for point in test_points:
            # Create feature vector for the test point
            features = np.array([
                point.real, point.imag, abs(point), np.angle(point),
                point.real**2, point.imag**2, point.real*point.imag,
                mandelbrot_iteration(point, 80) / 80
            ])
            
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            prediction = torch.sigmoid(model(features_tensor)).item()
            actual_iterations = mandelbrot_iteration(point, 80)
            is_in_set = actual_iterations == 80
            
            print(f"Point {point}: Prediction={prediction:.4f}, "
                  f"Actual={'In Set' if is_in_set else f'{actual_iterations} iterations'}, "
                  f"Match={'✓' if (prediction > 0.5) == is_in_set else '✗'}")
    
    # Plot training history
    print("\n5. PLOTTING TRAINING HISTORY")
    print("-" * 40)
    
    model.plot_training_history(history, save_path="mandelbrot_training_history.png")
    
    return model, history

def demonstrate_mandelbrot_prediction_visualization(model, bounds=(-2, 1, -1.5, 1.5), resolution=200):
    """
    Visualize neural network predictions on the complex plane
    
    :param model: Trained neural network model
    :param bounds: (xmin, xmax, ymin, ymax) for the complex plane
    :param resolution: Resolution of the visualization grid
    """
    print("\n6. VISUALIZING NEURAL NETWORK PREDICTIONS")
    print("-" * 40)
    
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create complex plane
    C = X + 1j * Y
    predictions = np.zeros((resolution, resolution))
    
    model.eval()
    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                point = C[i, j]
                features = np.array([
                    point.real, point.imag, abs(point), np.angle(point),
                    point.real**2, point.imag**2, point.real*point.imag,
                    mandelbrot_iteration(point, 80) / 80
                ])
                
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                prediction = torch.sigmoid(model(features_tensor)).item()
                predictions[i, j] = prediction
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot actual Mandelbrot set
    actual_mandelbrot = generate_mandelbrot_set(
        width=resolution, height=resolution,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, max_iter=80
    )
    
    im1 = ax1.imshow(actual_mandelbrot, extent=[xmin, xmax, ymin, ymax], 
                     cmap='hot', origin='lower')
    ax1.set_title('Actual Mandelbrot Set', fontsize=14)
    ax1.set_xlabel('Real axis')
    ax1.set_ylabel('Imaginary axis')
    plt.colorbar(im1, ax=ax1, label='Iterations')
    
    # Plot neural network predictions
    im2 = ax2.imshow(predictions, extent=[xmin, xmax, ymin, ymax], 
                     cmap='viridis', origin='lower')
    ax2.set_title('Neural Network Predictions', fontsize=14)
    ax2.set_xlabel('Real axis')
    ax2.set_ylabel('Imaginary axis')
    plt.colorbar(im2, ax=ax2, label='Probability of being in set')
    
    plt.tight_layout()
    plt.savefig("mandelbrot_vs_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison plot saved as 'mandelbrot_vs_predictions.png'")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_polymorphism()
    
    # Run epoch training examples
    try:
        history_reg, history_cls = demonstrate_epoch_training()
        demonstrate_theta_informed_training()
        
        # Run Mandelbrot set training
        mandelbrot_model, mandelbrot_history = demonstrate_mandelbrot_training()
        
        # Visualize neural network predictions vs actual Mandelbrot set
        demonstrate_mandelbrot_prediction_visualization(mandelbrot_model)
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("Files generated:")
        print("- mandelbrot_set.png (Original Mandelbrot set)")
        print("- mandelbrot_training_history.png (Training progress)")
        print("- mandelbrot_vs_predictions.png (NN predictions vs actual)")
        print("- mandelbrot_neural_network.pth (Trained model)")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during training demonstration: {e}")
        print("This might be due to missing dependencies or other issues.")
