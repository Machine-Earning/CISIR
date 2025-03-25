import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn.mixture import GaussianMixture
from tensorflow.keras.losses import Loss


def get_gmm(data, n_components=64):
    """
    Fit a Gaussian Mixture Model to the provided data.
    Matches the Torch implementation of Balancing MSE.
    
    Args:
        data: Target values to fit the GMM to (numpy array)
        n_components: Number of components in the mixture model
    
    Returns:
        Dictionary with GMM parameters as TensorFlow tensors
    """
    # Ensure data is proper shape for sklearn
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Fit GMM using scikit-learn
    gmm = GaussianMixture(n_components=n_components, random_state=42).fit(data)
    
    # Convert to tensor dictionary
    gmm_dict = {
        'means': tf.convert_to_tensor(gmm.means_, dtype=tf.float32),
        'weights': tf.convert_to_tensor(gmm.weights_, dtype=tf.float32),
        'variances': tf.convert_to_tensor(gmm.covariances_, dtype=tf.float32)
    }
    
    print(f"GMM fitted with {n_components} components")
    return gmm_dict


def gai_loss_md(y_pred, y_true, gmm, noise_var):
    """
    Implementation of the Generative Adversarial Imitation Loss for multi-dimensional data
    
    Args:
        y_pred: Model predictions
        y_true: True targets
        gmm: Gaussian mixture model parameters
        noise_var: Noise variance (sigma^2)
        
    Returns:
        Calculated loss value
    """
    # Get dimensionality of prediction
    pred_dim = tf.shape(y_pred)[-1]
    I = tf.eye(pred_dim, dtype=tf.float32)
    
    # MSE term: -log p(y_true|y_pred)
    mvn_pred = tfp.distributions.MultivariateNormalFullCovariance(
        loc=y_pred, 
        covariance_matrix=noise_var * I
    )
    mse_term = -mvn_pred.log_prob(y_true)
    
    # Balancing term using GMM components
    pred_expanded = tf.expand_dims(y_pred, 1)  # [batch, 1, dim]
    
    # Calculate covariance matrices for each component
    noise_covs = noise_var * I[tf.newaxis, :, :]  # shape (1, dim, dim)
    gmm_covs = gmm['variances'] + noise_covs  # shape (components, dim, dim)
    
    # Create distribution for GMM components
    mvn_gmm = tfp.distributions.MultivariateNormalFullCovariance(
        loc=gmm['means'],
        covariance_matrix=gmm_covs
    )
    
    # Calculate log probabilities and apply weights
    component_log_probs = mvn_gmm.log_prob(pred_expanded)
    weighted_log_probs = component_log_probs + tf.math.log(gmm['weights'])
    
    # Log-sum-exp for numerical stability
    balancing_term = tf.reduce_logsumexp(weighted_log_probs, axis=1)
    
    # Combine terms
    loss = mse_term + balancing_term
    
    # Scale by noise variance (detached from gradient)
    loss = loss * tf.stop_gradient(2.0 * noise_var)
    
    return tf.reduce_mean(loss)


class GAILossMD(Loss):
    """
    Keras Loss subclass for balanced MSE using GMM
    Matches the Torch implementation of Balancing MSE.
    """
    
    def __init__(self, init_noise_sigma=1.0, gmm=None, data=None, n_components=64, name='gai_loss_md'):
        """
        Initialize the GAILossMD loss
        
        Args:
            init_noise_sigma: Initial value for noise standard deviation
            gmm: Pre-computed GMM parameters (optional)
            data: Data to fit GMM if not provided (optional)
            n_components: Number of GMM components if fitting to data
            name: Name of the loss function
        """
        super(GAILossMD, self).__init__(name=name)
        
        # Either use provided GMM or fit one to the data
        if gmm is None and data is not None:
            self.gmm = get_gmm(data, n_components)
        else:
            self.gmm = gmm
            # Ensure GMM values are TensorFlow tensors
            self.gmm = {k: tf.convert_to_tensor(self.gmm[k], dtype=tf.float32) for k in self.gmm}
            
        # Create trainable noise sigma parameter
        self.noise_sigma = tf.Variable(init_noise_sigma, trainable=True, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        """Calculate loss between y_true and y_pred"""
        noise_var = tf.square(self.noise_sigma)
        # Note the correct parameter order: y_pred, y_true matches the function signature
        return gai_loss_md(y_pred, y_true, self.gmm, noise_var)


def create_gai_loss(data, n_components=64, init_noise_sigma=1.0):
    """
    Factory function to create a GAILossMD loss 
    to be used with model.compile()
    
    Args:
        data: Data to fit GMM to
        n_components: Number of GMM components
        init_noise_sigma: Initial value for noise standard deviation
    
    Returns:
        A loss function compatible with Keras
    """
    # Initialize the loss with GMM from data
    gai_loss = GAILossMD(
        init_noise_sigma=init_noise_sigma, 
        data=data,
        n_components=n_components
    )
    
    # Return a lambda function compatible with Keras compile
    return lambda y_true, y_pred: gai_loss(y_true, y_pred)