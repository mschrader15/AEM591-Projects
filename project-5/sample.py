# Set number of particles and preallocate data matrices.
import numpy as np

num_p = 10000
state_p = np.zeros((num_p, num_data))
weights_p = np.zeros((num_p, num_data))
state_pf = np.zeros((num_data, 1))
cov_pf = np.zeros((num_data, 1))
# Initialize particles about a zero-mean Gaussian with covariance of
# 2 with uniform weights.
state_p[:, 1] = 2 * np.random.randn(num_p, 1)
weights_p[:, 1] = 1 / num_p * np.ones((num_p, 1))
state_predict = np.zeros((num_p, 1))
measurements_predict = np.zeros((num_p, 1))
state_pf[1] = mean(state_p(: , 1))
cov_pf[1] = (std(state_p(: , 1))) ** 2
for k in np.arange(2, num_data+1).reshape(-1):
    # Sample from the state transition conditional PDF.
    for i in np.arange(1, num_p+1).reshape(-1):
        state_predict[i, 1] = 0.5 * state_p(i, 1) + 25 * state_p(i, 1) / (1 + state_p(
            i, 1) ** 2) + 8 * np.cos(1.2 * (k - 1)) + sigma_w * np.random.randn(1)
    
    # Compute normalized importance weights from likelihood function.
    for i in np.arange(1, num_p+1).reshape(-1):
        weights_p[i, k] = normpdf(
            1 / 20 * state_predict(i, 1) ** 2, measurements(k), sigma_v) * weights_p(i, k - 1)
    
    weights_p[:, k] = weights_p(:, k) / (sum(weights_p(: , k)))
    
    # Resample particles according to weight.
    sum_weights = cumsum(weights_p(: , k))
    for i in np.arange(1, num_p+1).reshape(-1):
        r = np.random.rand(1)
        j = 1
        while sum_weights(j) < r:

            j = j + 1

        state_p[i, k] = state_predict(j)
    # Reset weights to uniform.
    weights_p[:, k] = 1 / num_p * np.ones((num_p, 1))
    # Output the mean as the state estimate and the covariance.
    state_pf[k] = mean(state_p(: , k))
    cov_pf[k] = var(state_p(: , k))
    # Roughen particles.
    max_diff = np.abs(np.amax(state_p(: , k)) - np.amin(state_p(: , k)))
    for i in np.arange(1, num_p+1).reshape(-1):
        rough = np.sqrt(0.2 * max_diff / num_p) * np.random.randn(1)
        state_p[i, k] = state_p(i, k) + rough
