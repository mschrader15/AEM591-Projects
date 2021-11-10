% Set number of particles and preallocate data matrices.
num_p = 10000;
state_p =  zeros(num_p, num_data);
weights_p = zeros(num_p, num_data);
state_pf = zeros(num_data, 1);
cov_pf = zeros(num_data, 1);
% Initialize particles about a zero-mean Gaussian with covariance of
% 2 with uniform weights.
state_p(:, 1) = 2*randn(num_p, 1);
weights_p(:, 1) = 1/num_p*ones(num_p, 1);
state_predict = zeros(num_p, 1);
measurements_predict = zeros(num_p, 1);
state_pf(1) = mean(state_p(:, 1));
cov_pf(1) = (std(state_p(:, 1)))^2;
for k = 2:num_data
    % Sample from the state transition conditional PDF.
    for i = 1:num_p
        state_predict(i, 1) = 0.5*state_p(i, 1) + 25*state_p(i, 1)/(1 + state_p(i, 1)^2) + 8*cos(1.2*(k-1)) + sigma_w*randn(1);
    end
    % Compute normalized importance weights from likelihood function.
    for i = 1:num_p
        weights_p(i, k) = normpdf(1/20*state_predict(i, 1).^2, measurements(k), sigma_v)*weights_p(i, k-1);
    end
    weights_p(:, k) = weights_p(:, k)/(sum(weights_p(:, k)));
    % Resample particles according to weight.
    sum_weights = cumsum(weights_p(:, k));
    for i = 1:num_p
        r = rand(1);
        j = 1;
        while sum_weights(j) < r
            j = j + 1;
        end
        state_p(i, k) = state_predict(j);
    end
    % Reset weights to uniform.
    weights_p(:, k) = 1/num_p*ones(num_p, 1);
    % Output the mean as the state estimate and the covariance.
    state_pf(k) = mean(state_p(:, k));
    cov_pf(k) = var(state_p(:, k));
    % Roughen particles.
    max_diff = abs( max(state_p(:, k)) - min(state_p(:, k)) );
    for i = 1:num_p
        rough = sqrt(0.2*max_diff/num_p)*randn(1);
        state_p(i, k) = state_p(i, k) + rough;
    end
end

