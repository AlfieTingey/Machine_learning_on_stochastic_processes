%% In this script we define a function to perform Gibbs sampling given some input parameters.

% Input parameters: The model is the queueing network of which we aim to
% estimate the service demands. M is the number of stations. R is the
% number of job classes. P is the population of jobs in the network. Z is
% the 'thinktimes', which normally we just set to 1 for each class. I
% denotes the interval from which we pick theta samples. d_theta is the
% incremental change between theta values. S is the number of experiments
% we do. D is the number of samples we take. N is the queue length states.
% prior_theta denotes the prior distribution p(\theta). 

%% Sources to help implement Gibbs sampling: 

% We implement this function based on the work in: 
% Casale, G. Wang, W. & Sutton, C. (2016). A Bayesian Approach to Parameter Inference in Queueing Networks. ACM Trans. Model. Comput. Simul.,27(1).

%% Start of function:

function [theta_average, thetas] = gibbs_sampling_mcmc_algorithm(model, M, R, P, Z, I, d_theta, S, D, N, prior_theta)
    %% initialize size of variables. We choose initial theta to always be zero vector.
    
    theta_init = zeros(M,R);
    thetas = zeros(M,R,S);
    thetas(:,:,1) = theta_init; 
    
    %% begin for loop for experiments, queueing stations, and job classes
    for s = 1:S-1
        current_theta = thetas(:,:,s);
        for i = 1:M 
            for j = 1:R 
                interval = I(1):d_theta:I(2);
                log_probs = zeros(size(interval));

                for l = 1:length(interval)
                    % calculate posterior distribution P(thetaij |
                    % theta_(ij)^(s-1), N).
                    current_theta(i, j) = interval(l); % Estimate current theta,
                    log_prior = log(prior_theta(I, d_theta)); % Use prior information that is known.
                    g_nc = pfqn_mci(current_theta, P, Z, D); % Find normalizing constant using MCI method from LINE.
                    log_p =  D*N(i + 1, j)*log(current_theta(i,j))-D*log(g_nc)+log_prior; % Find posterior distribution.
                    
                    % Fill matrix with posterior distribution value.
                    log_probs(l) = log_p;
                end
                
                % Use exponential to remove logs. Only used logs above
                % because addition is easier (and less comp. expensive) than multiplication. 

                probs = exp(log_probs-max(log_probs));
                probs = probs / sum(probs);

                %% select a theta based on the theory of Gibbs sampling: 
                
                % choose 0 < u < 1 and find theta_{ij} that sits inbetween
                % the two probabilities as detailed in [Cas16] within interval I. 
                
                cumulative_prob = cumsum(probs); % Add all the probabilities
                u = rand(1); % Set a random value for u
                index_prob = find(u<cumulative_prob); % Find indexes where u is less that cumulative prob
                index_theta = index_prob(1); % Take the first index that is less than cumulative prob
                thetaij = interval(index_theta);

                % set theta(s) = (theta(s) thetaij theta (s-1))
                current_theta(i, j) = thetaij;
            end
        end
        thetas(:,:,s+1) = current_theta;
    end
    
    %% Calculate the visit values of our model. We need these to give service times. 
    
    visit_count = zeros(R, M);
    visits = model.getStruct.visits;
    
    for i = 1 : R
      visit_count(i, :) = visits{i}(2:M+1, i);
    end    
    
    % Calculate the service times by dividing by visit count
    
    for s = 1:S
        thetas(:,:,S) = thetas(:,:,S)./visit_count';
    end 
   
    % Discard first half of thetas because of 'burn in' time. 
    n = round(S / 2);
    thetas = thetas(:,:,n:end);
    
    % Take the mean of the S=20 experiments. 
    
    theta_average = mean(thetas,3);
end