%% Define Variables
clear all 

% Here we define some hyperparameters that we will use as well as values of
% population that we will predict accuracy for.

% hyperparameters: dthetas and sample sizes.

d_thetas = [0.001, 0.0005];
Ds = [5000, 20000, 60000]; 

% population vector

Ks = [20, 40, 60, 80];

% No. of experiments vector

S = 20;

%% Make assertions on hyper-parameters for testing: 
% we do this so that we throw an error for values that are nonsense.

assert(all(d_thetas > 0));
assert(all(Ds > 0));
assert(all(Ks > 0));
assert(S > 0)

%% Here we start our experimentation. 

% We made 5 examples that we experiment our Gibbs Sampling on. We only use
% PS sharing networks, but if the user is interested they can change the
% model they evaluate the gibbs sampling on. 

% evaluate_gibbs is a function that uses the gibbs_sampling_mcmc_algorithm
% function to evaluate the gibbs sampling for each model that we use. All
% experiments are commented out apart from the first one. 

%% M = 1, R = 1

demand = [1/20];

assert(all(demand > 0));

[M1R1_err_d_theta, M1R1_err_D, thetas_eval] = evaluate_gibbs(1, 1, demand, [1], d_thetas, Ds, Ks, S, @prior_theta);

subplot(2,1,1);
plot(Ks, M1R1_err_d_theta(1,:),'r');
hold on 
plot(Ks, M1R1_err_d_theta(2,:),'b');
xlabel('Number of Jobs');
ylabel('Error Value');
hold off
title('Evaluation of Gibbs Sampling with Different Value of dtheta');
legend('dtheta = 0.001', 'dtheta = 0.0005');
subplot(2,1,2);
plot(Ks, M1R1_err_D(1,:),'r');
hold on 
plot(Ks, M1R1_err_D(2,:),'b');
plot(Ks, M1R1_err_D(3,:),'k');
hold off
xlabel('Number of Jobs');
ylabel('Error Value');
title('Evaluation of Gibbs Sampling with Different Value of D');
legend('D = 5000','D = 20000','D = 60000');

%% M = 2, R = 1
% % 
% demands = [1/60; 1/40];

% assert(all(demand > 0));
% 
% [M2R1_err_d_theta, M2R1_err_D, thetas_eval] = evaluate_gibbs(2, 1, demands, [1], d_thetas, Ds, Ks, S, @prior_theta);
% 
% subplot(2,1,1);
% plot(Ks, M2R1_err_d_theta(1,:), 'r');
% hold on 
% plot(Ks, M2R1_err_d_theta(2,:), 'b');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Step-size');
% legend('dtheta = 0.001', 'dtheta = 0.0005');
% subplot(2,1,2);
% plot(Ks, M2R1_err_D(1,:),'r');
% hold on 
% plot(Ks, M2R1_err_D(2,:),'b');
% plot(Ks, M2R1_err_D(3,:),'k');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Samples');
% legend('D = 5000', 'D = 20000','D = 60000');
% 
%% M = 3, R = 1
% % 
% demands = [1/60; 1/40; 1/50];

% assert(all(demand > 0));
% 
% [M3R1_err_d_theta, M3R1_err_D, thetas_eval] = evaluate_gibbs(3, 1, demands, [1], d_thetas, Ds, Ks, S, @prior_theta);
% 
% subplot(2,1,1);
% plot(Ks, M3R1_err_d_theta(1,:), 'r');
% hold on 
% plot(Ks, M3R1_err_d_theta(2,:), 'b');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Step-size');
% legend('dtheta = 0.001', 'dtheta = 0.0005');
% subplot(2,1,2);
% plot(Ks, M3R1_err_D(1,:),'r');
% hold on 
% plot(Ks, M3R1_err_D(2,:),'b');
% plot(Ks, M3R1_err_D(3,:),'k');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Samples');
% legend('D = 5000', 'D = 20000','D = 60000');
% 
%% M = 2, R = 2
% % 
% demands = [1/60, 1/50; 1/40, 1/50];

% assert(all(demand > 0));
% 
% [M2R21_err_d_theta, M2R2_err_D, thetas_eval] = evaluate_gibbs(2, 2, demands, [1,1],  d_thetas, Ds, Ks, S, @prior_theta);
% 
% subplot(2,1,1);
% plot(Ks, M2R2_err_d_theta(1,:), 'r');
% hold on 
% plot(Ks, M2R2_err_d_theta(2,:), 'b');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Step-size');
% legend('dtheta = 0.001', 'dtheta = 0.0005');
% subplot(2,1,2);
% plot(Ks, M2R2_err_D(1,:),'r');
% hold on 
% plot(Ks, M2R2_err_D(2,:),'b');
% plot(Ks, M2R2_err_D(3,:),'k');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Samples');
% legend('D = 5000', 'D = 20000','D = 60000');
% 
%% M = 3, R = 2
% 
% demands = [1/60, 1/50; 1/40, 1/50; 1/60, 1/50]; 
% 
% assert(all(demand > 0));
% Ks = [20, 50, 80, 100];
% 
% [M3R2_err_d_theta, M3R2_err_D, thetas_eval] = evaluate_gibbs(3, 2, demands, [1,1], [], d_thetas, Ds, Ks, S, @prior_theta);
% 
% subplot(2,1,1);
% plot(Ks, M3R2_err_d_theta(1,:), 'r');
% hold on 
% plot(Ks, M3R2_err_d_theta(2,:), 'b');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Step-size');
% legend('dtheta = 0.001', 'dtheta = 0.0005');
% subplot(2,1,2);
% plot(Ks, M3R2_err_D(1,:),'r');
% hold on 
% plot(Ks, M3R2_err_D(2,:),'b');
% plot(Ks, M3R2_err_D(3,:),'k');
% hold off
% xlabel('Number of Jobs');
% ylabel('Error Value');
% title('Evaluation of Gibbs Sampling with Different Values of Samples');
% legend('D = 5000', 'D = 20000','D = 60000');

%% Function evaluate_gibbs

function [error_d_theta, error_D, thetas] = evaluate_gibbs(M, R, demands, thinkTimes, d_thetas, Ds, Ks, S, prior_theta)
    
    show_progress = true;
    error_d_theta = zeros(length(d_thetas), length(Ks));
    error_D = zeros(length(Ds), length(Ks));
    
    % Define interval range
    
    I = [0, 0.2];
    
    % make some assertions on the interval range for testing
    
    assert(I(1) < I(2) && length(I) == 2 && I(1) >= 0 && I(2) > 0);
    
    for k = 1:length(Ks)
        
        P = ones(1, R) * Ks(k);
        
        model = Network.cyclicPsInf(P, demands, thinkTimes);
        
        solver = SolverJMT(model, 'seed', 1, 'samples', 10000); 
        state = solver.getAvgQLen;
            
        for d = 1:length(d_thetas) 
            [theta_avg, thetas] = gibbs_sampling_mcmc_algorithm(model, M, R, P, thinkTimes, I, d_thetas(d), S, 10000, state, prior_theta);
            % Store the error values for each d_theta hyper-parameter
            error_d_theta(d, k) = mean_abs_per_error(theta_avg, demands);
        end
        
        % Show progress if we want 
        
        if show_progress
            fprintf('Finished for a step-size value: d_theta=%g', d_thetas(d));
            fprintf('\n');
        end
        
        for D = 1:length(Ds)
           solver = SolverJMT(model, 'seed', 1, 'samples', Ds(D)); 
           state = solver.getAvgQLen;
           [theta_avg, thetas] = gibbs_sampling_mcmc_algorithm(model, M, R, P, thinkTimes, I, 0.0001, S, Ds(D), state, prior_theta);
           % Store the error values for each sample value hyper-parameter
           error_D(D, k) = mean_abs_per_error(theta_avg, demands);
        end
        
        % Show progress if we want 
        if show_progress
            fprintf('Finished for a sample value: D=%g', Ds(D));
            fprintf('\n');
            fprintf('Finished for a population value: k=%g', Ks(k));
        end
    end 
end 

%% calculate prior distribution P(theta) given by user

function p = prior_theta(I, d_theta)
    % assume uniform distribution
    p = d_theta / (I(2) - I(1));
    
end

%% Calculate mean absolute percentage error. 

function [err] = mean_abs_per_error(theta_avg, demands)
   % mean absolute percentage error 
   error = (theta_avg - demands) ./ demands;
   err = sum(abs(error), 'all') / numel(error);
end
