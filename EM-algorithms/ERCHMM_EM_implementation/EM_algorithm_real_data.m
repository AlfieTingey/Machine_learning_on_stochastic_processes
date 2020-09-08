%% START OF EM ALGORITHM for real data %%

clear all

%% We set a seed, load the data, then initialise the trace using kpcfit_init algorithm in LINE. 

seed = 4; % We set a random seed so that we can use kpcfit_init to put the data into trace format.
%seed = 12;
%seed = 27;
rand('seed',seed);

%% Load either bcaug89 data or DEC-PKT-1-UDP data. 
load BCAUG89.mat;
% load DEC-PKT-1-UDP.mat;

%% Use kpcfit_init to initialize the data  
trace = kpcfit_init(S);

%% Likelihood functions and EM algorithm attempts using the sample S as the trace. Randomely choose P0 and P1

% Define the trace as a sample from the MAP defined in previous section
tr = S;

% We work out D0 and D1 using three different order values. 

[D0_estimated_8, D1_estimated_8] = EM_algorithm_using_ER_CHMM(tr, 8);
[D0_estimated_3, D1_estimated_3] = EM_algorithm_using_ER_CHMM(tr, 3);
[D0_estimated_5, D1_estimated_5] = EM_algorithm_using_ER_CHMM(tr, 5);
% [D0_estimated_8, D1_estimated_8] = MAPFromTrace(tr, 8);


estimated_map_8 = {D0_estimated_8,D1_estimated_8};
estimated_map_8 = map_normalize(estimated_map_8);
estimated_map_3 = {D0_estimated_3,D1_estimated_3};
estimated_map_3 = map_normalize(estimated_map_3);
estimated_map_5 = {D0_estimated_5,D1_estimated_5};
estimated_map_5 = map_normalize(estimated_map_5);

estimated_mean_3 = map_mean(estimated_map_3);
estimated_var_3 = map_var(estimated_map_3);

estimated_mean_5 = map_mean(estimated_map_5);
estimated_var_5 = map_var(estimated_map_5);

estimated_mean_8 = map_mean(estimated_map_8);
estimated_var_8 = map_var(estimated_map_8);

%% Plot results 

%% Plot autocorellation

% plot(map_acf(estimated_map,1:50),'r');
% hold on 
% plot(trace_acf(tr,1:50), 'g');
% hold off
% xlabel('lag');
% ylabel('Auto-Correlation');
% title('Autocorrelation of Predicted MAP and Trace');
% legend('predicted','trace')

%% Plot distribution

[dist_tr, x] = ecdf(tr);

subplot(2,1,1)
plot(map_acf(estimated_map_3,1:150),'r');
hold on 
plot(map_acf(estimated_map_5,1:150),'k');
plot(map_acf(estimated_map_8,1:150),'g');
% plot(trace_acf(tr,1:5000), 'b*'); 
plot(trace_acf(tr,1:150), 'b'); 
hold off
xlabel('lag k');
ylabel('Auto-Correlation');
title('Autocorrelation of Predicted MAP and Trace');
legend('predicted MAP 3','predicted MAP 5', 'predicted MAP 2','trace')
subplot(2,1,2)
plot(x, map_cdf(estimated_map_3,x),'r');
hold on 
plot(x, map_cdf(estimated_map_5,x),'k');
plot(x, map_cdf(estimated_map_8,x),'g');
plot(x, dist_tr, 'b'); 
hold off
xlabel('time');
ylabel('Comulative Distribution');
xlim([0,0.05]);
title('CDF of Predicted MAP and Trace');
legend('predicted MAP 3','predicted MAP 5', 'predicted MAP 2','trace')
