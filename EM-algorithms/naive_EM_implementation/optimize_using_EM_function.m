clear all

%% Define model and initial conditions

model = Network('MIP');

S = 2;
N = 5;
    
%% Block 1: nodes
delay = Delay(model,'WorkingState');
queue = Queue(model, 'RepairQueue', SchedStrategy.FCFS);
queue.setNumberOfServers(S);
%% Block 2: classes
cclass = ClosedClass(model, 'Machines', N, delay);
delay.setService(cclass, Exp(0.5));
queue.setService(cclass, Exp(2.0));

%% Block 3: topology
model.link(Network.serialRouting(delay,queue));

% model.jsimgView
% model = Network('M/G/1');
solver = SolverCTMC(model);
ctmcAvgTable = solver.getAvgTable;
StateSpace = solver.getStateSpace();
InfGen = full(solver.getGenerator());
Q = InfGen;

var_map_generated = Inf;

% MAP_random = map_rand(N+1);
%% Testing a little bit 
% D0 = [-3.721, 0.500, 0.020; 0.100, -1.206, 0.005;0.001, 0.002, -0.031];
% D1 = [0.200, 3.000, 0.001; 1.000, 0.100, 0.001; 0.005, 0.003, 0.020];
D1= [1.0,0.5,0,0,0,0; 0.5000,1,0.5,0,0,0; 0,1.0,1.5,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0;0,0,0,0,0,0];
D0 = Q - D1;
D0=Q-D1;
MAP={D0,D1};
% MAP = map_normalize(MAP);
mean_map_real = map_mean(MAP);
var_map_real = map_var(MAP);
% map_sample(MAP,1e3);
sample_trace = map_sample(MAP,1e3);

%% START OF EM ALGORITHM %%


%% Likelihood functions and EM algorithm attempts using the sample S as the trace. Randomely choose P0 and P1

% Define the trace as a sample from the MAP defined in previous section
trace = sample_trace;

local_opt_track = 0;

D1_old = zeros(N+1,N+1);
D0_old = zeros(N+1,N+1);
exp_max_chosen = {D0_old, D1_old};
var_map_chosen = 0;
mean_map_chosen = 0;
likelihood_value_old = 0;

while true
    
    [exp_max, mean_map_generated, var_map_generated, likelihood_value_comparison] = EM_algorithm_function(N,trace);
    
    if var_map_generated ~= Inf && abs(var_map_real - var_map_generated) <= 0.1 && abs(mean_map_real - mean_map_generated) <= 0.1
        exp_max_chosen = exp_max;
        var_map_chosen = var_map_generated;
        mean_map_chosen = mean_map_generated;
        break 
    end 
    
    %d(T,M)=α^m * a(m)* eT - display likelihood using this function to see
    % quality of prediction.
    
    if likelihood_value_comparison > likelihood_value_old
        disp('new MAP chosen')
        exp_max_chosen = exp_max;
        var_map_chosen = var_map_generated;
        mean_map_chosen = mean_map_generated;
    end
    
     local_opt_track = local_opt_track+1;
    if local_opt_track == 20
        break
    end 
end 


generated_MAP = exp_max_chosen
real_mean = mean_map_real
generated_mean = mean_map_chosen
real_var = var_map_real
generated_var = var_map_chosen

plot(map_acf(MAP,1:10));
hold on 
plot(map_acf(generated_MAP,1:10),'r');
hold off
xlabel('lag');
ylabel('Auto-Correlation');


