clear all

%% Description:

% We used this script to create examples that we use our improved
% EM-algorithm on. A reader can use this script to recreate our results. 

%% Choose the type of network that we want to optimize using our EM algorithm.

% Choose 1 if we want to analyse 'MIP'. Choose 2 if we want to analyse
% M/M/1. Choose 3 if we want to analyse M/G/1. One must set the transition
% probability function in each. These are just examples of our EM algorithm
% working on random MAPs. 

type_of_network = 1;

% Do some testing to make sure we don't get an error. 

if type_of_network ~= 1 && type_of_network ~= 2 && type_of_network && 3
    disp('Wrong value chosen for type of network: choose a value of 1 2 or 3')
end 

assert(type_of_network == 1 || type_of_network == 2 || type_of_network == 3);

if type_of_network == 1
    
    model = Network('MIP');
    
    % User can change these values.
    
    S = 3;
    N = 5;
    mu = [0.5,2.0];

    %% Block 1: nodes
    delay = Delay(model,'WorkingState');
    queue = Queue(model, 'RepairQueue', SchedStrategy.FCFS);
    queue.setNumberOfServers(S);
    %% Block 2: classes
    cclass = ClosedClass(model, 'Machines', N, delay);
    delay.setService(cclass, Exp(mu(1)));
    queue.setService(cclass, Exp(mu(2)));

    %% Block 3: topology
    model.link(Network.serialRouting(delay,queue));
    
    solver = SolverCTMC(model);
    ctmcAvgTable = solver.getAvgTable;
    StateSpace = solver.getStateSpace();
    InfGen = full(solver.getGenerator());
    Q = InfGen;
    
    D1 = [1.0000,  0.5000,  0.2000,  0.0100,  0.0100, 0.0200;
    0.5000, 1.0000,   0.5000,  0.2000,  0.1000, 0.2000;
    0.1000,  1.0000,  0.2000,  1.5000,  0.1000, 0.1000;
    0.5000,   1.0000,  0.1000,   2.0000,   0.5000, 0.1000;
    0.1000,  0.1000,  0.2000,  0.1000,  0.5000, 0.0100;
    0.4000,   0.0100, 0.7000,  0.8000,   0.0100, 1.0000];

end 

if type_of_network == 2
    
    model = Network('M/M/1');
    %% Block 1: nodes

    source = Source(model, 'mysource');
    queue = Queue(model, 'myQueue', SchedStrategy.FCFS);
    sink = Sink(model, 'mySink');

    %% Block 2: classes

    oclass = OpenClass(model, 'myClass');
    source.setArrival(oclass, Exp(1));
    queue.setService(oclass, Exp(2));


    %% Block 3: topology

    model.link(Network.serialRouting(source,queue,sink));
    
    solver = SolverCTMC(model,'cutoff',1);
    ctmcAvgTable = solver.getAvgTable;
    StateSpace = solver.getStateSpace();
    InfGen = full(solver.getGenerator());
    Q = InfGen;
    
    D1 = [0.2,2;3.5,0.8];
end 

if type_of_network == 3
    %% We now consider a more challenging variant of the first example. We assume that there are two classes of
    % incoming jobs with non-exponential service times. For the first class, service times are Erlang distributed
    % with unit rate and variance 1/3; they are instead read from a trace for the second class. Both classes have
    % exponentially distributed inter-arrival times with mean 2s.

    %% make sure Matlab directory is in the examples/ folder

    %% Nodes block:

    model = Network('M/G/1');
    source = Source(model, 'Source');
    queue = Queue(model, 'Queue', SchedStrategy.FCFS);
    sink = Sink(model, 'sink');

    %% Classes block: 

    jobclass1 = OpenClass(model, 'Class1');

    source.setArrival(jobclass1, Exp(0.5));

    queue.setService(jobclass1, Erlang.fitMeanAndSCV(1,1/3));
    
    P = model.initRoutingMatrix();
    P{jobclass1} = Network.serialRouting(source,queue,sink);
    model.link(P);
    
    solver = SolverCTMC(model,'cutoff',1,'verbose',true,'force',true);
    InfGen = full(solver.getGenerator());
    Q = InfGen;
    
    D1 = [0,0,0,0.1;2,0.5,0,0;0,1,2,0;0,0,1,0.5];
end 


% MAP_random = map_rand(N+1);
%% Testing a little bit 
D0 = Q - D1;
% D0=Q-D1;
MAP={D0,D1};
MAP_normal = map_normalize(MAP);
mean_map_real = map_mean(MAP_normal);
var_map_real = map_var(MAP_normal);
% map_sample(MAP,1e3);
sample_trace = map_sample(MAP_normal,1e5);

%% START OF EM ALGORITHM on created MAP ETC %%


%% Likelihood functions and EM algorithm attempts using the sample S as the trace. Randomely choose P0 and P1

% Define the trace as a sample from the MAP defined in previous section
tr = sample_trace;

% Use algorithm function 
[D0_estimated, D1_estimated] = EM_algorithm_using_ER_CHMM(tr, 6);

estimated_map = {D0_estimated,D1_estimated};
estimated_map = map_normalize(estimated_map);

estimated_mean = map_mean(estimated_map);
estimated_var = map_var(estimated_map);
var_map_real
mean_map_real

[dist_tr, x] = ecdf(tr);

%% Print graphs

% plot(map_acf(MAP_normal,1:10));
% hold on 
% plot(map_acf(estimated_map,1:10),'r');
% plot(trace_acf(tr,1:10), 'g');
% hold off
% xlabel('lag');
% ylabel('Auto-Correlation');
% title('Autocorrelation of Real and Predicted MAPs');
% legend('real','predicted','trace')

plot(x, map_cdf(estimated_map,x),'k');
hold on 
plot(x, map_cdf(MAP_normal,x), 'b');
plot(x, dist_tr, 'g'); 
hold off
xlabel('Time');
ylabel('Comulative Distribution');
xlim([0,4]);
title('CDF of Predicted MAP and Trace');
legend('Predicted MAP','Real MAP','trace')