
%% This function creates traces from an example QN with 3 stations

% We use this function to build queueing networks which we take synthetic
% traces from to input into our rnn-algorithm. We only explain this script,
% but the 5-station and 10-station examples use exactly the same technique.
% We set the transition probability matrix, and mu values to build the
% network. 

for i = 1:50
    x_range = 50:200;
    
    % define range for population values.
    
    x_1 = randsample(x_range, 1);
    x_2 = randsample(x_range, 1);
    x_3 = randsample(x_range, 1);
    
    % Choose random values for setting the initial population in the
    % network for each station.
    
    x = [x_1;x_2;x_3];
    
    assert(all(x>=0));
    
    % Choose server concurrency values that we use for learning
    
    s = [40,20,10];
    
    assert(all(s > 0));
    
    % Set the service rates for the generating network.
    
    mu = [100,50,50];
    
    assert(all(mu > 0));

    model = Network('myModel');

    %% Block 1: nodes
    
    % create the stations in the queue
    
    node{1} = Queue(model, 'QueueStation1', SchedStrategy.FCFS);
    node{2} = Queue(model, 'QueueStation2', SchedStrategy.FCFS);
    node{3} = Queue(model, 'QueueStation3', SchedStrategy.FCFS);

    %% Set number of servers for each queueing station node
    
    % set number of servers in the queue
    
    node{1}.setNumberOfServers(s(1));
    node{2}.setNumberOfServers(s(2));
    node{3}.setNumberOfServers(s(3));

    %% Block 2: Classes
    
    % create closed class of the model
    
    cclass1 = ClosedClass(model, 'class1', x(1), node{1});

    %% Might have to set service rates like this and create loads of them so let's see

    %% Set service rates for each queueing station node and pop

    node{1}.setService(cclass1, Exp(mu(1)));
    node{2}.setService(cclass1, Exp(mu(2)));
    node{3}.setService(cclass1, Exp(mu(3)));
    
    node{2}.setState(x(2));
    node{3}.setState(x(3));

    %% Block 3: Topology using Routing Matrix P
    
    % Set transition probability matrix of generating network that we aim to predict.
    T = [0,2/3,1/3;1/3,0,2/3;2/3,1/3,0];
    
    % initiate matrix in the queue
    P = model.initRoutingMatrix;
    P{cclass1} = T;

    model.link(P);

    %% Metrics
    
    % Find the transient average queue length metrics over 5s for the 3
    % station example.

    solver = SolverJMT(model);

    [QN,UN,RN,TN] = solver.getAvg();

    [Qt,Ut,Tt] = model.getTranHandles();

    [QNt,UNt,TNt] = SolverJMT(model,'force', true, 'timespan',[0,5]).getTranAvg(Qt,Ut,Tt);

    time_intervals = QNt{1,1}.t;
    av_q_len_station1 = QNt{1,1}.metric;
    av_q_len_station2 = QNt{2,1}.metric;
    av_q_len_station3 = QNt{3,1}.metric;

    total_size_1 = size(av_q_len_station1);
    total_size_2 = size(av_q_len_station2);
    total_size_3 = size(av_q_len_station3);

    time_size = size(time_intervals);

    total_length_1 = total_size_1(1);
    total_length_2 = total_size_2(1);
    total_length_3 = total_size_3(1);

    time_length = time_size(1);
    
    % Only take every 100th value so its not a huge vector.

    av_q_len_station1 = av_q_len_station1(1:100:total_length_1);
    av_q_len_station2 = av_q_len_station2(1:100:total_length_2);
    av_q_len_station3 = av_q_len_station3(1:100:total_length_3);

    time_intervals = time_intervals(1:100:time_length);

    test_size = size(av_q_len_station1);
    
    % Create trace in the form that we want: first column time intervals,
    % other columns the average q lengths for each station. 

    average_queue_length_trace = [time_intervals, av_q_len_station1, av_q_len_station2, av_q_len_station3];

    % save('average_queue_length_trace.mat','trace_1');
    
    % Save the traces in .mat format to use in python.
    save(['average_queue_length_trace_' num2str(i) '.mat'],'average_queue_length_trace')
end  