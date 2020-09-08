clear all 

%% Function that we use to create the graphs and find the error of our predicted 3-station model from rnn.

% We add in some extra comments in this script that are not present in the
% 5-station and 10-station scripts because they use the same method as this
% one. 

%% Set 'unseen' variables that we will predict our network parameters with.

x = [200;0;0];
s = [50,20,10];

model_generated = Network('myModel_generated');

%% mu in generating network

mu = [100,50,50];

%% Create Generated Model

%% Block 1: nodes
node{1} = Queue(model_generated, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_generated, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_generated, 'QueueStation3', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));

%% Block 2: Classes

cclass1 = ClosedClass(model_generated, 'class1', x(1), node{1});

node{2}.setState(x(2));
node{3}.setState(x(3));

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));

%% Block 3: Topology using Routing Matrix P

T = [0,2/3,1/3;1/3,0,2/3;2/3,1/3,0];

P = model_generated.initRoutingMatrix;
P{cclass1} = T;

model_generated.link(P);

[Qt_generated,Ut_generated,Tt_generated] = model_generated.getTranHandles();

[QNt_generated,UNt_generated,TNt_generated] = SolverJMT(model_generated,'force', true, 'timespan',[0,5]).getTranAvg(Qt_generated,Ut_generated,Tt_generated);

%% Create Predicted Model

model_predicted = Network('myModel_predicted');

%% To create the predicted model we use the parameters as estimated by our rnn network.

% These parameters are saved in a .txt as an output from our rnn algorithm.

% mu 98.61132 38.104538 51.377357
% P 0.0 0.6514631 0.3485369 0.105028175 0.0 0.89497185 0.99705505 0.0029449302 0.0

mu = [98.6, 38.1, 51.38];

%% Block 1: nodes
node{1} = Queue(model_predicted, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_predicted, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_predicted, 'QueueStation3', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));

%% Block 2: Classes

cclass1 = ClosedClass(model_predicted, 'class1', x(1), node{1});

node{2}.setState(x(2));
node{3}.setState(x(3));

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));

%% Block 3: Topology using Routing Matrix P

T = [0,0.6514631,0.3485369;0.105028175,0,0.89497185;0.99705505,0.0029449302,0];

P = model_predicted.initRoutingMatrix;
P{cclass1} = T;

model_predicted.link(P);

[Qt_predicted,Ut_predicted,Tt_predicted] = model_predicted.getTranHandles();

[QNt_predicted,UNt_predicted,TNt_predicted] = SolverJMT(model_predicted,'force', true, 'timespan',[0,5]).getTranAvg(Qt_predicted,Ut_predicted,Tt_predicted);

%% Now we start plotting graphs of average q len, util and abs diff in av q len. Uncomment the ones that you want.

% In the av. q. len. plot we find the error. 

%% Plot average queue length


% subplot(3,1,1); plot(QNt_generated{1,1}.t, QNt_generated{1,1}.metric)
% hold on 
% plot(QNt_predicted{1,1}.t, QNt_predicted{1,1}.metric,'r')
% hold off
% title('Station 1')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% xlim([0,1])
% legend('G. Net.','P. Net.');
% subplot(3,1,2); plot(QNt_generated{2,1}.t, QNt_generated{2,1}.metric)
% hold on 
% plot(QNt_predicted{2,1}.t, QNt_predicted{2,1}.metric,'r')
% hold off
% title('Station 2')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% xlim([0,1])
% legend('G. Net.','P. Net.');
% subplot(3,1,3); plot(QNt_generated{3,1}.t, QNt_generated{3,1}.metric)
% hold on 
% plot(QNt_predicted{3,1}.t, QNt_predicted{3,1}.metric,'r')
% hold off
% title('Station 3')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% xlim([0,1])
% legend('G. Net.','P. Net.');

%% plot accuracy and find the ERROR


% QNt_predicted_interp_station1 = interp1(QNt_predicted{1,1}.t,QNt_predicted{1,1}.metric,QNt_generated{1,1}.t);
% QNt_predicted_interp_station2 = interp1(QNt_predicted{2,1}.t,QNt_predicted{2,1}.metric,QNt_generated{2,1}.t);
% QNt_predicted_interp_station3 = interp1(QNt_predicted{3,1}.t,QNt_predicted{3,1}.metric,QNt_generated{3,1}.t);
% 
% length_predict = length(QNt_predicted_interp_station1)-100;
% pred_interp_1 = QNt_predicted_interp_station1(1:length_predict-100);
% pred_interp_2 = QNt_predicted_interp_station2(1:length_predict-100);
% pred_interp_3 = QNt_predicted_interp_station3(1:length_predict-100);
% 
% QNt_generated_1 = QNt_generated{1,1}.metric(1:length_predict-100);
% QNt_generated_2 = QNt_generated{2,1}.metric(1:length_predict-100);
% QNt_generated_3 = QNt_generated{3,1}.metric(1:length_predict-100);
% 
% mean_graph_values_1 = mean(abs(pred_interp_1 - QNt_generated_1))*ones(length_predict-100,1);
% mean_graph_values_2 = mean(abs(pred_interp_2 - QNt_generated_2))*ones(length_predict-100,1);
% mean_graph_values_3 = mean(abs(pred_interp_3 - QNt_generated_3))*ones(length_predict-100,1);
% 
% time_values = QNt_generated{1,1}.t(1:length_predict-100);
% 
% 
% subplot(3,1,1);
% plot(time_values, abs(pred_interp_1 - QNt_generated_1))
% hold on
% plot(time_values, mean_graph_values_1, 'r')
% hold off
% title('Absolute Difference in Average Queueing Lengths: Station 1')
% ylabel('Absolute Difference')
% xlabel('Time (s)')
% xlim([0,2.5])
% legend('Abs. Diff.','Mean Abs. Diff.')
% subplot(3,1,2);
% plot(time_values, abs(pred_interp_2 - QNt_generated_2))
% hold on
% plot(time_values, mean_graph_values_2, 'r')
% hold off
% title('Absolute Difference in Average Queueing Lengths: Station 2')
% ylabel('Absolute Difference')
% xlabel('Time (s)')
% xlim([0,2.5])
% legend('Abs. Diff.','Mean Abs. Diff.')
% subplot(3,1,3);
% plot(time_values, abs(pred_interp_3 - QNt_generated_3))
% hold on
% plot(time_values, mean_graph_values_3, 'r')
% hold off
% title('Absolute Difference in Average Queueing Lengths: Station 3')
% ylabel('Absolute Difference')
% xlabel('Time (s)')
% xlim([0,2.5])
% legend('Abs. Diff.','Mean Abs. Diff.')
% 
% QNt_predicted_truncated = [pred_interp_1, pred_interp_2, pred_interp_3];
% QNt_generated_truncated = [QNt_generated_1, QNt_generated_2, QNt_generated_3];
% % 
% error = 100*max(sum(abs(QNt_predicted_truncated-QNt_generated_truncated),2))/2/200

%% Plot util

subplot(3,1,1); plot(UNt_generated{1,1}.t, UNt_generated{1,1}.metric)
hold on 
plot(UNt_predicted{1,1}.t, UNt_predicted{1,1}.metric)
hold off
title('Station 1')
ylabel('Utilization')
xlabel('Time (s)')
legend('G. Net', 'P. Net')
subplot(3,1,2); plot(UNt_generated{2,1}.t, UNt_generated{2,1}.metric)
hold on
plot(UNt_predicted{2,1}.t, UNt_predicted{2,1}.metric)
hold off
title('Station 2')
ylabel('Utilization')
xlabel('Time (s)')
legend('G. Net', 'P. Net')
subplot(3,1,3); plot(UNt_generated{3,1}.t, UNt_generated{3,1}.metric)
hold on 
plot(UNt_predicted{3,1}.t, UNt_predicted{3,1}.metric)
hold off
title('Station 3')
ylabel('Utilization')
xlabel('Time (s)')
legend('G. Net', 'P. Net')
ylim([0,1.05])