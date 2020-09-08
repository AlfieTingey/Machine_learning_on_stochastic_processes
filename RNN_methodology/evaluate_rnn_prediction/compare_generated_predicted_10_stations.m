
clear all 

model_generated = Network('myModel_generated');

x = [250;0;0;0;0;0;0;0;0;0];
% Original server concurrency
% s = [70,10,40,10,20,40,20,30,60,20];
% Fix bottleneck server concurrency
s = [70,40,60,40,20,40,20,30,60,20];
mu = [200,100,50,100,200,500,200,100,200,100];

%% generated model

%% Block 1: nodes
node{1} = Queue(model_generated, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_generated, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_generated, 'QueueStation3', SchedStrategy.FCFS);
node{4} = Queue(model_generated, 'QueueStation4', SchedStrategy.FCFS);
node{5} = Queue(model_generated, 'QueueStation5', SchedStrategy.FCFS);
node{6} = Queue(model_generated, 'QueueStation6', SchedStrategy.FCFS);
node{7} = Queue(model_generated, 'QueueStation7', SchedStrategy.FCFS);
node{8} = Queue(model_generated, 'QueueStation8', SchedStrategy.FCFS);
node{9} = Queue(model_generated, 'QueueStation9', SchedStrategy.FCFS);
node{10} = Queue(model_generated, 'QueueStation10', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));
node{4}.setNumberOfServers(s(4));
node{5}.setNumberOfServers(s(5));
node{6}.setNumberOfServers(s(6));
node{7}.setNumberOfServers(s(7));
node{8}.setNumberOfServers(s(8));
node{9}.setNumberOfServers(s(9));
node{10}.setNumberOfServers(s(10));

%% Block 2: Classes

cclass1 = ClosedClass(model_generated, 'class1', x(1), node{1});
% cclass2 = ClosedClass(model, 'class2', x(2), queue2);
% cclass3 = ClosedClass(model, 'class3', x(3), queue3);

node{2}.setState(x(2));
node{3}.setState(x(3));
node{4}.setState(x(4));
node{5}.setState(x(5));
node{6}.setState(x(6));
node{7}.setState(x(7));
node{8}.setState(x(8));
node{9}.setState(x(9));
node{10}.setState(x(10));

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));
node{4}.setService(cclass1, Exp(mu(4)));
node{5}.setService(cclass1, Exp(mu(5)));
node{6}.setService(cclass1, Exp(mu(6)));
node{7}.setService(cclass1, Exp(mu(7)));
node{8}.setService(cclass1, Exp(mu(8)));
node{9}.setService(cclass1, Exp(mu(9)));
node{10}.setService(cclass1, Exp(mu(10)));

%% Block 3: Topology using Routing Matrix P

 T = [0,0.1,0.1,0.1,0.3,0,0.1,0.1,0.1,0.1;
        0.05,0,0.15,0.2,0.1,0.2,0.1,0,0.1,0.1;
        0.1,0.1,0,0.1,0.2,0.1,0.1,0.1,0.1,0.1;
        0.2,0.1,0,0.2,0.1,0,0.1,0.1,0.1,0.1;
        0.2,0.1,0.1,0.1,0,0.1,0.1,0.1,0.1,0.1;
        0.1,0.1,0.2,0.1,0.1,0,0.1,0.1,0.1,0.1;
        0.1,0.2,0.1,0.1,0.1,0.1,0,0.1,0.1,0.1;
        0.1,0.1,0.1,0.1,0.1,0,0.1,0.3,0,0.1;
        0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0,0.1;
        0.1,0.1,0.1,0.1,0.1,0.2,0.1,0.1,0.1,0];

P = model_generated.initRoutingMatrix;
P{cclass1} = T;
% P{cclass2} = T;
% P{cclass3} = T;
model_generated.link(P);

[Qt_generated,Ut_generated,Tt_generated] = model_generated.getTranHandles();

[QNt_generated,UNt_generated,TNt_generated] = SolverJMT(model_generated,'force', true, 'timespan',[0,10]).getTranAvg(Qt_generated,Ut_generated,Tt_generated);

%% Predicted Model

model_predicted = Network('myModel_predicted');

% mu 184.81764 14.707113 16.87524 13.4063 55.37143 22.420633 22.172466 15.760179 17.810358 13.4173975
% P 
% 0.0 0.13134432 0.13787897 0.1273863 0.18700635 0.023259081 0.06960563 0.13204777 0.07589676 0.11557479 
% 0.85356545 0.0 0.025023576 0.0076542073 0.07700181 0.0027224862 0.004315471 0.01936555 0.0040426166 0.0063088173 
% 0.7043182 0.028209284 0.0 0.0024168885 0.18401304 0.0006456974 0.051313717 0.013188386 0.0087256115 0.0071691005 
% 0.7009549 0.003688074 0.0036525952 0.0 0.16975836 0.00083296926 0.07304246 0.039487805 0.0051818383 0.0034010618 
% 0.30553648 0.020912193 0.5004262 0.117419526 0.0 0.016528849 0.00078984036 0.02503958 0.0015261878 0.011821157 
% 0.59193593 0.08260843 0.057509508 0.013046569 0.14890856 0.0 0.037668522 0.056149155 0.001833016 0.010340367 
% 0.8916271 0.011117203 0.017033473 0.013991601 0.028777469 0.0146850655 0.0 0.0127610015 0.0015729577 0.008434165 
% 0.8410079 0.00861835 0.056797437 0.011811606 0.056863204 0.011000446 0.004636507 0.0 0.0039114533 0.0053531574 
% 0.84918624 0.016750807 0.032410257 0.01651684 0.03132598 0.038480315 0.0010743563 0.009737488 0.0 0.004517707 
% 0.64690644 0.07163138 0.021162242 0.00697124 0.182991 0.010039489 0.002652981 0.055378743 0.0022664461 0.0

% mu 8.826743 13.0607815 9.28407 10.970014 14.628479

mu = [184.81764, 14.707113, 16.87524, 13.4063, 55.37143, 22.420633, 22.172466, 15.760179, 17.810358, 13.4173975];

%% Block 1: nodes
node{1} = Queue(model_predicted, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_predicted, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_predicted, 'QueueStation3', SchedStrategy.FCFS);
node{4} = Queue(model_predicted, 'QueueStation4', SchedStrategy.FCFS);
node{5} = Queue(model_predicted, 'QueueStation5', SchedStrategy.FCFS);
node{6} = Queue(model_predicted, 'QueueStation6', SchedStrategy.FCFS);
node{7} = Queue(model_predicted, 'QueueStation7', SchedStrategy.FCFS);
node{8} = Queue(model_predicted, 'QueueStation8', SchedStrategy.FCFS);
node{9} = Queue(model_predicted, 'QueueStation9', SchedStrategy.FCFS);
node{10} = Queue(model_predicted, 'QueueStation10', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));
node{4}.setNumberOfServers(s(4));
node{5}.setNumberOfServers(s(5));
node{6}.setNumberOfServers(s(6));
node{7}.setNumberOfServers(s(7));
node{8}.setNumberOfServers(s(8));
node{9}.setNumberOfServers(s(9));
node{10}.setNumberOfServers(s(10));

%% Block 2: Classes

cclass1 = ClosedClass(model_predicted, 'class1', x(1), node{1});
% cclass2 = ClosedClass(model, 'class2', x(2), queue2);
% cclass3 = ClosedClass(model, 'class3', x(3), queue3);

node{2}.setState(x(2))
node{3}.setState(x(3))
node{4}.setState(x(4))
node{5}.setState(x(5))
node{6}.setState(x(6));
node{7}.setState(x(7));
node{8}.setState(x(8));
node{9}.setState(x(9));
node{10}.setState(x(10));

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));
node{4}.setService(cclass1, Exp(mu(4)));
node{5}.setService(cclass1, Exp(mu(5)));
node{6}.setService(cclass1, Exp(mu(6)));
node{7}.setService(cclass1, Exp(mu(7)));
node{8}.setService(cclass1, Exp(mu(8)));
node{9}.setService(cclass1, Exp(mu(9)));
node{10}.setService(cclass1, Exp(mu(10)));


%% Block 3: Topology using Routing Matrix P

T = [0.0, 0.13134432, 0.13787897, 0.1273863, 0.18700635, 0.023259081, 0.06960563, 0.13204777, 0.07589676, 0.11557479; 
0.85356545, 0.0, 0.025023576, 0.0076542073, 0.07700181, 0.0027224862, 0.004315471, 0.01936555, 0.0040426166, 0.0063088173; 
0.7043182, 0.028209284, 0.0, 0.0024168885, 0.18401304, 0.0006456974, 0.051313717, 0.013188386, 0.0087256115, 0.0071691005; 
0.7009549, 0.003688074, 0.0036525952, 0.0, 0.16975836, 0.00083296926, 0.07304246, 0.039487805, 0.0051818383, 0.0034010618; 
0.30553648, 0.020912193, 0.5004262, 0.117419526, 0.0, 0.016528849, 0.00078984036, 0.02503958, 0.0015261878, 0.011821157; 
0.59193593, 0.08260843, 0.057509508, 0.013046569, 0.14890856, 0.0, 0.037668522, 0.056149155, 0.001833016, 0.010340367; 
0.8916271, 0.011117203, 0.017033473, 0.013991601, 0.028777469, 0.0146850655, 0.0, 0.0127610015, 0.0015729577, 0.008434165; 
0.8410079, 0.00861835, 0.056797437, 0.011811606, 0.056863204, 0.011000446, 0.004636507, 0.0, 0.0039114533, 0.0053531574; 
0.84918624, 0.016750807, 0.032410257, 0.01651684, 0.03132598, 0.038480315, 0.0010743563, 0.009737488, 0.0, 0.004517707; 
0.64690644, 0.07163138, 0.021162242, 0.00697124, 0.182991, 0.010039489, 0.002652981, 0.055378743, 0.0022664461, 0.0];

P = model_predicted.initRoutingMatrix;
P{cclass1} = T;
% P{cclass2} = T;
% P{cclass3} = T;
model_predicted.link(P);

[Qt_predicted,Ut_predicted,Tt_predicted] = model_predicted.getTranHandles();

[QNt_predicted,UNt_predicted,TNt_predicted] = SolverJMT(model_predicted,'force', true, 'timespan',[0,10]).getTranAvg(Qt_predicted,Ut_predicted,Tt_predicted);

%% Plot average queue length

% QNt_predicted_interp_station1 = interp1(QNt_predicted{1,1}.t,QNt_predicted{1,1}.metric,QNt_generated{1,1}.t);
% QNt_predicted_interp_station2 = interp1(QNt_predicted{2,1}.t,QNt_predicted{2,1}.metric,QNt_generated{2,1}.t);
% QNt_predicted_interp_station3 = interp1(QNt_predicted{3,1}.t,QNt_predicted{3,1}.metric,QNt_generated{3,1}.t);
% QNt_predicted_interp_station4 = interp1(QNt_predicted{4,1}.t,QNt_predicted{4,1}.metric,QNt_generated{4,1}.t);
% QNt_predicted_interp_station5 = interp1(QNt_predicted{5,1}.t,QNt_predicted{5,1}.metric,QNt_generated{5,1}.t);
% QNt_predicted_interp_station6 = interp1(QNt_predicted{6,1}.t,QNt_predicted{6,1}.metric,QNt_generated{6,1}.t);
% QNt_predicted_interp_station7 = interp1(QNt_predicted{7,1}.t,QNt_predicted{7,1}.metric,QNt_generated{7,1}.t);
% QNt_predicted_interp_station8 = interp1(QNt_predicted{8,1}.t,QNt_predicted{8,1}.metric,QNt_generated{8,1}.t);
% QNt_predicted_interp_station9 = interp1(QNt_predicted{9,1}.t,QNt_predicted{9,1}.metric,QNt_generated{9,1}.t);
% QNt_predicted_interp_station10 = interp1(QNt_predicted{10,1}.t,QNt_predicted{10,1}.metric,QNt_generated{10,1}.t);
% 
% subplot(5,2,1); plot(QNt_generated{1,1}.t, QNt_generated{1,1}.metric)
% hold on 
% plot(QNt_generated{1,1}.t, QNt_predicted_interp_station1, 'r')
% hold off
% title('Station 1')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,2); plot(QNt_generated{2,1}.t, QNt_generated{2,1}.metric)
% hold on 
% plot(QNt_generated{2,1}.t, QNt_predicted_interp_station2, 'r')
% hold off
% title('Station 2')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,3); plot(QNt_generated{3,1}.t, QNt_generated{3,1}.metric)
% hold on 
% plot(QNt_generated{3,1}.t, QNt_predicted_interp_station3, 'r')
% hold off
% title('Station 3')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,4); plot(QNt_generated{4,1}.t, QNt_generated{4,1}.metric)
% hold on 
% plot(QNt_generated{4,1}.t, QNt_predicted_interp_station4, 'r')
% hold off
% title('Station 4')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,5); plot(QNt_generated{5,1}.t, QNt_generated{5,1}.metric)
% hold on 
% plot(QNt_generated{5,1}.t, QNt_predicted_interp_station5, 'r')
% hold off
% title('Station 5')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,6); plot(QNt_generated{6,1}.t, QNt_generated{6,1}.metric)
% hold on 
% plot(QNt_generated{6,1}.t, QNt_predicted_interp_station6, 'r')
% hold off
% title('Station 6')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,7); plot(QNt_generated{7,1}.t, QNt_generated{7,1}.metric)
% hold on 
% plot(QNt_generated{7,1}.t, QNt_predicted_interp_station7, 'r')
% hold off
% title('Station 7')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,8); plot(QNt_generated{8,1}.t, QNt_generated{8,1}.metric)
% hold on 
% plot(QNt_generated{8,1}.t, QNt_predicted_interp_station8, 'r')
% hold off
% title('Station 8')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,9); plot(QNt_generated{9,1}.t, QNt_generated{9,1}.metric)
% hold on 
% plot(QNt_generated{9,1}.t, QNt_predicted_interp_station9, 'r')
% hold off
% title('Station 9')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')
% subplot(5,2,10); plot(QNt_generated{10,1}.t, QNt_generated{10,1}.metric)
% hold on 
% plot(QNt_generated{10,1}.t, QNt_predicted_interp_station10, 'r')
% hold off
% title('Station 10')
% ylabel('Av. Q. Len')
% xlabel('Time (s)')

%% plot Accuracy and Mean absolute difference


QNt_predicted_interp_station1 = interp1(QNt_predicted{1,1}.t,QNt_predicted{1,1}.metric,QNt_generated{1,1}.t);
QNt_predicted_interp_station2 = interp1(QNt_predicted{2,1}.t,QNt_predicted{2,1}.metric,QNt_generated{2,1}.t);
QNt_predicted_interp_station3 = interp1(QNt_predicted{3,1}.t,QNt_predicted{3,1}.metric,QNt_generated{3,1}.t);
QNt_predicted_interp_station4 = interp1(QNt_predicted{4,1}.t,QNt_predicted{4,1}.metric,QNt_generated{4,1}.t);
QNt_predicted_interp_station5 = interp1(QNt_predicted{5,1}.t,QNt_predicted{5,1}.metric,QNt_generated{5,1}.t);
QNt_predicted_interp_station6 = interp1(QNt_predicted{6,1}.t,QNt_predicted{6,1}.metric,QNt_generated{6,1}.t);
QNt_predicted_interp_station7 = interp1(QNt_predicted{7,1}.t,QNt_predicted{7,1}.metric,QNt_generated{7,1}.t);
QNt_predicted_interp_station8 = interp1(QNt_predicted{8,1}.t,QNt_predicted{8,1}.metric,QNt_generated{8,1}.t);
QNt_predicted_interp_station9 = interp1(QNt_predicted{9,1}.t,QNt_predicted{9,1}.metric,QNt_generated{9,1}.t);
QNt_predicted_interp_station10 = interp1(QNt_predicted{10,1}.t,QNt_predicted{10,1}.metric,QNt_generated{10,1}.t);
% 
length_predict = length(QNt_predicted_interp_station1)-100;
pred_interp_1 = QNt_predicted_interp_station1(1:length_predict-100);
pred_interp_2 = QNt_predicted_interp_station2(1:length_predict-100);
pred_interp_3 = QNt_predicted_interp_station3(1:length_predict-100);
pred_interp_4 = QNt_predicted_interp_station4(1:length_predict-100);
pred_interp_5 = QNt_predicted_interp_station5(1:length_predict-100);
pred_interp_6 = QNt_predicted_interp_station6(1:length_predict-100);
pred_interp_7 = QNt_predicted_interp_station7(1:length_predict-100);
pred_interp_8 = QNt_predicted_interp_station8(1:length_predict-100);
pred_interp_9 = QNt_predicted_interp_station9(1:length_predict-100);
pred_interp_10 = QNt_predicted_interp_station10(1:length_predict-100);

QNt_generated_1 = QNt_generated{1,1}.metric(1:length_predict-100);
QNt_generated_2 = QNt_generated{2,1}.metric(1:length_predict-100);
QNt_generated_3 = QNt_generated{3,1}.metric(1:length_predict-100);
QNt_generated_4 = QNt_generated{4,1}.metric(1:length_predict-100);
QNt_generated_5 = QNt_generated{5,1}.metric(1:length_predict-100);
QNt_generated_6 = QNt_generated{6,1}.metric(1:length_predict-100);
QNt_generated_7 = QNt_generated{7,1}.metric(1:length_predict-100);
QNt_generated_8 = QNt_generated{8,1}.metric(1:length_predict-100);
QNt_generated_9 = QNt_generated{9,1}.metric(1:length_predict-100);
QNt_generated_10 = QNt_generated{10,1}.metric(1:length_predict-100);

mean_graph_values_1 = mean(abs(pred_interp_1 - QNt_generated_1))*ones(length_predict-100,1);
mean_graph_values_2 = mean(abs(pred_interp_2 - QNt_generated_2))*ones(length_predict-100,1);
mean_graph_values_3 = mean(abs(pred_interp_3 - QNt_generated_3))*ones(length_predict-100,1);
mean_graph_values_4 = mean(abs(pred_interp_4 - QNt_generated_4))*ones(length_predict-100,1);
mean_graph_values_5 = mean(abs(pred_interp_5 - QNt_generated_5))*ones(length_predict-100,1);
mean_graph_values_6 = mean(abs(pred_interp_6 - QNt_generated_6))*ones(length_predict-100,1);
mean_graph_values_7 = mean(abs(pred_interp_7 - QNt_generated_7))*ones(length_predict-100,1);
mean_graph_values_8 = mean(abs(pred_interp_8 - QNt_generated_8))*ones(length_predict-100,1);
mean_graph_values_9 = mean(abs(pred_interp_9 - QNt_generated_9))*ones(length_predict-100,1);
mean_graph_values_10 = mean(abs(pred_interp_10 - QNt_generated_10))*ones(length_predict-100,1);
% 
time_values = QNt_generated{1,1}.t(1:length_predict-100);

subplot(5,2,1);
plot(time_values, abs(pred_interp_1 - QNt_generated_1))
hold on
plot(time_values, mean_graph_values_1, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 1')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,2);
plot(time_values, abs(pred_interp_2 - QNt_generated_2))
hold on
plot(time_values, mean_graph_values_2, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 2')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,3);
plot(time_values, abs(pred_interp_3 - QNt_generated_3))
hold on
plot(time_values, mean_graph_values_3, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 3')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,4);
plot(time_values, abs(pred_interp_4 - QNt_generated_4))
hold on
plot(time_values, mean_graph_values_4, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 4')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,5);
plot(time_values, abs(pred_interp_5 - QNt_generated_5))
hold on
plot(time_values, mean_graph_values_5, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 5')
ylabel('Absolute Difference')
xlabel('Time (s)')

subplot(5,2,6);
plot(time_values, abs(pred_interp_6 - QNt_generated_6))
hold on
plot(time_values, mean_graph_values_6, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 6')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,7);
plot(time_values, abs(pred_interp_7 - QNt_generated_7))
hold on
plot(time_values, mean_graph_values_7, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 7')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,8);
plot(time_values, abs(pred_interp_8 - QNt_generated_8))
hold on
plot(time_values, mean_graph_values_8, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 8')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,9);
plot(time_values, abs(pred_interp_9 - QNt_generated_9))
hold on
plot(time_values, mean_graph_values_9, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 9')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,2,10);
plot(time_values, abs(pred_interp_10 - QNt_generated_10))
hold on
plot(time_values, mean_graph_values_10, 'r')
hold off
title('Abs. Diff. in Av. Queueing Len: Station 10')
ylabel('Absolute Difference')
xlabel('Time (s)')

QNt_predicted_truncated = [pred_interp_1, pred_interp_2, pred_interp_3, pred_interp_4, pred_interp_5, pred_interp_6, pred_interp_7, pred_interp_8, pred_interp_9, pred_interp_10];
QNt_generated_truncated = [QNt_generated_1, QNt_generated_2, QNt_generated_3, QNt_generated_4, QNt_generated_5, QNt_generated_6, QNt_generated_7, QNt_generated_8, QNt_generated_9, QNt_generated_10];
% 
error = 100*max(sum(abs(QNt_predicted_truncated-QNt_generated_truncated),2))/2/250

%% Plot util

% subplot(5,2,1); plot(UNt_generated{1,1}.t, UNt_generated{1,1}.metric)
% hold on 
% plot(UNt_predicted{1,1}.t, UNt_predicted{1,1}.metric, 'r')
% hold off
% title('Station 1')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,2); plot(UNt_generated{2,1}.t, UNt_generated{2,1}.metric)
% hold on 
% plot(UNt_predicted{2,1}.t, UNt_predicted{2,1}.metric, 'r')
% hold off
% title('Station 2')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,3); plot(UNt_generated{3,1}.t, UNt_generated{3,1}.metric)
% hold on 
% plot(UNt_predicted{3,1}.t, UNt_predicted{3,1}.metric, 'r')
% hold off
% title('Station 3')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,4); plot(UNt_generated{4,1}.t, UNt_generated{4,1}.metric)
% hold on 
% plot(UNt_predicted{4,1}.t, UNt_predicted{4,1}.metric, 'r')
% hold off
% title('Station 4')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,5); plot(UNt_generated{5,1}.t, UNt_generated{5,1}.metric)
% hold on 
% plot(UNt_predicted{5,1}.t, UNt_predicted{5,1}.metric, 'r')
% hold off
% title('Station 5')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,6); plot(UNt_generated{6,1}.t, UNt_generated{6,1}.metric)
% hold on 
% plot(UNt_predicted{6,1}.t, UNt_predicted{6,1}.metric, 'r')
% hold off
% title('Station 6')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,7); plot(UNt_generated{7,1}.t, UNt_generated{7,1}.metric)
% hold on 
% plot(UNt_predicted{7,1}.t, UNt_predicted{7,1}.metric, 'r')
% hold off
% title('Station 7')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,8); plot(UNt_generated{8,1}.t, UNt_generated{8,1}.metric)
% hold on 
% plot(UNt_predicted{8,1}.t, UNt_predicted{8,1}.metric, 'r')
% hold off
% title('Station 8')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,9); plot(UNt_generated{9,1}.t, UNt_generated{9,1}.metric)
% hold on 
% plot(UNt_predicted{9,1}.t, UNt_predicted{9,1}.metric, 'r')
% hold off 
% title('Station 9')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
% subplot(5,2,10); plot(UNt_generated{10,1}.t, UNt_generated{10,1}.metric)
% hold on 
% plot(UNt_predicted{10,1}.t, UNt_predicted{10,1}.metric, 'r')
% hold off
% title('Station 10')
% ylabel('Utilization')
% xlabel('Time (s)')
% xlim([0,8])
