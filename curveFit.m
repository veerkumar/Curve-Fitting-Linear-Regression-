%start code for project #1: linear regression
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2018
%Bharadwaj Ravichandran, Jan 2020

%Your Details: (The below details should be included in every matlab script
%file that you create)
%{
    Name:Pramod Kumar
    PSU Email ID:pjk5502@psu.edu
    Description: Script tries to learn W* values from various(4) different approch, namely
   1)Linear regression
   2) Linear regerssion with regularization.
   3) Maximum likelihood
   4) Maximum Posterior
%}

addpath export_fig/

%load the data points
load data.mat
    
%% Start your curve fitting program here

M = 9;
N = 10;
lambda  = 0; %s lambda zero means non regularized
[X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
error_10
figure();
hold on
 shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','y','LineWidth',2},0);
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot (x(1:N),y(1:N),'r')
 hold on
 
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x                       Degree =9')
 ylabel('t,    y(x, W^*)')
 legend('Error Region','','','Optimized(at M = 9)', 'Data(10 pts)','Ground truth')
 export_fig 1_Task1_M9_N10 -png -transparent -r150
 hold off
 
 
 M = 0;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
Wstar_50
error_50
figure();
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 hold on
 plot(x(1:N),Y_op_50,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                       Degree M=0')
 ylabel('t,    y(x, W^*)')
 legend('Data(50 pts)',  'Optimized(at M = 0)','Ground truth')
 export_fig 2_Task1_M0_N50 -png -transparent -r150
 hold off
 
 M = 1;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
Wstar_50
error_50
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_50,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                  Degree M=1')
 ylabel('t,    y(x, W^*)')
 legend('Data(50 pts)',  'Optimized(at M = 1)','Ground truth')
 export_fig 3_Task1_M1_N50 -png -transparent -r150
 hold off
 
 M = 3;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
Wstar_50
error_50
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_50,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                       Degree M=3')
 ylabel('t,    y(x, W^*)')
 legend('Data(50 pts)',  'Optimized(at M = 3)','Ground truth')
 export_fig 4_Task1_M3_N50 -png -transparent -r150
 hold off
 
 M = 6;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
Wstar_50
error_50
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_50,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                       Degree M=6')
 ylabel('t,    y(x, W^*)')
 legend('Data(50 pts)',  'Optimized(at M = 6)','Ground truth')
 export_fig 5_Task1_M6_N50 -png -transparent -r150
 hold off
 
 M = 9;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
Wstar_50
error_50
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_50,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                       Degree M=9')
 ylabel('t,    y(x, W^*)')
 legend('Data(50 pts)',  'Optimized(at M = 9)','Ground truth')
 export_fig 6_Task1_M9_N50 -png -transparent -r150
 hold off
 
 %Bonus point 3
 load data100.mat
 M = 9;
 N = 15;
 lambda  = 0; %s lambda zero means non regularized
[X_15,Y_op_15, Wstar_15, error_15] = optimization(x, y, t, N, M,lambda);
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_15,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X')
 ylabel('t')
 legend('Data(15 pts)',  'Optimized(at M = 9)','Ground truth')
 export_fig 7_Task1_M9_N15 -png -transparent -r150
 hold off
 
  load data100.mat
 M = 9;
 N = 100;
 lambda  = 0; %s lambda zero means non regularized
[X_100,Y_op_100, Wstar_100, error_100] = optimization(x, y, t, N, M,lambda);
figure();
hold on
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot(x(1:N),Y_op_100,'black')
 plot (x(1:N),y(1:N),'b')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X ')
 ylabel('t')
 legend('Data(100 pts)',  'Optimized(at M = 9)','Ground truth')
 export_fig 8_Task1_M9_N100 -png -transparent -r150
 hold off
 
 
 
 
 %% Task 2: Regularigation 
 
load data.mat


%Find error

M = 9;
N = 40;
 
% Divinding data into training (40 points) and test (10point) 
train_x = x(1:40);
train_y = y(1:40);
train_t = t(1:40);
train_err = zeros(1,6);

test_x = x(41:50);
test_y = y(41:50);
test_t = t(41:50);
test_err = zeros(1,6);


% calculate w* on training data for various lambda
[X_40_25,Y_op_40_25, Wstar_40_25, train_err(1,1)] = optimization(train_x, train_y, train_t, N, M,exp(-25));
[X_40_18,Y_op_40_18, Wstar_40_18, train_err(1,2)] = optimization(train_x, train_y, train_t, N, M,exp(-18));
[X_40_15,Y_op_40_15, Wstar_40_15, train_err(1,3)] = optimization(train_x, train_y, train_t, N, M,exp(-15));
[X_40_13,Y_op_40_13, Wstar_40_13, train_err(1,4)] = optimization(train_x, train_y, train_t, N, M,exp(-13));
[X_40_0,Y_op_40_0, Wstar_40_1, train_err(1,5)] = optimization(train_x, train_y, train_t, N, M,exp(-1));
[X_40_0,Y_op_40_0, Wstar_40_1, train_err(1,6)] = optimization(train_x, train_y, train_t, N, M,1);
train_err
N=10;
[ test_err(1,1)] = test_error(test_x, test_t, N, M, Wstar_40_25, exp(-25));
[ test_err(1,2)] = test_error(test_x, test_t, N, M, Wstar_40_18, exp(-18));
[ test_err(1,3)] = test_error(test_x, test_t, N, M, Wstar_40_15, exp(-15));
[ test_err(1,4)] = test_error(test_x, test_t, N, M, Wstar_40_13, exp(-13));
[ test_err(1,5)] = test_error(test_x, test_t, N, M, Wstar_40_1, exp(-1));
[test_err(1,6)] = test_error(test_x, test_t, N, M, Wstar_40_1, exp(-1));
Lambda =[exp(-25), exp(-18), exp(-15), exp(-13),exp(-1),1];
test_err
figure();
% plot(x,t,'ro','MarkerSize',8,'LineWidth',1.5);
%  hold on
 plot(log(Lambda),train_err,'LineWidth',1.5) %Ploting Erms for training set
 hold on
 plot (log(Lambda),test_err,'LineWidth',1.5) %Ploting Erms for testing set
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('Lambda (ln)             Order =9')
 ylabel('Erms')
 legend('Training', 'Testing')
 hold on
 export_fig 1_Task2_ErmsvslnLambda -png -transparent -r150
 
 W=9;
 N=40;
 [X_40_1,Y_op_40_1, Wstar_40_1, error_40_L09] = optimization(train_x, train_y, train_t, N, M,0.9);
error_40_L09
 figure();
hold on
shadedErrorBar(x(1:N),Y_op_40_1,ones(size(x(1:N)))*error_40_L09,{'b-','color','b','LineWidth',2},0);
 plot (x(1:N),y(1:N),'r')
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X ')
 ylabel('t ' )
 legend('Data(40 pts, lambda = 0.9 )',  'Optimized(at M = 9)','Ground truth')
 export_fig 2_Task2_M9_N40_L_09 -png -transparent -r150
 hold off

 W=9;
 N=40;
train_err(1,1)
 figure();
hold on
shadedErrorBar(x(1:N),Y_op_40_25,ones(size(x(1:N)))*train_err(1,1),{'b-','color','b','LineWidth',2},0);
 plot (x(1:N),y(1:N),'r')
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X ')
 ylabel('t ')
 legend('Data(40 pts, ln(lambda) -25)',  'Optimized(at M = 9)','Ground truth')
 export_fig 3_Task2_M9_N40_L_-25 -png -transparent -r150
 hold off
 %% Task3 ML Maximum likelihood

M = 9;
N = 10;
lambda  = 0; %s lambda zero mean non regularized
% Utilizing same function for cacluating W* but error will be calculated
% differently.
[X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
T = t(1:N)';
error_10 = sqrt(sum((Y_op_10 - T).^2)/N)
%plot the groud truth curve
 figure();
  clf
  hold on;
  shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','y','LineWidth',1},0);
  plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
  plot (x(1:N),y(1:N),'b')
  % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x                    MLE order=9')
 ylabel('t,    y(x,w^*)')
 legend('Error Region',  '','','Optimized order 9 ','10 data pts','Ground truth')
% Save the image into a decent resolution
 export_fig Task3_MLEOrder_9_N=10 -png -transparent -r150
 
 M = 9;
 N = 50;
 lambda  = 0; %s lambda zero mean non regularized
% Utilizing same function for cacluating W* but error will be calculated
% differently.
[X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
T = t(1:N)';
error_10 = sqrt(sum((Y_op_10 - T).^2)/N);
%plot the groud truth curve
 figure();
  clf
  hold on;
  shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','b','LineWidth',1},0);
  plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
  plot (x(1:N),y(1:N),'y')
  % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x                    MLE order=9')
 ylabel('t,    y(x,w^*)')
 legend('Error region','','','Optimized','data pts','Ground truth')
% Save the image into a decent resolution
 export_fig Task3_MLEOrder_9_N=50 -png -transparent -r150
 
 M = 9;
 N = 50;
 lambda  = 0; %s lambda zero means non regularized
[X_50,Y_op_50, Wstar_50, error_50] = optimization(x, y, t, N, M,lambda);
figure();
hold on
shadedErrorBar(x(1:N),Y_op_50,ones(size(x(1:N)))*error_50,{'b-','color','b','LineWidth',1},0);
 plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 plot (x(1:N),y(1:N),'r')
 % Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('X                       Degree M=9')
 ylabel('t,    y(x, W^*)')
 legend('Error region', '','','Optimized','data pts','Ground truth')
 export_fig Task3_task1_M0_N50 -png -transparent -r150
 hold off

 
 
  %% MAP  Task 4 Maximum Posterior
 load data.mat
 M=9;
 N=10;
 alpha = 0.005;
 beta = 11.1 ;
 lambda = alpha/beta;
 T = t(1:N)';
 [X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
 error_10 = sqrt(sum((Y_op_10 - T).^2)/N); % Standard deviation
 figure();
  clf
  hold on;
 % shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','y','LineWidth',1},0);
  plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
  plot (x(1:N),Y_op_10,'r')
  plot (x(1:N),y(1:N),'b')
% Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x   Alpha = 0.005 & Beta = 11.1')
 ylabel('t  ')
 legend('Data = 10','Optimized ','Ground truth')
% Save the image into a decent resolution
 export_fig Task4_N10_M9_B005 -png -transparent -r150
 
 M=9;
 N=50;
 alpha = 0.005;
 beta = 11.1 ;
 lambda = alpha/beta;
 T = t(1:N)';
 [X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
 error_10 = sqrt(sum((Y_op_10 - T).^2)/N );
 figure();
  clf
  hold on;
  shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','b','LineWidth',1},0);
  plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
  plot (x(1:N),y(1:N),'r')
% Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x  Alpha = 0.005 & Beta = 11.1')
 ylabel('t')
 legend('Error region', '','', 'Optimized ','Data = 50','Ground truth')
% Save the image into a decent resolution
 export_fig Tast4_N50_M9_B005 -png -transparent -r150
 
 M=9;
 N=50;
 alpha = 6;
 beta = 11.1 ;
 lambda = alpha/beta;
 T = t(1:N)';
 [X_10,Y_op_10, Wstar_10, error_10] = optimization(x, y, t, N, M,lambda);
 error_10 = sqrt((sum((Y_op_10 - T).^2))/N);
 figure();
  clf
  hold on;
 % plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
 % plot (x(1:N),y(1:N),'b')
  shadedErrorBar(x(1:N),Y_op_10,ones(size(x(1:N)))*error_10,{'b-','color','b','LineWidth',1},0);
  plot(x(1:N),t(1:N),'ro','MarkerSize',8,'LineWidth',1.5);
  
  plot (x(1:N),y(1:N),'r')
  
% Make it look good
 grid on;
 set(gca,'FontWeight','bold','LineWidth',2)
 xlabel('x  Alpha = 6 & Beta = 11.1')
 ylabel('t')
 legend('Error Region','','', 'Optimized ','Data = 50','Ground truth')
% Save the image into a decent resolution
 export_fig Tast4_N50_M9_B6 -png -transparent -r150