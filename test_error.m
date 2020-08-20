function [ error] = test_error(x, t, N, M,Wstar,lambda)
T = t(1:N)';
X = (get_training_data(x,N,M));
Y_op = X*Wstar;
%Calculate root mean square error in the optimizated W*
error = sqrt(2*(sum((Y_op - T).^2)/2 + (lambda/2)*(sum(Wstar.*Wstar)))/N);

end