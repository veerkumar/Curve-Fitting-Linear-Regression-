function [X,Y_op, Wstar, error] = optimization(x, y, t, N, M,lambda)

% construct Nx(M+1) matrix
T = t(1:N)';
X = (get_training_data(x,N,M));
%identity matrix
I = eye(M+1,M+1);
Wstar = (X'*X + (lambda/2) * I)\(X'*T);
Y_op = X*Wstar;

%Calculate root mean square error in the optimizated W*
error = sqrt(2*(sum((Y_op - T).^2)/2 + (lambda/2)*(sum(Wstar.*Wstar)))/N);

end