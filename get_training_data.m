function [X] = get_training_data(x,N,M)
X = ones(1,N);
X = X';
for i =1:(M)
    X = horzcat(X,(x(1:N).^i)');
end

end