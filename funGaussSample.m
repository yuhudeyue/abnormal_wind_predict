function z = funGaussSample( mu, sigma, dim )
%GAUSSAMPLE Summary of this function goes here
%   Detailed explanation goes here

%R = chol(sigma);
%z = repmat(mu,dim(1),1) + randn(dim)*R;
z = mvnrnd(mu, sigma, dim(1));

end
