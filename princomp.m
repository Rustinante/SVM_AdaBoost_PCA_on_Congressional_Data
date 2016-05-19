% PRINcipal COMPonent calculator
%   Calculates the principal components of a collection of points.
% Input:
%   X - D-by-N data matrix of N points in D dimensions.
% Output:
%   PCs - A matrix containing the principal components of the data.

function [PCs] = princomp(X)

    cov_mat = cov(X);
    [U S V] = svd(cov_mat);
    PCs = U;
end