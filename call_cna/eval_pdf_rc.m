function results = eval_pdf_rc(data,mu,sigma)

if size(data,1) > size(data,2) %Nx1->1xN
    data = data';
end

% results = zeros(size(data));
% tv_zero = data == 0;
% results(~tv_zero) = normpdf(data(~tv_zero),mu,sigma);
% results(tv_zero) = 1-normcdf(mu/(sigma+eps));

results = normpdf(data,mu,sigma);

end