function [obslik,condi_probs_fluct] = get_obslik(data_rc,o,sigma,depend_table)

N = length(data_rc); %number of data points

tv_S = depend_table(:,2) == 1;
Y = depend_table(tv_S,3); %vector of copy numbers of different entries
mu = log2(Y/2)+o;

S = sum(tv_S);
obslik = zeros(S,N);
condi_probs_fluct = zeros(S,N);

fluct_prob = 1e-5;

for i = 1:length(Y)
    obslik_rc = eval_pdf_rc(data_rc,mu(i),sigma(i));
    if Y(i) == 2
        obslik_rc = 1.4*obslik_rc;
    end
    if Y(i) == 1
        obslik_rc = 1.2*obslik_rc;
    end
    obslik(i,:) = (1-fluct_prob)*obslik_rc+fluct_prob/6;
    condi_probs_fluct(i,:) = (fluct_prob/6)./obslik(i,:);
end

end

