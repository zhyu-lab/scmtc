function [LL,prior,transmat,o,sigma,nrIterations] = estimate_paras(init_paras,depend_table,thresh,max_iter,verbose)

global clamp_thres

previous_loglik = -inf;
converged = 0;
num_iter = 1;
LL = [];

prior = init_paras{1};
transmat = init_paras{2};
o = init_paras{3};
sigma = init_paras{4};

while (num_iter <= max_iter) && ~converged
    % perform EM algorithm
    [loglik,exp_num_trans,exp_num_visits1,o_u,sigma_u] = compute_ess(prior,transmat,o,sigma,depend_table);
    
    converged = em_converged_m(loglik,previous_loglik,verbose,thresh);
    
    % update parameters
    if init_paras{5}(1)
        prior = norm_trans(exp_num_visits1',0)';
    end
    if init_paras{5}(2) && ~isempty(exp_num_trans)
        % clamp_thres = 1-1e-4;
        transmat = norm_trans(exp_num_trans,clamp_thres);
    end
    if init_paras{5}(3) %update o here
        o = o_u;
    end
    if init_paras{5}(4) %update sigma here
        sigma = sigma_u;
    end
    
    if verbose
        disp(['sigma:' num2str(sigma) ', o:' num2str(o)]);
        fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik);
    end
    
    num_iter =  num_iter + 1;
    previous_loglik = loglik;
    LL = [LL loglik];
end
nrIterations = num_iter - 1;

end

%--------------------------------------------------------------------------
function [loglik,exp_num_trans,exp_num_visits1,o_u,sigma_u] = compute_ess(prior,transmat,o,sigma,depend_table)
global data_lrc_ds_sep
global gamma_sep
global condi_probs_fluct_sep

numex = length(data_lrc_ds_sep);
S_all = length(transmat); % number of all states 
exp_num_trans = zeros(S_all,S_all);
exp_num_visits1 = zeros(S_all,1);

%-----------------------E step-----------------------------
gamma_sep = cell(1,numex);
condi_probs_fluct_sep = cell(1,numex);
loglik = 0;
N = 0; % the size of the whole data set

for ex = 1:numex %
    
    % conditional probabilities
    [obslik,condi_probs_fluct] = get_obslik(data_lrc_ds_sep{ex},o,sigma,depend_table);
    % Forward and Backward algorithm
    [temp1,gamma,current_ll,temp2,xi_summed] = Forward_Backward_Algorithm(prior,transmat,obslik);
    
    clear temp1 temp2;
    loglik = loglik + current_ll;
    exp_num_trans = exp_num_trans + xi_summed;
    exp_num_visits1 = exp_num_visits1 + gamma(:,1);
    
    gamma_sep{ex} = gamma;
    clear gamma;
    condi_probs_fluct_sep{ex} = condi_probs_fluct;
    clear condi_probs_fluct;
end

%-----------------------M step-----------------------------
% %update paras
% [o_u,sigma_u] = update_parameters(o,sigma,depend_table);

%update o
o_u = update_o(o,depend_table);

%update sigma
sigma_u = update_sigma(o_u,sigma,depend_table);

end

%--------------------------------------------------------------------------
function sigma_u = update_sigma(o,sigma,depend_table)

global data_lrc_ds_sep
global gamma_sep
global condi_probs_fluct_sep

numex = length(data_lrc_ds_sep); % each row is a sample
tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries
mu_l = log2(Y/2)+o;
    
numerators = zeros(1,length(sigma));
denominators = zeros(1,length(sigma));
% numerator = 0;
% denominator = 0;

for ex = 1:numex
    obs_lrc = data_lrc_ds_sep{ex};
    post_probs = gamma_sep{ex}(1:length(Y),:).*(1-condi_probs_fluct_sep{ex}(1:length(Y),:));
    for i = 1:length(Y)         
        numerators(i) = numerators(i)+post_probs(i,:)*((obs_lrc-mu_l(i)).^2)';
        denominators(i) = denominators(i)+sum(post_probs(i,:)); 
%         numerator = numerator+post_probs(i,:)*((obs_lrc-mu_l(i)).^2)';
%         denominator = denominator+sum(post_probs(i,:)); 
    end
end

sigma_u = sqrt(numerators./denominators);
tv = isnan(sigma_u);
sigma_u(tv) = sigma(tv);
% sigma_u = sqrt(numerator/denominator);
% if isnan(sigma_u)
%     sigma_u = sigma;
% end

end

%--------------------------------------------------------------------------
function o_u = update_o(o,depend_table)
global data_lrc_ds_sep
global gamma_sep
global condi_probs_fluct_sep

numex = length(data_lrc_ds_sep); % each row is a sample
tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries

tmp1 = log2(Y/2);

numerator = 0;
denominator = 0;

for ex = 1:numex
    obs_lrc = data_lrc_ds_sep{ex};
    post_probs = gamma_sep{ex}(1:length(Y),:).*(1-condi_probs_fluct_sep{ex}(1:length(Y),:));
    for i = 1:length(Y)         
        numerator = numerator+post_probs(i,:)*(obs_lrc-tmp1(i))';
        denominator = denominator+sum(post_probs(i,:));
    end
end

o_u = numerator/denominator;
if isnan(o_u)
    o_u = o;
end

end

%--------------------------------------------------------------------------
function [o_u, sigma_u] = update_parameters(o,sigma,depend_table)

global data_lrc_ds_sep
global gamma_sep
global condi_probs_fluct_sep

numex = length(data_lrc_ds_sep); % each row is a sample
tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries
mu = Y/2*o;
z = mu./(sigma+eps);

o_numer = 0;
sigma_u = zeros(1,length(Y));
denom_total = 0;
for i = 1:length(Y)
    phi = normpdf(z(i));
    Phi = normcdf(z(i));
    lambda = phi/max(eps,1-Phi);
    sigma_numer = 0;
    n_total = 0;
    for ex = 1:numex
        obs_rc = data_rc_ds_sep{ex};
        post_probs = gamma_sep{ex}(1:length(Y),:).*(1-condi_probs_fluct_sep{ex}(1:length(Y),:));
        I1 = find(obs_rc > 0);
        I0 = find(obs_rc == 0);
        
        n1 = sum(post_probs(i,I1));
        n0 = sum(post_probs(i,I0));
        denom_total = denom_total+n1+n0;
        
        o_numer = o_numer+sum(post_probs(i,I1) .* (2 * obs_rc(I1) / Y(i)));
        y0_mean = mu(i) - sigma(i) * lambda;
        y0_sq = sigma(i)^2 + y0_mean^2;
        o_numer = o_numer + sum(post_probs(i,I0) .* (2 * y0_mean / Y(i)));

        sigma_numer1 = sum(post_probs(i,I1) .* (obs_rc(I1) - mu(i)).^2);
        sigma_numer0 = sum(post_probs(i,I0) .* y0_sq);
        sigma_numer = sigma_numer+sigma_numer1+sigma_numer0;
        n_total = n_total+n1+n0;
    end
    sigma_u(i) = sqrt(sigma_numer/n_total);
end

o_u = o_numer/denom_total;

if isnan(o_u)
    o_u = o;
end

tv = isnan(sigma_u);
sigma_u(tv) = sigma(tv);


end

