function paras = main(init_paras,depend_table,stepsize_ds,thres_EM,max_iter,verbose)

%--------------------------- screening -------------------------
global clamp_thres
global mc_w
clamp_thres = 1-1e-3;
mc_w = 0.8;

%-------------------------------------------------------------------
%               ---> d-sampling screening <---
%-------------------------------------------------------------------
%initialize parameters
thres_del = 0.009;

init_paras = init_hmm_paras(init_paras,depend_table,0);
[LL,paras,p_states,aCN] = screening...
    (stepsize_ds,init_paras,depend_table,thres_EM,max_iter,verbose);

tv_del = depend_table(depend_table(:,2)~=0,3)<1;
p_total_del = sum(p_states(tv_del,:),1);
% disp(num2str(aCN))

tv = (p_total_del<thres_del) & aCN < 6;
candi_indx = find(tv);
candi_ll = LL(tv);
candi_acn = aCN(tv);

[temp,I] = max(candi_ll);
best_indx = candi_indx(I);

[temp,indxs] = sort(candi_ll,'descend');
scores = 1;
acns = candi_acn(indxs(1));
for i = 2:length(candi_acn)
    j = indxs(i-1);
    pre_indxs = indxs(1:i-1);
    k = indxs(i);
    if abs(candi_ll(j)-candi_ll(k)) <= 2 && abs(candi_acn(j)-candi_acn(k)) <= 0.1
        continue;
    end
    score = (candi_acn(pre_indxs)-candi_acn(k)+eps)./(candi_ll(pre_indxs)-candi_ll(k)+eps);
    scores = [scores score(end)];
    acns = [acns candi_acn(k)];
    if sum(score >= 0.02) == i-1 %0.01
        best_indx = candi_indx(k);
    end
end

% tmp = (aCN(best_indx)-aCN(tv)+eps)./(LL(best_indx)-LL(tv)+eps);
%disp(['acn: ' num2str(acns)])
%disp(['score: ' num2str(scores)])
%disp(['ll: ' num2str(LL)])

% Now, use the optimal parameters to call CNA
%-------------------------------------------------------------------
init_paras = init_hmm_paras(paras,depend_table,best_indx);
[temp,paras] = screening(1,init_paras,depend_table,5*thres_EM,20,verbose);
%-------------------------------------------------------------------

end


function hmm_paras = init_hmm_paras(init_paras,depend_table,best_indx)

global clamp_thres
global var_l

hmm_paras = cell(1,5);
%parameter initialization
if best_indx == 0
    %---w---
    if isempty(init_paras{3})
        o_0 = [-1 -0.6 -0.3 0];
    else
        o_0 = init_paras{3};
    end
    
    N = length(o_0);
    hmm_paras{3} = mat2cell(o_0,1,ones(1,N));
    
    %---sigma--- 
    if isempty(init_paras{4})
        S = sum(depend_table(:,2) ~= 0);
        sigma_0 = ones(1,S)*sqrt(var_l);
    else
        sigma_0 = init_paras{4};
    end
    hmm_paras{4} = repmat({sigma_0},1,N);

    %---pi---
    if isempty(init_paras{1})
        S = sum(depend_table(:,2) ~= 0);
        prior_0 = 1/(S)*ones(S,1);
    else
        prior_0 = init_paras{1};
    end
    hmm_paras{1} = repmat({prior_0},1,N);

    %---A---
    if isempty(init_paras{2})
        transmat_0 = norm_trans(ones(S,S),clamp_thres);
    else
        transmat_0 = init_paras{2};
    end
    hmm_paras{2} = repmat({transmat_0},1,N);
    
    %---indicator vector---
    if isempty(init_paras{5}) %indicator vector: '1' for update '0' fixed
        adj_all = ones(1,4);
    else
        adj_all = init_paras{5};
    end
    hmm_paras{5} = repmat({adj_all},1,N);
else %parse the results from previous screening
    for i = 1:length(best_indx)
        %--pi--
        hmm_paras{1} = [hmm_paras{1} init_paras{1}(best_indx(i))];
        %--A--
        hmm_paras{2} = [hmm_paras{2} init_paras{2}(best_indx(i))];
         %--o--
        hmm_paras{3} = [hmm_paras{3} init_paras{3}(best_indx(i))];
        %--sigma--
        hmm_paras{4} = [hmm_paras{4} init_paras{4}(best_indx(i))];
        %--indicator vector--
        hmm_paras{5} = [hmm_paras{5} init_paras{5}(best_indx(i))];
    end
end

end
