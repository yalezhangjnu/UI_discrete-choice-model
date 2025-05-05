function app_givasf_scc3(ncores,ind)
% File name:   main_givasf_revise.m
% Description: This is the main file used for revising KZ(2025) empirical
%              part results
% Date:        2025/3/31
% Author:      Hiroaki Kaido and Yi Zhang   
% References:  Universal Inference for Incomplete Discrete Choice Model

%% Basic Parameter Setup
cv_alphlevel          = 0.05;         % critical value of test-statistic
As = {                                % This is the case for mu1w<mu0w with cdc events
    {'00', '01'};
    {'10', '11'};
    {'00', '01', '10'};
    {'01', '10', '11'};
};
%%%%%%%%%%%%% counterfactual (cf) parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cf_obj                = 'CASF';        % counterfactual objective function setup: ASF or CASF
cf_csize              = 'small';       % conditional sample with small, median, and large
cf_decision           = 1;            % counterfactual value of desicion: 0
cf_gridpoints         = 200;          % grid points to choose
cf_asfvalues          = linspace(0,1,cf_gridpoints); % gird space from [0,1] for ASF function
%%%%%%%%%%%%%%%%%%%%%%%% CHT(2007) parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cht_multistart        = 1;            % multistart in CHT method if equals to 1
cht_maxiter           = 2000;         % The maximum number of iteration for CHT
cht_k                 = 20;           % the number of clusters for x discretization
cht_ms                = 0;            % The moment selection criterion
%%%%%%%%%%%%%%%%%%%%%%%% RMLE parameter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmle_msnum            = 24;            % the number of multistart points we are going to use in rmle
rmle_maxiter          = 2000;         % The maximum number of iteration for RMLE
bank_thresholds        = [274000, 1098000];

rng('default')
%% Step 0: get clean data and discretization of X support
rawdat = readtable('rawdatafinal_2010.csv');   % read rawdata
[sampleD0, sampleD1, idx_perm, normalized_thresholds]=get_cleandata(rawdat, bank_thresholds);
discfile = 'disc_vars.mat';
if exist(discfile,'file') == 2
    load(discfile, 'xsupp_D1', 'xsupp_D0', 'idx_D1', 'idx_D0');
else
    [idx_D1,idx_allD1, all_centersD1,idx_D0,idx_allD0, all_centersD0,xsupp_D1,xsupp_D0]=get_discritization(sampleD0, sampleD1, cht_k, cht_maxiter);
    save(discfile,'xsupp_D1', 'xsupp_D0', 'idx_D1', 'idx_D0');
end
% save the subsample
csvwrite('sampleD0.csv', sampleD0);
csvwrite('sampleD1.csv', sampleD1);

%% Step 1 update: Calculate the CHT based on discritization of Xsupp with moment selections 
initfile = 'init_estd.mat';
if exist(initfile, 'file') == 2
  % Load the file if it exists
  load(initfile,'theta_hat0d','theta_hat1d');
else
   %Execute the functions and save the variables if the file doesn't exist
   rng('default')
   [theta_hat0d, cht_exitflags0d,cht_fvals0d, cht_thetahats0d, cht_allsolvers0d] = get_cht_givrevise(sampleD0, As, xsupp_D0, idx_D0, cht_multistart, cht_ms); 
   rng('default')
   [theta_hat1d, cht_exitflags1d,cht_fvals1d, cht_thetahats1d, cht_allsolvers1d] = get_cht_givrevise(sampleD1, As, xsupp_D1, idx_D1, cht_multistart, cht_ms); 
   save(initfile, 'theta_hat0d', 'theta_hat1d');
end

% Check ASF VALUES
%asf0=get_asf(sampleD0, theta_hat0, cf_obj, cf_decision);
%asf0d=get_asf(sampleD0, theta_hat0d,cf_obj,cf_decision);
%asf1=get_asf(sampleD1, theta_hat1,cf_obj, cf_decision);
%asf1d=get_asf(sampleD1, theta_hat1d,cf_obj, cf_decision);

%% Generate Indexes for small, median, and large banks based on CRA thresholds
small_D0 = (sampleD0(:,12) <= normalized_thresholds(1,1));
median_D0 = (normalized_thresholds(1,1) < sampleD0(:,12)) & (sampleD0(:,12) <= normalized_thresholds(1,2));
large_D0 = (sampleD0(:,12) > normalized_thresholds(1,2));

small_D1 = (sampleD1(:,12) <= normalized_thresholds(1,1));
median_D1 = (normalized_thresholds(1,1) < sampleD1(:,12)) & (sampleD1(:,12) <= normalized_thresholds(1,2));
large_D1 = (sampleD1(:,12) > normalized_thresholds(1,2));
%% Check Conditioanl ASF VALUES
casfD0_small=get_casf(sampleD0,small_D0,theta_hat0d,cf_obj,cf_decision);
casfD0_median=get_casf(sampleD0,median_D0,theta_hat0d,cf_obj,cf_decision);
casfD0_large=get_casf(sampleD0,large_D0,theta_hat0d,cf_obj,cf_decision);

casfD1_small=get_casf(sampleD1,small_D1,theta_hat1d,cf_obj,cf_decision);
casfD1_median=get_casf(sampleD1,median_D1,theta_hat0d,cf_obj,cf_decision);
casfD1_large=get_casf(sampleD1,large_D1,theta_hat1d,cf_obj,cf_decision);

if strcmp(cf_csize,'small')
    cidx0=small_D0;
    cidx1=small_D1;
elseif strcmp(cf_csize, 'median')
    cidx0=median_D0;
    cidx1=median_D1;
elseif strcmp(cf_csize, 'large')
    cidx0=large_D0;
    cidx1=large_D1;
else
    disp('wrong bank category')
end
%% Step 1.5 Get an Empirical Dist
[p_emp_D1,frequency_D1,n_cluster_D1]=get_empiricaldist_givrevise(sampleD1, idx_D1, cht_k, 4);
[p_emp_D0,frequency_D0,n_cluster_D0]=get_empiricaldist_givrevise(sampleD0, idx_D0, cht_k, 4);
%% Step 2: Construct the P values
p1 = get_pmat_giv(theta_hat1d, As, sampleD0, frequency_D0, idx_D0);
p0 = get_pmat_giv(theta_hat0d, As, sampleD1, frequency_D1, idx_D1);

% Use p1 and D0 construct logL1
[logL1, selecp1] = get_logp(sampleD0,p1);     
% Use p0 and D1 construct logL0
[logL0, selecp0] = get_logp(sampleD1,p0);

cf_asfvalue=cf_asfvalues(1,ind); % notice that cluster always need to change this line

rng('default')
[theta_rmle0asf,rmle_solutions0asf, rmle_exitflags0asf, rmle_fvals0asf, logL_rmle0asf] = get_rmle_asf(sampleD0, p1, cidx0, cf_obj, cf_asfvalue, cf_decision, rmle_msnum, rmle_maxiter);
rng('default')
[theta_rmle1asf,rmle_solutions1asf, rmle_exitflags1asf, rmle_fvals1asf, logL_rmle1asf] = get_rmle_asf(sampleD1, p0, cidx1, cf_obj, cf_asfvalue, cf_decision, rmle_msnum, rmle_maxiter);

%% Step 4: Calculate the test statistics
Tn_asf = exp(logL1-logL_rmle0asf);
Tn_swapasf = exp(logL0-logL_rmle1asf);
Sn_asf = (Tn_asf+Tn_swapasf)/2;
test_resultd=Sn_asf>1/cv_alphlevel;

if strcmp(cf_obj,'ASF')
    save(['/project/seteff/KZ24_UI_empirical/Results/app_parametricasf_d' num2str(cf_decision) '_cfsizevalue' num2str(0) '_ind',num2str(ind)]);
else
    save(['/project/seteff/KZ24_UI_empirical/Results/app_paraconasf_d' num2str(cf_decision) '_cfsizevalue' num2str(0) '_ind',num2str(ind)]);
end
end


function [sampleD0, sampleD1, idx_perm, normalized_thresholds]=get_cleandata(rawdat, size_threshold)
% Description:  Data processing fot rawdata
columns=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]; % select data we need from raw data set: in total 16, 2 outcomes, 12 covaraiates and 2 IV
dat=rawdat{:,columns};                         % dimension: 6201 by 16
[row_indices, ~] = find(isnan(dat));           % find missing value indexes
unique_missing = unique(row_indices);          % return a unique missing value indexes
dat_clean=dat;                                 % record clean data
dat_clean(unique_missing,:)=[];                % get data without missing value [D_i(1), Y_i(1), W_I(12), Z_i(2)], two possible IVs dimension: 5987 by 29
%% This section normalized the thresholds for bank size
banksize=dat_clean(:,12);
min_val = min(banksize);
max_val = max(banksize);
normalized_thresholds= (size_threshold-min_val)/(max_val - min_val);
dat_normalized = dat_clean;                    % record normalized data
for q = 3:16                                   % discrete choice model needs a min-max normalization of data between [0,1], discrete varable does not need for that
    if q ~= 5
        dat_normalized(:,q) = normalize_var(dat_clean(:,q));
    elseif q==5
        dat_normalized(:,q) = dat_clean(:,q); % data contains a negative sign for management quality
    end
end
[sampleD0, sampleD1, idx_perm]=get_splitsamplernd(dat_normalized); % This sample-splitting includes two IVs
end

function hatQ1 = get_samplecriterion_givrevise(sampleD, As, theta, xsupp_D, idx, mscondition)
% This function is used to compute the criterion function of theta_hat1 based on equation (2.11)
% Inputs:
% Sample_D      : spliting sample for D
% As            : the collection of finite number of evenets
% theta         : structural parameter in the first equation
% delta         : structural paremeter in the second equation
% xsupp_D       : clustering all support data
% idx           : clustering index of each individual
J=length(As);          % numebr of events we need to consider
[k,~] = size(xsupp_D); % number of support for discretization of support x, k is the same as number of discretization of clusters.
nu_A=zeros(k,J);       % store the lower bound for different As times different support x, first column for 0 outcome event, and second column for 1 outcome event
[p_emp,~,n_cluster]=get_empiricaldist_givrevise(sampleD, idx, k, J);   % p_emp dimension is k by J (K by 4 in our case)
m = NaN(k,J);          % store all moments inequalitys (the total number is the same as the lower bound)
conditions=NaN(k,J);
pc_emp=1-p_emp;
temp=p_emp.*pc_emp;
ms=NaN(k,J);
for i=1:k
    conditions(i,:)=n_cluster(i,1).* temp(i,:);    
end
if mscondition==0      % threshold with 0
    ms=conditions>0;
elseif mscondition==5  % threshold with 5
    ms=conditions>5; 
elseif mscondition==10  % threshold with 10
    ms=conditions>10; 
else
    %disp('wrongrestrictions')
end
for l=1:k
    xsupp=xsupp_D(l,:);
    for j=1:J
        nu_A(l,j) = get_nu_giv(theta, As{j}, xsupp);
        m(l,j)    = nu_A(l,j)-p_emp(l,j); % each moment is lower bound - empirical distribution
    end
end
% select moment inequality with different restrictions
final=m(ms);
hatQ1 = max(final);
end


function [idx_D1,idx_allD1, all_centersD1,idx_D0,idx_allD0, all_centersD0,xsupp_D1,xsupp_D0]=get_discritization(sampleD0, sampleD1,cht_k,cht_maxiter)
% Description: discritization of covariatates used for CHT step
D1_covariates = [sampleD1(:,3:end)]; % get data with W for D1 (xsupp) dimension: 2994 by 14 [W_I(12), Z_i(2)]
D0_covariates = [sampleD0(:,3:end)]; % get data with W for D0 (xsupp) dimension: 2993 by 14 [W_I(12), Z_i(2)]
tempD1 = [D1_covariates(:,1:2),-D1_covariates(:,3),D1_covariates(:,4:end)]; % Third column is management quality. 
tempD0 = [D0_covariates(:,1:2),-D0_covariates(:,3),D0_covariates(:,4:end)];
[~,col] = size(D1_covariates);                     % calculate the sample size n0 (or n1) and the total number of covariates col
inputtype = zeros(1,col);                          % specify the input types: 1 as categorical, and 0 as numerical, gen all zeros first
inputtype(1,3)=1;                                  % The third column is management quality is discrete 
tic                                                % For k=20 and iter_max=100, the approximate time is 820 seconds
[idx_D1,idx_allD1, all_centersD1, x_D1]=kmean_xsupportgen(tempD1, inputtype, cht_k, cht_maxiter);  % the xsupp returns continuous variable frist, then categorical variable follows sequentially
[idx_D0,idx_allD0, all_centersD0, x_D0]=kmean_xsupportgen(tempD0, inputtype, cht_k, cht_maxiter);
toc
xsupp_D1 = [x_D1(:,1:13), -x_D1(:,14)];
xsupp_D0 = [x_D0(:,1:13), -x_D0(:,14)];
end


function hatQ1 = get_sampleccp(sampleD, As, theta, xsupp_D, idx)
% This function is used to compute the criterion function of theta_hat1 based on equation (2.11)
% Inputs:
% Sample_D      : spliting sample for D
% As            : the collection of finite number of evenets
% theta         : structural parameter in the first equation
% delta         : structural paremeter in the second equation
% xsupp_D       : clustering all support data
% idx           : clustering index of each individual
J=length(As);          % numebr of events we need to consider
[k,~] = size(xsupp_D); % number of support for discretization of support x, k is the same as number of discretization of clusters.
Y = sampleD(:,2); 
d = sampleD(:,1);

frequency=zeros(k,J);
n_cluster=zeros(k,1);
for i=1:k
    n_cluster(i,1) = sum((idx==i));   
    num00 = sum((Y==0).*(d==0).*(idx==i));
    num01 = sum((Y==0).*(d==1).*(idx==i));
    num10 = sum((Y==1).*(d==0).*(idx==i));
    num11 = sum((Y==1).*(d==1).*(idx==i));
    frequency(i,1)=num00/n_cluster(i,1);
    frequency(i,2)=num01/n_cluster(i,1);
    frequency(i,3)=num10/n_cluster(i,1);
    frequency(i,4)=num11/n_cluster(i,1);
end
end

function [thetahat, exitflags, fvals, thetahats, allsolvers] = get_cht_givrevise(sample_D, As, xsupp_D, idx, multistartopt,cht_ms)
% Description: Compute the thetahat based on CHT(2007) :revise
% samplecriterion selection
% Inputs:
% sampleD           :  splitting sampleD
% As                :  collection of cdc events
% xsupp_D           :  discritization support of covariates x
% idx               :  index of clustering indicators
% multistartopt     :  1 or 0. whether use multistart options for compute the CHT
fun = @(theta) get_samplecriterion_givrevise(sample_D, As, theta, xsupp_D, idx, cht_ms); 
if multistartopt == 0
    lb = (-2)*ones(1,15);
    ub = 2*ones(1,15); % need a strictly negative value
    options = optimset('display','off', 'LargeScale','off', 'MaxFunEvals',100000, 'TolFun',1e-8, 'TolX',1e-8);
    initialguess =-0.1*ones(15,1);
    A=[1,1,zeros(1,13)
        1,zeros(1,14)]; % not a single constraint but a nx by 1 constrain for a single parameter? every beta0+beta1*size_var <= 1e-6
    b=[-1e-6;-1e-6];
    thetahat = fmincon(fun,initialguess,A,b,[],[],lb,ub,[], options);
else
    initialguess =-0.1*ones(15,1);
    lb = (-2)*ones(1,15);
    ub = 2*ones(1,15);
    opts = optimoptions(@fmincon,'Algorithm','sqp');
    A=[1,1,zeros(1,13)
        1,zeros(1,14)]; % not a single constraint but a nx by 1 constrain for a single parameter? every beta0+beta1*size_var <= 1e-6
    b=[-1e-6;-1e-6];
    problem = createOptimProblem('fmincon','objective',...
    fun,'x0',initialguess,'Aineq', A, 'bineq', b, 'lb',lb,'ub',ub,'options',opts);
    ms = MultiStart('FunctionTolerance',1e-8,'XTolerance',1e-8, 'Display', 'iter');
    [~,~,~,~,allsolvers] = run(ms,problem,50);
    exitflags=zeros(length(allsolvers),1);
    fvals    =zeros(length(allsolvers),1);
    thetahats=zeros(length(allsolvers),length(initialguess));
    for i=1:length(allsolvers)
        exitflags(i)=allsolvers(i).Exitflag;
        fvals(i)=allsolvers(i).Fval;
        thetahats(i,:)=allsolvers(i).X;
    end
    % use exitflag equals to 1 if possible
    valid_idx = find(exitflags == 1); 
    if ~isempty(valid_idx)
        [~, best_idx] = min(fvals(valid_idx));
        best_idx = valid_idx(best_idx);
    else
        [~, best_idx] = min(fvals);
    end
    thetahat = thetahats(best_idx, :);
end
end



function [D0,D1,idx_perm] = get_splitsamplernd(dat)
% This function is used to spliting the sample to get sample D0 and D1
% By random permutation, not take sequentially take first half and second
% half
n = size(dat,1);
if mod(n, 2) == 0 % If n is even
    half_n = n / 2;
    idx_perm = randperm(n);
    D0 = dat(idx_perm(1:half_n), :);
    D1 = dat;
    D1(idx_perm(1:half_n), :)=[];
else % If n is odd
    n_even = n - 1;
    half_n_even = n_even / 2;
    idx_perm = randperm(n_even);
    D0 = dat(idx_perm(1:half_n_even), :);
    D1 = dat;
    D1(idx_perm(1:half_n_even),:)=[];  % notice D1 take the rest of sample
end
end


function normalized_variable = normalize_var(variable)
% This is the function to normalize a variable to the range [0,1]
% The min-max normalization is hired in this function
min_val = min(variable);
max_val = max(variable);

% Normalize the variable to the [0,1] range
normalized_variable = (variable - min_val) / (max_val - min_val);
end


function p_new = move_to_interior(p, epsilon)
    % p: an m-by-4 matrix. Each row is a probability vector (sums to 1).
    % epsilon: a small positive number, must be less than 1/4.
    %
    % This function checks each row. If a row is not strictly in the interior
    % (i.e. all components > epsilon and < 1-epsilon), then it applies a transformation
    % to move that row into the interior of the probability simplex.
    
    [m, k] = size(p);  % k should be 4 in your case
    if k ~= 4
        error('This function expects p to have 4 columns.');
    end
    
    % Ensure that epsilon is feasible.
    if epsilon >= 1/k
        error('Epsilon must be less than 1/k (i.e. < %g for k = %d).', 1/k, k);
    end
    
    % Initialize the output matrix.
    p_new = p;
    
    % Set delta (a small number to counter numerical issues).
    delta = 1e-4;
    % Compute alpha. This is chosen so that even a zero entry becomes:
    %   (alpha/4 = epsilon + delta/4) which is strictly greater than epsilon.
    alpha = k * epsilon + delta;
    
    % Define the uniform probability vector (row vector of length 4).
    u = ones(1, k) / k;
    
    % Determine which rows are already strictly in the interior.
    interiorRows = all(p > epsilon, 2) & all(p < 1 - epsilon, 2);
    
    % For each row that is not in the interior, adjust it.
    for i = 1:m
        if ~interiorRows(i)
            p_new(i, :) = (1 - alpha) * p(i, :) + alpha * u;
        end
    end
    
    % Optional sanity check to ensure each row is now in the interior.
    if any(~(all(p_new > epsilon, 2) & all(p_new < 1 - epsilon, 2)))
        error('Transformation did not yield interior for some rows.');
    end
end

function [p_emp,frequency,n_cluster] = get_empiricaldist_givrevise(sampleD, idx, k, J)
% Descrption : compute the empirical distribution of outcomes
% Inputs:
% sample_D   : spliting sample D
% idx        : indexes of clustering that corresponding observation belongs to
% k          : number of clusters we consider for xsupp
% J          : the length of the possible events collection
% Outputs:
% phat       : emprirical probability of outcomes
Y = sampleD(:,2); 
d = sampleD(:,1);
p_emp=zeros(k,J); 
frequency=zeros(k,J);
n_cluster=zeros(k,1);
for i=1:k
    n_cluster(i,1) = sum((idx==i));   
    num00 = sum((Y==0).*(d==0).*(idx==i));
    num01 = sum((Y==0).*(d==1).*(idx==i));
    num10 = sum((Y==1).*(d==0).*(idx==i));
    num11 = sum((Y==1).*(d==1).*(idx==i));
    frequency(i,1)=num00/n_cluster(i,1);
    frequency(i,2)=num01/n_cluster(i,1);
    frequency(i,3)=num10/n_cluster(i,1);
    frequency(i,4)=num11/n_cluster(i,1);
    p_emp(i,1) = (num00+num01)/n_cluster(i,1);        % {00 and 01}
    p_emp(i,2) = (num10+num11)/n_cluster(i,1);        % {10 and 11} 
    p_emp(i,3) = (num00+num01+num10)/n_cluster(i,1);  % {00, 01 and 10}
    p_emp(i,4) = (num01+num10+num11)/n_cluster(i,1);  % {01, 10 and 11}
end
frequency = move_to_interior(frequency,1e-4);
end


function [logL, selecp]=get_logp(sampleD, p)
% This function is used to calculate the likelihood under alternative
% Inputs:
% SampleD    : the split sample data
% p           : the selected probability lower bound under alternative
Y=sampleD(:,2); % collection of outcome vector
dx=sampleD(:,1);
[nx,~]=size(sampleD);
selecp=zeros(nx,1);
for i=1:nx
    y=Y(i,1);
    d=dx(i,1);
    if y==0 && d==0
        selecp(i,1)=p(i,1);
    elseif y==0 && d==1         
        selecp(i,1)=p(i,2); % p(i,2) is the probability for {10 and 11} select (1,1) with 10%
    elseif y==1 && d==0
        selecp(i,1)=p(i,3);
    else
        selecp(i,1)=p(i,4);
    end
end
if any(selecp) == 0       % degenerate case can generate the same situation, with some p(.|x) equals to zero. large sample size may return a negative inf value
    logL = -5000;
else
    logL = sum(log(selecp)); %notice we take log in the final step
end
end


function [theta_asf, rmle_solutions, exitflags, fvals, Lval_asf]=get_rmle_asf(sampleD, p, cidx, cf_obj, cf_asfvalue, cf_decision, rmle_msnum, rmle_maxiter)
% Description: Compute the asf restricted estimator
% Inputs:
% sample_D     : split sample we use
% p            : the alternative selected probability p
% cidx         : conditional idx we used for different groups
% cf_obj       : objective function is ASF or other functional form
% cf_asfval    : the average structual function value we are interested in
% cf_decision  : counterfactual decision (cfd)
% rmle_msnum   : restricted MLE number of multistart
% rmle_maxiter : restricted MLE maximum number of iteration
fun=@(theta)-get_loglikelihood_givasf(sampleD, p, theta);
lb=-2.*ones(1,15);
ub=2*ones(1,15);
A=[1,1,zeros(1,13)
    1,zeros(1,14)]; % not a single constraint but a nx by 1 constrain for a single parameter? every beta0+beta1*size_var <= 1e-6
b=[-1e-6;-1e-6];
options = optimoptions(@fmincon,...
    'Algorithm','sqp',...            % Different Algorithms
    'Display','iter',...             % Display Iterated details
    'TolFun', 1e-4,...               % Set the tolerance
    'MaxFunctionEvaluations',rmle_maxiter);            % Use parallel computing
nonlcon = @(theta)cfcon(sampleD, cidx, theta, cf_obj, cf_asfvalue, cf_decision);
initialguess=(-0.1).*ones(1,15);
problem = createOptimProblem('fmincon','objective',...
    fun,'x0',initialguess,'nonlcon', nonlcon,'Aineq', A, 'bineq', b,'lb',lb,'ub',ub,'options',options);
ms = MultiStart('FunctionTolerance',1e-6,'XTolerance',1e-6,'UseParallel', true,'StartPointsToRun','bounds');
[~, ~, ~, ~, rmle_solutions] = run(ms, problem, rmle_msnum);
numsolvers=length(rmle_solutions);
exitflags=zeros(numsolvers,1);
fvals    =zeros(numsolvers,1);
all_thetaasf=zeros(numsolvers,length(initialguess));
for i=1:length(rmle_solutions)
    exitflags(i)=rmle_solutions(i).Exitflag;
    fvals(i)=rmle_solutions(i).Fval;
    all_thetaasf(i,:)=rmle_solutions(i).X;
end
% Select the solution with exitflag is 1
valid_indices = find(exitflags == 1); % Find solutions with exitflag = 1
if ~isempty(valid_indices)
    % If there are valid solutions, choose the one with the smallest fval
    [~, best_idx] = min(fvals(valid_indices));
    best_solution_idx = valid_indices(best_idx);
else
    % If no valid solutions, choose the one with the smallest fval overall
    disp('No optimizer found with exitflag equal to 1. Selecting the best available solution.');
    [~, best_solution_idx] = min(fvals);
end
theta_asf = all_thetaasf(best_solution_idx, :);
Lval_asf = -fvals(best_solution_idx);
% Display results
disp('Optimal solution found:');
disp(theta_asf);
disp('Log-likelihood value at optimal solution:');
disp(Lval_asf);
disp('Exit Flag:');
disp(exitflags(best_solution_idx));
disp('Number of solutions found:');
disp(numsolvers);
end

function logL=get_loglikelihood_givasf(sampleD, p, theta)
% Description: Compute the likelihood function value in giv case with asf
% Inputs:
% sampleD   : splitting sample
% p         : the alternative selected probability p
% theta     : strucutural paramter
Y=sampleD(:,2);
dx=sampleD(:,1);
[nx,J]=size(p); % provide both sample size and number of events    
lfp_q=zeros(nx,J);
llh=zeros(nx,1);
D_covariates = [sampleD(:,3:end)];
for l=1:nx
    D_covariate=D_covariates(l,:);
    lfp_q(l,:)=get_qtheta_giv(theta, p(l,:), D_covariate);
end
for l=1:nx
    if Y(l,:)==0 && dx(l,:)==0
        llh(l,1)=lfp_q(l,1);
    elseif Y(l,:)==0 && dx(l,:)==1
        llh(l,1)=lfp_q(l,2);
    elseif Y(l,:)==1 && dx(l,:)==0
        llh(l,1)=lfp_q(l,3);
    else
        llh(l,1)=lfp_q(l,4);
    end
end
logL=sum(log(llh));
end    

function qtheta = get_qtheta_giv(theta, p, D_covariate)
% Description  : compute LFP pairs based on Porposition 5.1
% Inputs:
% theta        :  parameter of estimation based on CHT
% As           :  collection of possible events
% p            :  probability p we find based on (2.15) in the paper
% D_covariate  :  the coviarate value based on observation data
beta0=theta(1);
beta1=theta(2);
beta2=theta(3:end);
J=length(p);
% p00=p(1,1)*0.1;    % alternative p probability
% p01=(1-p(1,1)*0.1-p(1,2)*0.1)/2;
% p10=(1-p(1,1)*0.1-p(1,2)*0.1)/2;
% p11=p(1,2)*0.1;
p00 = p(1);
p01 = p(2);
p10 = p(3);
p11 = p(4);
size_var=D_covariate(10);
w=[1,D_covariate(1:12)];
Phi = @(x)normcdf(x,0,1);

mu0w=Phi(w * beta2');
mu1w=Phi(beta0 + beta1 * size_var + w * beta2');
qtheta=zeros(1,J);
%% Based on KKT notes there are eight cases to be generated based on different conditions
if mu1w<mu0w
    if p10+p11 >= mu1w && p10+p11<mu0w && p11<=mu1w && p00<=1-mu0w % 4 conditions checked
        qtheta(:,1)=p00; 
        qtheta(:,2)=p01; 
        qtheta(:,3)=p10; 
        qtheta(:,4)=p11;
    elseif (p10/p11)>= ((mu0w-mu1w)/mu1w) && p10+p11>mu0w % 2 conditions checked
        qtheta(:,1)=(p00/(p00+p01))*(1-mu0w); 
        qtheta(:,2)=(p01/(p00+p01))*(1-mu0w); 
        qtheta(:,3)=(p10/(p10+p11))*mu0w; 
        qtheta(:,4)=(p11/(p10+p11))*mu0w;
    elseif (p10/p11) <((mu0w-mu1w)/mu1w) && (p10/(1-p11))>((mu0w-mu1w)/(1-mu1w)) % 2 conditions checked
        qtheta(:,1)=(p00/(p00+p01))*(1-mu0w); 
        qtheta(:,2)=(p01/(p00+p01))*(1-mu0w);
        qtheta(:,3)=mu0w-mu1w; 
        qtheta(:,4)=mu1w;
    elseif (p01/p00)>=((mu0w-mu1w)/(1-mu0w)) && p10+p11<mu1w % 2 conditions checked
        qtheta(:,1)=(p00/(p00+p01))*(1-mu1w); 
        qtheta(:,2)=(p01/(p00+p01))*(1-mu1w);
        qtheta(:,3)=(p10/(p10+p11))*mu1w; 
        qtheta(:,4)=(p11/(p10+p11))*mu1w;
    elseif (p01/(1-p00)) > ((mu0w-mu1w)/mu0w) && (p01/p00)<((mu0w-mu1w)/(1-mu0w)) % 2 conditions checked
        qtheta(:,1)=1-mu0w; 
        qtheta(:,2)=mu0w-mu1w;
        qtheta(:,3)=(p10/(p10+p11))*mu1w; 
        qtheta(:,4)=(p11/(p10+p11))*mu1w;
    elseif (p10/(1-p11))<=((mu0w-mu1w)/(1-mu1w)) && (p00/(1-p11))<=((1-mu0w)/(1-mu1w)) && p11>mu1w % 3 conditions checked
        qtheta(:,1)=(p00/(1-p11))*(1-mu1w); 
        qtheta(:,2)=(p01/(1-p11))*(1-mu1w);
        qtheta(:,3)=(p10/(1-p11))*(1-mu1w); 
        qtheta(:,4)=mu1w;
    elseif (p11/(1-p00))> (mu1w/mu0w) && (p00/(1-p11)) > ((1-mu0w)/(1-mu1w)) % 2 conditions checked
        qtheta(:,1)=(1-mu0w); 
        qtheta(:,2)=(p01/(p10+p01))*(mu0w-mu1w);
        qtheta(:,3)=(p10/(p10+p01))*(mu0w-mu1w); 
        qtheta(:,4)=mu1w;
    elseif (p01/(1-p00))<= ((mu0w-mu1w)/mu0w) && (p11/(1-p00)) <= (mu1w/mu0w) && p00>1-mu0w % 3 conditions checked
        qtheta(:,1)=(1-mu0w); 
        qtheta(:,2)=(p01/(p11+p10+p01))*mu0w;
        qtheta(:,3)=(p10/(p11+p10+p01))*mu0w; 
        qtheta(:,4)=(p11/(p11+p10+p01))*mu0w;
    else
        disp('not one of 8 cases in case A')
    end
elseif mu1w==mu0w
    qtheta(:,1)=(p00/(p00+p01))*(1-mu0w); 
    qtheta(:,2)=(p01/(p00+p01))*(1-mu0w); 
    qtheta(:,3)=(p10/(p10+p11))*mu1w; 
    qtheta(:,4)=(p11/(p10+p11))*mu1w;
end
end

function pmat = get_pmat_giv(theta, As, sampleD, frequency_D, idx_D)
% Description: compute lower bound for all possible events
% conditional on x.
% Inputs:
% theta             : structure parameter in the first equation
% As                : the collection of cdc events
% sample_D          : split sampleD
D_covariates = [sampleD(:,3:end)]; % get data with W
[nx,~] = size(D_covariates);
J=length(As);
pmat=NaN(nx,J);   %%%% for single event (0,0) (1,1)（1,0) (0,1)

temp = zeros(nx,J);
for l=1:nx
    pmat(l,:) = get_qtheta_giv(theta,frequency_D(idx_D(l),:),D_covariates(l,:));
end
end

function pout=get_p_giv(theta, As, x)
% Description: compute the probability p(.|x) in (2.16)
% Inputs:
% theta: parameter of estimation based on CHT
% As:    collection of events 
% x:     a given covariates for a single observation
J=length(As);
nu_A=zeros(1,J);   % only four cases we consider in our case.
for j=1:J
    nu_A(1,j)=get_nu_givobs(theta, As{j}, x);
end
pout=nu_A; % assign the lower bound as p
end


function nu=get_nu_givobs(theta, As, D_covariate)
% Descrption: computing the theoretical lower bound (belief
% function) based on real data support (not discretization xsupp)
% Inputs:
% theta: Structure parameters in the first equation (alpha, beta, eta)
% xsupp: one of the k support of covariates (not all support of xsupp_D) based on k-mean clustering 
% As   : a column vector representing the event
% Output?
% nu   : here nu is not true lower bound, it just represents a p that
%        satisfy problem
beta0=theta(1);
beta1=theta(2);
beta2=theta(3:end);
w=[1,D_covariate(1:12)];
size_var=D_covariate(10); % in discritization ninth column, without discritization tenth column

% Normal CDF
Phi = @(x) normcdf(x, 0, 1);

% Compute mu1w and mu0w
mu1w = Phi(beta0 + beta1 * size_var + w * beta2');
mu0w = Phi(w * beta2');
%% Based on equation 1.6-1.10, note that (1.7) is redundant
if isempty(As)
    nu=0;
elseif isequal(As, {'00', '01'})
    nu=1-mu0w;
elseif isequal(As, {'10', '11'})
    nu=mu1w;
elseif isequal(As, {'00', '01', '10'})
    nu=1-mu1w;
elseif isequal(As, {'01', '10', '11'})
    nu=mu0w;
else
    disp('error: not belongs to coredeter-class')
end
end


function nu = get_nu_giv(theta, As, xsupp)
% Description: Compute the theoretical lower bound (belief function) based on discretization xsupp
% Inputs:
% theta: Structure parameters (dimension is 15 by 1)
% xsupp: One of the k supports of covariates (from k-mean clustering)
% As   : A column vector representing the event

% Extract parameters
beta0 = theta(1);       % Parameter of d
beta1 = theta(2);       % Parameter of d*size
beta2 = theta(3:end);   % Parameter of 12 covariates

% Discretization: Change the variable sequence
w = [1, xsupp(1), xsupp(2), xsupp(14), xsupp(3:11)];  % 12 covariates
size_var = xsupp(9);  % Rename to avoid conflict with MATLAB's `size`

% Normal CDF
Phi = @(x) normcdf(x, 0, 1);

% Compute mu1w and mu0w
mu1w = Phi(beta0 + beta1 * size_var + w * beta2);
mu0w = Phi(w * beta2);
% Initialize nu
nu = NaN;  % Default value for debugging purposes
if isempty(As)
    nu=0;
elseif isequal(As, {'00', '01'})
    nu=1-mu0w;
elseif isequal(As, {'10', '11'})
    nu=mu1w;
elseif isequal(As, {'00', '01', '10'})
    nu=1-mu1w;
elseif isequal(As, {'01', '10', '11'})
    nu=mu0w;
else
    disp('error: not belongs to coredeter-class')
end
end


function asf = get_casf(sampleD, cidx, theta, cf_obj, cf_decision)
% Calculate the Average Structural Function (ASF)
if ~strcmp(cf_obj, 'CASF')
    error('Unsupported objective function type. Use ''CASF''.');
end

% Parameter decomposition
beta0 = theta(1);
beta1 = theta(2);
beta2 = theta(3:end);
D_covariates = sampleD(cidx,3:end);
[nx, ~] = size(D_covariates);

% Initialize container
inner_obj = zeros(nx, 1); 
Phi = @(x) max(min(normcdf(x,0,1), 1-1e-10), 1e-10); % Numerical safety bounds

% Main computation loop
for l = 1:nx
    D_covariate=D_covariates(l,:);
    w = [1, D_covariate(1:12)]; 
    size_var = D_covariate(10); 
    inner_part = beta0 * cf_decision + ...
                  beta1 * size_var * cf_decision + ...
                  w * beta2';
    % Probability calculation (with numerical protection)
    inner_obj(l) = Phi(inner_part); 
end
% Final result calculation (critical fix: moved outside loop)
asf = mean(inner_obj); 
end

function asf = get_asf(sampleD, theta, cf_obj, cf_decision)
% Calculate the Average Structural Function (ASF)
if ~strcmp(cf_obj, 'ASF')
    error('Unsupported objective function type. Use ''ASF''.');
end

% Parameter decomposition
beta0 = theta(1);
beta1 = theta(2);
beta2 = theta(3:end);
D_covariates = sampleD(:,3:end);
[nx, ~] = size(D_covariates);

% Initialize container
inner_obj = zeros(nx, 1); 
Phi = @(x) max(min(normcdf(x,0,1), 1-1e-10), 1e-10); % Numerical safety bounds

% Main computation loop
for l = 1:nx
    D_covariate=D_covariates(l,:);
    w = [1, D_covariate(1:12)]; 
    size_var = D_covariate(10); 
    inner_part = beta0 * cf_decision + ...
                  beta1 * size_var * cf_decision + ...
                  w * beta2';
    % Probability calculation (with numerical protection)
    inner_obj(l) = Phi(inner_part); 
end
% Final result calculation (critical fix: moved outside loop)
asf = mean(inner_obj); 
end


function mu1w = compute_mu1w(theta, sampleD)
beta0=theta(1);
beta1=theta(2);
beta2=theta(3:end);
D_covariates = [sampleD(:,3:end)];
[nx,~]=size(D_covariates);
Phi = @(x)normcdf(x,0,1);
mu1w=nan(nx,1);
for l=1:nx
    D_covariate=D_covariates(l,:);
    size_var=D_covariate(10);
    w=[1,D_covariate(1:12)];
    mu1w(l,1)=Phi(beta0+beta1*size_var+w*beta2');
end
end

function mu0w = compute_mu0w(theta, sampleD)
%beta0=theta(1);
%beta1=theta(2);
beta2=theta(3:end);
D_covariates = [sampleD(:,3:end)];
[nx,~]=size(D_covariates);
Phi = @(x)normcdf(x,0,1);
mu0w=nan(nx,1);
for l=1:nx
    D_covariate=D_covariates(l,:);
    %size_var=D_covariate(10);
    w=[1,D_covariate(1:12)];
    mu0w(l,1)=Phi(w*beta2');
end
end

function [c,ceq]=cfcon(sampleD, cidx, theta, cf_obj, cf_asfvalue, cf_decision)
% Description: Compute the nonlinear constraints.
% Inputs:
% sampleD      : splitting samples
% cidx         : conditional idx we used for different groups
% theta        : structural parameter
% cf_obj       : asf or other functional form
% cf_asfvalue  : asf function value we are intersted in
% cf_decision  ：counterfactual decision we use

% setup threshold
epsilon = 1e-2; % Lower bound for mu1w
mu1w = compute_mu1w(theta, sampleD); 
mu0w = compute_mu0w(theta, sampleD);
c1 = epsilon - min(mu1w); % Corresponds to min(mu1w) >= epsilon
c2 = epsilon - min(mu0w); % Corresponds to min(mu0w) >= epsilon
c3 = max(mu1w) - (1 - epsilon); % Corresponds to max(mu1w) <= 1 - epsilon
c4 = max(mu0w) - (1 - epsilon); % Corresponds to max(mu0w) <= 1 - epsilon
c = [c1; c2; c3; c4]; % Inequality constraints
if strcmp(cf_obj,'ASF')
   ceq=get_asf(sampleD, theta, cf_obj, cf_decision)-cf_asfvalue;
elseif strcmp(cf_obj,'CASF')
   ceq=get_casf(sampleD, cidx, theta, cf_obj,cf_decision)-cf_asfvalue;
else
   disp('specify the type of function')
end
end





































