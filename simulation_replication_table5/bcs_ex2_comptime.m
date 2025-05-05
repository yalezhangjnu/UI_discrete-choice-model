function bcs_ex2_comptime(ncores,ind)
S=100; % number of Monte Carlo replications
M = 20;
K = 5;
B = 500;
hgrid = linspace(0,0.15,M);
n = 5000;
DGP = 'IID';
eqsel.proc = DGP;
eqsel.p = 0.5;
rng('default')

xbar = K-(K+1)/2;
[X,Y]=meshgrid(-xbar:xbar,-xbar:xbar);
xsupp = [X(:),Y(:)];

As={
    00;
    11;
    10;
    };

m = ind;
h = hgrid(m);
theta_true = [h,h,-0.5,-0.5];
beta_true = [-0.5,-0.5];
[y,x1,x2] = generate_data(theta_true,S,n,eqsel,K);
W2_AA = norminv(rand(B,n));

kappa_list = sqrt(log(n));
options = optimset('Display','off','Algorithm','interior-point'); % from BCS
% significance level
alpha = 0.05;
% Placeholders
minQn_DR = NaN(size(kappa_list,1), B);
minQn_PR = NaN(size(kappa_list,1), B);
minQn_MR = NaN(size(kappa_list,1), B);
cn_MR  = NaN(1,size(kappa_list,1));
Tn_MRsim = NaN(S,1);
cn_MRsim = NaN(S,1);
comptime=zeros(S,1);
parpool(ncores)

for s=1:S
    tstart = tic;
    dat = [y(:,s),x1(:,s),x2(:,s)];
    % Pick a large value as initial function value for the minimization
    min_value = 10^10; % from BCS
    lb = [-3,-3];
    ub = [0,0];
    M = 49;
    start = [beta_true; lhsscale(M, 2, ub, lb)];% 1st starting point is oracle
    % placeholder for minimization results
    min_outcomes = zeros(M+1, 3);
    % loop over starting points for minimization
    for initial = 1:(M + 1)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % (Line 20 in BCS Algorithm 2.1)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Instead of set minimizer we have point estimator of nuisance par
        try
            [beta_est,Qn_aux,bandera] =  fmincon(@(x) Qn_function_X([0,0], x,y(:,s), x1(:,s), x2(:,s), 'disagg',xsupp),...
                start(initial, :),[],[],[],[],[-3,-3],[0,0],[],options);
            % check whether minimization is successful and reduced value
            if Qn_aux  < min_value && bandera >= 1
                Qn_minimizer = beta_est;
                min_value = Qn_aux;
            end
        catch
            min_value = NaN;
            bandera = 0;
        end

        % if minimization is successful, collect minimizer and its value;
        if bandera >= 1
            min_outcomes(initial,:) = [min_value, beta_est];
        end
    end
    % return the test statstic
    minQn = min_value;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test MR
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    kappa_type =1;
    parfor b = 1:B % loop over bootstrap draws
        % (1) DR test (equivalent to plugging in nuisance estimator and null value)
        % compute simulated DR criterion function
        minQn_DR(kappa_type, b) = Qn_MR_function_X([0,0],beta_est, y(:,s),x1(:,s),x2(:,s),kappa_list(kappa_type), W2_AA(b,:), 1, 'disagg',xsupp);

        % (2) PR test
        [delta_aux,min_value_PR] =  fmincon(@(x) Qn_MR_function_X([0,0],x, y(:,s),x1(:,s),x2(:,s), kappa_list(kappa_type), W2_AA(b,:), 2, 'disagg',xsupp),...
            [-0.5,-0.5],[],[],[],[],...
            [-3,-3],[0,0],[],options);
        % compute simulated PR criterion function
        minQn_PR(kappa_type, b) = min_value_PR ;

        % (3) MR test
        minQn_MR(kappa_type, b) = min(minQn_DR(kappa_type, b), minQn_PR(kappa_type, b));
    end
    % Compute critical values
    cn_MR(kappa_type) = quantile(minQn_MR(kappa_type,:), 1-alpha);
    cn_MRsim(s) = cn_MR;
    Tn_MRsim(s) = minQn;
    comptime(s)=toc(tstart);
end
save(['./Results/bcs_ex2_dat_',DGP,'n',num2str(n),'_ind',num2str(ind),'_ncores',num2str(ncores),'comptime'])
end

function [dat,x1,x2] = generate_data(theta,S,n,eqsel,K)
% description:   generate data for simulation
% theta:         structure parameters
% n:             number of observation
% eqsel:         equilibirum selection
% dat:           generated data, n by 1


%rng(123456)
beta = theta(3:4);
delta = theta(1:2);

u1=randn([n,S]);
u2=randn([n,S]);
x1=randi(K,n,S)-(K+1)/2;
x2=randi(K,n,S)-(K+1)/2;

switch eqsel.proc
    case 'IID'  % IID Data Generation %%%
        vi=rand(n,S)<=eqsel.p; %%% Bernoulli random variable

        %store data we generated
        G_result=NaN(n,S);
        G_result( (u1 < -x1.*delta(1))        & (u2 < -x2.*delta(2))        )=0;
        G_result( (u1 > -x1.*delta(1)-beta(1)) & (u2 > -x2.*delta(2)-beta(2)) ) = 11;
        G_result( (u1 > -x1.*delta(1)-beta(1)) & (u2 < -x2.*delta(2)-beta(2)) ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 <- x1.*delta(1)-beta(1)) & (u2<-x2.*delta(2)) ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (u2 > -x2.*delta(2)-beta(2))) = 01;
        G_result( (u1 < -x1.*delta(1)) & (u2>-x2.*delta(2)) ) = 01;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (-x2.*delta(2) < u2) & (u2 <-x2.*delta(2)-beta(2)) & vi ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (-x2.*delta(2) < u2) & (u2 <-x2.*delta(2)-beta(2)) & (~vi) ) = 01;
        dat=G_result;


    case 'Cluster'
        K = 5; % # of clusters
        nk = n/K;
        vk = rand(K,S)<=eqsel.p; %%% Bernoulli random variable
        vi = repelem(vk,nk,1);
        G_result=NaN(n,S);
        G_result( (u1 < -x1.*delta(1))        & (u2 < -x2.*delta(2))        )=0;
        G_result( (u1 > -x1.*delta(1)-beta(1)) & (u2 > -x2.*delta(2)-beta(2)) ) = 11;
        G_result( (u1 > -x1.*delta(1)-beta(1)) & (u2 < -x2.*delta(2)-beta(2)) ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 <- x1.*delta(1)-beta(1)) & (u2<-x2.*delta(2)) ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (u2 > -x2.*delta(2)-beta(2))) = 01;
        G_result( (u1 < -x1.*delta(1)) & (u2>-x2.*delta(2)) ) = 01;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (-x2.*delta(2) < u2) & (u2 <-x2.*delta(2)-beta(2)) & vi ) = 10;
        G_result( (-x1.*delta(1) < u1) & (u1 < -x1.*delta(1)-beta(1)) & (-x2.*delta(2) < u2) & (u2 <-x2.*delta(2)-beta(2)) & (~vi) ) = 01;
        dat=G_result;

end
end

% 6/17/2020 Shuowen Chen and Hiroaki Kaido
% Implements Latin Hypercube Sampling and rescale to produce initial
% guesses for restricted MLE
function output = lhsscale(nsamples, nvars, ux, lx)
% Inputs:
%   nsamples: the number of points for each parameter
%   nvars:    the dimension of the parameter
%   ux:       the upper bound of the parameter space (1 by nvars)
%   lx:       the lower bounds of the parameter space (1 by nvars)
rng(123)
% draw points on a cube (Latin hypercube sampling)
% the nsample points will be from (0,1/n), (1/n, 2/n),...,(1-1/n,1), where
% n is shorthand notation of nsamples. Note the intervals can be randomly
% permutated
temp = lhsdesign(nsamples, nvars);
% rescale to draw points on parameter space
output = zeros(nsamples, nvars);
for i = 1:nvars
    output(:, i) = repmat((ux(i)-lx(i)), nsamples, 1).*temp(:,i) + ...
        repmat(lx(i), nsamples, 1);
end
end

function value = Qn_function_X(delta, beta, data, X1, X2, mtype,xsupp)
%{
 - Inputs - 
   delta: covariate coefficient
   beta:  strategic parameter 
   data:  data
   X1:    covariate for player 1
   X2:    covariate for player 2
   mtype: whether use aggregate or disaggregate moments

 - Outputs - 
   value: function value 
%}

% determines sample size;
n = size(data,1);

% the number of moment inequalities (which should appear first);
if strcmp(mtype, 'disagg')
    p = 50;
elseif strcmp(mtype, 'agg')
    p = 2;
end

% studentizes and averages the data (4 by S)
[mbar_std, ~, ~] = dataOp(delta, beta, data, 1, X1, X2, mtype,xsupp); % No use for kappa, set to one.

% computes the sample criterion function;
value = S_function(sqrt(n)*mbar_std, p);
end

function value = Qn_MR_function_X(delta, beta, data, X1, X2, kappa, W2_AA, MR_type, moment_type,xsupp)
%{
 - Inputs - 
   delta:       coefficients on covariates
   beta:        strategic parameter 
   data:        data (outcome)
   X1:          covariate for player 1
   X2:          covariate for player 2
   kappa:       tuning parameter for GMS
   W2_AA:       a vector of random variables used to implement the (multiplier) bootstrap.
   MR_type:     the type of resampling, i.e., DR or PR
   moment_type: whether aggregate or disaggregate moments

 - Outputs - 
   value:       function value 
%}

% decide if the number of moments are
n = size(data,1); % sample size

% the number of moment inequalities (which should appear first) and total moments;
if strcmp(moment_type, 'disagg')
    p = 50;
    k = 100;
elseif strcmp(moment_type, 'agg')
    p = 2;
    k = 4;
end

[~, mData, xi] = dataOp(delta, beta, data, kappa, X1, X2, moment_type,xsupp);

if MR_type == 1 % DR method;
    value = S_function(W2_AA*zscore(mData, 1)/sqrt(n) + repmat(phi_function(xi',p,k-p),size(W2_AA,1),1), p);
elseif MR_type == 2 % PR method;
    value = S_function(W2_AA*zscore(mData, 1)/sqrt(n) + repmat(xi,size(W2_AA,1),1), p);
end
end

function [mbar_std, mData, xi] = dataOp(delta, beta, y, kappa, x1, x2, mtype,xsupp)
%{
 -Inputs-
 delta:    coefficient parameter
 beta:     strategic interaction
 data:     dataset (n by S)
 kappa:    tuning parameter
 X1:       covariate for player 1 (n by S)
 X2:       covariate for player 2 (n by S)
 mtype:    whether we use aggregate or disaggregate moments

 -Outputs- 
 Let k denote the total number of moments
 mbar_std: standardized sample moment inequalities and equalities (k by S)
 mData:    data for sample moments (n by k by S). It is used for
           constructing objective function for DR and PR tests.  
 xi:       slackness measure in GMS (k by S)
%}

% sample size (number of markets);
n = size(y, 1);
% number of simulations
S = 1;
K = 5;


% Model predictions (n by S)
% (1) equalities
modelP00 = normcdf(-x1*delta(1)) .* normcdf(-x2*delta(2));
modelP11 = normcdf(x1*delta(1)+beta(1)) .* normcdf(x2*delta(2)+beta(2));
% (2) inequalities
% region of multiplicity
mul = (normcdf(-x1*delta(1)-beta(1)) - normcdf(-x1*delta(1))) .* (normcdf(-x2*delta(2)-beta(2)) - normcdf(-x2*delta(2)));
modelP10_ub = normcdf(x1*delta(1)) .* normcdf(-x2*delta(2)-beta(2));
modelP10_lb = modelP10_ub - mul;

if strcmp(mtype,'disagg')
    [nx,~] = size(xsupp);
    datP00 = zeros(n,nx);
    datP11 = zeros(n,nx);
    datP10 = zeros(n,nx);
    pX = 1/K^2;
    epsilon = 0.000001;
    for l=1:nx
    datP00(y==0  & x1==xsupp(l,1) & x2==xsupp(l,2),l)=1;
    datP11(y==11 & x1==xsupp(l,1) & x2==xsupp(l,2),l)=1;
    datP10(y==10 & x1==xsupp(l,1) & x2==xsupp(l,2),l)=1;
    end
    mData1 = datP00 - repmat(modelP00*pX,1,nx);
    mData2 = datP11 - repmat(modelP11*pX,1,nx);
    mData3 = datP10 - repmat(modelP10_lb*pX,1,nx); 
    mData4 = repmat(modelP10_ub*pX,1,nx) - datP10; 
    mData = [mData3, mData4, mData1, mData2];

    mbar_std = mean(mData)./(std(mData) + epsilon); 
    xi = (1/kappa)*sqrt(n)*mbar_std;
end

if strcmp(mtype, 'agg')
    % Construct aggregate sample moments
    dataP00 = zeros(n, S); % placeholders
    dataP11 = zeros(n, S);
    dataP10 = zeros(n, S);
    for s = 1:S % loop over each simulation
        % From the data, count possible outcomes
        % (0, 0), (1, 1) and (1, 0)
        dataP00(data(:, s) == 0, s) = 1;
        dataP11(data(:, s) == 11, s) = 1;
        dataP10(data(:, s) == 10, s) = 1;
    end
    % Define moment (in)equalities data (n by S).
    mData_1q = dataP00 - modelP00;
    mData_2q = dataP11 - modelP11;
    mData_3q = dataP10 - modelP10_lb;
    mData_4q = modelP10_ub - dataP10;
    % Stack up moments for each simulation Note: inequalities should appear first
    mData = zeros(n, 4, S); % placeholders
    mbar_std = zeros(4, S);
    xi = zeros(4, S);
    epsilon = 0.000001; % introduces this parameter to avoid division by zero
    for s = 1:S
        mData(:, :, s) = [mData_3q(:, s), mData_4q(:, s), mData_1q(:, s), mData_2q(:, s)];
        % compute studentized sample averages of mData
        mbar_std(:, s)  = mean(mData(:, :, s))./(std(mData(:, :, s)) + epsilon);
        % Additional parameter needed in DR, PR, and MR inference
        xi(:, s) = (1/kappa)*sqrt(n)*mbar_std(:, s); % Slackness measure in GMS
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Defines the S function according to MMM in Eq. (2.6);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Qn,chi] = S_function(m_std, p)
% m_std denotes the studentized sample moment conditions;
% p denotes the number of moment inequalities (which should appear first);

chi = 2; % degree of homogeneity of criterion function in MMM

% take negative part of inequalities;
m_std(:, 1:p) = min(m_std(:, 1:p), zeros(size(m_std, 1), p));

% sample criterion function;
Qn = sum(abs(m_std).^chi,2);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Defines the GMS penalization function for DR inference as in Eq. (2.11)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function value = phi_function(xi,p,v)
% xi is the xi(1) vector in AS (2010) Eq. 4.5;
% p is the number of moment inequalities;
% v is the number of moment equalities;

% My definition of "infinity" (satisfies Inf*0 = 0 (as desired), while the built in "inf" has inf*0 = NaN)
Inf = 10^10;

% define GMS penalization;
value      = zeros(1, p+v); % zero for equalities and "close" inequalities;
value(1:p) = (xi(1:p)>1).*Inf; % infinity for "sufficiently violated" inequalities;
end