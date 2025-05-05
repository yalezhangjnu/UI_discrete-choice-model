clear
S=1000; % number of Monte Carlo replications
alph=0.05;% significance level
M =  30;
hgrid = linspace(0,1,M);
n = 200;
DGP = 'IID';
eqsel.proc = DGP;
eqsel.p = 0.5;

Tn_store = zeros(S,M);
Sn_store = zeros(S,M);
Tn_cht_store = zeros(S,M);
Sn_cht_store = zeros(S,M);
Tnrej_store = zeros(S,M);
Tnrej_cht_store = zeros(S,M);
Snrej_store = zeros(S,M);
Snrej_cht_store = zeros(S,M);
Tnrej_prob = zeros(M,1);
Tnrej_cht_prob = zeros(M,1);
Snrej_prob = zeros(M,1);
Snrej_cht_prob = zeros(M,1);
pibar = zeros(M,1);

for m=1:M
    h = hgrid(m);
    theta_true = [-h,-h];
    dat=generate_data(theta_true,S,n,eqsel);

    parfor s=1:S
        % split samples
        [D0,D1] = get_splitsample(dat(:,s));

        % get unrestricted estimators
        theta_hat1 = get_mle(D1);
        theta_hat0 = get_mle(D0);
        theta_cht_hat1 = get_cht(D1);
        theta_cht_hat0 = get_cht(D0);

        % get restricted estimators
        theta_rmle0 = [0,0];
        theta_rmle1 = [0,0];

        % evaluate split sample stats
        Tn = get_splitLR(D0,theta_rmle0,theta_hat1);
        Tn_swap = get_splitLR(D1,theta_rmle1,theta_hat0);
        Tn_store(s,m) = Tn;
        Tn_cht = get_splitLR(D0,theta_rmle0,theta_cht_hat1);
        Tn_cht_swap = get_splitLR(D1,theta_rmle1,theta_cht_hat0);
        Tn_cht_store(s,m) = Tn_cht;

        % evaluate cross-fit stats
        Sn = (Tn+Tn_swap)/2;
        Sn_store(s,m) = Sn;
        Sn_cht = (Tn_cht+Tn_cht_swap)/2;
        Sn_cht_store(s,m) = Sn_cht;

        % store results
        Tnrej_store(s,m) = Tn > 1/alph;
        Snrej_store(s,m) = Sn > 1/alph;
        Tnrej_cht_store(s,m) = Tn_cht > 1/alph;
        Snrej_cht_store(s,m) = Sn_cht > 1/alph;
    end
    Tnrej_prob(m) = sum(Tnrej_store(:,m))/S;
    Snrej_prob(m) = sum(Snrej_store(:,m))/S;
    Tnrej_cht_prob(m) = sum(Tnrej_cht_store(:,m))/S;
    Snrej_cht_prob(m) = sum(Snrej_cht_store(:,m))/S;

    % power envelope
    if m==1
        pibar(m) = alph;
    else
        [Q0,Q1]=get_LFP(theta_true);
        [Cn,log_LH]=get_cv(Q0,Q1,n,alph);
        pibar(m) = get_power(Q1,log_LH,n,Cn);
    end
end
%disp([Tnrej_prob,Snrej_prob,Tnrej_cht_prob,Snrej_cht_prob,pibar])
odd_indices = 1:2:M;
Snrej_prob_odd = Snrej_prob(odd_indices);
Snrej_cht_prob_odd = Snrej_cht_prob(odd_indices);

% Table results
disp(['Sample size: ', num2str(n)])
disp('LR-test(MLE theta_hat1):')
disp(Snrej_prob_odd')
disp('LR-test(moment-based theta_hat1):')
disp(Snrej_cht_prob_odd')
save(['./Results/crossfit_ex1_dat_',DGP,'n',num2str(n)])


function dat = generate_data(theta,S,n,eqsel)
% description:   generate data for simulation
% theta:         structure parameters
% n:             number of observation
% eqsel:         equilibirum selection
% dat:           generated data, n by 1

rng('default')
%rng(123456)

u1=randn([n,S]); % Notice it is not normal distribution, but random number from normal distribution
u2=randn([n,S]);


switch eqsel.proc
    case 'IID'  % IID Data Generation %%%
        vi=rand(n,S)<=eqsel.p; %%% Bernoulli random variable

        %store data we generated
        G_result=NaN(n,S);
        G_result( (u1 < 0)        & (u2 < 0)        )=0;
        G_result( (u1 > -theta(1)) & (u2 > -theta(2)) ) = 11;
        G_result( (u1 > -theta(1)) & (u2 < -theta(2)) ) = 10;
        G_result( (0 < u1) & (u1 < -theta(1)) & (u2<0) )       = 10;
        G_result( (0 < u1) & (u1 < -theta(1)) & (u2 > -theta(2))) = 01;
        G_result( (u1 < 0) & (u2>0) ) = 01;
        G_result( (0 < u1) & (u1 < -theta(1)) & (0 < u2) & (u2 <-theta(2)) & vi ) = 10;
        G_result( (0 < u1) & (u1 < -theta(1)) & (0 < u2) & (u2 <-theta(2)) & (~vi) ) = 01;
        dat=G_result;
end
end

function p = get_freq(D)
[nD,~] = size(D);
p = [sum(D==0)/nD,sum(D==11)/nD,sum(D==10)/nD,sum(D==1)/nD];
end

function qtheta = get_qtheta_cf(theta,p)
Phi = @(x)normcdf(x,0,1);
Phi1 = Phi(0);
Phi2 = Phi(0);
Phi1b = Phi(theta(1));
Phi2b = Phi(theta(2));
nu_A=[(1-Phi1)*(1-Phi2),Phi1b*Phi2b,(1-Phi2b)*Phi1b+(1-Phi2)*(Phi1-Phi1b)];
nu_conj_A=[(1-Phi1)*(1-Phi2),Phi1b*Phi2b,(1-Phi2b)*Phi1];

prel10 = p(3)/(p(3)+p(4));
eta1 = 1-nu_A(1)-nu_A(2);
eta2 = nu_conj_A(3);
eta3 = nu_A(3);

if prel10 > eta3/eta1 && prel10 < eta2/eta1
    temp = prel10*eta1;
elseif prel10 >= eta2/eta1
    temp = eta2;
else
    temp = eta3;
end
qtheta = [nu_A(1),nu_A(2),temp,eta1-temp];
end

function [Q_0, Q_1] = get_LFP(theta_alt)
% Check three cases
Phi = @(x)normcdf(x,0,1);

Z1 = Phi(theta_alt(1))*(1-Phi(theta_alt(2)));
Z2 = Phi(theta_alt(2))*(1-Phi(theta_alt(2)));
Q_0=[1/4, 1/4, 1/4, 1/4];

if Z1 <= 4 && Z2 <= 1/4
    Q_1 = [1/4, Phi(theta_alt(1))*Phi(theta_alt(2)),...
        3/8-Phi(theta_alt(1))*Phi(theta_alt(2))/2, 3/8-Phi(theta_alt(1))*Phi(theta_alt(2))/2];
elseif Z1 > 1/4
    Q_1 = [1/4, Phi(theta_alt(1))*Phi(theta_alt(2)),...
        1/4-Phi(theta_alt(1))*(Phi(theta_alt(2))-1/2), (1-Phi(theta_alt(1)))/2];
elseif Z2 > 1/4
    Q_1 = [1/4, Phi(theta_alt(1))*Phi(theta_alt(2)),...
        (1-Phi(theta_alt(2)))/2, 1/4-Phi(theta_alt(2))*(Phi(theta_alt(1))-1/2)];
end
end

function qtheta = get_qtheta(theta,D)
if theta == [0,0] % null
    qtheta = [1/4, 1/4, 1/4, 1/4];
else
    p = get_freq(D);
    qtheta = get_qtheta_cf(theta,p);
end
end

function [D0,D1] = get_splitsample(dat)
n = size(dat,1);
n0 = floor(n/2);
D0 = dat(1:n0,1);
D1 = dat(n0+1:end,1);
end

function logL = get_loglikelihood(D,theta)
qtheta = get_qtheta(theta,D);
Li =((D==0).*qtheta(1)) + ((D==11).*qtheta(2)) + ((D==10).*qtheta(3))+...
    ((D==01).*qtheta(4));
logL = sum(log(Li));
end

function hatQ1 = get_samplecriterion(D,theta)
    Phi = @(x)normcdf(x,0,1);
    [ns,~] = size(D);
    Phat11 = sum(D==11)/ns;
    Phat10 = sum(D==10)/ns;
    nu11 = Phi(theta(1))*Phi(theta(2));
    nu10 = Phi(theta(1))*(1-Phi(theta(2)))+(1/2-Phi(theta(1)))*1/2;
    nu_conj10 = (1-Phi(0))*Phi(-theta(2));
    m = [abs(nu11-Phat11), nu10-Phat10, Phat10 - nu_conj10,0];
    hatQ1 = max(m);
end

function thetahat = get_cht(D)
fun = @(theta) get_samplecriterion(D,theta);
lb = [-5,-5];
ub = [0,0];
thetahat = fmincon(fun,[0,0],[],[],[],[],lb,ub);
end

function thetahat = get_mle(D)
fun = @(theta) -get_loglikelihood(D,theta);
lb = [-5,-5];
ub = [0,0];
thetahat = fmincon(fun,[0,0],[],[],[],[],lb,ub);
end

function Tn = get_splitLR(D0,thetahat0,thetahat1)
lnL0_thetahat1 = get_loglikelihood(D0,thetahat1);
lnL0_thetahat0 = get_loglikelihood(D0,thetahat0);
logTn = lnL0_thetahat1 - lnL0_thetahat0;
Tn = exp(logTn);
end

function [Cn, log_LH]=get_cv(Q_0,Q_1,n,alph)
log_LH=log(Q_1./Q_0); % calculate log-likelihood ratio

% step 2: calculate expectation and variance of log-likelihood ratio 
mean_LH=Q_0*log_LH'; % calculate expectation
var_LH=Q_0*(log_LH.^2)'-(mean_LH)^2;


% step 3: calculate cv
Cn=exp(sqrt(n)*norminv(1-alph)*sqrt(var_LH)+n*mean_LH);

%%% Since there exists an numerical issue
%Cn=norminv(1-alph)*sqrt(var_LH);
end

function pibar = get_power(Q_1,log_LH,n,c)
Phi = @(x)normcdf(x,0,1);
mean_LH = Q_1*log_LH';
var_LH=Q_1*(log_LH.^2)'-(mean_LH)^2;
pibar = 1-Phi((log(c)-n*mean_LH)/(sqrt(n)*sqrt(var_LH)));
end

