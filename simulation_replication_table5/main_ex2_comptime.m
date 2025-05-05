function main_ex2_comptime(ncores)
% This is for an entry game with covariates
% addpath('./cvx','-end')
% cvx_setup

S=1000; % number of Monte Carlo replications
alph=0.05;% significant level
M = 20;
K = 5;
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

Tn_store = zeros(S,1);
Tnrej_store = zeros(S,1);
Sn_store = zeros(S,1);
Snrej_store = zeros(S,1);

m = 1;
h = hgrid(m);
theta_true = [h,h,-0.5,-0.5];
[y,x1,x2] = generate_data(theta_true,S,n,eqsel,K);
comptime=zeros(S,1);
parpool(ncores)
parfor s=1:S
%for s=1:S
    tstart = tic;
    dat = [y(:,s),x1(:,s),x2(:,s)];
    [D0,D1] = get_splitsample(dat);
    theta_hat1 = get_cht(D1,As,xsupp);
    theta_hat0 = get_cht(D0,As,xsupp);
    p1 = get_pmat(theta_hat1,As,xsupp);
    p0 = get_pmat(theta_hat0,As,xsupp);
    [theta_rmle0,logL0] = get_rmle(D0,p1,As,xsupp);
    [theta_rmle1,logL1] = get_rmle(D1,p0,As,xsupp);
    Tn = get_splitLR(D0,logL0,p1,xsupp);
    Tn_swap = get_splitLR(D1,logL1,p0,xsupp);
    Tn_store(s) = Tn;
    Sn = (Tn+Tn_swap)/2;
    Sn_store(s) = Sn;
    Tnrej_store(s) = Tn > 1/alph;
    Snrej_store(s) = Sn > 1/alph;
    comptime(s)=toc(tstart);
end

Tnrej_prob = sum(Tnrej_store)/S;
Snrej_prob = sum(Snrej_store)/S;

disp([Tnrej_prob,Snrej_prob])
disp(median(comptime))
save(['./Results/crossfit_ex2_dat_',DGP,'n',num2str(n),'comptime'])
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

end
end

function [D0,D1] = get_splitsample(dat)
n = size(dat,1);
n0 = floor(n/2);
D0 = dat(1:n0,:);
D1 = dat(n0+1:end,:);
end

function phat = get_empiricaldist(D,x1,x2)
Y = D(:,1);
nx = sum((D(:,2)==x1).*(D(:,3)==x2));
num00 = sum((Y==0).*(D(:,2)==x1).*(D(:,3)==x2));
num11 = sum((Y==11).*(D(:,2)==x1).*(D(:,3)==x2));
num10 = sum((Y==10).*(D(:,2)==x1).*(D(:,3)==x2));
phat = [num00/nx, num11/nx, num10/nx, 1-(num00+num11+num10)/nx];
end


function hatQ1 = get_samplecriterion(D,As,theta,xsupp)
J=length(As);
nu_A=zeros(J);
nu_conj_A=zeros(J);
[nx,~] = size(xsupp);
m = zeros(nx,3);
for l=1:nx
    phat = get_empiricaldist(D,xsupp(l,1),xsupp(l,2));
    for j=1:J
        nu_A(j)=get_nu(theta,As{j},xsupp(l,1),xsupp(l,2));
        nu_conj_A(j)=get_nu_conj(theta, As{j},xsupp(l,1),xsupp(l,2));
    end
    m(l,:) = [abs(nu_A(2)-phat(2)), nu_A(3)-phat(3), phat(3) - nu_conj_A(3)];
end
hatQ1 = max(max(m));
end

function thetahat = get_cht(D,As,xsupp)
fun = @(theta) get_samplecriterion(D,As,theta,xsupp);
lb = [-3,-3,-3,-3];
ub = [3,3,0,0];
options = optimoptions('fmincon','Display','off');
thetahat = fmincon(fun,[0,0,0,0],[],[],[],[],lb,ub,[],options);
end

function pmat = get_pmat(theta,As,xsupp)
[nx,~] = size(xsupp);
temp = zeros(nx,4);
for l=1:nx
    temp(l,:) = get_p(theta,As,xsupp(l,1),xsupp(l,2));
end
pmat = temp;
end

function pout = get_p(theta,As,x1,x2)
J=length(As);
nu_A=zeros(J);
nu_conj_A=zeros(J);
for j=1:J
    nu_A(j)=get_nu(theta,As{j},x1,x2);
    nu_conj_A(j)=get_nu_conj(theta, As{j},x1,x2);
end

% inequality constraints 
A=[0,0,-1,0;
    0,0,1,0];
b=[-nu_A(3)
    nu_conj_A(3)];
% Equality restriction: probabilities sum up equal to 1 under null and
% alternative hypotheses
Aeq=[1,1,1,1
    1,0,0,0
    0,1,0,0];
beq=[1
    nu_A(1)
    nu_A(2)];
params.A = A;
params.b = b;
params.Aeq = Aeq;
params.beq = beq;
settings.verbose = 0;
% Exercise the high-speed solver.
[vars, status] = csolve(params,settings);  % solve, saving results.

% Check convergence, and display the optimal variable value.
if ~status.converged, error 'failed to converge'; end
pout = vars.p';
end


function qtheta = get_qtheta_cf(theta,As,p,x1,x2)
J = length(As);
nu_A=zeros(J);
nu_conj_A=zeros(J);
for j=1:J
    nu_A(j)=get_nu(theta,As{j},x1,x2);
    nu_conj_A(j)=get_nu_conj(theta, As{j},x1,x2);
end

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

function qtheta = get_qtheta(theta,As,p,x1,x2)
J=length(As);
nu_A=zeros(J);
nu_conj_A=zeros(J);
for j=1:J
    nu_A(j)=get_nu(theta,As{j},x1,x2);
    nu_conj_A(j)=get_nu_conj(theta, As{j},x1,x2);
end

n=4;

cvx_begin quiet
variables q(n);
% Restrictions we are going to use
% Inequalities: upper and lower bound for composite events under null and
% alternative hypotheses
A=[0,0,-1,0;
    0,0,1,0];
b=[-nu_A(3)
    nu_conj_A(3)];
% Equality restriction: probabilities sum up equal to 1 under null and
% alternative hypotheses
Aeq=[1,1,1,1
    1,0,0,0
    0,1,0,0];
beq=[1
    nu_A(1)
    nu_A(2)];

%Inequalities: upper and lower probability for singleton events under null
%and alternative hypotheses
lb=zeros(n,1); %lower bound
ub=ones(n,1);


minimize(rel_entr(q(1)+p(1),q(1))+rel_entr(q(2)+p(2),q(2))+rel_entr(q(3)+p(3),q(3))+rel_entr(q(4)+p(4),q(4)));
subject to
A*q <= b;
Aeq*q == beq;
lb <= q <=ub;

cvx_end

qtheta=[q(1), q(2), q(3), q(4)];
end

function nu=get_nu(theta,A,x1,x2)
% computes the theoretical lower bound or belief function
% theta: structure parameters
% mu:    the mean of error term
% sigma: the sd of error term
% A    : a column vector representing the event

beta = theta(3:4);
delta = theta(1:2);
Phi = @(x)normcdf(x,0,1);
Phi1 = Phi(x1.*delta(1));
Phi2 = Phi(x2.*delta(2));
Phi1b = Phi(x1.*delta(1)+beta(1));
Phi2b = Phi(x2.*delta(2)+beta(2));

if isempty(A)
    nu=0;
elseif A==0
    nu=(1-Phi1)*(1-Phi2);
elseif A == 11
    nu=Phi1b*Phi2b;
elseif A==10
    nu=(1-Phi2b)*Phi1b+(1-Phi2)*(Phi1-Phi1b);
else
    disp('error:wrong event')
end
end

function nu_conj= get_nu_conj(theta,A,x1,x2)

beta = theta(3:4);
delta = theta(1:2);
Phi = @(x)normcdf(x,0,1);
Phi1 = Phi(x1.*delta(1));
Phi2 = Phi(x2.*delta(2));
Phi1b = Phi(x1.*delta(1)+beta(1));
Phi2b = Phi(x2.*delta(2)+beta(2));

if isempty(A)
    nu_conj=0;
elseif A==0
    nu_conj=(1-Phi1)*(1-Phi2);
elseif A == 11
    nu_conj=Phi1b*Phi2b;
elseif A==10
    nu_conj=(1-Phi2b)*Phi1;
else
    disp('error:wrong event')
end
end

function logL = get_loglikelihood(D,p,As,theta,xsupp)
y = D(:,1);
x1 = D(:,2);
x2 = D(:,3);
[nx,~] = size(xsupp);
qtheta = zeros(nx,4);
freq = zeros(nx,4);
for l=1:nx
    qtheta(l,:) = get_qtheta_cf(theta,As,p(l,:),xsupp(l,1),xsupp(l,2));
    freq(l,1) = sum((y==0).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,2) = sum((y==11).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,3) = sum((y==10).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,4) = sum((y==01).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
end
logL = sum(sum(freq.*log(qtheta)));
end


function [thetahat,Lval] = get_rmle(D,p,As,xsupp)
fun = @(beta) -get_loglikelihood(D,p,As,[0,0,beta],xsupp);
lb = [-3,-3];
ub = [0,0];
options = optimoptions('fmincon','Display','off');
[betahat,fval] = fmincon(fun,[-1,-1],[],[],[],[],lb,ub,[],options);
thetahat = [0,0,betahat];
Lval = -fval;
end

function [thetahat,Lval] = get_rmle_bopt(D,p,As,xsupp)
beta1 = optimizableVariable('b1',[-3,0]);
beta2 = optimizableVariable('b2',[-3,0]);
fun = @(theta) -get_loglikelihood(D,p,As,[0,0,theta.b1,theta.b2],xsupp);
results = bayesopt(fun,[beta1,beta2],'Verbose',0,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'UseParallel',true,'PlotFcn',[]);
thetahat = [0,0,results.XAtMinObjective.b1,results.XAtMinObjective.b2];
Lval = -results.MinObjective;
end

function Tn = get_splitLR(D0,logL0,p1,xsupp)
y = D0(:,1);
x1 = D0(:,2);
x2 = D0(:,3);
[nx,~] = size(xsupp);
freq = zeros(nx,4);
for l=1:nx
    freq(l,1) = sum((y==0).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,2) = sum((y==11).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,3) = sum((y==10).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
    freq(l,4) = sum((y==01).*(x1==xsupp(l,1)).*(x2==xsupp(l,2)));
end
logL1 = sum(sum(freq.*log(p1)));
logTn = logL1 - logL0;
Tn = exp(logTn);
end