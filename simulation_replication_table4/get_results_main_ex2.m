
M=20;
hgrid = linspace(0,2,M);
Snrej_prob_store = zeros(M,2);
DGP = 'IID';
samplesize = [50,100,200,300];
for j=1:4
    for ind =1:M
        n = samplesize(j);
        load(['./Results/crossfit_ex2_dat_',DGP,'n',num2str(n),'_ind',num2str(ind)])
        Snrej_prob_store(ind,j) = Snrej_prob;
    end
end
disp(Snrej_prob_store)
%plot(hgrid,Snrej_prob_store)
%xlabel('h')
%legend('n=50','n=100','n=200','n=300')
%print('./Figures/power_ex2.pdf','-dpdf')
%tabledata = [hgrid(1:15);Snrej_prob_store(1:15,:)']

%Mval=20;
%hgrid = linspace(0,0.15,Mval);
%Snrej_prob_store = zeros(Mval,1);
%BCSrej_prob_store = zeros(Mval,1);
%DGP = 'IID';
%samplesize = [5000,7500];
%for j=1:2
%    for ind =1:Mval
%        n = samplesize(j);
%        load(['./Results/crossfit_ex2_dat_',DGP,'n',num2str(n),'_ind',num2str(ind)])
%        Snrej_prob_store(ind,j) = Snrej_prob;
%        load(['./Results/bcs_ex2_dat_',DGP,'n',num2str(n),'_ind',num2str(ind)])
%        BCSrej_prob_store(ind,j) = sum(Tn_MRsim>cn_MRsim)/S;
%    end
%end
%plot(hgrid,Snrej_prob_store,hgrid,BCSrej_prob_store)
%xlabel('h')
%legend('cross-fit LR test (n=5000)','cross-fit LR test (n=7500)',...
%    'moment-based test (n=5000)','moment-based test (n=7500)','Location','SouthEast')
%print('./Figures/power_ex2_bcs_comparison.pdf','-dpdf')
