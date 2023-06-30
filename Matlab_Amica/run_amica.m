load("pink_sinus_data.mat", "x")
load("pink_sinus_data.mat", "s")
%load("eeg_data.mat", "x") %also change in amica.m!!!
%B = [1 4;4 1];
%s = gg6(2,1000,[0 0],[1 3],[1.5 1.5]);

%x = [1 4; 4 1]*[1.0 2 3; 4 5 6]
%x = B*s

%[A,W,S,khinds,c,LL,Ltall,gm,alpha,mu,beta,rho] = amica10(x,1,2,100,1,1);

%[A,c,LL,Lt,gm,alpha,mu,beta,rho] = amica_hardcoded(x,2,3,3,1);


tic
[A,c,LL,Lt,gm,alpha,mu,beta,rho] = amica_a(x,2,3,538,1);
toc

tiledlayout(3,1)
nexttile
plot(x(:,1:100)')
title('mixed data')
nexttile

unmixed = pinv(A(:,:,1))*x;
plot(unmixed(:,1:100)')
title('unmixed data')
nexttile

plot(LL)
title('LL')







%Inputs: amica_a
%
%       x           --  data, channels (n) by time points (N)
%       M           --  number of ICA mixture models (default is 1)
%       m           --  number of source density mixtures (default is 3)
%       maxiter     --  maximum number of iterations to perform (default is 500)
%       update_rho  --  1 = update the generalized gaussian shape parameters (default)
%                       0 = set all density shapes to 2 (Gaussian), do not update shape
%                       -rho0 = set density shapes to rho0, do not update shape
%       mindll      --  stop when avg log likelihood changing less than this (default is 1e-8)
%       iterwin     --  window over which to do moving avg of log likelihood (default is 50)
%       do_newton   --  1 = use Newton update for upmixing matrices
%                       0 = use natural gradient updates
%       remove_mean --  make the data zero mean (default is 1 == yes)
%       As          --  (optional) true generating basis for plotting purposes
%       cs          --  (optional) true model centers

%Inputs: amica10(x,M,m,maxiter,do_sphere,do_newton,As,cs,Ainit,cinit,kinit)
%
%       x           --  data, channels (n) by time points (N)
%       M           --  number of ICA mixture models (default is 1)
%       m           --  number of source density mixtures (default is 3)
%       maxiter     --  maximum number of iterations to perform (default is 100)
%       do_sphere   --  1 = remove mean and sphere data (default)
%                       2 = only remove mean and normalize channels
%                       0 = do not remove mean or sphere
%       do_newton   --  1 = use Newton update for upmixing matrices (default)
%                       0 = use natural gradient updates
%       As          --  (optional) true generating basis for plotting purposes
%       cs          --  (optional) true model centers for plotting purposes
