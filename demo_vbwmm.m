% demo vb wmm
addpath utils
addpath ~/Documents/dynamic-brain/utils/plots

clear, close all

SEED = 56894;
rng(SEED)

Kpos = 1:10;
RESTARTS = 10;

p = 100;
T_SCALE = 1000;
N=T_SCALE*[0.4 0.3 0.2 0.1];
ZS = [1,2,3,1];
%N=T_SCALE*repmat(0.1,[1,10]);
%ZS = [1,2,3,4,5,4,3,2,1,2];
K = length(unique(ZS));

wl = 25;
L = T_SCALE/wl;
C = nan(p,p,L);
Ctest = nan(p,p,L);

%% Generate Synthetic Data

% generate covariance parameters
for n = 1:length(unique(ZS))
    R{n}=triu(randn(p));
end
[X,X_test,zt] = generateSynthData(p,N,ZS,R);

% Extract scatter matrices
zl = [];
for l = 1:L
    window_idx = (1:wl)+(l-1)*wl;
    %C(:,:,l) = X(:, window_idx )*X(:, window_idx )';
    %Ctest(:,:,l) = X_test(:, window_idx )*X_test(:, window_idx )';
    C(:,:,l) = corr(X(:, window_idx )');
    Ctest(:,:,l) = corr(X_test(:, window_idx )');
    zl=[zl, round(mean(zt(window_idx)))]; 
end
nu = wl*ones(1,L);

%% run  VB
prediction = nan(length(Kpos),RESTARTS);
found_ss = cell(length(Kpos),RESTARTS);
num_states = nan(length(Kpos),RESTARTS);
nmi_towards_true = nan(length(Kpos),RESTARTS);
lower_bound = nan(length(Kpos),RESTARTS); 

for kk = Kpos
    for r = 1:RESTARTS
        tic
        [expectations, other,priors] = vbwmm( C, nu , kk, 'run_gpu',false,...
            'verbose', 'off', 'init_method', 'kmeans', 'symmetric_tol', 1e-6, 'update_z', 'stochastic_search');
        toc
       prediction(kk,r) = sum(vbwmm_predictiveLikelihood(Ctest,nu, expectations, priors, true),1); 
       
       [~,z_vbwmm]  = max(expectations.Z,[],2);
       found_ss{kk,r} = z_vbwmm;
       num_states(kk,r) = max(unique(z_vbwmm));
       nmi_towards_true(kk,r) = calcNMI( createAssignmentMatrix(z_vbwmm,kk),  createAssignmentMatrix(zl)  );
       lower_bound(kk,r) = other.lower_bound(end);
    end
end



%% Plots

figure('Position',[993     1   927   973]),
subplot(3,1,1)
plot(Kpos,mean(prediction,2), 'bo-'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
ylabel({'Mean Log-Predictive Likelihood', 'over Restarts'})

subplot(3,1,2)
boxplot(nmi_towards_true'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
ylabel('NMI towards True')

subplot(3,1,3)
plot(Kpos,mean(lower_bound,2), 'bo-'), hold on
line([K K],get(gca,'YLim'),'Color',[0 1 0]), hold off
xlabel('Number of States')
ylabel('Mean ELBO over Restarts')



%% Train optimal model


%[~,z_vbwmm]  = max(expectations.Z,[],2);


%figure,
%stateplot(createAssignmentMatrix(z_vbwmm))