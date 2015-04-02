 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve C = AB, Omega = Observations with
% min_{A,B>=0} D_Kullback-Leibler ( Omega * ( C || AB ) ) 
%                  + gamma_A ||A||_TV + gamma_B ||B||_TV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function test_NMF_KL_TV_MC_v1


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Create Recommendation Matrix C (Netflix-style) from A,B s.t. C = AB
% % Generate Graph GB for Matrix B (noisy community graph)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% if 2==1
%     % Parameters
%     n = 256;
%     m = 128;
%     rA = 12;
%     rB = 15;
%     r = min( rA , rB ); % rank
%     [C,A,B,WB] = create_recommendation_data(n,m,r,rA,rB);
%     save('mat/recommendation_data.mat','C','A','B','WB');
% else
%     load('mat/recommendation_data.mat','C','A','B','WB');
%     [n,m] = size(C)
%     r = size(A,2)
% end
% 
% % Display
% if 1==1
%     cpt_fig = 1;
%     figure(cpt_fig);
%     subplot(131);
%     imagesc(A); colorbar;
%     title('A');
%     subplot(132);
%     imagesc(B); colorbar;
%     title('B');
%     subplot(133);
%     imagesc(C); colorbar;
%     title('C');
%     %pause
% end
% 
% % Save ground truth
% Agt = A;
% Bgt = B;
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Observation Mask for Matrix Completion 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% O = zeros(n,m); % O = Omega
% R = rand(n,m);
% % th = keep th % of values
% th = 0.8;
% th = 0.5;
% th = 0.2;
% th = 0.1;
% th = 0.05;
% %th = 0.025;
% O(R<=th) = 1;
% perc = nnz(O)/(n*m);
% 
% % Save it
% if 2==1
%     save('mat/Omega_mask.mat','O','perc');
% end
% load('mat/Omega_mask.mat','O','perc');
% 
% % Display
% if 1==1
%     cpt_fig = 3;
%     figure(cpt_fig); clf;
%     subplot(121);
%     imagesc(C); colorbar;
%     title('C');
%     subplot(122);
%     imagesc(O.*C,[min(C(:)),max(C(:))]); colorbar;
%     title(['O*C - Perc observed values= ', num2str(perc) ]);
%     %return
% end
% 
% % NO COMPLETION
% %O = ones(n,m);
% 
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Compute graph gradient operator of WB
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Save it
% if 2==1
%     KB = compute_graph_gradient(WB);
%     save('mat/gradient_KB.mat','KB');
% end
% load('mat/gradient_KB.mat','KB');
% 
% 
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Compute graph of A from ratings and graph gradient
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Save it
% if 2==1
%     kNN = 10; % nb of nearest neighbors
%     WA = compute_graph_from_ratings(O.*C,kNN);
%     KA = compute_graph_gradient(WA);
%     save('mat/gradient_KA.mat','KA');
% end
% load('mat/gradient_KA.mat','KA');
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Init: A^{n=0}, B^{n=0}
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Ainit = rand(n,r);
% Binit = rand(r,m);
% 
% % Column and Row L2 Normalization
% A = Ainit;
% A = bsxfun(@rdivide, A, sqrt(sum(A.^2,1)));
% Ainit = A;
% B = Binit;
% B = bsxfun(@rdivide, B', sqrt(sum(B'.^2,1))); B = B';
% Binit = B;
% 
% % Save it
% if 2==1
%     save('mat/initAB.mat','Ainit','Binit');
% end
% load('mat/initAB.mat','Ainit','Binit');
% 
% % Display
% if 1==1
%     cpt_fig = 4;
%     figure(cpt_fig); clf;
%     subplot(121);
%     imagesc(Ainit); colorbar;
%     title('Init A');
%     subplot(122);
%     imagesc(Binit); colorbar;
%     title('Init B');
%     %return
% end












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOR PYTHON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 2==1
    
    normA = normest(Agt);
    normK = normest(KB);
    
    save('recom_data.mat','C','Agt','Bgt','Ainit','Binit','WB','O','perc','KB','KA','normA','normK')
    
    return
    
end

load('recom_data.mat','C','Agt','Bgt','Ainit','Binit','WB','O','perc','KB','KA','normA','normK')










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proximal algorithm for recommendation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters for NMF + TV
lambdaTV = 1e-4*5; % NMF + TV
%lambdaTV = 1e-6;   % NMF ONLY (NO TV)
%lambdaTV = 1e-2;    % TV ONLY (NO NMF)

% Init
A = Ainit;
B = Binit;
% for niter=1:20
%     
%     
%     niter
    
    
    
    %%%%%%%%%%%%%%%%%%%
    % Update B
    %%%%%%%%%%%%%%%%%%%
    
    % TO TEST INDEPENDENTLY OF A
    A = Agt;
    
    % Row, Column L2 Normalization
    A = bsxfun(@rdivide, A, sqrt( sum(A.^2,1))+1e-6 );
    B = bsxfun(@rdivide, B', sqrt( sum(B'.^2,1))+1e-6 ); B = B'; 
    
    % Init Y,B
    Bb = B; Bold = B;
    Y1 = A* B;
    Y2 = KB* B';
    
%     % Norm of A, KB
%     normA = normest(A);
%     normK = normest(KB);
    
    % Init time steps
    sigma1 = 1e0/normA;
    tau1 = 1e0/normA;
    sigma2 = 1e0/normK;
    tau2 = 1e0/normK;
    
    % Gammas
    gamma1 = 1e-1;
    gamma2 = 1e-1;

    % Parameter
    %lambdaTV = 1e-4*5;
    %lambdaTV = 1e-6;
    
    % Loop
    NbInnerIter = 1000; % 1000 5000
    PlotError = zeros(NbInnerIter,1);
    for i=1:NbInnerIter
        
        % New Y1 (NMF)
        Y1 = Y1 + sigma1.* (A* Bb);
        Y1 = 0.5* ( Y1 + O - ( (Y1-O).^2 + 4*sigma1.* O.* C ).^0.5 );
        
        % New Y2 (TV)
        Y2 = Y2 + sigma2.* (KB*Bb');
        Y2 = Y2 - sigma2.* Shrink( Y2/sigma2 , lambdaTV/sigma2 );
        
        % New B
        B = B - tau1.* ( A'*Y1 ) - tau2.* ( KB'* Y2 )';
        B = max(B,0);
       
        % Acceleration
        theta1 = 1./sqrt(1+2*gamma1*tau1);
        tau1 = tau1.* theta1;
        sigma1 = sigma1./ theta1;
        theta2 = 1./sqrt(1+2*gamma2*tau2);
        tau2 = tau2.* theta2;
        sigma2 = sigma2./ theta2;
        
        % New Bb
        Bb = B + 1/2* theta1* ( B - Bold ) + 1/2* theta2* ( B - Bold );
        
        % No Acceleration
        %Bb = 2*B - Bold;
        
        % New Bold
        Bold = B;
        
        % Error Plot
        PlotError(i) = norm(Bgt-B,2);
        
    end
    
    % Display
    figure(10); clf;
    subplot(321);
    imagesc(Bgt); colorbar;
    title('Bgt');
    subplot(322);
    imagesc(B); colorbar;
    title(['B, i= ', num2str(i) ,' error= ',num2str( norm(Bgt-B,2) )]);
    subplot(323);
    imagesc(Agt*Bgt); colorbar;
    title('Agt*Bgt');
    subplot(324);
    imagesc(Agt*B); colorbar;
    title(['Agt*B, i= ', num2str(i) ,' error= ',num2str( norm( Agt*Bgt - Agt*B ,2) )]);
    subplot(325);
    imagesc(Agt*Bgt); colorbar;
    title('Agt*Bgt');
    subplot(326);
    imagesc(A*B); colorbar;
    title(['A*B, i= ', num2str(i) ,' error= ',num2str( norm( Agt*Bgt - A*B ,2) )]);

    figure(11); clf; 
    plot(PlotError,'r-'); 
    title('Error vs iter');
    pause(0.25)
    %pause
    return
     
    
    
    
    %%%%%%%%%%%%%%%%%%%
    % Update A
    %%%%%%%%%%%%%%%%%%%
    
    % TO TEST INDEPENDTLY OF H
    %B = Bgt;
    
    % Row, Column L2 Normalization
    A = bsxfun(@rdivide, A, sqrt( sum(A.^2,1))+1e-6 );
    B = bsxfun(@rdivide, B', sqrt( sum(B'.^2,1))+1e-6 ); B = B'; 
    
    % Init Y,B
    At = A'; Atb = A'; Atold = A'; 
    Y1t = (A* B)';
    Y2t = KA* Atb';
    Ot = O';
    
    % Norm of B, KA
    warning off;
    normB = normest(B);
    normK = normest(KA);
    
    % Init time steps
    sigma1 = 1e0/normB;
    tau1 = 1e0/normB;
    sigma2 = 1e0/normK;
    tau2 = 1e0/normK;
    
    % Gammas
    gamma1 = 1e-1;
    gamma2 = 1e-1;

    % Parameter
    %lambdaTV = 1e-4*5;
    %lambdaTV = 1e-6;
    
    
    % Loop
    NbInnerIter = 1000; % 1000 5000
    PlotError = zeros(NbInnerIter,1);
    for i=1:NbInnerIter
        
        % New Y1 (NMF)
        Y1t = Y1t + sigma1.* (B'* Atb);
        Y1t = 0.5* ( Y1t + Ot - ( (Y1t-Ot).^2 + 4*sigma1.* Ot.* C' ).^0.5 );
        
        % New Y2 (TV)
        Y2t = Y2t + sigma2.* (KA* Atb');
        Y2t = Y2t - sigma2.* Shrink( Y2t/sigma2 , lambdaTV/sigma2 );
        
        % New At
        At = At - tau1.* ( B* Y1t ) - tau2.* ( KA'* Y2t )';
        At = max(At,0);
        
        % Acceleration
        theta1 = 1./sqrt(1+2*gamma1*tau1);
        tau1 = tau1.* theta1;
        sigma1 = sigma1./ theta1;
        theta2 = 1./sqrt(1+2*gamma2*tau2);
        tau2 = tau2.* theta2;
        sigma2 = sigma2./ theta2;
        
        % New Atb
        Atb = At + 1/2* theta1* ( At - Atold ) + 1/2* theta2* ( At - Atold );
        
        % No Acceleration
        %Atb = 2*At - Atold;
        
        % New Bold
        Atold = At;
        
        % New A
        A = At';
        
        % Error Plot
        PlotError(i) = norm(Agt-A,2);
        
    end
%     
%     % Display
%     figure(20); clf;
%     subplot(321);
%     imagesc(Agt); colorbar;
%     title('Agt');
%     subplot(322);
%     imagesc(A); colorbar;
%     title(['A, i= ', num2str(i) ,' error= ',num2str( norm(Agt-A,2) )]);
%     subplot(323);
%     imagesc(Agt*Bgt); colorbar;
%     title('Agt*Bgt');
%     subplot(324);
%     imagesc(A*Bgt); colorbar;
%     title(['A*Bgt, i= ', num2str(i) ,' error= ',num2str( norm( Agt*Bgt - A*Bgt ,2) )]);
%     subplot(325);
%     imagesc(Agt*Bgt); colorbar;
%     title('Agt*Bgt');
%     subplot(326);
%     imagesc(A*B); colorbar;
%     title(['A*B, i= ', num2str(i) ,' error= ',num2str( norm( Agt*Bgt - A*B ,2) )]);
% 
%     figure(21); clf; 
%     plot(PlotError,'r-'); 
%     title('Error vs iter');
%     pause(0.25)
%     
%     cpt_fig = 22;
%     figure(cpt_fig); clf;
%     subplot(131);
%     imagesc(C); colorbar;
%     title('C');
%     subplot(132);
%     imagesc(O.*C,[min(C(:)),max(C(:))]); colorbar;
%     title(['O*C - Perc observed values= ', num2str(perc) ]);
%     subplot(133);
%     imagesc(A*B); colorbar;
%     title(['Our Solution - error= ', num2str( norm( Agt*Bgt - A*B ,2) ) , ' - lamdaTV= ', num2str( lambdaTV ) ]);
%     
%     pause(0.25)
%     %pause
%     %return
%     
%     
%     % For Talk
%     cpt_fig = 100;
%     figure(cpt_fig); clf;
%     subplot(121);
%     imagesc(O.*C,[min(C(:)),max(C(:))]); axis off;
%     subplot(122);
%     imagesc(A*B); axis off;
%     
%     
%     
% end




end














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sub-Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [C,A,B,WB] = create_recommendation_data(n,m,r,rA,rB)

%%%%%%%%%%%%%%%%%%%%%%%
% Compute A
%%%%%%%%%%%%%%%%%%%%%%%
A = zeros(n,r);
t = 0.5 + 0.1*randn(rA,1);
t = t/ sum(t);
t = floor( t * n);
t(rA) = n - sum(t(1:rA-1));
for i=1:r
    shift = 0;
    for j=1:rA
        A(shift+1:shift+t(j),i) = 1 + (5-1)*rand;
        shift = shift + t(j);
    end
end

% Column L2 normalization
A = bsxfun(@rdivide, A, sqrt(sum(A.^2,1)));



%%%%%%%%%%%%%%%%%%%%%%%
% Compute B
%%%%%%%%%%%%%%%%%%%%%%%
B = zeros(r,m);
t = 0.5 + 0.1*randn(rB,1);
t = t/ sum(t);
t = floor( t * m);
t(rB) = m - sum(t(1:rB-1));
for i=1:r
    shift = 0;
    for j=1:rB
        B(i,shift+1:shift+t(j)) = 1 + (5-1)*rand;
        shift = shift + t(j);
    end
end

% Row L2 normalization L2
B = bsxfun(@rdivide, B', sqrt(sum(B'.^2,1))); B = B';




%%%%%%%%%%%%%%%%%%%%%%%
% Compute C
%%%%%%%%%%%%%%%%%%%%%%%
C = A*B;

% Display
cpt_fig = 1;
figure(cpt_fig);
subplot(131);
imagesc(A); colorbar;
title('A');
subplot(132);
imagesc(B); colorbar;
title('B');
subplot(133);
imagesc(C); colorbar;
title('C');

rankA = rank(A)
rankB = rank(B)
rankC = rank(C)







%%%%%%%%%%%%%%%%%%%%%%%
% Compute Graph GB for Matrix B
%%%%%%%%%%%%%%%%%%%%%%%

% Ground truth graph
Wgt = zeros(m,m);
shift = 0;
for j=1:rB
    Wgt( shift+1:shift+t(j) , shift+1:shift+t(j) ) = 1;
    shift = shift + t(j);
end
% Construct k-NN graph with neighbors in the same class and
%                           neighbors in a different class
nbNNsameClass = 10;
nbNNdiffClass = 1;
Wgt = triu(Wgt,1);
WB = zeros(m,m);
for i=1:m
    Gi = find(Wgt(i,:));
    rem_Gi = setdiff(1:m,[Gi,i]);
    % Same class
    idx = randperm(length(Gi));
    len = min( nbNNsameClass/2 , length(Gi) );
    idx = idx(1:len);
    Gi = Gi(idx);
    WB(i,Gi) = 1;
    % Different class
    idx = randperm(length(rem_Gi));
    idx = idx(1:nbNNdiffClass);
    rem_Gi = rem_Gi(idx);
    WB(i,rem_Gi) = 1;
end

% Symmetrize and diag=1
WB = max(WB,WB');
WB = WB + eye(m,m);
WB = sparse(WB);

% Display
figure(3);
imagesc(WB); colorbar;
title('Graph of B');


end





function K = compute_graph_gradient(W)

% Compute gradient operator
n = size(W,1);
Wadj = W;
triangularW = triu(Wadj,1);
[I, J, v] = find(triangularW);
m_edges = length(v); % number of edges in the graph
KI = [1:m_edges 1:m_edges];
KJ(1:m_edges) = I;
KJ(m_edges+1:2*m_edges) = J;
Kv(1:m_edges) = v;
Kv(m_edges+1:2*m_edges) = -v;
K = sparse(KI,KJ,Kv,m_edges,n);

end






function G = compute_graph_from_ratings(X,kNN)

% Compute distances between xi and xj for COMMON ratings
fprintf('Compute pairwise distances... (can be speed up!)\n')
n = size(X,1)
NNDist = -ones(n,n);
for i=1:n
    Xi = X(i,:);
    idxi = find(Xi>eps);
    for j=i+1:n
        Xj = X(j,:);
        idxj = find(Xj>eps);
        idx_common = intersect(idxi,idxj);
        if length(idx_common)>0
            Xit = Xi(idx_common);
            Xjt = Xj(idx_common);
            NNDist(i,j) = norm(Xit-Xjt,2)/ sqrt(length(idx_common));
        else
            NNDist(i,j) = -1;
        end
    end
end
NNDist(NNDist<0) = max(NNDist(:));
idx = tril(true(size(NNDist)),-1);
NNDist(idx) = 0;
NNDist = max(NNDist,NNDist');

% Eps-graph
if 2==1
    NNDist_sort = sort(NNDist,2);
    eps_graph = NNDist_sort(:,kNN);
    eps_graph = sqrt(1)* mean(eps_graph)
    WadjW = exp( - (NNDist.^2)./ eps_graph^2 );
    % Truncated if needed
    WadjW( WadjW < exp(-1/2) ) = 0;
    % Fix issue when no common ratings
    [I,~] = find(NNDist_sort(:,1)>0.95*max(NNDist(:)));
    for i=1:length(I)
        idx = randperm(n); idx = idx(1:kNN);
        WadjW(I(i),idx) = 1;
    end
    % Symmetrize and add diagonal
    WadjW = max(WadjW,WadjW');
    WadjW = WadjW + speye(n,n);
    WadjW = sparse(WadjW);
    perc_non_zeros_Wadj = nnz(WadjW)/ n^2
    %return
end

% kNN-graph (SEEMS BEST)
if 1==1
    [NNDist_sort,NNIdxs_sort] = sort(NNDist,2);
    NNDist_sort_crop = NNDist_sort(:,1:kNN);
    NNIdxs_sort_crop = NNIdxs_sort(:,1:kNN);
    % Scale
    scale = 1 * mean( NNDist_sort_crop(:,kNN) )
    % Adjacency Matrix
    idx = 1;
    idxi = zeros(kNN*n,1);
    idxj = zeros(kNN*n,1);
    entries = zeros(kNN*n,1);
    for k = 1:n
        % Indexes
        idxi(idx:idx+kNN-1) = k;
        idxj(idx:idx+kNN-1) = NNIdxs_sort_crop(k,:);
        % Gaussian weight function
        entries(idx:idx+kNN-1)= exp( -(NNDist_sort_crop(k,:).^2)./ scale^2 );
        % Update current index
        idx = idx + kNN;
    end
    WadjW = sparse(idxi,idxj,entries,n,n);
    % Fix issue when no common ratings
    [I,~] = find(NNDist_sort_crop(:,1)>0.95*max(NNDist(:)));
    for i=1:length(I)
        idx = randperm(n); idx = idx(1:kNN);
        WadjW(I(i),idx) = 1;
    end
    % Symmetrize and add diagonal
    WadjW = max(WadjW,WadjW');
    WadjW = WadjW + speye(n,n);
    perc_non_zeros_Wadj = nnz(WadjW)/ n^2
end

G = WadjW;

% Display
cpt_fig = 5;
figure(cpt_fig); clf;
subplot(221);
imagesc(X,[min(X(:)),max(X(:))]); colorbar;
title('X');
subplot(222);
imagesc(NNDist); colorbar;
title('NNDist');
subplot(223);
imagesc(NNDist_sort); colorbar;
title('NNDist sort');
subplot(224);
imagesc(WadjW); colorbar;
title('Wadj of W');

end




function res = Shrink(x,lambda)

s = sqrt(x.^2);
ss = s - lambda;
ss = ss.* ( ss>0 );
s = s + ( s<lambda );
ss = ss./s;
res = ss.* x;

end








    
    