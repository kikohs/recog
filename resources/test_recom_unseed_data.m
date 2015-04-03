




function test_recom_unseed_data




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Recommend unseen data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if 1==1
    
    % Load
    load('mat/result_AB.mat','C','O','Agt','Bgt','A','B','lambdaTV');
    
    % Display
    cpt_fig = 22;
    figure(cpt_fig); clf;
    subplot(131);
    imagesc(C); colorbar;
    title('C');
    subplot(132);
    imagesc(O.*C,[min(C(:)),max(C(:))]); colorbar;
    perc = nnz(O)/prod(size(O));
    title(['O*C - Perc observed values= ', num2str(perc) ]);
    subplot(133);
    imagesc(A*B); colorbar;
    title(['Our Solution - error= ', num2str( norm( Agt*Bgt - A*B ,2) ) , ' - lamdaTV= ', num2str( lambdaTV ) ]);
    
    % select one row of C
    [n,m] = size(C)
    t = floor( 1 + (n-1)* rand)
    c_test = C(t,:);
    
    % mask of observations
    o_test = zeros(m,1);
    th = 0.25;
    th = 0.05;
    R = rand(m,1);
    o_test(R<=th) = 1;
    perc = nnz(o_test)/m
    
    % solution
    [r,~] = size(B);
    Bt = B';
    ct = c_test';
    Ot = diag(o_test);
    tic
    c_recom = ( Bt'*Ot*Bt + 1e-3*eye(r) )\ (Bt'*Ot*ct);
    time = toc
    c_recom = c_recom';
    
    % Display
    cpt_fig = 30;
    figure(cpt_fig); clf;
    subplot(121);
    plot(c_test); hold on;
    plot(o_test'.*c_test,'r+');
    title('c ground truth');
    subplot(122);
    plot(c_recom*B); hold on;
    plot(o_test'.*c_test,'r+');
    title(['c computed, perc observation= ', num2str( perc ), ' - error= ', num2str( norm(c_test-c_recom*B)/norm(c_test) ), ' - computing time (sec)= ', num2str( time ) ]);
    
    return
    
end















    
    