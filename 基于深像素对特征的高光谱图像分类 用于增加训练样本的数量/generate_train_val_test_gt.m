%train_samples: the training samples in each class, 200 in demo.
%val_indices is used in validation.
function generate_train_val_test_gt(train_samples)
    load 'PaviaU_gt'
    train_test_gt = paviaU_gt;
    N = 9;
    rng('default');
    rng(1);
    for k = 1:N
        indices = find(train_test_gt==k);
        n = size(indices, 1); %size 返回维度1的长度
        rndIDX = randperm(n);
        s = int32(train_samples);%32 位有符号整数数组
        train_indices = indices(rndIDX(1:s)); %indices 索引被替换为从索引rndIDX 的1到s。
    %     val_indices = indices(rndIDX(s+1:train_samples));
        train_test_gt(train_indices) = N+k;
    %     train_test_gt(val_indices) = 2*(N)+k;
    end

    % imagesc(train_test_gt);
    save train_test_gt.mat train_test_gt
end