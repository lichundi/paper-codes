% uncomment those val lines to generate validation data
function hyperDataClassfication()
    load('train_test_gt');
    load 'PaviaU'
    %% preprocessing
    pre = paviaU;
    pre = pre - min(pre(:)); %min数组的最小元素
    pre = pre / max(pre(:));
    %%
    [h, w, dim] = size(pre); %size(A) 返回一个行向量，其元素包含 A 的相应维度的长度。
    save pre.mat pre;
    % MaxIndex = (max(max(train_test_gt)))/ 3; %with val
    MaxIndex = (max(max(train_test_gt)))/ 2; %without val

    trainclass = cell(MaxIndex, 1);%返回由空矩阵构成的MaxIndex × 1元胞数组
    testclass = cell(MaxIndex, 1);
    % valclass = cell(MaxIndex, 1);

    data = reshape(pre, [h*w dim]); %reshape(A,[2,3]) 将 A 重构为一个 2×3 矩阵。
    %% test class
    for j = 1:MaxIndex
        testclass{j} = data((train_test_gt==j), :);
    end
    %% train class
    for j = MaxIndex+1:2*MaxIndex
        trainclass{j-MaxIndex} = data((train_test_gt==j), :);
    end
    % %% val class
    % for j = 2*MaxIndex+1:3*MaxIndex
    %     valclass{j-2*MaxIndex} = data((train_test_gt==j), :);
    % end
    %% save
    save trainclass.mat trainclass
    save testclass.mat testclass
    % save valclass.mat valclass
end