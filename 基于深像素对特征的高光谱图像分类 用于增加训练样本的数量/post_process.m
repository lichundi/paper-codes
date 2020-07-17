function post_process()
    load train_data
    % load val_data
    load test_label
    % extract subset from each class
    N = size(train_data, 1);
    trainlabel = [];
    % vallabel = [];
    for j=1:N
        nRows = size(train_data{j}, 1);
        trainlabel = cat(1, trainlabel, ones(nRows, 1)*j); %cat(dim, A1, A2, A3, A4, ...) 沿数组维度 dim 串联所有输入数组
    end
    traindata = (cell2mat(train_data)); %%cell2mat 将元胞数组转换为普通数组

    % for j=1:N
    %     nRows = size(val_data{j}, 1);
    %     vallabel = cat(1, vallabel, ones(nRows, 1)*j);
    % end
    % valdata = cell2mat(val_data);
    %% save data and label
    trainlabel = int64(trainlabel);
    testlabel = int64(test_label2);
    % vallabel = int64(vallabel);

    % save
    save traindata.mat traindata
    save trainlabel.mat trainlabel
    save testlabel.mat testlabel
    % save valdata.mat valdata
    % save vallabel.mat vallabel
end