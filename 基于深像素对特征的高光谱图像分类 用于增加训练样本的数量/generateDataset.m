% this script is used to generate paired data for training
% and the slide-window data for a default windowsize 5.
function generateDataset()
    load trainclass
    load testclass
    % load valclass
    load train_test_gt
    load pre

    deepth = 103;
    N = size(trainclass, 1);
    classes = N;
    %% generate train and val data
    train_data = cell(N+1, 1); %返回由空矩阵构成的（N+1）×1元胞数组
    n = size(trainclass{1}, 1);%n取trainclass{1}的长度
    index = 1;
    %Class 0. if you change the number of training samples, you should
    %change the following line.
    train_data{1, 1} = zeros(n*classes*(classes-1)*3, 1, 2, deepth);
    rng('default');
    rng(1);
    for m = 1:N
        for j = 1:N
            if m==j
                continue
            end
            n = size(trainclass{1}, 1);
            for k = 1:n
                samples = datasample(trainclass{j}, 3, 1, 'Replace', false); %datasample从数据集中随机抽取样本函数
                train_data{1, 1}(index, 1, 1, :) = trainclass{m}(k, :);
                train_data{1, 1}(index, 1, 2, :) = samples(1,:);
                index = index + 1;
                train_data{1, 1}(index, 1, 1, :) = trainclass{m}(k, :);
                train_data{1, 1}(index, 1, 2, :) = samples(2,:);
                index = index + 1;
                train_data{1, 1}(index, 1, 1, :) = trainclass{m}(k, :);
                train_data{1, 1}(index, 1, 2, :) = samples(3,:);
                index = index + 1;
            end
        end
    end
    for j = 2:N+1
        n = size(trainclass{j-1}, 1);
        train_data{j, 1} = zeros(n*(n-1), 1, 2, deepth);
        index = 1;
        for k = 1:n
            for m = 1:n
                if k == m
                    continue
                end
                train_data{j, 1}(index, 1, 1, :) = trainclass{j-1}(k, :);
                train_data{j, 1}(index, 1, 2, :) = trainclass{j-1}(m, :);
                index = index + 1;
            end
        end
    end
    % % val data
    % val_data = cell(N, 1);
    % n = size(valclass{1}, 1);
    % val_data{1, 1} = zeros(n*classes*(classes-1)/2, 1, 2, deepth);
    % index = 1;
    % for j = 2:N
    %     n = int32(size(valclass{1}, 1) / 2);
    %     for k = 1:n
    %         val_data{1, 1}(index, 1, 1, :) = valclass{1}(k, :);
    %         val_data{1, 1}(index, 1, 2, :) = datasample(valclass{j}, 1, 1);
    %         index = index + 1;
    %     end
    % end
    % for j = 2:N
    %     n = size(valclass{j}, 1);
    %     val_data{j, 1} = zeros(n*(n-1), 1, 2, deepth);
    %     index = 1;
    %     for k = 1:n
    %         for m = 1:n
    %             if k == m
    %                 continue
    %             end
    %             val_data{j, 1}(index, 1, 1, :) = valclass{j}(k, :);
    %             val_data{j, 1}(index, 1, 2, :) = valclass{j}(m, :);
    %             index = index + 1;
    %         end
    %     end
    % end

    %% test data
    num_of_testdata = 0;
    for j = 1:N
        num_of_testdata = num_of_testdata + size(testclass{j}, 1);
    end
    test_data2 = cell(num_of_testdata, 1);
    test_label2 = zeros(num_of_testdata, 1);
    index = 1;
    for j = 1:N
        for k = 1:size(testclass{j}, 1)
            % test data
            [test_data2{index}, train_test_gt] = generate4dTensor(pre, train_test_gt, j);
            test_label2(index) = j;
            index = index + 1;
        end
    end
    test_label_width = [];
    for k = 1:size(test_data2, 1)
        test_label_width = cat(1, test_label_width, size(test_data2{k}, 1)); %cat(dim, A1, A2, A3, A4, ...) 沿数组维度 dim 串联所有输入数组
    end
    test_label_width = int32(test_label_width);%int32数组转换为 int32 类型的数组
    testdata = cell2mat(test_data2); %cell2mat 将元胞数组转换为普通数组
    %% save data
    save train_data.mat train_data
    % save val_data.mat val_data
    save testdata5.mat testdata -v7.3
    save test_label.mat test_label2
    save testlabelwidth5.mat test_label_width
end