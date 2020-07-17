function [tensor, gt] = generate4dTensor(pre, gt, j)
    [row, col]= find(gt==j, 1, 'first');
    deepth = size(pre, 3);
    raw_pixel = pre(row, col, :);
    gt(row, col) = nan;%nan  Not A Number就是代bai表不是一个数据 (1)处理缺失的数据时就会跳过或者其他处理(2)绘图的时候，如果我们想挖掉一部分
    raw_pixel = reshape(raw_pixel, [1, deepth]);% reshape(A,[2,3]) 将 A 重构为一个 2×3 矩阵。
    [h, w] = size(gt);
    total = 0;
    % change window size here, 1 means a 3*3 window, 2 means a 5*5 window 
    % etc.
    window_size = 2;
    for j = -window_size:1:window_size
        for k = -window_size:1:window_size
            if j==0 && k==0
                continue
            end
            r = row + j;
            c = col + k;
            if r>0 && r<=h && c>0 && c<=w
                total = total + 1;
            end
        end
    end
    tensor = zeros(total, 1, 2, deepth);
    index = 1;
    for j = -window_size:1:window_size
        for k = -window_size:1:window_size
            if j==0 && k == 0
                continue
            end
            r = row + j;
            c = col + k;
            if r>0 && r<=h && c>0 && c<=w
                tensor(index, 1, 1, :) = raw_pixel;
                tensor(index, 1, 2, :) = reshape(pre(r, c, :), [1, deepth]);
                index = index + 1;
            end
        end
    end
end