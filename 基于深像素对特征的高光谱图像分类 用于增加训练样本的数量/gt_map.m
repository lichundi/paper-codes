%%
%this script is used to generate classification map after getting prediction.txt
%%
clear;
close all;

f = fopen('prediction.txt');
prediction = fscanf(f, '%d\n'); %读取文本文件中的数据
load train_test_gt
load testclass
load PaviaU_gt

index = 1;
for j=1:9
    for k=1:size(testclass{j},1) %size 返回维度1的长度
        [row, col]= find(train_test_gt==j, 1, 'first'); %find 查找与train_test_gt==j中的非零元素对应的第一个索引
        train_test_gt(row, col) = 16;
        paviaU_gt(row, col) = prediction(index)+1;
        index = index + 1;
    end
end
%imagesc将paviaU_gt中的元素数值按大小转化为不同颜色
% axis image将坐标轴显bai示du的框调整到显示数据最紧凑的情况,同时xy比例一致
%axis off把坐标系设为不可见,但把坐标系的Title设为可见
imagesc(paviaU_gt),axis image,axis off;
