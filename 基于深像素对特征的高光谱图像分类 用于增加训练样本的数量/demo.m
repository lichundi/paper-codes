%     demo for CNN-PPF classification algorithm
%--------------Brief description-------------------------------------------
%
% 
% This demo implements the  CNN-PPF hyperspectral image classification [1]
%
%
% More details in:
%
% [1] W. Li, G. Wu, F. Zhang, Q. Du. Hyperspectral Image Classification
% Using Deep Pixel-Pair Features.
% IEEE Transactions on Geoscience and Remote Sensing.
% DOI: 10.1109/TGRS.2016.2616355
%
% contact: liwei089@ieee.org (Wei Li)
% contact: 495482236@qq.com (Guodong Wu)
% 
% Note: testing with University of Pavia data with names of PaviaU.mat, PaviaU_gt.mat 
%  http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes


% sample train test data in gt, 200 samples each class
generate_train_val_test_gt(200);
hyperDataClassfication();
generateDataset();
post_process();