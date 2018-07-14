load('Indian_pines_gt.mat');
load('Indian_pines_corrected.mat');
total_x = reshape((indian_pines_corrected),[],200);
total_y = reshape(uint8(indian_pines_gt),[],1);

number_classes = max(total_y);
test_indexs = [];

for i = 0:number_classes
    all_indexs = find(total_y == i);
    pick_20 = randsample(all_indexs,floor(size(all_indexs,1)*0.25));
    test_indexs = [test_indexs;pick_20];
end

total_mask = zeros(size(total_y));
total_mask(test_indexs) = 1;
test_y = total_y(total_mask == 1);
test_x = total_x(total_mask == 1, :);
train_y = total_y(total_mask == 0);
train_x = total_x(total_mask == 0, :);
testLabels = svmClassify(train_x,train_y,total_x);
testLabels = testLabels-1;
correct = testLabels==total_y;
before_denoise_acc = sum(correct)/size(testLabels,1);

% plot the result image
result_image = reshape(testLabels, [145, 145]);
result_image = denoise(result_image);
                            
im = imagesc(result_image);
axis off
colorbar
title("Kernel : Polynomial. After Denoise");



