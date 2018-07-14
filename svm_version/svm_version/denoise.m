function result_image = denoise(result_image)
    load('Indian_pines_gt.mat');
    correct = result_image==indian_pines_gt;
    disp('Before denoise global accuracy')
    sum(sum(correct))/(size(indian_pines_gt,1)*size(indian_pines_gt,2))
    for i = 1:144
        for j = 1:144
            class = -1;
            flag=0;
            for m = -1:1
                if(flag == 1)
                    break
                end
                for n = -1:1
                    if ((m == 0 && n == 0) || (i+m <= 0 || j+n <= 0))
                        continue
                    end
                    if(result_image(i+m,j+n) == result_image(i,j))
                        flag = 1;
                        break;
                    else
                        if (class == -1)
                            class = result_image(i+m,j+n);
                        elseif (result_image(i+m,j+n) ~= class)
                            flag = 1;
                            break;
                        end
                    end
                    if(m == 1 && n == 1 && class ~= -1)
                        result_image(i,j) = class;
                    end
                end
            end
        end
    end
    correct = result_image==indian_pines_gt;
    disp('After denoise global accuracy')
    sum(sum(correct))/(size(indian_pines_gt,1)*size(indian_pines_gt,2))



