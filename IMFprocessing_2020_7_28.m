function IMFprocessing_2020_7_28()

folder = 'IMF_2020_7_28';

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.mat'))];

for i = 1 : length(filepaths)
    load(fullfile(folder,filepaths(i).name));
    [s_1,s_2] = size(IMF);
    IMF_processing = zeros(s_1,15);
    for j = 2
        IMF_processing(:,1) = IMF(:,j);
    end
    for j = 3
        IMF_processing(:,2) = IMF(:,j);
    end
    for j = 4
        IMF_processing(:,3) = IMF(:,j);
    end
    for j = 5
        IMF_processing(:,4) = IMF(:,j);
    end 
    for j = 6
        IMF_processing(:,5) = IMF(:,j);
    end    
    for j = 7
        IMF_processing(:,6) = IMF(:,j);
    end
    for j = 8
        IMF_processing(:,7) = IMF(:,j);
    end
    for j = 9
        IMF_processing(:,8) = IMF(:,j);
    end
    for j = 10
        IMF_processing(:,9) = IMF(:,j);
    end 
    for j = 11
        IMF_processing(:,10) = IMF(:,j);
    end    
    for j = 12
        IMF_processing(:,11) = IMF(:,j);
    end
    for j = 13
        IMF_processing(:,12) = IMF(:,j);
    end
    for j = 14
        IMF_processing(:,13) = IMF(:,j);
    end
    for j = 15
        IMF_processing(:,14) = IMF(:,j);
    end 
    for j = 16
        IMF_processing(:,15) = IMF(:,j);
    end    

    %12345,6789,101112,13141516
    save(['entropy_processing_2020_7_28/entropy_processing_' filepaths(i).name],'IMF_processing');
end