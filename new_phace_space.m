function new_phace_space()

folder = 'phase_space';

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.mat'))];

for i = 1 : length(filepaths)
%     if (isequal(filepaths(i).name,'.')||...
%         isequal(filepaths(i).name,'..')||...
%         ~filepaths(i).isdir)
%         continue;
%     end
    load(fullfile(folder,filepaths(i).name));
    filepaths(i).name
    if mod(i,8)==0
        Y_temp(:,:,8) = Y;
    else
        Y_temp(:,:,mod(i,8)) = Y;
    end
    
    if mod(i,8)==0
        Y = Y_temp;
        save(['new_phase_space/phasespace_' num2str(fix(i/8)) '_' filepaths(i).name],'Y');
    end
end


end