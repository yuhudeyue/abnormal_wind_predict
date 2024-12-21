load('out_y_6_2020_7_20_selected_detected.mat');
the_data = out_y_6_2020_7_20_selected_detected;
[n,m] = size(the_data);

for i = 1 : 8

IMF = eemd(the_data(i,:)',0,50);
save(['IMF_out_y_6_2020_7_20_selected_detected/IMF_out_y_6_2020_7_20_selected_detected_' num2str(i) '.mat'],'IMF');
end
[n,m] = size(IMF);
for i = 1 : m
    
    figure(),plot(IMF(:,i));
    
end
