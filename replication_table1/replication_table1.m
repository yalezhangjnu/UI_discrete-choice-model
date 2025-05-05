% This file is used to generate the table 1 for the empirical application
% Table 1 Descriptive Statistics
rawdat = readtable('rawdatafinal_2010.csv');   % read rawdata
columns=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]; % select data we need from raw data set: in total 16, 2 outcomes, 12 covaraiates and 2 IV
data=rawdat{:,columns};                         % dimension: 6136 by 16
mean_values = mean(data, 'omitnan');
percentile_25 = prctile(data, 25);
median_values = median(data, 'omitnan');
percentile_75 = prctile(data, 75);
std_dev = std(data, 'omitnan');
num_observations = sum(~isnan(data));

display_order = [2, 1, size(data, 2)-1, size(data, 2), 3:size(data, 2)-2];
%% adjust all statsitics in the order of Table 1
ordered_mean_values = mean_values(display_order);
ordered_percentile_25 = percentile_25(display_order);
ordered_median_values = median_values(display_order);
ordered_percentile_75 = percentile_75(display_order);
ordered_std_dev = std_dev(display_order);
ordered_num_observations = num_observations(display_order);

variables = {'Enforcement action', 'Lobbying status', 'Distance to DC', 'Initial Market size', 'Capital adequacy', ...
             'Asset Quality', 'Management Quality', 'Earning', 'Liquidity', 'Sensitivity to Market Size'...
             'Deposit to asset ratio', 'Leverage', 'Total Core Deposit', 'Size','Age', 'Personal Income Growth'};


% use fprint function to print the table title
fprintf('%-25s %-15s %-18s %-15s %-18s %-15s %-18s\n', ...
    'Variable', 'Mean', '25th_Percentile', 'Median', ...
    '75th_Percentile', 'Std_Dev', 'Num_Observations');

% loop fprint every line
for i = 1:length(variables)
    fprintf('%-25s %-15.3f %-18.3f %-15.3f %-18.3f %-15.3f %-18d\n', ...
        variables{i}, ordered_mean_values(i), ordered_percentile_25(i), ...
        ordered_median_values(i), ordered_percentile_75(i), ...
        ordered_std_dev(i), ordered_num_observations(i));
end