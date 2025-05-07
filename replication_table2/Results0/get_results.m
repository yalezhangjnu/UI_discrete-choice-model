
cf_asfvalues = linspace(0,1,200);
Sn_asf_vals = zeros(200, 1);
Tn_asf_vals = zeros(200, 1);
Tn_swapasf_vals = zeros(200, 1);
test_resultd_vals = zeros(200, 1);

% Preallocate cell arrays for vectors (each cell can have different sizes)
rmle_exitflags0asf_vals = cell(200, 1);
rmle_exitflags1asf_vals = cell(200, 1);

% Loop over the indices (adjust range if necessary)
for idx = 1:200
    % Construct file name
    filename = sprintf('C:/Users/YI/Desktop/RobustLR1030/Bankdata/Results0/app_parametricasf_d0_cfsizevalue0_ind%d.mat', idx);
    
    % Load the .mat file
    data = load(filename);
    
    % Store scalar values, checking for emptiness
    if isempty(data.Sn_asf)
        Sn_asf_vals(idx) = Inf;
    else
        Sn_asf_vals(idx) = data.Sn_asf;
    end
    
    if isempty(data.Tn_asf)
        Tn_asf_vals(idx) = Inf;
    else
        Tn_asf_vals(idx) = data.Tn_asf;
    end
    
    if isempty(data.Tn_swapasf)
        Tn_swapasf_vals(idx) = Inf;
    else
        Tn_swapasf_vals(idx) = data.Tn_swapasf;
    end
    
    if isempty(data.test_resultd)
        test_resultd_vals(idx) = NaN;  % or whatever makes sense for you
    else
        test_resultd_vals(idx) = data.test_resultd;
    end
    
    % Store variable-length vectors into cell arrays
    rmle_exitflags0asf_vals{idx} = data.rmle_exitflags0asf;
    rmle_exitflags1asf_vals{idx} = data.rmle_exitflags1asf;
end

% Find smallest and largest indices with test_resultnd == 0
zero_indices = find(test_resultd_vals == 0);

if isempty(zero_indices)
    disp('No instances where test_resultnd_vals is zero.');
    smallest_idx = NaN;
    largest_idx = NaN;
else
    smallest_idx = min(zero_indices);
    largest_idx = max(zero_indices);
end

% Check if zero_indices are found first
if isempty(zero_indices)
    disp('No instances where test_resultnd_vals is zero.');
    cf_value_smallest = NaN;
    cf_value_largest = NaN;
else
    % Get the cf_asfvalues corresponding to smallest and largest idx
    cf_value_smallest = cf_asfvalues(smallest_idx);
    cf_value_largest = cf_asfvalues(largest_idx);
    
    fprintf('cf_asfvalue for smallest_idx (%d): %.4f\n', smallest_idx, cf_value_smallest);
    fprintf('cf_asfvalue for largest_idx (%d): %.4f\n', largest_idx, cf_value_largest);
end

% Calculate Status-quo for the table
rawdat = readtable('rawdatafinal_2010.csv');   % read rawdata
columns=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]; % select data we need from raw data set: in total 16, 2 outcomes, 12 covaraiates and 2 IV
dat=rawdat{:,columns};
[row_indices, ~] = find(isnan(dat));           % find missing value indexes
unique_missing = unique(row_indices);          % return a unique missing value indexes
dat_clean=dat;   
% This is for the entire population
mean_values = mean(dat_clean, 'omitnan');
display_order = [2, 1, size(dat_clean, 2)-1, size(dat_clean, 2), 3:size(dat_clean, 2)-2];
ordered_mean_values = mean_values(display_order);
statusquo=ordered_mean_values(:,1); 
fprintf('cf_asfvalue for smallest_idx (%d)-(status-quo): %.4f\n', smallest_idx, cf_value_smallest-statusquo);
fprintf('cf_asfvalue for largest_idx (%d)-(status-quo): %.4f\n', largest_idx, cf_value_largest-statusquo);


% OPTIONAL: save the aggregated results into a new .mat file
save('aggregated_asf_results.mat', 'Sn_asf_vals', 'Tn_asf_vals', ...
    'Tn_swapasf_vals', 'test_resultd_vals', ...
    'rmle_exitflags0asf_vals', 'rmle_exitflags1asf_vals');