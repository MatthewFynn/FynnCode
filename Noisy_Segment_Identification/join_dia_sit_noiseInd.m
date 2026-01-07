%This function will join all Dia1, Dia2 and Dia3 together,
%It will return set of good indicies (of concatenated signals)
%Hence, we know what part of the signal recordings to extract for %downstream tasks including ML/DL
clear
save_ind_table=1;
for ccc = 1:6
    addpath('/home/sparc/Desktop/TH_alldata')
    fs_new = 2000;           % Sampling frequency (Hz)
    
    %get a list of all filenames and exclude Bell, Lie, 10s
    folder_path = '/home/sparc/Desktop/TH_alldata'; 
    files = dir(folder_path); % Get list of all files and folders
    file_names = {files.name};
    
    order_1 = [1,2,4,5,6,7,8,11];
    order_23= [2,4,5,6,8,7,9,11];

    filtered_files = file_names;
    
    data = readtable('REFERENCE_ALLROUNDS_ExclusionCriteria.csv'); %subjects here
    subs = table2cell(data(:,1)); %create vector of all subjects
    for s = 1:length(subs) %for each subject
        %extract all of the files from this subject
        sub = subs{s};
        Sub_files = filtered_files(contains(filtered_files, sub)) %get all files for current subject
        
        data_concat = []; %this is where channel-wise concatenated data will be stored
        len_array(1) = 0; %so i can access it in the for loop after
        bad_ind = [];
        for f = 1:length(Sub_files) %for each .wav file belonging to particular subject
            file = Sub_files{f};
            
            %channel order determined by data collection round
            if strcmp(file(2),'c') ==1 || strcmp(file(2),'v') == 1
                order = order_23;
            else
                order = order_1;
            end
    
            %read in data and resample
            [data_wav1,fs] = audioread(file);
            data_wav = resample(data_wav1,fs_new,fs);
    
            % Below functions will Identify noisy indicies
            % ind_medPow_HM_chan1 = segment_signal_noise_multi_MedPow(data_wav(:,order(1)), 1.5, fs_new, 2.5, 1, 0,0);
            % ind_medPow_HM_chan2 = segment_signal_noise_multi_MedPow(data_wav(:,order(2)), 1.5, fs_new, 2.5, 1, 0,0);
            % ind_medPow_HM_chan3 = segment_signal_noise_multi_MedPow(data_wav(:,order(3)), 1.5, fs_new, 2.5, 1, 0,0);
            % ind_medPow_HM_chan4 = segment_signal_noise_multi_MedPow(data_wav(:,order(4)), 1.5, fs_new, 2.5, 1, 0,0);
            % ind_medPow_HM_chan5 = segment_signal_noise_multi_MedPow(data_wav(:,order(5)), 1.5, fs_new, 2.5, 1, 0,0);
            % ind_medPow_HM_chan6 = segment_signal_noise_multi_MedPow(data_wav(:,order(6)), 1.5, fs_new, 2.5, 1, 0,0);
            ind_medPow_HM_chan = segment_signal_noise_multi_MedPow(data_wav(:,order(ccc)), 1.5, fs_new, 2.5, 1, 0,0);
    
            ind_medPow_NM = segment_signal_noise_multi_MedPow(data_wav(:,order(4)+8),0.25,fs_new, 2.5, 1 , 1,0);

            %We do spike removal for each PCG channel
            for chan = 1:7
                col = order(chan);
                temp_chan = data_wav(:,col);
                temp_chan(fs_new-200:end-1000) = schmidt_spike_removal(temp_chan(fs_new-200:end-1000),fs_new);
                data_wav(:,col) = temp_chan;

            end
            
            %We channel-wise concatenate the data
            data_concat = [data_concat; data_wav];  
    
    
            %WE NEED TO ADD LENGTH OF PREVIOUS FILE TO INDICIES
            % bad_ind_HM_chan1 = ind_medPow_HM_chan1 + sum(len_array);
            % bad_ind_HM_chan2 = ind_medPow_HM_chan2 + sum(len_array);
            % bad_ind_HM_chan3 = ind_medPow_HM_chan3 + sum(len_array);
            % bad_ind_HM_chan4 = ind_medPow_HM_chan4 + sum(len_array);
            % bad_ind_HM_chan5 = ind_medPow_HM_chan5 + sum(len_array);
            % bad_ind_HM_chan6 = ind_medPow_HM_chan6 + sum(len_array);
            bad_ind_HM_chan = ind_medPow_HM_chan + sum(len_array);
            bad_ind_NM = ind_medPow_NM + sum(len_array); %as we are concatenating different .wav files together
    
            %update this for next files use
            len_array(f+1) = length(data_wav);
    
            %NOW WE JOIN ALL NOIST INDICIES IN ONE VARIABLE - OVERLAPS MAY EXIST - TO BE HANDLED AFTER FOR LOOP
            % bad_ind = [bad_ind;bad_ind_NM;bad_ind_HM3;bad_ind_HM4];    
            bad_ind = [bad_ind;bad_ind_NM;bad_ind_HM_chan];
    
            clearvars data_wav
        end
        %calculate start and end of file indicies here (count them as bad indices)
        chop = floor(1.5*fs_new)+1;
        x=1;
        %OUR FINAL OUTPUT NEEDS TO TELL US WHERE GOOD INDICIES ARE - NOT BAD ONES
        for i = 1:length(len_array)-1
            good_ind(x,1) = chop + sum(len_array(1:i)); x=x+1;
            good_ind(x,1) = sum(len_array(1:i+1)) - chop; x=x+1;
        end
        
        %FIRST WE HAVE TO MERGE THE BAD IND (could get same indicies from HM and BNM in same recording or across different channels
        bad_ind_s = sortrows(bad_ind);
        merged_bad_ind = [];
        current_range = bad_ind_s(1,:);
        
        for i = 2:size(bad_ind_s, 1)
            this_range = bad_ind_s(i,:);
            
            if this_range(1) <= current_range(2) + 1
                % Overlapping or adjacent ranges, merge them
                current_range(2) = max(current_range(2), this_range(2));
            else
                % No overlap, store the current range
                merged_bad_ind = [merged_bad_ind; current_range];
                current_range = this_range;
            end
        end
        
        % Don't forget to add the last range
        merged_bad_ind = [merged_bad_ind; current_range];
    
        %NOW WE HAVE TO CHANGE GOOD IND
        % Step 2: Subtract bad intervals from good intervals
        new_good_ind = [];
        
        for i = 1:2:length(good_ind)
            good_start = good_ind(i);
            good_end = good_ind(i+1);
            
            % Find bad intervals that overlap with this good interval
            overlapping = merged_bad_ind(merged_bad_ind(:,2) >= good_start & merged_bad_ind(:,1) <= good_end, :);
            
            if isempty(overlapping)
                % No bad parts, keep the whole good interval
                new_good_ind = [new_good_ind; good_start; good_end];
            else
                % Split the good interval around bad parts
                current_start = good_start;
                for j = 1:size(overlapping,1)
                    bad_start = max(overlapping(j,1), good_start); % Clip to good interval
                    bad_end = min(overlapping(j,2), good_end);
                    
                    if current_start < bad_start
                        new_good_ind = [new_good_ind; current_start; bad_start - 1];
                    end
                    current_start = bad_end + 1;
                end
                
                % Add any remaining good part after last bad interval
                if current_start <= good_end
                    new_good_ind = [new_good_ind; current_start; good_end];
                end
            end
        end
        
        % Make sure it's a vector in correct dimension for csv file
        new_good_ind = new_good_ind(:);
    
       
        ind_cell{s,1} = sub; %SUBJECT
        ind_cell{s,2} = new_good_ind; %GOOD INDICIES
        clearvars data_* len_array good_ind good_ind new* bad* merged*
    end
    
    %BELOW WRITES TO CSV FILE
    ind_table = cell2table(ind_cell);
    ind_table.Properties.VariableNames{'ind_cell1'} = 'subject';
    
    if save_ind_table == 1
        writetable(ind_table,['Ind_Table_chan',num2str(ccc),'_NM4.csv']);
    end
    
    clearvars -except ccc save_data save_ind_table

end