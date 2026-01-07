%This function will join all Dia1, Dia2 and Dia3 recordinsg together
%Then it will do spike removal on each PCG Channel
%Then it will save concatenetd signals channel-wise

clear
save_data=1;

outdir = '/home/sparc/Desktop/TH_data_joined_MATLAB_SpRe/';
mkdir(outdir)
addpath('/home/sparc/Desktop/TH_alldata')
fs_new = 2000;           % Sampling frequency (Hz)

%get a list of all filenames and exclude Bell, Lie, 10s
folder_path = '/home/sparc/Desktop/TH_alldata'; 
files = dir(folder_path); % Get list of all files and folders
file_names = {files.name};

order_1 = [1,2,4,5,6,7,8,11];
order_23= [2,4,5,6,8,7,9,11];

filtered_files = file_names;

data = readtable('REFERENCE_ALLROUNDS_ExclusionCriteria.csv');
subs = table2cell(data(:,1));
for s = 1:length(subs)
    %extract all of the files from this subject
    sub = subs{s};
    Sub_files = filtered_files(contains(filtered_files, sub))

    data_concat = [];
    for f = 1:length(Sub_files)
        file = Sub_files{f};
        if strcmp(file(2),'c') ==1 || strcmp(file(2),'v') == 1
            order = order_23;
        else
            order = order_1;
        end

        [data_wav1,fs] = audioread(file);
        data_wav = resample(data_wav1,fs_new,fs);

        %spike removal for each PCG channel - standrd preprcoessing
        %technique
        for chan = 1:7
            col = order(chan);
            temp_chan = data_wav(:,col);
            temp_chan(fs_new-200:end-1000) = schmidt_spike_removal(temp_chan(fs_new-200:end-1000),fs_new);
            data_wav(:,col) = temp_chan;

        end

        data_concat = [data_concat; data_wav];  

        clearvars data_wav
    end


    fname_save = [outdir,sub,'_Sit'];
    if save_data==1
        audiowrite([fname_save,'.wav'],data_concat,fs_new, 'BitsPerSample',32); % ONLY UNCOMMENT IF WAV FILES CHANGE
    end

    clearvars data_*
end
    

