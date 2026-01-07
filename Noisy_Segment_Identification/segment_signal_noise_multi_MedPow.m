function ind_flag_m = segment_signal_noise_multi_MedPow(multi_signal, frame_len_sec, fs_new, threshold, total_sig,nm, fig)
    if fig~=0;figure(fig);clf;end
    for c = 1:total_sig
        signal = multi_signal(:,c);
        len_signal = length(signal);
        frame_len_sample = frame_len_sec * fs_new;
        no_frames = floor(len_signal / frame_len_sample);
        
        En = zeros(1, no_frames);
        
        for i = 1:no_frames
            sig = signal((i-1)*frame_len_sample+1 : i*frame_len_sample);
            En(i) = sum(sig.^2);
        end
        
        med_val = median(En(2:end-1));
        % med_val = mean(En(2:end-1));

        j = 1;
        ind_flag = [];
        
        for i = 1:length(En)
            if En(i) > threshold * med_val
                ind_flag(j,1) = (i-1) * frame_len_sample + 1;
                ind_flag(j,2) = i * frame_len_sample;
                j = j + 1;
            end
        end

        if nm==1
            k = find(En == max(En));
            ind_flag(j,1) = (k-1)*frame_len_sample;
            ind_flag(j,2) = k*frame_len_sample;
        end

        if fig ~= 0
            figure(fig);
            subplot(total_sig,1,c)
            t = 1:len_signal;
            signal=bandpass(signal,[20 500],fs_new);
            plot(t, signal, 'b');
            xlim([400 len_signal]);
            hold on;
            
            for i = 1:size(ind_flag, 1)
                idx_range = ind_flag(i, 1):ind_flag(i, 2);
                plot(t(idx_range), signal(idx_range), 'r', 'LineWidth', 2);
            end
            
            hold off;
        end
        if total_sig>1
            ind_flag_m{c} = ind_flag;
        else
            ind_flag_m = ind_flag;
        end
    end
end
