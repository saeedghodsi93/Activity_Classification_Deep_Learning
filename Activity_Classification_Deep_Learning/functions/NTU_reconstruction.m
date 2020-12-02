function [  ] = NTU_reconstruction( reload )

    if reload==1
        
        % read the data
        load('dataset\NTU_Denoised_2.mat','data');

        number_of_actions = 60;
        number_of_joints = 25;

        % init order map object
        spikes = containers.Map;

        spikes_val = zeros(number_of_actions,[]);
        spikes_cnt = zeros(number_of_actions,1);
        spikes_med = zeros(number_of_actions,1);
        spikes_std = zeros(number_of_actions,1);
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % file parameters
            action = str2num(baseFileName(18:20));

            % extract the bodies data
            bodies = data(baseFileName);

            % calculate the spike percentage
            low_thresh = 0.05;
            high_thresh = 0.15;
            ks = keys(bodies);
            spike = containers.Map;
            for k = 1:length(ks)
                body = bodies(ks{k});
                counter = 0;
                dis = zeros([],1);
                for f = 1:size(body,1)-1
                    if ~any(any(isnan(body(f+1,:,:)))) && ~any(any(isnan(body(f,:,:))))
                        for j = 1:number_of_joints
                             dis(counter*number_of_joints+j,1) = norm(permute(body(f+1,j,:)-body(f,j,:),[3 1 2]));
                        end
                        counter = counter + 1;
                    end
                end
                small_spikes = sum(dis>low_thresh)/counter;
                large_spikes = sum(dis>high_thresh)/counter;
                spike(ks{k}) = small_spikes*large_spikes;
                spikes_cnt(action,1) = spikes_cnt(action,1) + 1;
                spikes_val(action,spikes_cnt(action,1)) = spike(ks{k});
            end
            spikes(baseFileName) = spike;

            if action==1
                display(file);
            end
            
        end

        for action = 1:number_of_actions
            spikes_med(action,1) = median(spikes_val(action,1:spikes_cnt(action,1)));
            spikes_std(action,1) = std(spikes_val(action,1:spikes_cnt(action,1)));
        end
        
        save('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std','-v7.3');
    
    elseif reload==2
        
        % read the data
        load('dataset\NTU_Denoised_2.mat','data');
        load('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std');

        % thresholds
        sthresh = 2;
            
        % checked files
        BaseFileNames = {};

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the proccessed samples
            if any(find(strcmp(BaseFileNames,baseFileName)))
                continue;
            end

            % file parameters
            action = str2num(baseFileName(18:20));

            % all views
            camera = [1 2 3];
            c = str2num(baseFileName(8));
            camera(c) = [];
            BaseFileName = {};
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(1));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(2));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end

            % remove incomplete samples
            remove_idx = [];
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                if ~isKey(spikes,baseFileName)
                    remove(data,baseFileName);
                    remove_idx(end+1) = baseFileName_idx;
                end
            end
            BaseFileName(remove_idx) = [];

            % threshold the samples
            remove_idx = [];
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                bodies = data(baseFileName);
%                 dist = match_dist(baseFileName);
                spike = spikes(baseFileName);

                ks = keys(bodies);
                for k = 1:length(ks)
                    if spike(ks{k}) > spikes_med(action,1)+sthresh
                        remove(bodies,ks{k});
                    end
                end

                if ~isempty(keys(bodies))
                    data(baseFileName) = bodies;
                else
                    remove(data,baseFileName);
                    remove_idx(end+1) = baseFileName_idx;
                end

            end
            BaseFileName(remove_idx) = [];

            % remove the insufficient samples
            remove_idx = [];
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                if (action < 50 && length(keys(data(baseFileName))) < 1) || (action >= 50 && length(keys(data(baseFileName))) < 2)
                    remove(data,baseFileName);
                    remove_idx(end+1) = baseFileName_idx;
                end
            end
            BaseFileName(remove_idx) = [];
            
        end
        
        save('dataset\NTU_Selected_1.mat','data','-v7.3');
    
    elseif reload==3
        
        % read the data
        load('dataset\NTU_Selected_1.mat','data');
        load('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std');

        % checked files
        BaseFileNames = {};

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the proccessed samples
            if any(find(strcmp(BaseFileNames,baseFileName)))
                continue;
            end

            % file parameters
            action = str2num(baseFileName(18:20));

            % all views
            camera = [1 2 3];
            c = str2num(baseFileName(8));
            camera(c) = [];
            BaseFileName = {};
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(1));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(2));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end

            min_spike = inf;
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                bodies = data(baseFileName);
                spike = spikes(baseFileName);

                ks = keys(bodies);
                s = zeros(length(ks),1);
                for k = 1:length(ks)
                    s(k,1) = spike(ks{k});
                end
                [mn,ix] = sort(s);

                if action < 50
                    if mn(1) < min_spike
                        min_spike = mn(1);
                        min_baseFileName = baseFileName;
                        min_body = {ks{ix(1)}};
                    end
                elseif action >= 50
                    if mn(1)+mn(2) < min_spike
                        min_spike = mn(1)+mn(2);
                        min_baseFileName = baseFileName;
                        min_body = {ks{ix(1)},ks{ix(2)}};
                    end
                end

            end

            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                if ~strcmp(baseFileName,min_baseFileName)
                    remove(data,baseFileName);
                end
            end

            bodies = data(min_baseFileName);
            ks = keys(bodies);
            for k = 1:length(ks)
                if ~any(ismember(min_body,ks{k}))
                    remove(bodies,ks{k});
                end
            end
            data(min_baseFileName) = bodies;

            if action==1
                display(length(BaseFileNames));
            end

        end    
        
        save('dataset\NTU_Selected_2.mat','data','-v7.3');
       
    elseif reload==4
        
        % read the data
        load('dataset\NTU_Selected_1.mat','data');

        % checked files
        BaseFileNames = {};

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the proccessed samples
            if any(find(strcmp(BaseFileNames,baseFileName)))
                continue;
            end

            % file parameters
            action = str2num(baseFileName(18:20));

            % all views
            camera = [1 2 3];
            c = str2num(baseFileName(8));
            camera(c) = [];
            BaseFileName = {};
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(1));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end
            baseFileName(8) = num2str(camera(2));
            if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
                BaseFileName{end+1} = baseFileName;
                BaseFileNames{end+1} = baseFileName;
            end

            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                bodies = data(baseFileName);

                if (action<50 && length(keys(bodies))~=1) || (action>=50 && length(keys(bodies))~=2)
                    remove(data,baseFileName);
                end

            end

            if action==1
                display(length(BaseFileNames));
            end

        end    
        
        save('dataset\NTU_Selected_3.mat','data','-v7.3');
       
    end
    
end
