function [  ] = NTU_remove_noisy( reload )

    if reload==1
    
        % read the data
        load('dataset\NTU_Data.mat','data');

        % body joints
        spine = 1;
        neck = 3;
        left_knee = 14;
        right_knee = 18;
        left_foot = 16;
        right_foot = 20;
        
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % file parameters
            action = str2num(baseFileName(18:20));
            
            % extract the bodies data
            bodies = data(baseFileName);

            % remove the incomplete tracked skeletons
            mthresh = 0.8;
            hthresh = 1;
            ks = keys(bodies);
            for k = 1:length(ks)
                body = bodies(ks{k});
                
                % remove short time tracked bodies
                nans = sum(sum(isnan(body),3),2) > 0;
                missing = sum(nans) / size(body,1);
                if missing > mthresh
                    sprintf('missing: %s %g',baseFileName,missing)
                    remove(bodies,ks{k});
                    continue;
                end
                
                % remove bodies with unusual heigth
                heights = zeros([],1);
                for frame_idx = 1:size(body,1)
                    if ~any(any(isnan(body(frame_idx,:,:))))
                        h1 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,neck,:),[2 3 1]));
                        h2 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,left_knee,:),[2 3 1]));
                        h3 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,right_knee,:),[2 3 1]));
                        h4 = norm(permute(body(frame_idx,left_knee,:)-body(frame_idx,left_foot,:),[2 3 1]));
                        h5 = norm(permute(body(frame_idx,right_knee,:)-body(frame_idx,right_foot,:),[2 3 1]));
                        heights(end+1,1) = h1 + (h2+h3)/2 + (h4+h5)/2;
                    end
                end
                if ((action < 50 && length(ks) > 1) || (action >= 50 && length(ks) > 2)) && (median(heights) < hthresh)
                    sprintf('height: %s %g %g',baseFileName,length(ks),median(heights))
                    remove(bodies,ks{k});
                    continue;
                end
                
            end
            
            if ~isempty(bodies)
                data(baseFileName) = bodies;
            else
                remove(data,baseFileName);
            end
                
            if action == 1
                display(file);
            end
            
        end

        save('dataset\NTU_Denoised_1.mat','data','-v7.3');
   
    elseif reload==2
    
        % read the data
        load('dataset\NTU_Denoised_1.mat','data');

        % checked files
        BaseFileNames = {};

        % init order map object
        match_dist = containers.Map;
        match_id = containers.Map;

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the missing samples
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

            distances = {};
            for i = 1:length(BaseFileName)
                baseFileName1 = BaseFileName{i};
                bodies1 = data(baseFileName1);
                for j = i+1:length(BaseFileName)
                    baseFileName2 = BaseFileName{j};
                    bodies2 = data(baseFileName2);
                    distances{end+1} = NTU_check_skeleton_similarity(bodies1,bodies2); 
                end
            end
            
            % leave empty if there were just one camera
            % two cameras
            if length(distances)==1
                ks1 = keys(data(BaseFileName{1}));
                ks2 = keys(data(BaseFileName{2}));
                
                dist = containers.Map;
                id = containers.Map;
                [d,i] = min(distances{1},[],2);
                for k = 1:length(ks1)
                    dist(ks1{k}) = {d(k)};
                    id(ks1{k}) = {ks2{i(k)}};
                end
                match_dist(BaseFileName{1}) = dist;
                match_id(BaseFileName{1}) = id;

                dist = containers.Map;
                id = containers.Map;
                [d,i] = min(distances{1},[],1);
                for k = 1:length(ks2)
                    dist(ks2{k}) = {d(k)};
                    id(ks2{k}) = {ks1{i(k)}};
                end
                match_dist(BaseFileName{2}) = dist;
                match_id(BaseFileName{2}) = id;

            % three cameras
            elseif length(distances)==3
                ks1 = keys(data(BaseFileName{1}));
                ks2 = keys(data(BaseFileName{2}));
                ks3 = keys(data(BaseFileName{3}));
                
                dist = containers.Map;
                id = containers.Map;
                [d2,i2] = min(distances{1},[],2);
                [d3,i3] = min(distances{2},[],2);
                for k = 1:length(ks1)
                    dist(ks1{k}) = {d2(k),d3(k)};
                    id(ks1{k}) = {ks2{i2(k)},ks3{i3(k)}};
                end
                match_dist(BaseFileName{1}) = dist;
                match_id(BaseFileName{1}) = id;
                
                dist = containers.Map;
                id = containers.Map;
                [d1,i1] = min(distances{1},[],1);
                [d3,i3] = min(distances{3},[],2);
                for k = 1:length(ks2)
                    dist(ks2{k}) = {d1(k),d3(k)};
                    id(ks2{k}) = {ks1{i1(k)},ks3{i3(k)}};
                end
                match_dist(BaseFileName{2}) = dist;
                match_id(BaseFileName{2}) = id;
                
                dist = containers.Map;
                id = containers.Map;
                [d1,i1] = min(distances{2},[],1);
                [d2,i2] = min(distances{3},[],1);
                for k = 1:length(ks3)
                    dist(ks3{k}) = {d1(k),d2(k)};
                    id(ks3{k}) = {ks1{i1(k)},ks2{i2(k)}};
                end
                match_dist(BaseFileName{3}) = dist;
                match_id(BaseFileName{3}) = id;
                
            end

            if action==1
                display(length(BaseFileNames));
            end

        end
        
        save('dataset\NTU_Matching_1.mat','match_dist','match_id','-v7.3');
       
    elseif reload==3
    
        % read the data
        load('dataset\NTU_Denoised_1.mat','data');
        load('dataset\NTU_Matching_1.mat','match_dist','match_id');
        
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            if ~isKey(match_dist,baseFileName)
                continue;
            end
            
            % file parameters
            action = str2num(baseFileName(18:20));

            bodies = data(baseFileName);
            dist = match_dist(baseFileName);
            
            % threshold the distance
            thresh1_high = 10;
            thresh1_low = 5;
            thresh2 = 1.5;
            ks = keys(bodies);
            if action < 50 && length(ks) > 1
                d = zeros(length(ks),1);
                for k = 1:length(ks)
                    d(k,1) = min(cell2mat(dist(ks{k})));
                end
                [m,i] = sort(d);
                if (m(1) < thresh1_low && m(2) > thresh2*m(1)) || (m(2) > thresh1_high && m(2) > thresh2*m(1))
                    K = 1:length(ks);
                    K(i(1)) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g',baseFileName,m(1),m(2))
                end
            elseif action >= 50 && length(ks) > 2
                d = zeros(length(ks),1);
                for k = 1:length(ks)
                    d(k,1) = min(cell2mat(dist(ks{k})));
                end
                [m,i] = sort(d);
                if (m(2) < thresh1_low && m(3) > thresh2*m(2)) || (m(3) > thresh1_high && m(3) > thresh2*m(2))
                    K = 1:length(ks);
                    K([i(1),i(2)]) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g %g',baseFileName,m(1),m(2),m(3))
                end
            end
            
            if ~isempty(bodies)
                data(baseFileName) = bodies;
            else
                remove(data,baseFileName);
            end
            
        end
        
        save('dataset\NTU_Denoised_2.mat','data','-v7.3');

    elseif reload==4
    
        % read the data
        load('dataset\NTU_Denoised_2.mat','data');

        number_of_setups = 17;
        number_of_cameras = 3;
        number_of_dimensions = 3;
        rel_pos = cell(number_of_setups,1);
        for s = 1:number_of_setups
            rel_pos{s} = zeros(number_of_cameras,number_of_dimensions,[]);
        end

        hip_joint = 1;

        % checked files
        BaseFileNames = {};

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the proccessed samples
            if any(find(strcmp(BaseFileNames,baseFileName)))
                continue;
            end

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

            % file parameters
            setup = str2num(baseFileName(2:4));
            action = str2num(baseFileName(18:20));

            % just keep one subject actions
            if action >= 50
                continue;
            end

            % skip if more/less than three cameras
            if ~(length(BaseFileName) == 3)
                continue;
            end

            % skip if more/less than one sample per camera
            flag = false;
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                if ~(length(keys(data(baseFileName))) == 1)
                    flag = true;
                end
            end
            if flag == true
                continue;
            end

            % find min length
            min_frames = Inf;
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                bodies = data(baseFileName);
                ks = keys(bodies);
                body = bodies(ks{1});
                if size(body,1) < min_frames
                    min_frames = size(body,1);
                end
            end

            % add hip locations
            pos = zeros(number_of_cameras,number_of_dimensions,[]);
            for baseFileName_idx = 1:length(BaseFileName)
                baseFileName = BaseFileName{baseFileName_idx};
                camera = str2num(baseFileName(6:8));
                bodies = data(baseFileName);
                ks = keys(bodies);
                body = bodies(ks{1});
                pos(camera,:,1:min_frames) = permute(body(1:min_frames,hip_joint,:),[2 3 1]);
            end

            % remove missing frames
            remove_idx = [];
            for frame_idx = 1:min_frames
                if any(any(isnan(pos(:,:,frame_idx))))
                    remove_idx(end+1) = frame_idx;
                end
            end
            pos(:,:,remove_idx) = [];
    
            % find relative pos
            pos = cat(1,pos,pos(1,:,:));
            pos = diff(pos,1,1);
            
            % find relative median pos
            pos_med = zeros(number_of_cameras,number_of_dimensions);
            for camera_idx = 1:number_of_cameras
                for dimension_idx = 1:number_of_dimensions
                    pos_med(camera_idx,dimension_idx) = median(pos(camera_idx,dimension_idx,:));
                end
            end
            rel_pos{setup}(:,:,end+1) = pos_med;
            
            if action==1
                display(length(BaseFileNames));
            end

        end

        relative_med = zeros(number_of_setups,number_of_cameras,number_of_dimensions);
        relative_std = zeros(number_of_setups,number_of_cameras,number_of_dimensions);
        for setup_idx = 1:length(rel_pos)
            for camera_idx = 1:number_of_cameras
                for dimension_idx = 1:number_of_dimensions
                    relative_med(setup_idx,camera_idx,dimension_idx) = median(rel_pos{setup_idx}(camera_idx,dimension_idx,:));
                    relative_std(setup_idx,camera_idx,dimension_idx) = std(rel_pos{setup_idx}(camera_idx,dimension_idx,:));
                end
            end
        end

        save('dataset\NTU_Relative.mat','relative_med','relative_std','-v7.3');
        
    elseif reload==5
    
        % read the data
        load('dataset\NTU_Denoised_2.mat','data');
        load('dataset\NTU_Relative.mat','relative_med','relative_std');

        % checked files
        BaseFileNames = {};

        % init order map object
        match_dist = containers.Map;
        match_id = containers.Map;

        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            % ignore the missing samples
            if any(find(strcmp(BaseFileNames,baseFileName)))
                continue;
            end

            % file parameters
            setup = str2num(baseFileName(2:4));
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

            distances = {};
            for i = 1:length(BaseFileName)
                baseFileName1 = BaseFileName{i};
                camera1 = str2num(baseFileName1(6:8));
                bodies1 = data(baseFileName1);
                for j = i+1:length(BaseFileName)
                    baseFileName2 = BaseFileName{j};
                    camera2 = str2num(baseFileName2(6:8));
                    bodies2 = data(baseFileName2);
                    distances{end+1} = NTU_check_relative(bodies1,bodies2,relative_med,relative_std,setup,camera1,camera2); 
                end
            end
            
            % leave empty if there were just one camera
            % two cameras
            if length(distances)==1
                ks1 = keys(data(BaseFileName{1}));
                ks2 = keys(data(BaseFileName{2}));
                
                dist = containers.Map;
                id = containers.Map;
                [d,i] = min(distances{1},[],2);
                for k = 1:length(ks1)
                    dist(ks1{k}) = {d(k)};
                    id(ks1{k}) = {ks2{i(k)}};
                end
                match_dist(BaseFileName{1}) = dist;
                match_id(BaseFileName{1}) = id;

                dist = containers.Map;
                id = containers.Map;
                [d,i] = min(distances{1},[],1);
                for k = 1:length(ks2)
                    dist(ks2{k}) = {d(k)};
                    id(ks2{k}) = {ks1{i(k)}};
                end
                match_dist(BaseFileName{2}) = dist;
                match_id(BaseFileName{2}) = id;

            % three cameras
            elseif length(distances)==3
                ks1 = keys(data(BaseFileName{1}));
                ks2 = keys(data(BaseFileName{2}));
                ks3 = keys(data(BaseFileName{3}));
                
                dist = containers.Map;
                id = containers.Map;
                [d2,i2] = min(distances{1},[],2);
                [d3,i3] = min(distances{2},[],2);
                for k = 1:length(ks1)
                    dist(ks1{k}) = {d2(k),d3(k)};
                    id(ks1{k}) = {ks2{i2(k)},ks3{i3(k)}};
                end
                match_dist(BaseFileName{1}) = dist;
                match_id(BaseFileName{1}) = id;
                
                dist = containers.Map;
                id = containers.Map;
                [d1,i1] = min(distances{1},[],1);
                [d3,i3] = min(distances{3},[],2);
                for k = 1:length(ks2)
                    dist(ks2{k}) = {d1(k),d3(k)};
                    id(ks2{k}) = {ks1{i1(k)},ks3{i3(k)}};
                end
                match_dist(BaseFileName{2}) = dist;
                match_id(BaseFileName{2}) = id;
                
                dist = containers.Map;
                id = containers.Map;
                [d1,i1] = min(distances{2},[],1);
                [d2,i2] = min(distances{3},[],1);
                for k = 1:length(ks3)
                    dist(ks3{k}) = {d1(k),d2(k)};
                    id(ks3{k}) = {ks1{i1(k)},ks2{i2(k)}};
                end
                match_dist(BaseFileName{3}) = dist;
                match_id(BaseFileName{3}) = id;
                
            end

            if action==1
                display(length(BaseFileNames));
            end

        end
        
        save('dataset\NTU_Matching_2.mat','match_dist','match_id','-v7.3');

    elseif reload==6
    
        % read the data
        load('dataset\NTU_Denoised_2.mat','data');
        load('dataset\NTU_Matching_2.mat','match_dist','match_id');
        
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            if ~isKey(match_dist,baseFileName)
                continue;
            end
            
            % file parameters
            action = str2num(baseFileName(18:20));

            bodies = data(baseFileName);
            dist = match_dist(baseFileName);
            
            % threshold the distance
            thresh1_high = 8;
            thresh1_low = 3;
            thresh2 = 2;
            ks = keys(bodies);
            if action < 50 && length(ks) > 1
                d = zeros(length(ks),1);
                for k = 1:length(ks)
                    d(k,1) = min(cell2mat(dist(ks{k})));
                end
                [m,i] = sort(d);
                if (m(1) < thresh1_low && m(2) > thresh2*m(1)) || (m(2) > thresh1_high && m(2) > thresh2*m(1))
                    K = 1:length(ks);
                    K(i(1)) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g',baseFileName,m(1),m(2))
                end
            elseif action >= 50 && length(ks) > 2
                d = zeros(length(ks),1);
                for k = 1:length(ks)
                    d(k,1) = min(cell2mat(dist(ks{k})));
                end
                [m,i] = sort(d);
                if (m(2) < thresh1_low && m(3) > thresh2*m(2)) || (m(3) > thresh1_high && m(3) > thresh2*m(2))
                    K = 1:length(ks);
                    K([i(1),i(2)]) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g %g',baseFileName,m(1),m(2),m(3))
                end
            end
            
            if ~isempty(bodies)
                data(baseFileName) = bodies;
            else
                remove(data,baseFileName);
            end
            
        end
        
        save('dataset\NTU_Denoised_3.mat','data','-v7.3');
        
    elseif reload==7
    
        % read the data
        load('dataset\NTU_Denoised_3.mat','data');
        load('dataset\NTU_Matching_1.mat','match_dist','match_id');
        match_dist_1 = match_dist;
        load('dataset\NTU_Matching_2.mat','match_dist','match_id');
        match_dist_2 = match_dist;
        
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            if ~isKey(match_dist,baseFileName)
                continue;
            end
            
            % file parameters
            action = str2num(baseFileName(18:20));

            bodies = data(baseFileName);
            dist_1 = match_dist_1(baseFileName);
            dist_2 = match_dist_2(baseFileName);
            
            ks = keys(bodies);
            d = zeros(length(ks),1);
            for k = 1:length(ks)
                body = bodies(ks{k});
                nans = sum(sum(isnan(body),3),2) > 0;
                m = 1 - sum(nans) / size(body,1);
                d1 = min(cell2mat(dist_1(ks{k})));
                d2 = min(cell2mat(dist_2(ks{k})));
                d(k) = d1*d2/m;
            end
            
            % threshold the distance
            thresh = 1.5;
            if action < 50 && length(ks) > 1
                [mx,ix] = sort(d);
                if mx(2) > thresh*mx(1)
                    K = 1:length(ks);
                    K(ix(1)) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g',baseFileName,mx(1),mx(2))
                end
            elseif action >= 50 && length(ks) > 2
                [mx,ix] = sort(d);
                if mx(3) > thresh*mx(2)
                    K = 1:length(ks);
                    K([ix(1),ix(2)]) = [];
                    for k = K
                        remove(bodies,ks{k});
                    end
                    sprintf('matching: %s %g %g %g',baseFileName,mx(1),mx(2),mx(3))
                end
            end
            
            if ~isempty(bodies)
                data(baseFileName) = bodies;
            else
                remove(data,baseFileName);
            end
            
        end
        
        save('dataset\NTU_Denoised_4.mat','data','-v7.3');
        
    elseif reload==8
    
        % read the data
        load('dataset\NTU_Denoised_4.mat','data');
        load('dataset\NTU_Matching_1.mat','match_dist','match_id');
        match_dist_1 = match_dist;
        load('dataset\NTU_Matching_2.mat','match_dist','match_id');
        match_dist_2 = match_dist;
        
        fks = keys(data);
        for file = 1:length(fks)
            baseFileName = fks{file};

            if ~isKey(match_dist,baseFileName)
                continue;
            end
            
            % file parameters
            action = str2num(baseFileName(18:20));

            bodies = data(baseFileName);
            dist_1 = match_dist_1(baseFileName);
            dist_2 = match_dist_2(baseFileName);
            
            ks = keys(bodies);
            d = zeros(length(ks),1);
            for k = 1:length(ks)
                body = bodies(ks{k});
                nans = sum(sum(isnan(body),3),2) > 0;
                m = 1 - sum(nans) / size(body,1);
                d1 = min(cell2mat(dist_1(ks{k})));
                d2 = min(cell2mat(dist_2(ks{k})));
                d(k) = d1*d2/m;
            end
            
            thresh = 2;
            ks = keys(bodies);
            for k = 1:length(ks)
                if d(k) > thresh * min(d)
                    remove(bodies,ks{k});
                end
            end
            
            ks = keys(bodies);
            d = zeros(length(ks),1);
            for k = 1:length(ks)
                body = bodies(ks{k});
                nans = sum(sum(isnan(body),3),2) > 0;
                m = 1 - sum(nans) / size(body,1);
                d1 = min(cell2mat(dist_1(ks{k})));
                d2 = min(cell2mat(dist_2(ks{k})));
                d(k) = d1*d2/m;
            end
            
            % threshold the distance
            if action < 50 && length(ks) > 1
                [mx,ix] = sort(d);
                K = 1:length(ks);
                K(ix(1)) = [];
                for k = K
                    remove(bodies,ks{k});
                end
                sprintf('matching: %s %g %g',baseFileName,mx(1),mx(2))
            elseif action >= 50 && length(ks) > 2
                [mx,ix] = sort(d);
                K = 1:length(ks);
                K([ix(1),ix(2)]) = [];
                for k = K
                    remove(bodies,ks{k});
                end
                sprintf('matching: %s %g %g %g',baseFileName,mx(1),mx(2),mx(3))
            end
            
            if ~isempty(bodies)
                data(baseFileName) = bodies;
            else
                remove(data,baseFileName);
            end
            
        end
        
        save('dataset\NTU_Denoised_5.mat','data','-v7.3');
        
    end
    
end
