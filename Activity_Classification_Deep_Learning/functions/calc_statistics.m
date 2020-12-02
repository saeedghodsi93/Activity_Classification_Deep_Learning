function [ enough_counter,extra_counter ] = calc_statistics( data )

    load('dataset\NTU_Matching_1.mat','match_dist','match_id');
    match_dist_1 = match_dist;
    load('dataset\NTU_Matching_2.mat','match_dist','match_id');
    match_dist_2 = match_dist;

    % checked files
    BaseFileNames = {};

    % body joints
    spine = 1;
    neck = 3;
    left_knee = 14;
    right_knee = 18;
    left_foot = 16;
    right_foot = 20;

    enough_counter = zeros(60,4);
    extra_counter = zeros(60,4);
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
        
        enough_flag = 1;
        for baseFileName_idx = 1:length(BaseFileName)
            baseFileName = BaseFileName{baseFileName_idx};
            
            bodies = data(baseFileName);
            ks = keys(bodies);
            if (action < 50 && length(ks) >= 1) || (action >= 50 && length(ks) >= 2)
                enough_flag = enough_flag + 1;
            end
            
        end
        enough_counter(action,enough_flag) = enough_counter(action,enough_flag) + 1;

        extra_flag = 1;
        for baseFileName_idx = 1:length(BaseFileName)
            baseFileName = BaseFileName{baseFileName_idx};
            
            bodies = data(baseFileName);
            ks = keys(bodies);
            if (action < 50 && length(ks) > 1) || (action >= 50 && length(ks) > 2)
                extra_flag = extra_flag + 1;
                bodies_match_dist_1 = match_dist_1(baseFileName);
                bodies_match_dist_2 = match_dist_2(baseFileName);
                for k = 1:length(ks)
                    body = bodies(ks{k});
                    nans = sum(sum(isnan(body),3),2) > 0;
                    missing = sum(nans) / size(body,1);
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
                    sprintf('%s %g %g %g %g %g',baseFileName,min(cell2mat(bodies_match_dist_1(ks{k}))),min(cell2mat(bodies_match_dist_2(ks{k}))),missing,median(heights),min(cell2mat(bodies_match_dist_1(ks{k})))*min(cell2mat(bodies_match_dist_2(ks{k})))/(1-missing))
                end
            end
            
        end
        extra_counter(action,extra_flag) = extra_counter(action,extra_flag) + 1;

    end
    
end

