
% reset
clc
% clear all
close all
    
% functions path
addpath(sprintf('%s',strcat(pwd,'\functions')));
    
%%
% load('dataset\NTU_Displacements.mat','displacements','counter');
    
% m = zeros(60,[]);
% d = zeros(60,1);
% for a = 1:60
%     m(a,1:size(displacements,3)) = mean(permute(displacements(a,1:counter(a,1),:),[2 3 1]));
%     for s = 1:counter(a,1)
%         d(a,1) = d(a,1) + norm(permute(displacements(a,s,:),[2 3 1])-m(a,:))/counter(a,1);
%     end
% end
% 
% c = zeros(60,60);
% for a = 1:60
%     for b = 1:60
%         c(a,b) = norm(m(b,:)-m(a,:));
%     end
% end

% features = zeros([],size(displacements,3));
% labels = zeros([],1);
% for action_idx = 1:size(displacements,1)
%     for sample_idx = 1:counter(action_idx,1)
%         features(size(features,1)+1,:) = permute(displacements(action_idx,sample_idx,:),[2 3 1]);
%         labels(size(labels,1)+1,1) = action_idx;
%     end
% end
% 
% idx = randperm(size(labels,1));
% features = features(idx,:);
% labels = labels(idx,:);
% train_num = round(0.8*size(labels,1));
% train_features = features(1:train_num,:);
% train_labels = labels(1:train_num,1);
% test_features = features(train_num+1:end,:);
% test_labels = labels(train_num+1:end,1);
% 
% options = statset('UseParallel',false);
% classifier=TreeBagger(100,train_features,train_labels,'OOBPrediction','Off','Method','classification','Options',options,'Prior','Uniform');
% predicted_labels=predict(classifier,test_features);
% 
% if(iscell(predicted_labels))
%     predicted_labels=cellfun(@str2num,predicted_labels);
% end
% 
% % update confusion matrix
% confusion_matrix = zeros(60,60);
% for i = 1:size(predicted_labels,1)
%     confusion_matrix(test_labels(i),predicted_labels(i))=confusion_matrix(test_labels(i),predicted_labels(i))+1;
% end
% 
% acc = zeros(60,1);
% for i = 1:60
%     acc(i,1) = confusion_matrix(i,i)/sum(confusion_matrix(i,:));
% end

%%
% number_of_joints=25;
% number_of_dimensions=3;
% 
% % extract the bodies data
% Setup = [2];
% Camera = [1,2,3];
% Person = [10];
% Repeat = [1];
% Action = [59];
% for setup = Setup
%     for camera = Camera
%         for person = Person
%             for repeat = Repeat
%                 for action = Action
%                     
%                     path='E:\Saeed\Programming\Dataset\NTU\Skeleton';
%                     baseFileName = sprintf('S%03gC%03gP%03gR%03gA%03g.skeleton',setup,camera,person,repeat,action)
%                     fullFileName = fullfile(path, baseFileName);
% %                     bodies = alignment(bodies);
% %                     draw_skeleton(bodies,10);
%                     
%                     % open the file and read
%                     fileid = fopen(fullFileName);
%                     framecount = fscanf(fileid,'%d',1); % no of the recorded frames
%                     bodyinfo=[]; % to store multiple skeletons per frame
%                     for f=1:framecount
%                         bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
%                         for b=1:bodycount
%                             clear body;
%                             body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
%                             arrayint = fscanf(fileid,'%d',6); % read 6 integers
%                             body.clipedEdges = arrayint(1);
%                             body.handLeftConfidence = arrayint(2);
%                             body.handLeftState = arrayint(3);
%                             body.handRightConfidence = arrayint(4);
%                             body.handRightState = arrayint(5);
%                             body.isResticted = arrayint(6);
%                             lean = fscanf(fileid,'%f',2);
%                             body.leanX = lean(1);
%                             body.leanY = lean(2);
%                             body.trackingState = fscanf(fileid,'%d',1);
% 
%                             body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
%                             for j=1:body.jointCount
%                                 jointinfo = fscanf(fileid,'%f',11);
%                                 joint=[];
% 
%                                 % 3D location of the joint j
%                                 joint.x = jointinfo(1);
%                                 joint.y = jointinfo(2);
%                                 joint.z = jointinfo(3);
% 
%                                 % 2D location of the joint j in corresponding depth/IR frame
%                                 joint.depthX = jointinfo(4);
%                                 joint.depthY = jointinfo(5);
% 
%                                 % 2D location of the joint j in corresponding RGB frame
%                                 joint.colorX = jointinfo(6);
%                                 joint.colorY = jointinfo(7);
% 
%                                 % The orientation of the joint j
%                                 joint.orientationW = jointinfo(8);
%                                 joint.orientationX = jointinfo(9);
%                                 joint.orientationY = jointinfo(10);
%                                 joint.orientationZ = jointinfo(11);
% 
%                                 % The tracking state of the joint j
%                                 joint.trackingState = fscanf(fileid,'%d',1);
% 
%                                 body.joints(j)=joint;
%                             end
%                             bodyinfo(f).bodies(b)=body;
%                         end
%                     end
%                     fclose(fileid);
% 
%                     % determine all the tracked subjects
%                     ids = {};
%                     for f = 1:length(bodyinfo)
%                         for s = 1:length(bodyinfo(f).bodies)
%                             id = int2str(bodyinfo(f).bodies(s).bodyID);
%                             if ~any(ismember(id,ids))
%                                 ids{length(ids)+1} = id;
%                             end
%                         end
%                     end
%                     
%                     % store the bodies in a map object
%                     bodies = containers.Map;
%                     for s = 1:length(ids)
%                         body = NaN(length(bodyinfo),number_of_joints,number_of_dimensions);
%                         bodies(ids{s}) = body;
%                     end
%                     for f = 1:length(bodyinfo)
%                         for s = 1:length(bodyinfo(f).bodies)
%                             id = int2str(bodyinfo(f).bodies(s).bodyID);
%                             skeleton = bodies(id);
%                             for j = 1:number_of_joints
%                                 skeleton(f,j,1) = bodyinfo(f).bodies(s).joints(j).x;
%                                 skeleton(f,j,2) = bodyinfo(f).bodies(s).joints(j).y;
%                                 skeleton(f,j,3) = bodyinfo(f).bodies(s).joints(j).z;
%                             end
%                             bodies(id) = skeleton;
%                         end
%                     end
% 
%                     % interpolate for the missing frame
%                     ks = keys(bodies);
%                     for k = 1:length(ks)
%                         body = bodies(ks{k});
%                         noisy = sum(sum(sum(isnan(body))))/(size(body,1)*size(body,2)*size(body,3));
%                         if noisy > 0
%                             for j = 1:number_of_joints
%                                 for d = 1:number_of_dimensions
%                                     subsignal = body(:,j,d);
%                                     x = subsignal;
%                                     bd = isnan(subsignal);
%                                     gd = find(~bd);
%                                     bd([1:(min(gd)-1) (max(gd)+1):end]) = 0;
%                                     x(bd) = interp1(gd,subsignal(gd),find(bd));
%                                     x(1:(min(gd)-1)) = subsignal(min(gd));
%                                     x((max(gd)+1):end) = subsignal(max(gd));
%                                     body(:,j,d) = x;
%                                 end
%                             end
%                             bodies(ks{k}) = body;
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end

%%
% % read the missing file names
% missing_samples = textread('dataset\NTU_missing_samples.txt','%s','delimiter','\n');
% 
% % init data map object
% data = containers.Map;
% 
% path='E:\Saeed\Programming\Dataset\NTU\Skeleton';
% files = dir(fullfile(path,'*.skeleton'));
% for file = 1:length(files)
%     baseFileName = files(file).name;
%     fullFileName = fullfile(path, baseFileName);
% 
%     % ignore the missing samples
%     if any(find(strcmp(missing_samples,baseFileName(1:20))))
%         continue;
%     end
%                     
%                     
%     % open the file and read
%     fileid = fopen(fullFileName);
%     framecount = fscanf(fileid,'%d',1); % no of the recorded frames
%     bodyinfo=[]; % to store multiple skeletons per frame
%     for f=1:framecount
%         bodycount = fscanf(fileid,'%d',1); % no of observerd skeletons in current frame
%         for b=1:bodycount
%             clear body;
%             body.bodyID = fscanf(fileid,'%ld',1); % tracking id of the skeleton
%             arrayint = fscanf(fileid,'%d',6); % read 6 integers
%             body.clipedEdges = arrayint(1);
%             body.handLeftConfidence = arrayint(2);
%             body.handLeftState = arrayint(3);
%             body.handRightConfidence = arrayint(4);
%             body.handRightState = arrayint(5);
%             body.isResticted = arrayint(6);
%             lean = fscanf(fileid,'%f',2);
%             body.leanX = lean(1);
%             body.leanY = lean(2);
%             body.trackingState = fscanf(fileid,'%d',1);
% 
%             body.jointCount = fscanf(fileid,'%d',1); % no of joints (25)
%             for j=1:body.jointCount
%                 jointinfo = fscanf(fileid,'%f',11);
%                 joint=[];
% 
%                 % 3D location of the joint j
%                 joint.x = jointinfo(1);
%                 joint.y = jointinfo(2);
%                 joint.z = jointinfo(3);
% 
%                 % 2D location of the joint j in corresponding depth/IR frame
%                 joint.depthX = jointinfo(4);
%                 joint.depthY = jointinfo(5);
% 
%                 % 2D location of the joint j in corresponding RGB frame
%                 joint.colorX = jointinfo(6);
%                 joint.colorY = jointinfo(7);
% 
%                 % The orientation of the joint j
%                 joint.orientationW = jointinfo(8);
%                 joint.orientationX = jointinfo(9);
%                 joint.orientationY = jointinfo(10);
%                 joint.orientationZ = jointinfo(11);
% 
%                 % The tracking state of the joint j
%                 joint.trackingState = fscanf(fileid,'%d',1);
% 
%                 body.joints(j)=joint;
%             end
%             bodyinfo(f).bodies(b)=body;
%         end
%     end
%     fclose(fileid);
% 
%     % determine all the tracked subjects
%     ids = {};
%     for f = 1:length(bodyinfo)
%         for s = 1:length(bodyinfo(f).bodies)
%             id = int2str(bodyinfo(f).bodies(s).bodyID);
%             if ~any(ismember(id,ids))
%                 ids{length(ids)+1} = id;
%             end
%         end
%     end
%         
%     % store the bodies in a map object
%     bodies = containers.Map;
%     for s = 1:length(ids)
%         body = NaN(length(bodyinfo),number_of_joints,number_of_dimensions);
%         bodies(ids{s}) = body;
%     end
%     for f = 1:length(bodyinfo)
%         for s = 1:length(bodyinfo(f).bodies)
%             id = int2str(bodyinfo(f).bodies(s).bodyID);
%             skeleton = bodies(id);
%             for j = 1:number_of_joints
%                 skeleton(f,j,1) = bodyinfo(f).bodies(s).joints(j).x;
%                 skeleton(f,j,2) = bodyinfo(f).bodies(s).joints(j).y;
%                 skeleton(f,j,3) = bodyinfo(f).bodies(s).joints(j).z;
%             end
%             bodies(id) = skeleton;
%         end
%     end
% 
%     % remove the incomplete tracked skeletons
%     thresh = 0;
%     ks = keys(bodies);
%     for k = 1:length(ks)
%         body = bodies(ks{k});
%         nans = sum(sum(isnan(body),3),2) > 0;
%         t = diff([false;nans==1;false]);
%         p = find(t==1);
%         q = find(t==-1);
%         [maxlen,ix] = max(q-p);
%         first = p(ix);
%         last = q(ix)-1;
%         miss = last-first + 1;
%         if isempty(miss)
%             miss = 0;
%         end
%         miss = miss / size(body,1);
%         missing = sum(nans) / size(body,1);
%         if missing > thresh
%             sprintf('%g %g %g %g %g',miss,missing,first,last,size(body,1))
% %             remove(bodies,ks{k});
%         end
%     end
%     
%     % interpolate for the missing frame
%     ks = keys(bodies);
%     noisy = zeros(1,length(ks));
%     for k = 1:length(ks)
%         body = bodies(ks{k});
%         noisy(1,k) = sum(sum(sum(isnan(body))))/(size(body,1)*size(body,2)*size(body,3));
%         if noisy(1,k) > 0
%             for j = 1:number_of_joints
%                 for d = 1:number_of_dimensions
%                     subsignal = body(:,j,d);
%                     x = subsignal;
%                     bd = isnan(subsignal);
%                     gd = find(~bd);
%                     bd([1:(min(gd)-1) (max(gd)+1):end]) = 0;
%                     x(bd) = interp1(gd,subsignal(gd),find(bd));
%                     x(1:(min(gd)-1)) = subsignal(min(gd));
%                     x((max(gd)+1):end) = subsignal(max(gd));
%                     body(:,j,d) = x;
%                 end
%             end
%             bodies(ks{k}) = body;
%         end
%     end
%     
% end

%%
% read the data
% load('dataset\NTU_Data.mat','data');
%             
% % extract the bodies data
% Setup = [2];
% Camera = [1,2,3];
% Person = [10];
% Repeat = [1];
% Action = [59];
% for setup = Setup
%     for camera = Camera
%         for person = Person
%             for repeat = Repeat
%                 for action = Action
%                     baseFileName = sprintf('S%03gC%03gP%03gR%03gA%03g.skeleton',setup,camera,person,repeat,action);
%                     bodies = data(baseFileName);
% %                     bodies = alignment(bodies);
% %                     draw_skeleton(bodies,10);
%                 end
%             end
%         end
%     end
% end

%%
% % read the data
% load('dataset\NTU_Denoised.mat','data');
% 
% counter = zeros(60,2);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName1 = fks{file};
%     baseFileName2 = baseFileName1;
%     baseFileName3 = baseFileName1;
%     
%     % other views
%     camera = [1 2 3];
%     c = str2num(baseFileName1(8));
%     camera(c) = [];
%     baseFileName2(8) = num2str(camera(1));
%     baseFileName3(8) = num2str(camera(2));
% %     sprintf('%s\t%s\t%s',baseFileName1,baseFileName2,baseFileName3)
%     
%     % file parameters
%     action = str2num(baseFileName1(18:20));
% 
%     % count valid samples
%     if isKey(data,baseFileName1)
%         bodies1 = data(baseFileName1);
%         ks = keys(bodies1);
%         if (action<50 && length(ks)==1) || (action>=50 && length(ks)==2)
%             counter(action,1) = counter(action,1) + 1;
%             continue;
%         end
%     end
%     if isKey(data,baseFileName2)
%         bodies2 = data(baseFileName2);
%         ks = keys(bodies2);
%         if (action<50 && length(ks)==1) || (action>=50 && length(ks)==2)
%             counter(action,1) = counter(action,1) + 1;
%             continue;
%         end
%     end
%     if isKey(data,baseFileName3)
%         bodies3 = data(baseFileName3);
%         ks = keys(bodies3);
%         if (action<50 && length(ks)==1) || (action>=50 && length(ks)==2)
%             counter(action,1) = counter(action,1) + 1;
%             continue;
%         end
%     end
%     counter(action,2) = counter(action,2) + 1;
%     
% %     display(baseFileName1);
% %     draw_skeleton(alignment(data(baseFileName1)),10);
% %     draw_skeleton(alignment(data(baseFileName2)),10);
% %     draw_skeleton(alignment(data(baseFileName3)),10);
%     
% end

%%
% % read the data
% load('dataset\NTU_Data.mat','data');
% 
% % body joints
% spine = 1;
% neck = 3;
% left_knee = 14;
% right_knee = 18;
% left_foot = 16;
% right_foot = 20;
% 
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
% 
%     % file parameters
%     action = str2num(baseFileName(18:20));
% 
%     % extract the bodies data
%     bodies = data(baseFileName);
% 
%     % remove the incomplete tracked skeletons
%     mthresh = 0.8;
%     hthresh = 1;
%     ks = keys(bodies);
%     for k = 1:length(ks)
%         body = bodies(ks{k});
% 
%         % remove short time tracked bodies
%         nans = sum(sum(isnan(body),3),2) > 0;
%         missing = sum(nans) / size(body,1);
%         if missing > mthresh
%             sprintf('missing: %s %g',baseFileName,missing)
%             remove(bodies,ks{k});
%             continue;
%         end
% 
%         % remove bodies with unusual heigth
%         heights = zeros([],1);
%         for frame_idx = 1:size(body,1)
%             if ~any(any(isnan(body(frame_idx,:,:))))
%                 h1 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,neck,:),[2 3 1]));
%                 h2 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,left_knee,:),[2 3 1]));
%                 h3 = norm(permute(body(frame_idx,spine,:)-body(frame_idx,right_knee,:),[2 3 1]));
%                 h4 = norm(permute(body(frame_idx,left_knee,:)-body(frame_idx,left_foot,:),[2 3 1]));
%                 h5 = norm(permute(body(frame_idx,right_knee,:)-body(frame_idx,right_foot,:),[2 3 1]));
%                 heights(end+1,1) = h1 + (h2+h3)/2 + (h4+h5)/2;
%             end
%         end
%         if median(heights) < hthresh
%             sprintf('height: %s %g',baseFileName,median(heights))
%             remove(bodies,ks{k});
%             continue;
%         end
% 
%     end
%     if ~isempty(bodies)
%         data(baseFileName) = bodies;
%     else
%         remove(data,baseFileName);
%     end
% 
%     if action == 1
%         display(file);
%     end
% 
% end

%%
% % read the data
% % load('dataset\NTU_Denoised_2.mat','data');
% % load('dataset\NTU_Matching_2.mat','match_dist','match_id');
% % load('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std');
% % load('dataset\NTU_Missings.mat','missings');
% 
% % checked files
% BaseFileNames = {};
% 
% counter = zeros(60,2);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
% 
%     % ignore the proccessed samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     % file parameters
%     action = str2num(baseFileName(18:20));
% 
%     % other views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     BaseFileName = {};
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
% 
%     dthresh = 4;
%     sthresh = 2;
%     mthresh = 0.01;
%     flag = 2;
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         if ~isKey(match_dist,baseFileName) || ~isKey(match_id,baseFileName) || ~isKey(spikes,baseFileName) || ~isKey(missings,baseFileName)
%             continue;
%         end
%         bodies = data(baseFileName);
%         dist = match_dist(baseFileName);
%         id = match_id(baseFileName);
%         spike = spikes(baseFileName);
%         missing = missings(baseFileName);
%         
%         ks = keys(bodies);
%         if action < 50 && length(ks) >= 1
%             d = zeros(length(ks),1);
%             s = zeros(length(ks),1);
%             m = zeros(length(ks),1);
%             for k = 1:length(ks)
%                 d(k,1) = min(cell2mat(dist(ks{k})));
%                 s(k,1) = spike(ks{k});
%                 m(k,1) = missing(ks{k});
%             end
%             [mn,ix] = sort(d);
%             if mn(1) < dthresh && s(ix(1)) < spikes_med(action,1)+sthresh && m(ix(1)) < mthresh
%             	flag = 1;
%             else
%                 sprintf('%g %g %g',mn(1),s(ix(1)),m(ix(1)))
%             end
%         elseif action >= 50 && length(ks) >= 2
%             d = zeros(length(ks),1);
%             s = zeros(length(ks),1);
%             m = zeros(length(ks),1);
%             for k = 1:length(ks)
%                 d(k,1) = min(cell2mat(dist(ks{k})));
%                 s(k,1) = spike(ks{k});
%                 m(k,1) = missing(ks{k});
%             end
%             [mn,ix] = sort(d);
%             if mn(2) < dthresh  && s(ix(1)) < spikes_med(action,1)+sthresh && s(ix(2)) < spikes_med(action,1)+sthresh && m(ix(1)) < mthresh && m(ix(2)) < mthresh
%                 flag = 1;
%             else
%                 sprintf('%g %g %g %g %g %g',mn(1),mn(2),s(ix(1)),s(ix(2)),m(ix(1)),m(ix(2)))
%             end
%         else
%             
%         end
%     end
%     counter(action,flag) = counter(action,flag) + 1;
%     
% %     % count valid samples
% %     bodies = data(baseFileName);
% %     ks = keys(bodies);
% %     if (action<50 && length(ks)==1) || (action>=50 && length(ks)==2)
% %         counter(action,1) = counter(action,1) + 1;
% %     else
% %         counter(action,2) = counter(action,2) + 1;
% %     end
% 
% end

%%
% % read the data
% % load('dataset\NTU_Denoised_2.mat','data');
% % load('dataset\NTU_Matching_2.mat','match_dist','match_id');
% % load('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std');
% % load('dataset\NTU_Missings.mat','missings');
% 
% % thresholds
% dthresh = 4;
% sthresh = 2;
% mthresh = 0.01;
% 
% % checked files
% BaseFileNames = {};
% 
% counter = zeros(60,1);
% n_samples = zeros(60,[]);
% valid_samples = zeros(60,1);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
% 
%     % ignore the proccessed samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     % file parameters
%     action = str2num(baseFileName(18:20));
% 
%     % all views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     BaseFileName = {};
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
% 
%     % remove incomplete samples
%     remove_idx = [];
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         if ~isKey(match_dist,baseFileName) || ~isKey(spikes,baseFileName) || ~isKey(missings,baseFileName)
%             remove(data,baseFileName);
%             remove_idx(end+1) = baseFileName_idx;
%         end
%     end
%     BaseFileName(remove_idx) = [];
% 
%     % threshold the samples
%     remove_idx = [];
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         bodies = data(baseFileName);
%         dist = match_dist(baseFileName);
%         spike = spikes(baseFileName);
%         missing = missings(baseFileName);
% 
%         ks = keys(bodies);
%         for k = 1:length(ks)
%             if min(cell2mat(dist(ks{k}))) > dthresh || spike(ks{k}) > spikes_med(action,1)+sthresh || missing(ks{k}) > mthresh
%                 remove(bodies,ks{k});
%             end
%         end
% 
%         if ~isempty(keys(bodies))
%             data(baseFileName) = bodies;
%         else
%             remove(data,baseFileName);
%             remove_idx(end+1) = baseFileName_idx;
%         end
% 
%     end
%     BaseFileName(remove_idx) = [];
%     
%     % remove the insufficient samples
%     remove_idx = [];
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         if (action < 50 && length(keys(data(baseFileName))) < 1) || (action >= 50 && length(keys(data(baseFileName))) < 2)
%             remove(data,baseFileName);
%             remove_idx(end+1) = baseFileName_idx;
%         end
%     end
%     BaseFileName(remove_idx) = [];
% 
%     flag = 1;
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         if (action < 50 && length(keys(data(baseFileName))) >= 1) || (action >= 50 && length(keys(data(baseFileName))) >= 2)
%             flag = 2;
%         end
%         counter(action) = counter(action) + 1;
%         n_samples(action,counter(action)) = length(keys(data(baseFileName)));
%     end
%     
%     if flag==2
%         valid_samples(action) = valid_samples(action) + 1;
%     end
%     
%     if action==1
%         display(sum(counter));
%     end
%     
% end

%%
% % read the data
% % load('dataset\NTU_Selected_1.mat','data');
% % load('dataset\NTU_Matching_2.mat','match_dist','match_id');
% % load('dataset\NTU_Spikes.mat','spikes','spikes_med','spikes_std');
% % load('dataset\NTU_Missings.mat','missings');
% 
% % checked files
% BaseFileNames = {};
% 
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
% 
%     % ignore the proccessed samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     % file parameters
%     action = str2num(baseFileName(18:20));
% 
%     % all views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     BaseFileName = {};
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     
%     min_spike = inf;
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         bodies = data(baseFileName);
%         dist = match_dist(baseFileName);
%         id = match_id(baseFileName);
%         spike = spikes(baseFileName);
%         missing = missings(baseFileName);
%         
%         ks = keys(bodies);
%         if action < 50
%             d = zeros(length(ks),1);
%             s = zeros(length(ks),1);
%             m = zeros(length(ks),1);
%             for k = 1:length(ks)
%                 d(k,1) = min(cell2mat(dist(ks{k})));
%                 s(k,1) = spike(ks{k});
%                 m(k,1) = missing(ks{k});
%             end
%             [mn,ix] = sort(d);
%             if mn(1) < dthresh && s(ix(1)) < spikes_med(action,1)+sthresh && m(ix(1)) < mthresh
%             	flag = 1;
%             else
%                 sprintf('%g %g %g',mn(1),s(ix(1)),m(ix(1)))
%             end
%         elseif action >= 50
%             d = zeros(length(ks),1);
%             s = zeros(length(ks),1);
%             m = zeros(length(ks),1);
%             for k = 1:length(ks)
%                 d(k,1) = min(cell2mat(dist(ks{k})));
%                 s(k,1) = spike(ks{k});
%                 m(k,1) = missing(ks{k});
%             end
%             [mn,ix] = sort(d);
%             if mn(2) < dthresh  && s(ix(1)) < spikes_med(action,1)+sthresh && s(ix(2)) < spikes_med(action,1)+sthresh && m(ix(1)) < mthresh && m(ix(2)) < mthresh
%                 flag = 1;
%             else
%                 sprintf('%g %g %g %g %g %g',mn(1),mn(2),s(ix(1)),s(ix(2)),m(ix(1)),m(ix(2)))
%             end
%         else
%             
%         end
%     end
%     
% end

%%
% % read the data
% % load('dataset\NTU_Denoised.mat','data');
% 
% % checked files
% BaseFileNames = {};
% 
% % init order map object
% match_dist = containers.Map;
% match_idx = containers.Map;
% 
% counter = zeros(60,2);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     
%     % ignore the missing samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     BaseFileName = {baseFileName};
%     BaseFileNames{end+1} = baseFileName;
%     
%     % file parameters
%     action = str2num(baseFileName(18:20));
%     
%     % other views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName)
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName)
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     
%     distances = {};
%     for i = 1:length(BaseFileName)
%         baseFileName1 = BaseFileName{i};
%         bodies1 = data(baseFileName1);
%         for j = i+1:length(BaseFileName)
%             baseFileName2 = BaseFileName{j};
%             bodies2 = data(baseFileName2);
%             distances{end+1} = check_skeleton_similarity(bodies1,bodies2); 
%         end
%     end
%         
%     % two cameras
%     if length(distances)==1
%         [m,i] = min(distances{1},[],2);
%         match_dist(BaseFileName{1}) = m';
%         match_idx(BaseFileName{1}) = i';
%         
%         [m,i] = min(distances{1},[],1);
%         match_dist(BaseFileName{2}) = m;
%         match_idx(BaseFileName{2}) = i;
%         
% %         display(min(distances{1},[],2)');
% %         display(min(distances{1},[],1));
%         
%     % three cameras
%     elseif length(distances)==3
%         [m1,i1] = min(distances{1},[],2);
%         [m2,i2] = min(distances{2},[],2);
%         match_dist(BaseFileName{1}) = [m1';m2'];
%         match_idx(BaseFileName{1}) = [i1';i2'];
%         
%         [m1,i1] = min(distances{1},[],1);
%         [m2,i2] = min(distances{3},[],2);
%         match_dist(BaseFileName{2}) = [m1;m2'];
%         match_idx(BaseFileName{2}) = [i1;i2'];
%         
%         [m1,i1] = min(distances{2},[],1);
%         [m2,i2] = min(distances{3},[],1);
%         match_dist(BaseFileName{3}) = [m1;m2];
%         match_idx(BaseFileName{3}) = [i1;i2];
%         
% %         display([min(distances{1},[],2)';min(distances{2},[],2)']);
% %         display([min(distances{1},[],1);min(distances{3},[],2)']);
% %         display([min(distances{2},[],1);min(distances{3},[],1)]);
%         
%     end
%     
%     if action==1
%         display(length(BaseFileNames));
%     end
%     
% end

%%
% read the data
% % load('dataset\NTU_Denoised_1.mat','data');
% % load('dataset\NTU_Matching_1.mat','match_dist','match_idx');
% 
% % checked files
% BaseFileNames = {};
% 
% counter = zeros(1,6);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     
%     if ~isKey(match_dist,baseFileName)
%         continue;
%     end
%     
%     % ignore the missing samples
% %     if any(find(strcmp(BaseFileNames,baseFileName)))
% %         continue;
% %     end
% 
%     BaseFileName = {baseFileName};
%     BaseFileNames{end+1} = baseFileName;
%     
%     % file parameters
%     action = str2num(baseFileName(18:20));
%     
%     % other views
% %     camera = [1 2 3];
% %     c = str2num(baseFileName(8));
% %     camera(c) = [];
% %     baseFileName(8) = num2str(camera(1));
% %     if isKey(data,baseFileName)
% %         BaseFileName{end+1} = baseFileName;
% %         BaseFileNames{end+1} = baseFileName;
% %     end
% %     baseFileName(8) = num2str(camera(2));
% %     if isKey(data,baseFileName)
% %         BaseFileName{end+1} = baseFileName;
% %         BaseFileNames{end+1} = baseFileName;
% %     end
% 
%     bodies = data(baseFileName);
%     dist = match_dist(baseFileName);
%     idx = match_idx(baseFileName);
%     
%     ks = keys(bodies);
%     if action < 50 && length(ks) > 1
%         [m,i] = sort(min(dist,[],1));
%         if m(1) < 5
%             counter(1,1) = counter(1,1) + 1;
% %             sprintf('%s %g %g',baseFileName,m(1),m(2))
%         elseif m(1) < 7.5
%             counter(1,2) = counter(1,2) + 1;
%         else
%             counter(1,3) = counter(1,3) + 1;
%         end
%     elseif action >= 50 && length(ks) > 2
%         [m,i] = sort(min(dist,[],1));
%         if m(2) < 5
%             counter(1,4) = counter(1,4) + 1;
%         elseif m(2) < 7.5
%             counter(1,5) = counter(1,5) + 1;
%         else
%             counter(1,6) = counter(1,6) + 1;
%         end
%     end
%     
% end

%%
% % read the data
% % load('dataset\NTU_Denoised_2.mat','data');
% 
% number_of_joints = 25;
% 
% % checked files
% BaseFileNames = {};
% 
% spikes = zeros([],1);
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     
%     % file parameters
%     action = str2num(baseFileName(18:20));
%     
%     % ignore the missing samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     % other views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     BaseFileName = {};
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
% 
%     for i = 1:length(BaseFileName)
%         baseFileName = BaseFileName{i};
%         
%         % count spikes
%         bodies = data(baseFileName);
% 
%         % calculate the subsignal spikes percentage
%         low_thresh = 0.05;
%         high_thresh = 0.15;
%         ks = keys(bodies);
%         for k = 1:length(ks)
%             body = bodies(ks{k});
%             counter = 0;
%             dis = zeros([],1);
%             for f = 1:size(body,1)-1
%                 if ~any(any(isnan(body(f+1,:,:)))) && ~any(any(isnan(body(f,:,:))))
%                     counter = counter + 1;
%                     for j = 1:number_of_joints
%                          dis(end+1,1) = norm(permute(body(f+1,j,:)-body(f,j,:),[3 1 2]));
%                     end
%                 end
%             end
%             small_spikes = sum(dis>low_thresh)/counter;
%             large_spikes = sum(dis>high_thresh)/counter;
%             spikes(end+1,1) = small_spikes*large_spikes;
%         end
%         
%     end
%     
% end

%%
% read the data
% load('dataset\NTU_Denoised_2.mat','data');
% 
% number_of_setups = 17;
% number_of_cameras = 3;
% number_of_dimensions = 3;
% rel_pos = cell(number_of_setups,1);
% for s = 1:number_of_setups
%     rel_pos{s} = zeros(number_of_cameras-1,number_of_dimensions,[]);
% end
% 
% hip_joint = 1;
% 
% % checked files
% BaseFileNames = {};
% 
% fks = keys(data);
% for file = 1:length(fks)
%     baseFileName = fks{file};
% 
%     % ignore the proccessed samples
%     if any(find(strcmp(BaseFileNames,baseFileName)))
%         continue;
%     end
% 
%     % all views
%     camera = [1 2 3];
%     c = str2num(baseFileName(8));
%     camera(c) = [];
%     BaseFileName = {};
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(1));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
%     baseFileName(8) = num2str(camera(2));
%     if isKey(data,baseFileName) && ~isempty(keys(data(baseFileName)))
%         BaseFileName{end+1} = baseFileName;
%         BaseFileNames{end+1} = baseFileName;
%     end
% 
%     % file parameters
%     setup = str2num(baseFileName(2:4));
%     action = str2num(baseFileName(18:20));
% 
%     if action >= 50
%         continue;
%     end
% 
%     % skip if more/less than three cameras
%     if ~(length(BaseFileName) == 3)
%         continue;
%     end
% 
%     % skip if more/less than one sample per camera
%     flag = false;
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         if ~(length(keys(data(baseFileName))) == 1)
%             flag = true;
%         end
%     end
%     if flag == true
%         continue;
%     end
% 
%     % find min length
%     min_frames = Inf;
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         bodies = data(baseFileName);
%         ks = keys(bodies);
%         body = bodies(ks{1});
%         if size(body,1) < min_frames
%             min_frames = size(body,1);
%         end
%     end
% 
%     % add hip locations
%     pos = zeros(number_of_cameras,number_of_dimensions,[]);
%     for baseFileName_idx = 1:length(BaseFileName)
%         baseFileName = BaseFileName{baseFileName_idx};
%         camera = str2num(baseFileName(6:8));
%         bodies = data(baseFileName);
%         ks = keys(bodies);
%         body = bodies(ks{1});
%         pos(camera,:,1:min_frames) = permute(body(1:min_frames,hip_joint,:),[2 3 1]);
%     end
% 
%     % remove missing frames
%     remove_idx = [];
%     for frame_idx = 1:min_frames
%         if any(any(isnan(pos(:,:,frame_idx))))
%             remove_idx(end+1) = frame_idx;
%         end
%     end
%     pos(:,:,remove_idx) = [];
%     
%     % find median pos
%     pos_med = zeros(number_of_cameras,number_of_dimensions);
%     for camera_idx = 1:number_of_cameras
%         for dimension_idx = 1:number_of_dimensions
%             pos_med(camera_idx,dimension_idx) = median(pos(camera_idx,dimension_idx,:));
%         end
%     end
%     
%     % add relative median pos
%     rel_pos{setup}(:,:,end+1) = diff(pos_med);
%     
%     if action==1
%         display(length(BaseFileNames));
%     end
%     
% end

% for setup_idx = 1:length(rel_pos)
%     figure;
%     hold on;
%     for camera_idx = 2:number_of_cameras-1
%         for dimension_idx = 1:number_of_dimensions
%             histogram(rel_pos{setup_idx}(camera_idx,dimension_idx,:));
% %             median(rel_pos{setup_idx}(camera_idx,dimension_idx,:))
% %             mean(rel_pos{setup_idx}(camera_idx,dimension_idx,:))
% %             std(rel_pos{setup_idx}(camera_idx,dimension_idx,:))
%         end
%     end
%     hold off;
% end
% 
% for setup_idx = 1:length(rel_pos)
%     figure;
%     hold on;
%     for camera_idx = 2:number_of_cameras-1
%         distance = zeros(1,size(rel_pos{setup_idx},3));
%         for sample_idx = 1:size(rel_pos{setup_idx},3)
%             distance(1,sample_idx) = norm(rel_pos{setup_idx}(camera_idx,:,sample_idx));
%         end
%         histogram(distance);
%     end
%     hold off;
% end

%%
% % read the data
% load('dataset\NTU_Matching_2.mat','match_dist','match_id');
% values_2 = zeros(1,[]);
% fks = keys(match_dist);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     dist = match_dist(baseFileName);
%     ks = keys(dist);
%     for k = 1:length(ks)
%         values_2(1,end+1) = min(cell2mat(dist(ks{k})));
%     end 
% end
% 
% load('dataset\NTU_Matching_3.mat','match_dist','match_id');
% values_3 = zeros(1,[]);
% fks = keys(match_dist);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     dist = match_dist(baseFileName);
%     ks = keys(dist);
%     for k = 1:length(ks)
%         values_3(1,end+1) = min(cell2mat(dist(ks{k})));
%     end    
% end
% 
% load('dataset\NTU_Lengths.mat','samples_length');
% values_3 = zeros(1,[]);
% fks = keys(samples_length);
% for file = 1:length(fks)
%     baseFileName = fks{file};
%     bodies_length = samples_length(baseFileName);
%     ks = keys(bodies_length);
%     for k = 1:length(ks)
%         values_3(1,end+1) = bodies_length(ks{k});
%     end 
% end

%%
% read the data
% load('dataset\NTU_Data.mat','data');
% [Data_enough_counter,Data_extra_counter] = calc_statistics(data);
% Data_counter = [Data_enough_counter,Data_extra_counter];
% clear data;
% 
% load('dataset\NTU_Denoised_1.mat','data');
% [Denoised_1_enough_counter,Denoised_1_extra_counter] = calc_statistics(data);
% Denoised_1_counter = [Denoised_1_enough_counter,Denoised_1_extra_counter];
% clear data;
% 
% load('dataset\NTU_Denoised_2.mat','data');
% [Denoised_2_enough_counter,Denoised_2_extra_counter] = calc_statistics(data);
% Denoised_2_counter = [Denoised_2_enough_counter,Denoised_2_extra_counter];
% clear data;
% 
% load('dataset\NTU_Denoised_3.mat','data');
% [Denoised_3_enough_counter,Denoised_3_extra_counter] = calc_statistics(data);
% Denoised_3_counter = [Denoised_3_enough_counter,Denoised_3_extra_counter];
% clear data;
% 
% load('dataset\NTU_Denoised_4.mat','data');
% [Denoised_4_enough_counter,Denoised_4_extra_counter] = calc_statistics(data);
% Denoised_4_counter = [Denoised_4_enough_counter,Denoised_4_extra_counter];
% clear data;
%
% 
% load('dataset\NTU_Denoised_5.mat','data');
% [Denoised_5_enough_counter,Denoised_5_extra_counter] = calc_statistics(data);
% Denoised_5_counter = [Denoised_5_enough_counter,Denoised_5_extra_counter];
% clear data;
% 
% load('dataset\NTU_Selected_1.mat','data');
% [Selected_1_enough_counter,Selected_1_extra_counter] = calc_statistics(data);
% Selected_1_counter = [Selected_1_enough_counter,Selected_1_extra_counter];
% clear data;
% 
% load('dataset\NTU_Selected_2.mat','data');
% [Selected_2_enough_counter,Selected_2_extra_counter] = calc_statistics(data);
% Selected_2_counter = [Selected_2_enough_counter,Selected_2_extra_counter];
% clear data;

load('dataset\NTU_Selected_3.mat','data');
% [Selected_3_enough_counter,Selected_3_extra_counter] = calc_statistics(data);
% Selected_3_counter = [Selected_3_enough_counter,Selected_3_extra_counter];

%%
% load('dataset\NTU_Selected_1.mat','data');
% 
% display(length(keys(data)));
