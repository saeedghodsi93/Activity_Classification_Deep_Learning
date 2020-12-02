function [  ] = load_NTU_dataset( path,base_joints,reload )
    
    if reload==1
        disp('Reloading Dataset...');
        
        NTU_read_data(path);
        
    elseif reload==2
        disp('Reloading Dataset...');
        
        NTU_remove_noisy(8);
    
    elseif reload==3
        disp('Reloading Dataset...');
        
        NTU_reconstruction(4);
    
    elseif reload==4
        disp('Reloading Dataset...');
        
        load('dataset\NTU_Selected_3.mat','data');
        
        % dataset parameters
        number_of_subjects = 40;
        number_of_actions = 60;
        number_of_joints=25;
        number_of_dimensions=3;
        scale_factor=100;
        
        for subject_idx = 1:number_of_subjects
            
            display(subject_idx);
            
            skeleton = zeros(number_of_actions,[],[],[],number_of_joints,number_of_dimensions);
            number_of_samples = zeros(number_of_actions,1);
            number_of_bodies = zeros(number_of_actions,[]);
            action_length = zeros(number_of_actions,[]);

            fks = keys(data);
            for file = 1:length(fks)
                baseFileName = fks{file};

                % file parameters
                person = str2num(baseFileName(10:12));
                action = str2num(baseFileName(18:20));

                if person ~= subject_idx
                    continue;
                end
                
                % extract the bodies data
                bodies = data(baseFileName);
                ks = keys(bodies);
                if (action < 50 && length(ks)~=1) || (action >= 50 && length(ks)~=2)
                    sprintf('%s %g',baseFileName,length(ks))
                    continue
                end

                action_idx = action;
                number_of_samples(action_idx) = number_of_samples(action_idx) + 1;
                number_of_bodies(action_idx,number_of_samples(action_idx)) = length(ks);
                for k = 1:length(ks)
                    body = bodies(ks{k});
                    
                    if sum(sum(sum(isnan(body)))) > 0
                        for j = 1:size(body,2)
                            for d = 1:size(body,3)
                                subsignal = body(:,j,d);
                                x = subsignal;
                                bd = isnan(subsignal);
                                gd = find(~bd);
                                bd([1:(min(gd)-1) (max(gd)+1):end]) = 0;
                                x(bd) = interp1(gd,subsignal(gd),find(bd));
                                x(1:(min(gd)-1)) = subsignal(min(gd));
                                x((max(gd)+1):end) = subsignal(max(gd));
                                body(:,j,d) = x;
                            end
                        end
                    end

                    action_length(action_idx,number_of_samples(action_idx)) = size(body,1);
                    skeleton(action_idx,number_of_samples(action_idx),k,1:size(body,1),1:size(body,2),1:size(body,3)) = body;
                    
                end
                
            end

            skeleton = skeleton .* scale_factor;
            
            skeleton = alignment(skeleton,number_of_samples,number_of_bodies,action_length,base_joints);
    
            save(sprintf('dataset\\NTU\\subject%g.mat',subject_idx),'skeleton','number_of_samples','number_of_bodies','action_length');

        end
    
    end
    
end
