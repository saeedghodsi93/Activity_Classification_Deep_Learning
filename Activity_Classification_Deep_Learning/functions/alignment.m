function [ aligned_skeleton ] = alignment( skeleton,number_of_samples,number_of_bodies,action_length,base_joints )
    
    for action_idx = 1:size(skeleton,1)
        for test_idx = 1:number_of_samples(action_idx)
            displacement = zeros(number_of_bodies(action_idx,test_idx),action_length(action_idx,test_idx)-1);
            for body_idx = 1:number_of_bodies(action_idx,test_idx)
                for frame_idx = 1:action_length(action_idx,test_idx)-1
                    if ~any(any(isnan(skeleton(action_idx,test_idx,body_idx,frame_idx+1,:,:)))) && ~any(any(isnan(skeleton(action_idx,test_idx,body_idx,frame_idx,:,:))))
                        for joint_idx = 1:size(skeleton,5)
                            displacement(body_idx,frame_idx) = displacement(body_idx,frame_idx) + norm(permute(skeleton(action_idx,test_idx,body_idx,frame_idx+1,joint_idx,:)-skeleton(action_idx,test_idx,body_idx,frame_idx,joint_idx,:),[6 1 2 3 4 5]));
                        end
                    end
                end
            end
            
            per = 0.5;
            total_displacement = zeros(number_of_bodies(action_idx,test_idx),1);
            for body_idx = 1:number_of_bodies(action_idx,test_idx)
                total_displacement(body_idx) = sum(displacement(body_idx,1:round(per*size(displacement,1))));
            end
            [~,order] = sort(total_displacement,'descend');
            bodies = skeleton(action_idx,test_idx,1:number_of_bodies(action_idx,test_idx),:,:,:);
            skeleton(action_idx,test_idx,1:number_of_bodies(action_idx,test_idx),:,:,:) = bodies(:,:,order,:,:,:);
            
        end
    end
    
    aligned_skeleton = zeros(size(skeleton,1),size(skeleton,2),size(skeleton,3),size(skeleton,4),size(skeleton,5)+1,size(skeleton,6));
    for action_idx = 1:size(skeleton,1)
        for test_idx = 1:number_of_samples(action_idx)
            for body_idx = 1:number_of_bodies(action_idx,test_idx)
                base_joint_center = base_joints(1);
                base_joint_right = base_joints(2);
                base_joint_left = base_joints(3);
                for frame_idx = 1:action_length(action_idx,test_idx)
                    pos=permute(skeleton(action_idx,test_idx,body_idx,frame_idx,base_joint_center,:),[5 6 1 2 3 4]);
                    body_x_direction=permute((skeleton(action_idx,test_idx,body_idx,frame_idx,base_joint_right,:)-skeleton(action_idx,test_idx,body_idx,frame_idx,base_joint_left,:)),[6 1 2 3 4 5]);
                    angle=atan2(body_x_direction(3),body_x_direction(1));
                    rotation_matrix=[cos(angle),0,-sin(angle);0,1,0;sin(angle),0,cos(angle)];
                    for joint_idx = 1:size(skeleton,5)
                        r=permute(skeleton(action_idx,test_idx,body_idx,frame_idx,joint_idx,:),[5 6 1 2 3 4])-pos;
                        aligned_skeleton(action_idx,test_idx,body_idx,frame_idx,joint_idx,:)=r*rotation_matrix;
                    end
                    base_body = 1;
                    base_pos=permute(skeleton(action_idx,test_idx,base_body,frame_idx,base_joint_center,:),[5 6 1 2 3 4]);
                    aligned_skeleton(action_idx,test_idx,body_idx,frame_idx,size(skeleton,5)+1,:)=pos-base_pos;
                end
                    
            end
        end
    end

end
