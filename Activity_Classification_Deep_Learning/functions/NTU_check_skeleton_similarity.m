function [ distances ] = NTU_check_skeleton_similarity( bodies1,bodies2 )
    
    number_of_joints=25;
    
    bodies1 = NTU_body_alignment(bodies1);
    bodies2 = NTU_body_alignment(bodies2);
    
    ks1 = keys(bodies1);
    ks2 = keys(bodies2);
    distances = zeros(length(ks1),length(ks2));
    for k1 = 1:length(ks1)
        body1 = bodies1(ks1{k1});
        for k2 = 1:length(ks2)
            body2 = bodies2(ks2{k2});
            len = min(size(body1,1),size(body2,1));
            dis = 0;
            count = 0;
            for frame_idx=1:len
                if ~any(any(isnan(body1(frame_idx,:,:)))) && ~any(any(isnan(body2(frame_idx,:,:))))
                    for joint_idx=1:number_of_joints
                        dis = dis + norm(permute(body1(frame_idx,joint_idx,:)-body2(frame_idx,joint_idx,:),[2 3 1]));
                    end
                    count = count + 1;
                end
            end
            distances(k1,k2) = dis / count;
        end
    end
    
end
