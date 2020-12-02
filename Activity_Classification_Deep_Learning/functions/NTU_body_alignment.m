function [ aligned_bodies ] = NTU_body_alignment( bodies )
    
    base_joint_center=1;
    base_joint_left=5;
    base_joint_right=9;

    aligned_bodies = containers.Map;
    
    ks = keys(bodies);
    for k = 1:length(ks)
        body = bodies(ks{k});
        aligned_body = zeros(size(body));
        for frame_idx=1:size(body,1)
            pos=permute(body(frame_idx,base_joint_center,:),[2 3 1]);
            body_x_direction=permute((body(frame_idx,base_joint_right,:)-body(frame_idx,base_joint_left,:)),[3 1 2]);
            angle=atan2(body_x_direction(3),body_x_direction(1));
            rotation_matrix=[cos(angle),0,-sin(angle);0,1,0;sin(angle),0,cos(angle)];
            for joint_idx=1:size(body,2)
                r=permute(body(frame_idx,joint_idx,:),[2 3 1])-pos;
                aligned_body(frame_idx,joint_idx,:)=r*rotation_matrix;
            end
        end
        aligned_bodies(ks{k}) = aligned_body;
    end
    
end
