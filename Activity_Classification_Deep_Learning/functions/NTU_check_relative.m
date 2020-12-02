function [ distances ] = NTU_check_relative( bodies1,bodies2,relative_med,relative_std,setup,camera1,camera2 )
    
    number_of_dimensions = 3;
    hip_joint = 1;
        
    if (camera1==1 && camera2==2) || (camera1==2 && camera2==1)
        camera_idx = 1;
    elseif (camera1==2 && camera2==3) || (camera1==3 && camera2==2)
        camera_idx = 2;
    elseif (camera1==3 && camera2==1) || (camera1==1 && camera2==3)
        camera_idx = 3;
    end
    m = permute(relative_med(setup,camera_idx,:),[3 1 2]);
    s = permute(relative_std(setup,camera_idx,:),[3 1 2]);
    
    ks1 = keys(bodies1);
    ks2 = keys(bodies2);
    distances = zeros(length(ks1),length(ks2));
    for k1 = 1:length(ks1)
        body1 = bodies1(ks1{k1});
        for k2 = 1:length(ks2)
            body2 = bodies2(ks2{k2});
            len = min(size(body1,1),size(body2,1));
            pos = zeros(number_of_dimensions,[]);
            for frame_idx=1:len
                if ~any(any(isnan(body1(frame_idx,:,:)))) && ~any(any(isnan(body2(frame_idx,:,:))))
                    pos(:,end+1) = permute(body2(frame_idx,hip_joint,:)-body1(frame_idx,hip_joint,:),[3 1 2]);
                end
            end
            pos_med = zeros(number_of_dimensions,1);
            for dimension_idx = 1:number_of_dimensions
                pos_med(dimension_idx,1) = median(pos(dimension_idx,:));
            end
            distances(k1,k2) = norm((pos_med(:,1) - m) ./ s);
        end
    end
    
end
