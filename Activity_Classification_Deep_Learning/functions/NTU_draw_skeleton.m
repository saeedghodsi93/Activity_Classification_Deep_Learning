function [ ] = NTU_draw_skeleton( bodies,frame_rate )

    joint_pair=[ 1  2  3  3  3  5  6   9 10   1  1 13 14  17 18 ;
                 2  3  4  5  9  6  7  10 11  13 17 14 15  18 19 ];
    
    h=figure;
    hold off;
    line_width=5;
    line_color=[39	64	139];
    
    ks = keys(bodies);
    for k = 1:length(ks)
        body = bodies(ks{k});
        
        body = body * 100;
        for frame_idx=1:size(body,1)

            S=permute(body(frame_idx,:,:),[2 3 1]);
            J=S;
            J([8,12,16,20,21:25],:)=[];

            plot3(J(:,1),J(:,3),J(:,2),'r.','markersize',5);
            for j=1:size(joint_pair,2)
                c1=joint_pair(1,j);
                c2=joint_pair(2,j);
                line([S(c1,1) S(c2,1)], [S(c1,3) S(c2,3)], [S(c1,2) S(c2,2)],'color',line_color/255,'linewidth',line_width);
            end

            xlim = [-75 75];
            ylim = [-75 75];
            zlim = [-120 120];
            set(gca, 'xlim',xlim, 'ylim',ylim, 'zlim',zlim);
            set(gca,'DataAspectRatio',[1 1 1]);
    %         axis([xlim ylim zlim]);
    %         rotate(h,[0 45], -180);

            pause(1/frame_rate);

        end
        pause(1);
    end
    
    close(h);
    
end
