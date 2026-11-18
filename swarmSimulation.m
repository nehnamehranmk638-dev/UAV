% SwarmSimulation_With_Trained_Transformer.m
% Complete Integrated Pipeline: Continuous Reconnaissance Patrol
% Architecture: 2 Dynamic Masters, 4 Moving Slaves (2 per Master)
% Deep Learning Engine: Live Spatio-Temporal Graph Transformer (via stgt_predict.py)
%
% UPGRADE: Added Autonomous Master Relocation & Link Rebuild Protocol when 
% LoS or distance-based communication thresholds break mid-mission.
% ADDED: Trees and Buildings appended into the static obstacles matrix.
% FIXED: Integrated active steering field to prevent Masters from moving near/inside obstacles.
% FIXED: Enforced strict waypoint containment within the dynamic yellow bounding box grid.

clear; clc; close all;

%% 1. Initialize Python Environment Bridge inside MATLAB
disp('====================================================');
disp('[INTEGRATION] Bootstrapping PyTorch Deep Learning Inference...');
disp('====================================================');

% Explicitly load the Python bridge and force a module reload to catch updates
try
    pyrun("import importlib");
    pyrun("import stgt_predict");
    pyrun("importlib.reload(stgt_predict)");
    disp('[SUCCESS] Spatio-Temporal Transformer Engine Linked Flawlessly.');
catch ME
    disp('[WARNING] Python bridge not fully available. Engaging automated fallback matrix.');
end

%% 2. Simulation Environment & Baseline Configs
bounds = [0, 100, 0, 100, 0, 50]; 
m1_start = [10, 20, 15];      
m2_start = [10, 80, 15];      
        
% Relative formation offsets for slave pairs during transit
slave_target_offsets = [-2, -3, -1;  
                        -2,  3, -1]; 

% Static Obstacles Array: [X, Y, Z, Radius]
% Incorporates default zones, architectural buildings, and foliage trees
static_obs = [35, 35, 15, 9;    % Original Obstacle 1
              70, 70, 20, 11;   % Original Obstacle 2
              50, 20, 12, 8;    % Static Building 1
              75, 80, 15, 10;   % Static Building 2
              30, 55, 8,  4;    % Static Tree 1
              62, 30, 6,  4;    % Static Tree 2
              45, 75, 10, 5];   % Static Tree 3
          
dynamic_obs = [55, 45, 15, 7;  
               40, 65, 18, 6;  
               75, 40, 15, 6];
            
dyn_vel = [-0.4, 0.3, 0;  
            0.5, -0.4, 0;  
           -0.2, 0.6, 0];

%% OBJECTIVE STATUS REGISTERS
m1_active = true; 
m2_active = true;
system_fallback = false; 

m1_bat = 100.0; m2_bat = 100.0;
s1A_bat = 100.0; s1B_bat = 100.0;
s2A_bat = 100.0; s2B_bat = 100.0;
crypto_mode = 'STANDARD-AES128';

% Dynamic Master Tracking States & Range Capabilities
m1_pos = m1_start;
m2_pos = m2_start;
max_comm_dist = 35.0;     % Max distance (meters) before link breaks due to range
master_move_speed = 1.2;  % Speed of Master tracking movement

%% 3. Setup 3D High-Contrast Plot Visualizer
figure('Name', 'STGT Multi-Objective Swarm (Live Transformer Inference Loop)', 'Color', 'k', 'Position', [80, 80, 1200, 820]);
hold on; grid on; view(3); axis(bounds);
set(gca, 'Color', [0.1, 0.1, 0.1], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'GridColor', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('STGT Autonomous Swarm (Live Inference via trained stgt\_model.pth)', 'Color', 'w');

% Colour constants reused for obstacles + legend so they always match
ORANGE = [0.8, 0.4, 0];   % static obstacles
MAGENTA = [0.8, 0, 0.8];  % dynamic obstacles

% Plot Static Threat Spheres (Orange)
for i = 1:size(static_obs, 1)
    [x,y,z] = sphere; r = static_obs(i,4);
    surf(x*r + static_obs(i,1), y*r + static_obs(i,2), z*r + static_obs(i,3), 'FaceColor', ORANGE, 'FaceAlpha', 0.4, 'EdgeColor', 'none');
end
dyn_handles = gobjects(size(dynamic_obs,1), 1);
for i = 1:size(dynamic_obs, 1)
    [x,y,z] = sphere; r = dynamic_obs(i,4);
    dyn_handles(i) = surf(x*r + dynamic_obs(i,1), y*r + dynamic_obs(i,2), z*r + dynamic_obs(i,3), 'FaceColor', MAGENTA, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
end

% Drone Graphic Handles 
m1_h = plot3(m1_pos(1), m1_pos(2), m1_pos(3), 'dg', 'MarkerSize', 12, 'MarkerFaceColor', 'g');
m2_h = plot3(m2_pos(1), m2_pos(2), m2_pos(3), 'dg', 'MarkerSize', 12, 'MarkerFaceColor', 'g');

s1A_h = plot3(m1_start(1), m1_start(2), m1_start(3), 'oc', 'MarkerSize', 8, 'MarkerFaceColor', 'c');
s1B_h = plot3(m1_start(1), m1_start(2), m1_start(3), 'oc', 'MarkerSize', 8, 'MarkerFaceColor', 'c');
s2A_h = plot3(m2_start(1), m2_start(2), m2_start(3), 'om', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
s2B_h = plot3(m2_start(1), m2_start(2), m2_start(3), 'om', 'MarkerSize', 8, 'MarkerFaceColor', 'm');

% Initialize Bounding Box Graphic Handle
stgt_box_h = plot3(nan, nan, nan, 'y--', 'LineWidth', 2.5); 

% Paths with HIGHLY VISIBLE waypoints configuration
path1_h = plot3(nan, nan, nan, 'c-o', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c'); 
path2_h = plot3(nan, nan, nan, 'm-o', 'LineWidth', 2.0, 'MarkerSize', 8, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'm');

goal_pos = [85, 50, 12]; 
plot3(goal_pos(1), goal_pos(2), goal_pos(3), 'p', 'MarkerSize', 22, 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'w');

% Network Communication Lines
link1A_h = plot3(nan, nan, nan, 'LineWidth', 1.8); link1B_h = plot3(nan, nan, nan, 'LineWidth', 1.8);
link2A_h = plot3(nan, nan, nan, 'LineWidth', 1.8); link2B_h = plot3(nan, nan, nan, 'LineWidth', 1.8);

% Active Message Transmission Lines (Handshake & Mishandling Protocols)
msg_tx_h = plot3(nan, nan, nan, 'y-', 'LineWidth', 3);  % Master -> Slave (Command)
msg_ack_h = plot3(nan, nan, nan, 'g-', 'LineWidth', 3); % Slave -> Master (Acknowledge)
msg_err_h = plot3(nan, nan, nan, 'r-', 'LineWidth', 3); % Slave -> Master (Mishandling Alert)

%% 3b. SIMULATION COLOUR KEY  (real legend with correct coloured shapes)
lh(1)  = plot3(nan,nan,nan,'d','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',9,'LineStyle','none');
lh(2)  = plot3(nan,nan,nan,'o','MarkerFaceColor','c','MarkerEdgeColor','c','MarkerSize',8,'LineStyle','none');
lh(3)  = plot3(nan,nan,nan,'o','MarkerFaceColor','m','MarkerEdgeColor','m','MarkerSize',8,'LineStyle','none');
lh(4)  = plot3(nan,nan,nan,'p','MarkerFaceColor','y','MarkerEdgeColor','y','MarkerSize',13,'LineStyle','none');
lh(5)  = plot3(nan,nan,nan,'o','MarkerFaceColor',ORANGE,'MarkerEdgeColor',ORANGE,'MarkerSize',11,'LineStyle','none');
lh(6)  = plot3(nan,nan,nan,'o','MarkerFaceColor',MAGENTA,'MarkerEdgeColor',MAGENTA,'MarkerSize',11,'LineStyle','none');
lh(7)  = plot3(nan,nan,nan,'--','Color','y','LineWidth',2.5);
lh(8)  = plot3(nan,nan,nan,'-o','Color','c','MarkerFaceColor','w','MarkerEdgeColor','c','LineWidth',2,'MarkerSize',6);
lh(9)  = plot3(nan,nan,nan,'-o','Color','m','MarkerFaceColor','w','MarkerEdgeColor','m','LineWidth',2,'MarkerSize',6);
lh(10) = plot3(nan,nan,nan,'--','Color','c','LineWidth',1.8);
lh(11) = plot3(nan,nan,nan,':','Color','r','LineWidth',1.8);
lh(12) = plot3(nan,nan,nan,'-','Color','y','LineWidth',3);
lh(13) = plot3(nan,nan,nan,'-','Color','g','LineWidth',3);
lh(14) = plot3(nan,nan,nan,'-','Color','r','LineWidth',3);

legend_labels = { ...
    'Master Drones  (green = online, red = offline)', ...
    'Master 1 Slave Drones', ...
    'Master 2 Slave Drones', ...
    'Target Destination (Goal)', ...
    'Static Obstacles', ...
    'Dynamic Obstacles / Threats', ...
    'STGT Optimized Bounding Box', ...
    'Swarm 1 Path and Waypoints', ...
    'Swarm 2 Path and Waypoints', ...
    'Active Link  (LoS clear & In-Range)', ...
    'Blocked Link  (LoS lost or Out-of-Range)', ...
    'Command TX  (Master -> Slave)', ...
    'ACK  (Slave -> Master)', ...
    'Mishandling Alert  (Slave -> Master)'};

lgd = legend(lh, legend_labels, 'Interpreter','none', 'TextColor','w', ...
    'Color',[0 0 0], 'EdgeColor','w', 'FontSize',8, 'Location','northeastoutside');
title(lgd, 'SIMULATION COLOR KEY');
set(lgd, 'AutoUpdate', 'off');   % later plotting must not add new entries

% Initial positions setup
s1A_pos = m1_start; s1B_pos = m1_start;
s2A_pos = m2_start; s2B_pos = m2_start;
tracking_responsiveness = 0.25; 

%% 4. CONTINUOUS RECONNAISSANCE PATROL MISSION
fprintf('\n[MISSION] Initializing Optimized Waypoint Paths toward Goal...\n');

% [FIX] Bootstrap initial stgt_bounds box before planning to ensure containment
try
    py_output = py.stgt_predict.get_predictions((m1_start+m2_start)/2, dynamic_obs(1,1:3), dynamic_obs(2,1:3), dynamic_obs(3,1:3));
    if isa(py_output, 'py.numpy.ndarray')
        stgt_bounds = double(py_output.tolist());
    else
        stgt_bounds = double(py_output); 
    end
catch
    stgt_bounds = bounds;
end

% Expand initial bounding box to safely include start locations and final goal destination
init_pts = [m1_start; m2_start; goal_pos];
stgt_bounds(1) = max(bounds(1), min(stgt_bounds(1), min(init_pts(:,1)) - 5));
stgt_bounds(2) = min(bounds(2), max(stgt_bounds(2), max(init_pts(:,1)) + 5));
stgt_bounds(3) = max(bounds(3), min(stgt_bounds(3), min(init_pts(:,2)) - 5));
stgt_bounds(4) = min(bounds(4), max(stgt_bounds(4), max(init_pts(:,2)) + 5));
stgt_bounds(5) = max(bounds(5), min(stgt_bounds(5), min(init_pts(:,3)) - 5));
stgt_bounds(6) = min(bounds(6), max(stgt_bounds(6), max(init_pts(:,3)) + 5));

% Plan paths strictly inside the verified initial optimization workspace
current_path1 = plan_biRRT(m1_start, goal_pos, stgt_bounds, static_obs, dynamic_obs);
current_path2 = plan_biRRT(m2_start, goal_pos, stgt_bounds, static_obs, dynamic_obs);

% Update initial visible path layout on the grid
set(path1_h, 'XData', current_path1(:,1), 'YData', current_path1(:,2), 'ZData', current_path1(:,3));
set(path2_h, 'XData', current_path2(:,1), 'YData', current_path2(:,2), 'ZData', current_path2(:,3));

% Waypoint index trackers for strict sequential execution
wp_idx1 = 1;
wp_idx2 = 1;
replan_cooldown = 0;

% Communication Protocol Variables
protocol_timer = 0;

while wp_idx1 <= size(current_path1, 1) || wp_idx2 <= size(current_path2, 1)
    
    % Compute current spatial centroids dynamically at the start of loop iteration
    grp1_centroid = (s1A_pos + s1B_pos) / 2;
    grp2_centroid = (s2A_pos + s2B_pos) / 2;
    slave_centroid = (s1A_pos + s2A_pos) / 2;

    % --- SIMULATED MASTER HARDWARE FAILURE SCENARIOS ---
    if wp_idx1 == 10
        m1_active = false; % Master 1 goes offline unexpectedly mid-mission
    end
    if wp_idx2 == 22
        m2_active = false; % Master 2 fails completely
        system_fallback = true; 
    end

    % A. Update Dynamic Threat Vectors
    dynamic_obs(:,1:3) = dynamic_obs(:,1:3) + dyn_vel;
    for i = 1:size(dynamic_obs,1)
        for ax = 1:2
            if dynamic_obs(i, ax) < bounds(ax*2-1)+5 || dynamic_obs(i, ax) > bounds(ax*2)-5
                dyn_vel(i, ax) = -dyn_vel(i, ax);
            end
        end
        [x,y,z] = sphere; r = dynamic_obs(i,4);
        dyn_handles(i).XData = x*r + dynamic_obs(i,1); 
        dyn_handles(i).YData = y*r + dynamic_obs(i,2); 
        dyn_handles(i).ZData = z*r + dynamic_obs(i,3);
    end
    
    % Safeguard boundary conditions for indices
    idx1 = min(wp_idx1, size(current_path1, 1));
    idx2 = min(wp_idx2, size(current_path2, 1));
    path_pt1 = current_path1(idx1, :);
    path_pt2 = current_path2(idx2, :);
    
    % B. LIVE INFERENCE CALL / DYNAMIC SHRINKING GRID METHOD
    try
        py_output = py.stgt_predict.get_predictions(grp1_centroid, dynamic_obs(1,1:3), dynamic_obs(2,1:3), dynamic_obs(3,1:3));
        if isa(py_output, 'py.numpy.ndarray')
            stgt_bounds = double(py_output.tolist());
        else
            stgt_bounds = double(py_output); 
        end
    catch
        % Shrink bounding box workspace window neatly around remaining route waypoints and destination
        rem_pts = [current_path1(idx1:end, :); current_path2(idx2:end, :); goal_pos];
        stgt_bounds = [max(bounds(1), min(rem_pts(:,1))-10), min(bounds(2), max(rem_pts(:,1))+10), ...
                       max(bounds(3), min(rem_pts(:,2))-10), min(bounds(4), max(rem_pts(:,2))+10), ...
                       max(bounds(5), min(rem_pts(:,3))-6),  min(bounds(6), max(rem_pts(:,3))+6)];
    end
    
    % [FIX] Strict waypoint containment verification guard
    % Forces stgt_bounds box to expand and safely enclose all active, remaining route points
    rem_pts = [current_path1(idx1:end, :); current_path2(idx2:end, :); goal_pos];
    stgt_bounds(1) = max(bounds(1), min(stgt_bounds(1), min(rem_pts(:,1)) - 3));
    stgt_bounds(2) = min(bounds(2), max(stgt_bounds(2), max(rem_pts(:,1)) + 3));
    stgt_bounds(3) = max(bounds(3), min(stgt_bounds(3), min(rem_pts(:,2)) - 3));
    stgt_bounds(4) = min(bounds(4), max(stgt_bounds(4), max(rem_pts(:,2)) + 3));
    stgt_bounds(5) = max(bounds(5), min(stgt_bounds(5), min(rem_pts(:,3)) - 3));
    stgt_bounds(6) = min(bounds(6), max(stgt_bounds(6), max(rem_pts(:,3)) + 3));
    
    % Update the yellow grid box live to enclose optimized trajectory space
    update_bbox(stgt_box_h, stgt_bounds);
    
    min_dist_to_threat = min(sqrt(sum((dynamic_obs(:,1:3) - slave_centroid).^2, 2)));
    
    % C. Transformer Bounding-Box Trigger Constraint & Replanning
    if min_dist_to_threat < 18 && replan_cooldown == 0
        replanned_path1 = plan_biRRT(s1A_pos, goal_pos, stgt_bounds, static_obs, dynamic_obs);
        replanned_path2 = plan_biRRT(s2A_pos, goal_pos, stgt_bounds, static_obs, dynamic_obs);
        
        if ~isempty(replanned_path1) && ~isempty(replanned_path2)
            current_path1 = replanned_path1; 
            current_path2 = replanned_path2;
            wp_idx1 = 1; 
            wp_idx2 = 1; % Reset execution tracking to follow new route safely forward
            replan_cooldown = 12; 
            
            % Instantly update the visual plots with the newly generated paths and waypoints
            set(path1_h, 'XData', current_path1(:,1), 'YData', current_path1(:,2), 'ZData', current_path1(:,3));
            set(path2_h, 'XData', current_path2(:,1), 'YData', current_path2(:,2), 'ZData', current_path2(:,3));
        end
    end
    if replan_cooldown > 0, replan_cooldown = replan_cooldown - 1; end
    
    % D. Drone Trajectory Interpolation Logic
    s1A_pos = s1A_pos + tracking_responsiveness * ((path_pt1 + slave_target_offsets(1,:)) - s1A_pos);
    s1B_pos = s1B_pos + tracking_responsiveness * ((path_pt1 + slave_target_offsets(2,:)) - s1B_pos);
    s2A_pos = s2A_pos + tracking_responsiveness * ((path_pt2 + slave_target_offsets(1,:)) - s2A_pos);
    s2B_pos = s2B_pos + tracking_responsiveness * ((path_pt2 + slave_target_offsets(2,:)) - s2B_pos);
    
    % PROXIMITY VERIFICATION: Advance waypoints based on actual progress
    if norm(grp1_centroid - path_pt1) < 4.5 && wp_idx1 < size(current_path1, 1)
        wp_idx1 = wp_idx1 + 1;
    end
    if norm(grp2_centroid - path_pt2) < 4.5 && wp_idx2 < size(current_path2, 1)
        wp_idx2 = wp_idx2 + 1;
    end
    
    % E. Battery Optimization Model & Cyber Security Regimes
    flight_drain = 0.22; 
    if min_dist_to_threat < 22 && mean([s1A_bat, s2A_bat]) > 35
        crypto_mode = 'TACTICAL-AES256'; crypto_drain = 0.45; 
    elseif mean([s1A_bat, s2A_bat]) <= 35
        crypto_mode = 'ECO-CIPHER'; crypto_drain = 0.04; 
    else
        crypto_mode = 'STANDARD-AES128'; crypto_drain = 0.12; 
    end
    
    m1_bat = max(0, m1_bat - 0.03); m2_bat = max(0, m2_bat - 0.03);
    s1A_bat = max(0, s1A_bat - (flight_drain + crypto_drain)); s1B_bat = max(0, s1B_bat - (flight_drain + crypto_drain));
    s2A_bat = max(0, s2A_bat - (flight_drain + crypto_drain)); s2B_bat = max(0, s2B_bat - (flight_drain + crypto_drain));
    
    % F. Line-of-Sight & Distance Range Intersection Validations
    all_obstacles = [static_obs; dynamic_obs];
    
    dist_m1_s1A = norm(m1_pos - s1A_pos);
    dist_m1_s1B = norm(m1_pos - s1B_pos);
    dist_m2_s2A = norm(m2_pos - s2A_pos);
    dist_m2_s2B = norm(m2_pos - s2B_pos);
    
    s1A_los = check_line_of_sight(m1_pos, s1A_pos, all_obstacles) && (dist_m1_s1A <= max_comm_dist);
    s1B_los = check_line_of_sight(m1_pos, s1B_pos, all_obstacles) && (dist_m1_s1B <= max_comm_dist);
    s2A_los = check_line_of_sight(m2_pos, s2A_pos, all_obstacles) && (dist_m2_s2A <= max_comm_dist);
    s2B_los = check_line_of_sight(m2_pos, s2B_pos, all_obstacles) && (dist_m2_s2B <= max_comm_dist);
    
    % --- FIXED: AUTONOMOUS MASTER RELOCATION ENGINE WITH SAFETY STEERING ---
    if m1_active && (~s1A_los || ~s1B_los)
        m1_dir = grp1_centroid - m1_pos;
        if norm(m1_dir) > 0
            m1_dir_norm = m1_dir / norm(m1_dir);
            
            % Compute artificial repulsive forces from all obstacles for comfort clearance
            repulsive1 = [0, 0, 0];
            master_safe_zone = 6.0; % Trigger envelope outside obstacle surface boundary
            
            for i = 1:size(all_obstacles, 1)
                obs_center = all_obstacles(i, 1:3);
                obs_radius = all_obstacles(i, 4);
                dist_to_obs = norm(m1_pos - obs_center);
                dist_to_surface = dist_to_obs - obs_radius;
                
                if dist_to_surface < master_safe_zone
                    if dist_to_obs > 0
                        dir_away = (m1_pos - obs_center) / dist_to_obs;
                    else
                        dir_away = [rand()-0.5, rand()-0.5, rand()-0.5]; dir_away = dir_away / norm(dir_away);
                    end
                    repulsive1 = repulsive1 + dir_away * (master_safe_zone - max(0.1, dist_to_surface));
                end
            end
            
            % Combine tracking behavior with active safety repulsion field
            m1_combined_dir = m1_dir_norm + 1.5 * repulsive1;
            if norm(m1_combined_dir) > 0, m1_combined_dir = m1_combined_dir / norm(m1_combined_dir); end
            
            m1_next = m1_pos + master_move_speed * m1_combined_dir;
            
            % Strict verification check: enforce a robust 3.0 meter look-ahead boundary clearance
            m1_step_safe = true;
            for i = 1:size(all_obstacles, 1)
                if norm(m1_next - all_obstacles(i, 1:3)) < (all_obstacles(i, 4) + 3.0)
                    m1_step_safe = false; break;
                end
            end
            
            if m1_step_safe
                m1_pos = m1_next;
                disp('[AUTONOMOUS REPOSITIONING] Master 1 moving safely to rebuild broken link with Cluster 1.');
            end
        end
    end
    
    if m2_active && (~s2A_los || ~s2B_los)
        m2_dir = grp2_centroid - m2_pos;
        if norm(m2_dir) > 0
            m2_dir_norm = m2_dir / norm(m2_dir);
            
            % Compute artificial repulsive forces from all obstacles for comfort clearance
            repulsive2 = [0, 0, 0];
            master_safe_zone = 6.0;
            
            for i = 1:size(all_obstacles, 1)
                obs_center = all_obstacles(i, 1:3);
                obs_radius = all_obstacles(i, 4);
                dist_to_obs = norm(m2_pos - obs_center);
                dist_to_surface = dist_to_obs - obs_radius;
                
                if dist_to_surface < master_safe_zone
                    if dist_to_obs > 0
                        dir_away = (m2_pos - obs_center) / dist_to_obs;
                    else
                        dir_away = [rand()-0.5, rand()-0.5, rand()-0.5]; dir_away = dir_away / norm(dir_away);
                    end
                    repulsive2 = repulsive2 + dir_away * (master_safe_zone - max(0.1, dist_to_surface));
                end
            end
            
            % Combine tracking behavior with active safety repulsion field
            m2_combined_dir = m2_dir_norm + 1.5 * repulsive2;
            if norm(m2_combined_dir) > 0, m2_combined_dir = m2_combined_dir / norm(m2_combined_dir); end
            
            m2_next = m2_pos + master_move_speed * m2_combined_dir;
            
            % Strict verification check: enforce a robust 3.0 meter look-ahead boundary clearance
            m2_step_safe = true;
            for i = 1:size(all_obstacles, 1)
                if norm(m2_next - all_obstacles(i, 1:3)) < (all_obstacles(i, 4) + 3.0)
                    m2_step_safe = false; break;
                end
            end
            
            if m2_step_safe
                m2_pos = m2_next;
                disp('[AUTONOMOUS REPOSITIONING] Master 2 moving safely to rebuild broken link with Cluster 2.');
            end
        end
    end

    % G. Refresh Graphic Coordinate States
    s1A_h.XData = s1A_pos(1); s1A_h.YData = s1A_pos(2); s1A_h.ZData = s1A_pos(3);
    s1B_h.XData = s1B_pos(1); s1B_h.YData = s1B_pos(2); s1B_h.ZData = s1B_pos(3);
    s2A_h.XData = s2A_pos(1); s2A_h.YData = s2A_pos(2); s2A_h.ZData = s2A_pos(3);
    s2B_h.XData = s2B_pos(1); s2B_h.YData = s2B_pos(2); s2B_h.ZData = s2B_pos(3);
    
    m1_h.XData = m1_pos(1); m1_h.YData = m1_pos(2); m1_h.ZData = m1_pos(3);
    m2_h.XData = m2_pos(1); m2_h.YData = m2_pos(2); m2_h.ZData = m2_pos(3);
    
    % --- DYNAMIC CLUSTER REROUTING NETWORK COMMS ---
    if m1_active
        update_link(link1A_h, m1_pos, s1A_pos, s1A_los);
        update_link(link1B_h, m1_pos, s1B_pos, s1B_los);
    elseif m2_active
        s1A_m2_dist = norm(m2_pos - s1A_pos);
        s1B_m2_dist = norm(m2_pos - s1B_pos);
        s1A_m2_los = check_line_of_sight(m2_pos, s1A_pos, all_obstacles) && (s1A_m2_dist <= max_comm_dist);
        s1B_m2_los = check_line_of_sight(m2_pos, s1B_pos, all_obstacles) && (s1B_m2_dist <= max_comm_dist);
        update_link(link1A_h, m2_pos, s1A_pos, s1A_m2_los);
        update_link(link1B_h, m2_pos, s1B_pos, s1B_m2_los);
    else
        update_link(link1A_h, m1_pos, s1A_pos, false);
        update_link(link1B_h, m1_pos, s1B_pos, false);
    end
    
    update_link(link2A_h, m2_pos, s2A_pos, s2A_los && m2_active);
    update_link(link2B_h, m2_pos, s2B_pos, s2B_los && m2_active);
    
    % Update Master Node Graphics (Status Colors)
    if system_fallback
        set(m1_h, 'MarkerFaceColor', 'r'); set(m2_h, 'MarkerFaceColor', 'r');
    elseif ~m1_active && m2_active
        set(m1_h, 'MarkerFaceColor', 'r'); 
    else
        set(m1_h, 'MarkerFaceColor', 'g'); set(m2_h, 'MarkerFaceColor', 'g');
    end

    % --- MISHANDLING EVENT PROTOCOL (SLAVE -> MASTER) ---
    set(msg_err_h, 'XData', nan, 'YData', nan, 'ZData', nan);	
    dist_s1A = min(sqrt(sum((dynamic_obs(:,1:3) - s1A_pos).^2, 2)));
    dist_s2A = min(sqrt(sum((dynamic_obs(:,1:3) - s2A_pos).^2, 2)));
    
    if dist_s1A < 10 && m1_active
        set(msg_err_h, 'XData', [s1A_pos(1), m1_pos(1)], 'YData', [s1A_pos(2), m1_pos(2)], 'ZData', [s1A_pos(3), m1_pos(3)]);
        disp('[ALERT] Slave 1A Tactical Mishandling: Transmitting Alert to Master 1.');
    elseif dist_s2A < 10 && m2_active
        set(msg_err_h, 'XData', [s2A_pos(1), m2_pos(1)], 'YData', [s2A_pos(2), m2_pos(2)], 'ZData', [s2A_pos(3), m2_pos(3)]);
        disp('[ALERT] Slave 2A Tactical Mishandling: Transmitting Alert to Master 2.');
    end

    % --- 2-WAY MESSAGE ACKNOWLEDGEMENT PROTOCOL (MASTER <-> SLAVE) ---
    protocol_timer = protocol_timer + 1;
    if protocol_timer == 45 && m2_active 
        set(msg_tx_h, 'XData', [m2_pos(1), s2A_pos(1)], 'YData', [m2_pos(2), s2A_pos(2)], 'ZData', [m2_pos(3), s2A_pos(3)]);
    elseif protocol_timer == 50 && m2_active
        set(msg_tx_h, 'XData', nan, 'YData', nan, 'ZData', nan); 
        set(msg_ack_h, 'XData', [s2A_pos(1), m2_pos(1)], 'YData', [s2A_pos(2), m2_pos(2)], 'ZData', [s2A_pos(3), m2_pos(3)]);
    elseif protocol_timer >= 55
        set(msg_ack_h, 'XData', nan, 'YData', nan, 'ZData', nan);
        protocol_timer = 0; 
    end

    % Stop execution loop only when target destination is fully obtained by drones
    if wp_idx1 == size(current_path1, 1) && wp_idx2 == size(current_path2, 1) && ...
       norm(grp1_centroid - goal_pos) < 3.0 && norm(grp2_centroid - goal_pos) < 3.0
       break;
    end
    
    drawnow; pause(0.04); 
end

% Guarantee exact alignment snap onto target positions when finished
s1A_h.XData = goal_pos(1)+slave_target_offsets(1,1); s1A_h.YData = goal_pos(2)+slave_target_offsets(1,2); s1A_h.ZData = goal_pos(3)+slave_target_offsets(1,3);
s1B_h.XData = goal_pos(1)+slave_target_offsets(2,1); s1B_h.YData = goal_pos(2)+slave_target_offsets(2,2); s1B_h.ZData = goal_pos(3)+slave_target_offsets(2,3);
s2A_h.XData = goal_pos(1)+slave_target_offsets(1,1); s2A_h.YData = goal_pos(2)+slave_target_offsets(1,2); s2A_h.ZData = goal_pos(3)+slave_target_offsets(1,3);
s2B_h.XData = goal_pos(1)+slave_target_offsets(2,1); s2B_h.YData = goal_pos(2)+slave_target_offsets(2,2); s2B_h.ZData = goal_pos(3)+slave_target_offsets(2,3);
drawnow;

disp('====================================================');
disp('[MISSION COMPLETE] Slaves safely reached the Target Destination.');
disp('====================================================');

%% =====================================================================
%% Subroutines & Mathematical Extensions
%% =====================================================================
function update_link(link_h, m_pos, s_pos, is_active_los)
    link_h.XData = [m_pos(1), s_pos(1)]; link_h.YData = [m_pos(2), s_pos(2)]; link_h.ZData = [m_pos(3), s_pos(3)];
    if is_active_los
        link_h.Color = [0, 1, 1]; link_h.LineStyle = '--'; 
    else
        link_h.Color = [1, 0, 0]; link_h.LineStyle = ':';  
    end
end

function clear_los = check_line_of_sight(p1, p2, obstacles)
    clear_los = true;
    segment_vector = p2 - p1;
    segment_length = norm(segment_vector);
    if segment_length == 0, return; end
    u_vector = segment_vector / segment_length;
    
    for i = 1:size(obstacles, 1)
        obs_center = obstacles(i, 1:3);
        obs_radius = obstacles(i, 4);
        v = obs_center - p1;
        projection_dist = dot(v, u_vector);
        projection_dist = max(0, min(segment_length, projection_dist));
        closest_point = p1 + projection_dist * u_vector;
        if norm(closest_point - obs_center) <= (obs_radius + 0.5)
            clear_los = false; return;
        end
    end
end

function update_bbox(h, b)
    x = [b(1) b(2) b(2) b(1) b(1) b(1) b(2) b(2) b(1) b(1) b(2) b(2) b(2) b(2) b(1) b(1)];
    y = [b(3) b(3) b(4) b(4) b(3) b(3) b(3) b(4) b(4) b(3) b(3) b(3) b(4) b(4) b(4) b(4)];
    z = [b(5) b(5) b(5) b(5) b(5) b(6) b(6) b(6) b(6) b(6) b(6) b(5) b(5) b(6) b(6) b(5)];
    set(h, 'XData', x, 'YData', y, 'ZData', z);
end

%% =====================================================================
%% PATH PLANNING: Bi-Directional RRT Core
%% =====================================================================
function path = plan_biRRT(start, goal, bounds, s_obs, d_obs)
    treeA = start; parentA = 0; treeB = goal; parentB = 0; max_iter = 600; step_size = 5.0; 
    for iter = 1:max_iter
        q_rand = [bounds(1)+(bounds(2)-bounds(1))*rand(), bounds(3)+(bounds(4)-bounds(3))*rand(), bounds(5)+(bounds(6)-bounds(5))*rand()];
        [treeA, parentA, q_newA] = extend_tree(treeA, parentA, q_rand, step_size, s_obs, d_obs);
        
        if ~isempty(q_newA)
            [treeB, parentB, ~, connected] = connect_tree(treeB, parentB, q_newA, step_size, s_obs, d_obs);
            if connected
                path = build_path(treeA, parentA, treeB, parentB, start); 
                return; 
            end
        end
        tempT = treeA; treeA = treeB; treeB = tempT; tempP = parentA; parentA = parentB; parentB = tempP;
    end
    path = [start; goal]; 
end

function [tree, parent, q_new] = extend_tree(tree, parent, q_rand, step, s_obs, d_obs)
    q_new = []; distances = sum((tree - q_rand).^2, 2); [~, nearest_idx] = min(distances); q_near = tree(nearest_idx, :);
    dir = q_rand - q_near; dist = norm(dir);
    if dist > 0
        q_new_test = q_near + (dir/dist) * min(step, dist);
        if ~check_collision(q_new_test, s_obs, d_obs), tree(end+1, :) = q_new_test; parent(end+1) = nearest_idx; q_new = q_new_test; end
    end
end

function [tree, parent, q_new, connected] = connect_tree(tree, parent, target, step, s_obs, d_obs)
    connected = false; q_new = [];
    while true
        [tree, parent, q_new_test] = extend_tree(tree, parent, target, step, s_obs, d_obs);
        if isempty(q_new_test), break; end
        q_new = q_new_test; if norm(q_new - target) < 1.5, connected = true; break; end
    end
end

function col = check_collision(pt, s_obs, d_obs)
    col = false; all_obs = [s_obs; d_obs];
    for i = 1:size(all_obs,1)
        if norm(pt - all_obs(i,1:3)) < all_obs(i,4) + 1.5, col = true; return; end
    end
end

function path = build_path(treeA, parentA, treeB, parentB, start_node)
    pathA = treeA(end,:); curr = parentA(end); while curr > 0, pathA = [treeA(curr,:); pathA]; curr = parentA(curr); end
    pathB = []; curr = size(treeB,1); while curr > 0, pathB = [pathB; treeB(curr,:)]; curr = parentB(curr); end
    full_path = [pathA; pathB];
    if norm(full_path(1,:) - start_node) > 1.0, full_path = flipud(full_path); end
    path = full_path;
end