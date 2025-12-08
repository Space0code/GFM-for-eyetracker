# eSEED_v2_conversion.py
"""
Data structure in mat file (see readme.txt in eSEED_v2 dataset):
- Data (10, 4)
    - data{#}.video{i}
        - gaze (N, 21)
            (
            gaze_timestamp, world_index, confidence, 
            norm_pos_x, norm_pos_y, 
            base_data, 
            gaze_point_3d_x, gaze_point_3d_y, gaze_point_3d_z, 
            eye_center0_3d_x, eye_center0_3d_y, eye_center0_3d_z, 
            gaze_normal0_x, gaze_normal0_y, gaze_normal0_z, 
            eye_center1_3d_x, eye_center1_3d_y, eye_center1_3d_z, 
            gaze_normal1_x, gaze_normal1_y, gaze_normal1_z
            )
        - pupil (M, 34)
            (
            pupil_timestamp, world_index, eye_id, confidence, 
            norm_pos_x, norm_pos_y,
            diameter, 
            method,
            ellipse_center_x, ellipse_center_y, ellipse_axis_a, ellipse_axis_b, ellipse_angle, 
            diameter_3d, 
            model_confidence, model_id, 
            sphere_center_x, sphere_center_y, sphere_center_z, sphere_radius, 
            circle_3d_center_x, circle_3d_center_y, circle_3d_center_z, 
            circle_3d_normal_x, circle_3d_normal_y, circle_3d_normal_z, 
            circle_3d_radius, theta, phi, 
            projected_sphere_center_x, projected_sphere_center_y, projected_sphere_axis_a, projected_sphere_axis_b, 
            projected_sphere_angle
            )
        - blinks
            (id, start_timestamp, duration, end_timestamp, start_frame_index, index, end_frame_index, confidence, filter_response, base_data)
        - annotation
            (Anger, Tenderness, Disgust, Sadness)
    - data{#}.questionnaires
    - data{#}.subject_info
"""