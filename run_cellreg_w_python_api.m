function run_cellreg(data_path, microns_per_pixel, alignment_type, reference_session_index)

    clearvars -except data_path microns_per_pixel alignment_type reference_session_index
    memory_efficient_run = 1;

    % Setting paths
    fileroot = data_path;
    results_directory = fullfile(fileroot, 'Results');
    figures_directory = fullfile(results_directory, 'Figures');
    if exist(figures_directory, 'dir') ~= 7
        mkdir(figures_directory);
    end

    figures_visibility = 'on';

    % Load all footprint files dynamically
    mat_files = dir(fullfile(fileroot, '*.mat'));
    number_of_sessions = numel(mat_files);
    file_names = cell(1, number_of_sessions);
    
    for file_index = 1: number_of_sessions
        file_names{file_index} = fullfile(mat_files(file_index).folder, mat_files(file_index).name);
    end

    %for it = 1:number_of_sessions
    %    file_names{it} = fullfile(fileroot, mat_files(it).name);
    %end

    if memory_efficient_run
        temp_dir = fullfile(figures_directory, 'temp'); 
        if ~exist(temp_dir, 'dir')
            mkdir(temp_dir);
        end
    end

    %% Stage 1 - Load spatial footprints
    disp('Stage 1 - Loading sessions')
    if memory_efficient_run
        spatial_footprints = file_names;
    else
        [spatial_footprints,number_of_sessions] = load_multiple_sessions(file_names);
    end

    [footprints_projections] = compute_footprints_projections(spatial_footprints);
    plot_all_sessions_projections(footprints_projections, figures_directory, figures_visibility);
    disp('Done')
    clear footprints_projections

    %% Stage 2 - Align sessions
    use_parallel_processing = true;
    maximal_rotation = 30;
    transformation_smoothness = 2;
    %reference_session_index = 1;

    disp('Stage 2 - Aligning sessions')
    if memory_efficient_run
        normalized_spatial_footprints = normalize_spatial_footprints(spatial_footprints, temp_dir);
    else
        normalized_spatial_footprints = normalize_spatial_footprints(spatial_footprints);
    end
    clear spatial_footprints

    [adjusted_spatial_footprints,adjusted_FOV,adjusted_x_size,adjusted_y_size,adjustment_zero_padding] = adjust_FOV_size(normalized_spatial_footprints);
    clear normalized_spatial_footprints

    [adjusted_footprints_projections] = compute_footprints_projections(adjusted_spatial_footprints);
    [centroid_locations] = compute_centroid_locations(adjusted_spatial_footprints, microns_per_pixel);
    [centroid_projections] = compute_centroids_projections(centroid_locations, adjusted_spatial_footprints);

    sufficient_correlation_centroids = 0.2;
    sufficient_correlation_footprints = 0.3;

    if strcmp(alignment_type,'Translations and Rotations')
        [spatial_footprints_corrected,centroid_locations_corrected,footprints_projections_corrected,centroid_projections_corrected,maximal_cross_correlation,alignment_translations,overlapping_FOV] = align_images(adjusted_spatial_footprints, centroid_locations, adjusted_footprints_projections, centroid_projections, adjusted_FOV, microns_per_pixel, reference_session_index, alignment_type, sufficient_correlation_centroids, sufficient_correlation_footprints, use_parallel_processing, maximal_rotation);
    elseif strcmp(alignment_type,'Non-rigid')
        [spatial_footprints_corrected,centroid_locations_corrected,footprints_projections_corrected,centroid_projections_corrected,maximal_cross_correlation,alignment_translations,overlapping_FOV,displacement_fields] = align_images(adjusted_spatial_footprints, centroid_locations, adjusted_footprints_projections, centroid_projections, adjusted_FOV, microns_per_pixel, reference_session_index, alignment_type, sufficient_correlation_centroids, sufficient_correlation_footprints, use_parallel_processing, transformation_smoothness);
    else
        [spatial_footprints_corrected,centroid_locations_corrected,footprints_projections_corrected,centroid_projections_corrected,maximal_cross_correlation,alignment_translations,overlapping_FOV] = align_images(adjusted_spatial_footprints, centroid_locations, adjusted_footprints_projections, centroid_projections, adjusted_FOV, microns_per_pixel, reference_session_index, alignment_type, sufficient_correlation_centroids, sufficient_correlation_footprints, use_parallel_processing);
    end

    [all_projections_correlations,number_of_cells_per_session] = evaluate_data_quality(spatial_footprints_corrected, centroid_projections_corrected, footprints_projections_corrected, maximal_cross_correlation, alignment_translations, reference_session_index, sufficient_correlation_footprints, alignment_type);

    if strcmp(alignment_type,'Non-rigid')
        plot_alignment_results(adjusted_spatial_footprints, centroid_locations, spatial_footprints_corrected, centroid_locations_corrected, adjusted_footprints_projections, footprints_projections_corrected, reference_session_index, all_projections_correlations, maximal_cross_correlation, alignment_translations, overlapping_FOV, alignment_type, number_of_cells_per_session, figures_directory, figures_visibility, displacement_fields);
    else
        plot_alignment_results(adjusted_spatial_footprints, centroid_locations, spatial_footprints_corrected, centroid_locations_corrected, adjusted_footprints_projections, footprints_projections_corrected, reference_session_index, all_projections_correlations, maximal_cross_correlation, alignment_translations, overlapping_FOV, alignment_type, number_of_cells_per_session, figures_directory, figures_visibility);
    end

    if use_parallel_processing
        delete(gcp('nocreate'));
    end

    disp('Done')

    %% Stage 3 (part a) - Calculating the similarities distributions from the data:
    % This stage uses the ditribtuions of centroid distance and spatial correlations
    % to compute the probabilities of neighboring cell-pairs to be the same cell (P_same).

    % part a includes the calculation of the distributions of centroid distances and spatial
    % correlations from the data.

    % Defining the parameters for the probabilstic modeling:
    maximal_distance=12; % cell-pairs that are more than 12 micrometers apart are assumed to be different cells
    normalized_maximal_distance=maximal_distance/microns_per_pixel;
    p_same_certainty_threshold=0.95; % certain cells are those with p_same>threshld or <1-threshold

    % Computing correlations and distances across days:
    disp('Stage 3 - Calculating a probabilistic model of the data')
    [number_of_bins,centers_of_bins] = estimate_number_of_bins(adjusted_spatial_footprints,normalized_maximal_distance);

    [all_to_all_indexes,all_to_all_spatial_correlations,all_to_all_centroid_distances,...
        neighbors_spatial_correlations,neighbors_centroid_distances,neighbors_x_displacements,...
        neighbors_y_displacements,NN_spatial_correlations,NNN_spatial_correlations,...
        NN_centroid_distances,NNN_centroid_distances]=...
        compute_data_distribution(spatial_footprints_corrected,centroid_locations_corrected,...
        normalized_maximal_distance);

    % Plotting the (x,y) displacements:
    plot_x_y_displacements(neighbors_x_displacements,neighbors_y_displacements,...
        microns_per_pixel,normalized_maximal_distance,number_of_bins,centers_of_bins,...
        figures_directory,figures_visibility);
    disp('Part a done')

    %% Stage 3 (part b) - Compute a probabilistic model:
    % Modeling the data as a weighted sum of same cells and different cells,
    % and estimating the attainable registration accuracy:

    disp('Calculating a probabilistic model of the data')
    % Modeling the distribution of centroid distances:
    [centroid_distances_model_parameters,p_same_given_centroid_distance,...
        centroid_distances_distribution,centroid_distances_model_same_cells,...
        centroid_distances_model_different_cells,centroid_distances_model_weighted_sum,...
        MSE_centroid_distances_model,centroid_distance_intersection]=...
        compute_centroid_distances_model(neighbors_centroid_distances,microns_per_pixel,centers_of_bins);

    % Modeling the distribution of spatial correlations:
    [spatial_correlations_model_parameters,p_same_given_spatial_correlation,...
        spatial_correlations_distribution,spatial_correlations_model_same_cells,...
        spatial_correlations_model_different_cells,spatial_correlations_model_weighted_sum,...
        MSE_spatial_correlations_model,spatial_correlation_intersection]=...
        compute_spatial_correlations_model(neighbors_spatial_correlations,centers_of_bins);

    % estimating registration accuracy:
    [p_same_centers_of_bins,uncertain_fraction_centroid_distances,...
        cdf_p_same_centroid_distances,false_positive_per_distance_threshold,...
        true_positive_per_distance_threshold,uncertain_fraction_spatial_correlations,...
        cdf_p_same_spatial_correlations,false_positive_per_correlation_threshold,...
        true_positive_per_correlation_threshold]=...
        estimate_registration_accuracy(p_same_certainty_threshold,neighbors_centroid_distances,...
        centroid_distances_model_same_cells,centroid_distances_model_different_cells,...
        p_same_given_centroid_distance,centers_of_bins,neighbors_spatial_correlations,...
        spatial_correlations_model_same_cells,spatial_correlations_model_different_cells,...
        p_same_given_spatial_correlation);

    % Checking which model is better according to a defined cost function:
    [best_model_string]=choose_best_model(MSE_centroid_distances_model,...
        centroid_distances_model_same_cells,centroid_distances_model_different_cells,...
        p_same_given_centroid_distance,MSE_spatial_correlations_model,...
        spatial_correlations_model_same_cells,spatial_correlations_model_different_cells,...
        p_same_given_spatial_correlation);

    % Plotting the probabilistic models and estimated registration accuracy:
    plot_models(centroid_distances_model_parameters,NN_centroid_distances,...
        NNN_centroid_distances,centroid_distances_distribution,...
        centroid_distances_model_same_cells,centroid_distances_model_different_cells,...
        centroid_distances_model_weighted_sum,centroid_distance_intersection,...
        centers_of_bins,microns_per_pixel,normalized_maximal_distance,figures_directory,...
        figures_visibility,spatial_correlations_model_parameters,NN_spatial_correlations,...
        NNN_spatial_correlations,spatial_correlations_distribution,...
        spatial_correlations_model_same_cells,spatial_correlations_model_different_cells,...
        spatial_correlations_model_weighted_sum,spatial_correlation_intersection)

    plot_estimated_registration_accuracy(p_same_centers_of_bins,p_same_certainty_threshold,...
        p_same_given_centroid_distance,centroid_distances_distribution,...
        cdf_p_same_centroid_distances,uncertain_fraction_centroid_distances,...
    true_positive_per_distance_threshold,false_positive_per_distance_threshold,...
    centers_of_bins,normalized_maximal_distance,microns_per_pixel,figures_directory,...
    figures_visibility,p_same_given_spatial_correlation,spatial_correlations_distribution,...
    cdf_p_same_spatial_correlations,uncertain_fraction_spatial_correlations,...
    true_positive_per_correlation_threshold,false_positive_per_correlation_threshold)

    % Computing the P_same for each neighboring cell-pair according to the different models:
    [all_to_all_p_same_centroid_distance_model,all_to_all_p_same_spatial_correlation_model]=...
        compute_p_same(all_to_all_centroid_distances,p_same_given_centroid_distance,...
        centers_of_bins,all_to_all_spatial_correlations,p_same_given_spatial_correlation);

    disp('Done')

    %% Stage 4 - Initial cell registration
    % This stage performs an initial cell registration according to an
    % optimized threshold of either spatial correlations or centroid distances.

    % Defining the parameters for initial registration:
    initial_registration_type=best_model_string; % either 'Spatial correlation', 'Centroid distance', or 'best_model_string';
    % The threshold that corresponds to p_same=0.5 is automatically chosen.
    % if a specific distance/correlation threshold is to be used - change the
    % initial threshold manually in the next few lines.

    % Computing the initial registration according to a simple threshold:
    disp('Stage 4 - Performing initial registration')
    if strcmp(initial_registration_type,'Spatial correlation') % if spatial correlations are used
        if exist('spatial_correlation_intersection','var')
            initial_threshold=spatial_correlation_intersection; % the threshold for p_same=0.5;
        else
            initial_threshold=0.65; % a fixed correlation threshold not based on the model
        end   
            [cell_to_index_map,registered_cells_spatial_correlations,non_registered_cells_spatial_correlations]=...
                initial_registration_spatial_correlations(normalized_maximal_distance,...
                initial_threshold,spatial_footprints_corrected,centroid_locations_corrected);
            plot_initial_registration(cell_to_index_map,number_of_bins,spatial_footprints_corrected,initial_registration_type,figures_directory,figures_visibility,registered_cells_spatial_correlations,non_registered_cells_spatial_correlations)
    else % if centroid distances are used
        if exist('centroid_distance_intersection','var')
            initial_threshold=centroid_distance_intersection; % the threshold for p_same=0.5;
        else
            initial_threshold=5; % a fixed distance threshold not based on the model
        end
        normalized_distance_threshold=initial_threshold/microns_per_pixel;
        [cell_to_index_map,registered_cells_centroid_distances,non_registered_cells_centroid_distances]=...
            initial_registration_centroid_distances(normalized_maximal_distance,normalized_distance_threshold,centroid_locations_corrected);
        plot_initial_registration(cell_to_index_map,number_of_bins,spatial_footprints_corrected,initial_registration_type,figures_directory,figures_visibility,registered_cells_centroid_distances,non_registered_cells_centroid_distances,microns_per_pixel,normalized_maximal_distance)
    end

    disp([num2str(size(cell_to_index_map,1)) ' cells were found'])
    disp('Done')

    %% Stage 5 - Final cell registration:
    % This stage performs the final cell registration with a clustering algorithm 
    % that is based on the probability model for same cells and different cells. 
    % P_same can be either according to centroid distances or spatial
    % correlations.

    % Defining the parameters for final registration:
    registration_approach='Probabilistic'; % either 'Probabilistic' or 'Simple threshold'
    model_type=best_model_string; % either 'Spatial correlation' or 'Centroid distance'
    p_same_threshold=0.5; % only relevant if probabilistic approach is used

    % Deciding on the registration threshold:
    transform_data=false;
    if strcmp(registration_approach,'Simple threshold') % only relevant if a simple threshold is used
        if strcmp(model_type,'Spatial correlation')
            if exist('spatial_correlation_intersection','var')
                final_threshold=spatial_correlation_intersection; % the threshold for p_same=0.5;
            else
                final_threshold=0.65; % a fixed correlation threshold not based on the model
            end
        elseif strcmp(model_type,'Centroid distance')
            if exist('centroid_distance_intersection','var')
                final_threshold=centroid_distance_intersection; % the threshold for p_same=0.5;
            else
                final_threshold=5; % a fixed distance threshold not based on the model
            end
            normalized_distance_threshold=(maximal_distance-final_threshold)/maximal_distance;
            transform_data=true;
        end
    else
        final_threshold=p_same_threshold;
    end

    % Registering the cells with the clustering algorithm:
    disp('Stage 5 - Performing final registration')
    if strcmp(registration_approach,'Probabilistic')    
        if strcmp(model_type,'Spatial correlation')
            [optimal_cell_to_index_map,registered_cells_centroids,cell_scores,...
                cell_scores_positive,cell_scores_negative,cell_scores_exclusive,...
                p_same_registered_pairs]=...
                cluster_cells(cell_to_index_map,all_to_all_p_same_spatial_correlation_model,...
                all_to_all_indexes,normalized_maximal_distance,p_same_threshold,...
                centroid_locations_corrected,registration_approach,transform_data);
        elseif strcmp(model_type,'Centroid distance')
            [optimal_cell_to_index_map,registered_cells_centroids,cell_scores,...
                cell_scores_positive,cell_scores_negative,cell_scores_exclusive,...
                p_same_registered_pairs]=...
                cluster_cells(cell_to_index_map,all_to_all_p_same_centroid_distance_model,...
                all_to_all_indexes,normalized_maximal_distance,p_same_threshold,...
                centroid_locations_corrected,registration_approach,transform_data);
        end
        plot_cell_scores(cell_scores_positive,cell_scores_negative,cell_scores_exclusive,cell_scores,p_same_registered_pairs,figures_directory,figures_visibility)
    elseif strcmp(registration_approach,'Simple threshold')
        if strcmp(model_type,'Spatial correlation')
            [optimal_cell_to_index_map,registered_cells_centroids]=...
                cluster_cells(cell_to_index_map,all_to_all_spatial_correlations,...
                all_to_all_indexes,normalized_maximal_distance,final_threshold,...
                centroid_locations_corrected,registration_approach,transform_data);
        elseif strcmp(model_type,'Centroid distance')
            [optimal_cell_to_index_map,registered_cells_centroids]=...
                cluster_cells(cell_to_index_map,all_to_all_centroid_distances,...
                all_to_all_indexes,normalized_maximal_distance,normalized_distance_threshold,...
                centroid_locations_corrected,registration_approach,transform_data);
        end
    end
    [is_in_overlapping_FOV]=check_if_in_overlapping_FOV(registered_cells_centroids,overlapping_FOV);

    % Plotting the registration results with the cell maps from all sessions:
    plot_all_registered_projections(spatial_footprints_corrected,...
        optimal_cell_to_index_map,figures_directory,figures_visibility)

    % move corrected_spatial footprints 
    if memory_efficient_run
        for file_n = 1:length(spatial_footprints_corrected)
            split_name = strsplit(spatial_footprints_corrected{file_n},filesep);
            f_name = split_name{end};
            footprints = get_spatial_footprints(spatial_footprints_corrected{file_n});
            footprints = footprints.load_footprints;
            footprints = footprints.footprints;
            footprint = mat_to_sparse_cell(footprints);
            spatial_footprints_corrected{file_n} = fullfile(results_directory,f_name);
            save(fullfile(results_directory,f_name),'footprint');
        end
    end
    % saving the final registration results:
    disp('Saving the results')
    cell_registered_struct=struct;
    cell_registered_struct.cell_to_index_map=optimal_cell_to_index_map;
    if strcmp(registration_approach,'Probabilistic')
        cell_registered_struct.cell_scores=cell_scores';
        cell_registered_struct.true_positive_scores=cell_scores_positive';
        cell_registered_struct.true_negative_scores=cell_scores_negative';
        cell_registered_struct.exclusivity_scores=cell_scores_exclusive';
        cell_registered_struct.p_same_registered_pairs=p_same_registered_pairs';
    end
    cell_registered_struct.is_cell_in_overlapping_FOV=is_in_overlapping_FOV';
    cell_registered_struct.registered_cells_centroids=registered_cells_centroids';
    cell_registered_struct.centroid_locations_corrected=centroid_locations_corrected';
    cell_registered_struct.spatial_footprints_corrected=spatial_footprints_corrected';
    cell_registered_struct.spatial_footprints_corrected=spatial_footprints_corrected';
    cell_registered_struct.alignment_x_translations=alignment_translations(1,:);
    cell_registered_struct.alignment_y_translations=alignment_translations(2,:);
    if strcmp(alignment_type,'Translations and Rotations')
        cell_registered_struct.alignment_rotations=alignment_translations(3,:);
    end
    cell_registered_struct.adjustment_x_zero_padding=adjustment_zero_padding(1,:);
    cell_registered_struct.adjustment_y_zero_padding=adjustment_zero_padding(2,:);
    save(fullfile(results_directory,['cellRegistered_' datestr(clock,'yyyymmdd_HHMMss') '.mat']),'cell_registered_struct','-v7.3')

    % Saving a log file with all the chosen parameters:
    comments=''; % anything written here will be added to the log file
    if strcmp(registration_approach,'Probabilistic')
        if strcmp(model_type,'Spatial correlation')
            save_log_file(results_directory,file_names,microns_per_pixel,adjusted_x_size,adjusted_y_size,alignment_type,reference_session_index,maximal_distance,number_of_bins,initial_registration_type,initial_threshold,registration_approach,model_type,final_threshold,optimal_cell_to_index_map,cell_registered_struct,comments,uncertain_fraction_spatial_correlations,false_positive_per_correlation_threshold,true_positive_per_correlation_threshold,MSE_spatial_correlations_model)
        elseif strcmp(model_type,'Centroid distance')
            save_log_file(results_directory,file_names,microns_per_pixel,adjusted_x_size,adjusted_y_size,alignment_type,reference_session_index,maximal_distance,number_of_bins,initial_registration_type,initial_threshold,registration_approach,model_type,final_threshold,optimal_cell_to_index_map,cell_registered_struct,comments,uncertain_fraction_centroid_distances,false_positive_per_distance_threshold,true_positive_per_distance_threshold,MSE_centroid_distances_model)
        end
    elseif strcmp(registration_approach,'Simple threshold')
        if strcmp(model_type,'Spatial correlation')
            save_log_file(results_directory,file_names,microns_per_pixel,adjusted_x_size,adjusted_y_size,alignment_type,reference_session_index,maximal_distance,number_of_bins,initial_registration_type,initial_threshold,registration_approach,model_type,final_threshold,optimal_cell_to_index_map,cell_registered_struct,comments)
        elseif strcmp(model_type,'Centroid distance')
            save_log_file(results_directory,file_names,microns_per_pixel,adjusted_x_size,adjusted_y_size,alignment_type,reference_session_index,maximal_distance,number_of_bins,initial_registration_type,initial_threshold,registration_approach,model_type,final_threshold,optimal_cell_to_index_map,cell_registered_struct,comments)
        end
    end
    disp([num2str(size(optimal_cell_to_index_map,1)) ' cells were found'])
    disp('End of cell registration procedure')

    if memory_efficient_run
        rmdir(temp_dir,'s');
    end
end
