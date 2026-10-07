#conda activate base
#cd /d D:\Projects\low-resource-language-review

# clear
rm -rf meta_data_pretrain
rm -rf meta_data_posttrain

rm -rf meta_data_pretrain_filtered
rm -rf meta_data_posttrain_filtered

rm -rf meta_data_pretrain_grouped
rm -rf meta_data_posttrain_grouped

rm -rf meta_data_pretrain_reduced
rm -rf meta_data_posttrain_reduced

rm -rf meta_data_pretrain_clustered
rm -rf meta_data_posttrain_clustered

# map
python map_datasets.py data_summaries_pretrain meta_data_pretrain
python map_datasets.py data_summaries meta_data_posttrain # FT data between 1990-2024.8
python map_datasets.py data_summaries_new meta_data_posttrain # FT data between 2024.9-2026.9

# filter
python filter_datasets.py meta_data_pretrain meta_data_pretrain_filtered
python filter_datasets.py meta_data_posttrain meta_data_posttrain_filtered

# group
python group_datasets.py meta_data_pretrain_filtered meta_data_pretrain_grouped --data_type pretrain
python group_datasets.py meta_data_posttrain_filtered meta_data_posttrain_grouped --data_type posttrain

# reduce
python reduce_attributes.py meta_data_pretrain_grouped meta_data_pretrain_reduced --data_type pretrain
python reduce_attributes.py meta_data_posttrain_grouped meta_data_posttrain_reduced --data_type posttrain

# grid search the best clustering parameter
#/Users/jiluo-wendu/anaconda3/bin/python -B cluster_grid_search.py --alphas 0.025 0.05 0.075 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --n-clusters 2 3 4 5 6 --seeds 42 43 44 45 46 47 48 49 50 51
#--data-type pretrain

# cluster
python cluster_analysis.py meta_data_pretrain_reduced meta_data_pretrain_clustered --data_type pretrain
python cluster_analysis.py meta_data_posttrain_reduced meta_data_posttrain_clustered --data_type posttrain


# join
#python -B join_language_clusters.py --data-type pretrain
#python -B join_language_clusters.py --data-type posttrain
python -B join_language_clusters.py