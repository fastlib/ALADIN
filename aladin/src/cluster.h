
#pragma once

#include "common.h"
#include "helpers.h"

using namespace std;

class Clustering {
    public:
        Clustering(float rho_min, float alpha, float beta);
        void cluster_qrs(vector<std::shared_ptr<QRS>> &beats, float _fs);
        void cluster_p(vector<std::shared_ptr<P>> &beats, float _fs);

        int get_number_of_clusters() { return clusters.size(); }
        std::shared_ptr<Cluster> get_cluster(int i) { return clusters[i]; }
        void list_clusters();

    private:
        void identify_qrs(bool *mask, bool *noise_mask);
        void process_beat(std::shared_ptr<Component> beat);
        void process_p_wave(std::shared_ptr<Component> wave);

        std::vector<std::tuple<int, int, double>> completeLinkageClustering(std::vector<std::vector<double>>& distanceMatrix, bool distance_matrix);
        void compare_qrs(std::shared_ptr<Component> self, std::shared_ptr<Component> other, float &similarity, float &normalized_similarity);
        void compare_p(std::shared_ptr<Component> self, std::shared_ptr<Component> other, float &similarity, float &normalized_similarity);
        float piecewise_simmilarity(std::shared_ptr<Component> self, std::shared_ptr<Component> other);
        float concordance_ratio(vector<float> &q, vector<float> &qc, int q_j_min, int q_j, int q_j_max, int qc_j_min, int qc_j_max, bool isconvex);
        float local_dissimilarity(vector<float> &q, vector<float> &qc, int qhat_j_min, int qhat_j, int qhat_j_max, bool isconvex);
        float sigmoid(float x);

        void qrs_characterization(std::shared_ptr<Component> beat, vector<int> &dominance_min, vector<int> &dominance_max, vector<int> &search_win_min, vector<int> &search_win_max, vector<float> &curvature);
        void calculate_curvature_and_dominance(vector<float> &x, int j, float& max_cosine, int& dominance_min, int& dominance_plus, int& search_win_min, int& search_win_plus);
        void calculate_I_sets(vector<float> &x, int j, vector<int> &I_minus, vector<int> &I_plus);
        float cosine_angle(vector<float> &x, int i, int j, int k);
        float delta_ecg(vector<float> &x, int i, int j);
        vector<double> delta_ecg(vector<float> &x, int j, vector<int> &indices);
        void calculate_dominant_points(std::shared_ptr<Component> beat, vector<int> &dominance_min, vector<int> &dominance_max, vector<int> &search_win_min, vector<int> &search_win_max, vector<float> &curvature);
        void determine_support_region(std::shared_ptr<Component> beat);

        std::shared_ptr<Cluster> create_new_cluster(std::shared_ptr<Component> beat, std::shared_ptr<Cluster> closest);
        void update_cluster(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat);
        void update_cluster_template(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat);
        void merge_clusters(std::shared_ptr<Cluster> self, std::shared_ptr<Cluster> other);
        void merge_with_closest(std::shared_ptr<Cluster> cluster, float norm_threshold);
        void assign_to_cluster(std::shared_ptr<Component> beat);
        bool find_most_similar_cluster(std::shared_ptr<Component> beat, int tstart, int tend, std::shared_ptr<Cluster> &cluster, std::shared_ptr<Cluster> &next_closest);
        void get_clusters_inside_temporal_ctx(vector<std::shared_ptr<Cluster>> &clusters, int tstart, int tend);
        void get_cluster_similarity(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat, float &similarity, float &normalized_similarity);
        void get_clusters_similarity(std::shared_ptr<Cluster> self, std::shared_ptr<Cluster> other, float &similarity, float &normalized_similarity);

        void process_clusters();

        Tools tools;

        float *ecg;
        float *decg;
        int size;
        float fs;

        float median_beat_range;
        int theta;
        int tau;
        float rho_min;
        float alpha;
        float beta;
        int cluster_index = 0;
        bool verbose;

        vector<std::shared_ptr<Cluster>> clusters;
        std::vector<bool> use_beat;
};