
#pragma once

#include "common.h"
#include "helpers.h"
#include "cluster.h"
#include "asra/asra.h"

using namespace std;


class Reflection {
    public:
        Reflection();
        Reflection(std::shared_ptr<Record> record);
        void initialize(std::shared_ptr<Record> record);

        void match_peaks(std::vector<int> &asra_peaks, std::vector<int> &aladin_peaks);
        void reflect_on_noise();

        void reflect_on_qrs();
        float qrs_median_range();
        float p_median_range();
        void identify_qrs();
        void calculate_rr_intervals();
        void identify_abnormality(std::shared_ptr<Cluster> cluster, int number_of_clusters);

        void reflect_on_afib();
        void extend_afib_if_possible();
        void correct_afib_for_pattern(std::regex pat);
        void correct_afib_for_ivr_or_vt();
        void correct_afib_for_p();
        void correct_afib_for_number_of_qrs(vector<tuple<int, int>> regions);
        void correct_afib_for_uncertainty(vector<tuple<int, int>> regions);
        int get_number_of_qrs_waves_inside_region(tuple<int, int> region);
        float get_median_hr_afib();
        float get_median_hr_inside_region(tuple<int, int> region);
        float get_cosen_afib();
        float get_cosen_inside_region(tuple<int, int> region);
        float get_cv_afib();
        float get_cv_inside_region(tuple<int, int> region);
        void print_regions(vector<tuple<int, int>> regions);

        void reflect_on_p_waves();
        void process_p_wave_uncertainty();
        vector<int> identify_possible_p_waves();
        float analyse_threshold(float threshold, vector<int> possible_peaks);
        vector<shared_ptr<P>> identify_p_waves();
        tuple<int, int> get_p_wave_streak(vector<float> &pibis);
        vector<int> get_predicted_locations(vector<int> &pcenters, vector<float> &pibis, tuple<int, int> longeststreak);
        tuple<int,int,int> analyse_predictions(vector<int> &predictions, vector<int> pcenters);
        void get_ecg_no_qrst();
        void identify_polarity(std::shared_ptr<Cluster> cluster, int number_of_clusters);
        void identify_polarity_of_unclustered_beats();
        void match_p_waves_to_qrs();

        void reflect(std::shared_ptr<Record> record);
        void reflect();

        void reset();

        int get_number_of_qrs_clusters() { return clusterer_qrs->get_number_of_clusters(); }
        std::shared_ptr<Cluster> get_qrs_cluster(int i) { std::shared_ptr<Cluster> shared(clusterer_qrs->get_cluster(i)); return shared; }

        int get_number_of_qrs_beats() { return beats.size(); }
        std::shared_ptr<QRS> get_qrs_beat(int i) { return beats[i]; }

        int get_number_of_p_clusters() { return clusterer_p->get_number_of_clusters(); }
        std::shared_ptr<Cluster> get_p_cluster(int i) { std::shared_ptr<Cluster> shared(clusterer_p->get_cluster(i)); return shared; }

        int get_number_of_p_beats() { return p_waves.size(); }
        std::shared_ptr<P> get_p_beat(int i) { return p_waves[i]; }


    private:
        std::unique_ptr<Clustering> clusterer_qrs;
        std::unique_ptr<Clustering> clusterer_p;
        vector<std::shared_ptr<QRS>> beats;
        vector<std::shared_ptr<P>> p_waves;
        vector<std::shared_ptr<T>> t_waves;
        std::shared_ptr<Record> record;
        std::vector<bool> afib_false_positive;
        Tools tools;
        bool verbose;
};


class BatchReflection {
    public:
        BatchReflection() {};
        void reflect_on_record(std::shared_ptr<Record> record);
        void reflect_on_batch(std::vector<std::shared_ptr<Record>> records, std::function<void(int)> progress_callback);
};