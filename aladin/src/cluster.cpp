
#include "cluster.h"


Cluster::Cluster() {
    cluster_id = -1;
    last_updated = -1;
    template_beat = nullptr;
    closest = nullptr;
    peak = 0;
}

Cluster::Cluster(int _cluster_id, int _last_updated, std::shared_ptr<Component> _template_beat) {
    cluster_id = _cluster_id;
    last_updated = _last_updated;
    template_beat = _template_beat;
    closest = nullptr;
    peak = 0;
}

void Cluster::set_template(std::shared_ptr<Component> beat) { 
    template_beat = beat; last_updated = beat->id; 
}

void Cluster::set_width() {
    // Set the width of the cluster based on the template beat
    vector<float> onsets;
    vector<float> offsets;
    for (int i = 0; i < beats.size(); ++i) {
        onsets.push_back((float)beats[i]->wave_start);
        offsets.push_back((float)beats[i]->wave_end);
    }
    onset = Tools::mean(onsets.data(), onsets.size());
    offset = Tools::mean(offsets.data(), offsets.size());
    width = offset - onset;
    for (int i = 0; i < beats.size(); ++i) {
        beats[i]->width = beats[i]->wave_end - beats[i]->wave_start + 1;
    }
}

void Cluster::determine_peak() {

    float max_delta = 0.0f;
    for (int i = 0; i < template_beat->dominant_points.size(); ++i) {
        float delta = template_beat->dominant_points[i].delta;
        if (delta > max_delta) {
            max_delta = delta;
            peak = template_beat->dominant_points[i].j;
        }
    }
}

Clustering::Clustering(float rho_min, float alpha, float beta) {
    this->rho_min = rho_min;
    this->alpha = alpha;
    this->beta = beta;
    this->tools = Tools();
    this->verbose = false;
    //std::cout << "Rho min: " << rho_min << ", Alpha: " << alpha << ", Beta: " << beta << std::endl;
}

void Clustering::cluster_qrs(vector<std::shared_ptr<QRS>> &beats, float fs) {
    this->fs = fs;

    this->theta = int(0.1 * fs);
    this->tau = 15; //temporal context of 15 beats

    //std::cout << "Theta: " << this->theta << std::endl;


    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < beats.size(); ++i) {
        //std::cout << "Processing beat " << i << std::endl;
        use_beat.push_back(true);
        this->process_beat(beats[i]);
        //break;
    }
    //return;
    auto tstop = chrono::high_resolution_clock::now();

    //std:: cout << "Time taken to process beats: " << chrono::duration_cast<chrono::milliseconds>(tstop - t0).count() << " ms" << std::endl;
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < beats.size(); ++i) {
        if (!use_beat[i]) {
            //std::cout << "Skipping beat " << i << " as it is not used" << std::endl;
            continue;
        }
        //std::cout << "Comparing beats " << i << " and 0" << std::endl;
        beats[i]->cluster = nullptr;
        this->assign_to_cluster(beats[i]);
    }
    tstop = chrono::high_resolution_clock::now();

    // std::cout << "Time taken to compare beats: " << chrono::duration_cast<chrono::milliseconds>(tstop - t1).count() << " ms" << std::endl;
    // std::cout << "Time taken in total: " << chrono::duration_cast<chrono::milliseconds>(tstop - t0).count() << " ms" << std::endl;

    //process_clusters();
    if (verbose) {
        list_clusters();
    }
}


std::vector<std::tuple<int, int, double>> Clustering::completeLinkageClustering(std::vector<std::vector<double>>& distanceMatrix, bool distance_matrix=true) {
    int n = distanceMatrix.size();
    std::vector<int> cluster(n);
    for (int i = 0; i < n; ++i) cluster[i] = i;

    std::vector<std::tuple<int, int, double>> merges;

    int numClusters = n;
    for(int i=0; i<n; i++) {
        if (!use_beat[i]) {
            numClusters--;
            cluster[i] = -1;  // Mark as unused
        }
    }

    while (numClusters > 1) {
        double maxSim = 0;
        double minSim = std::numeric_limits<double>::max();
        int a = -1, b = -1;

        // Find the closest pair of clusters
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (use_beat[i] && use_beat[j] && cluster[i] != cluster[j] && distanceMatrix[i][j] > maxSim && distance_matrix == false) {
                    maxSim = distanceMatrix[i][j];
                    a = i;
                    b = j;
                }
                if (use_beat[i] && use_beat[j] && cluster[i] != cluster[j] && distanceMatrix[i][j] < minSim && distance_matrix == true) {
                    minSim = distanceMatrix[i][j];
                    a = i;
                    b = j;
                }
            }
        }

        if (a == -1 || b == -1) break;

        int clusterA = cluster[a];
        int clusterB = cluster[b];

        // Merge clusters: unify cluster IDs
        for (int i = 0; i < n; ++i) {
            if (cluster[i] == clusterB) {
                // if(distance_matrix==false) {
                //     std::cout << "Updating cluster ID for " << i << " from " << clusterB << " to " << clusterA << " with similarity " << maxSim << std::endl;
                // } else {
                //     std::cout << "Updating cluster ID for " << i << " from " << clusterB << " to " << clusterA << " with dissimilarity " << minSim << std::endl;
                // }
                cluster[i] = clusterA;
            }
        }

        if(distance_matrix==false) {
            merges.emplace_back(a, b, maxSim);
        } else {
            merges.emplace_back(a, b, minSim);
        }
        //std::cout << "Starting to update distances, a=" << a << std::endl;

        for (int j=0; j<n; ++j) {
            if (cluster[j] != clusterA) {
                double minDist = std::numeric_limits<double>::max();
                double maxDist = 0.0f;
                for (int i=0; i<n; ++i) {
                    if (cluster[i] == clusterA && distance_matrix == false) {
                        minDist = min(minDist,min(distanceMatrix[j][i], distanceMatrix[a][b]));
                    }
                    if (cluster[i] == clusterA && distance_matrix == true) {
                        maxDist = max(maxDist,max(distanceMatrix[j][i], distanceMatrix[a][b]));
                    }
                }
                for (int i=0; i<n; ++i) {
                    if (cluster[i] == clusterA && distance_matrix == false) {
                        distanceMatrix[j][i] = minDist;
                        distanceMatrix[i][j] = minDist;
                    }
                    if (cluster[i] == clusterA && distance_matrix == true) {
                        distanceMatrix[j][i] = maxDist;
                        distanceMatrix[i][j] = maxDist;
                    }
                }
            }
        }

        --numClusters;
    }

    return merges;  // Can be used to build dendrogram
}

// void Clustering::cluster_qrs(vector<std::shared_ptr<QRS>> &beats, float _fs) {

//     fs = _fs;
//     theta = int(0.1 * fs);

//     if(beats.size() == 0) {
//         std::cout << "No QRS waves detected" << std::endl;
//         return;
//     }

//     use_beat = new bool[beats.size()];
//     memset(use_beat, 1, beats.size() * sizeof(bool));
    
//     auto t0 = chrono::high_resolution_clock::now();
//     for (int i = 0; i < beats.size(); ++i) {
//         //std::cout << "Processing beat " << i << std::endl;
//         this->process_beat(beats[i]);
//         //break;
//     }
//     auto t1 = chrono::high_resolution_clock::now();
//     std:: cout << "Time taken to process beats: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

//     float min_similarity = 0.4f;
//     bool done = false;
//     int num_clusters = 0;
//     int iter = 0;

//     std::vector<std::vector<double>> distanceMatrix(beats.size(), std::vector<double>(beats.size(), 0.0));
//     for (int i = 0; i < beats.size(); i++) {
//         for (int j = i + 1; j < beats.size(); j++) {
//             float similarity = 0.0f;
//             float norm = 0.0f;
//             this->compare_qrs(beats[i], beats[j], similarity, norm);
//             distanceMatrix[i][j] = norm;
//             distanceMatrix[j][i] = norm;
//         }
//     }
//     auto t2 = chrono::high_resolution_clock::now();
//     std:: cout << "Time taken to calculate distance matrix: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
//     std::vector<std::tuple<int, int, double>> merges = completeLinkageClustering(distanceMatrix, false);

//     for (int i = 0; i < merges.size(); ++i) {
//         int a = std::get<0>(merges[i]);
//         int b = std::get<1>(merges[i]);
//         double similarity = std::get<2>(merges[i]);

//         if (similarity > min_similarity) {
//             if (beats[a]->cluster == nullptr && beats[b]->cluster == nullptr) {
//                 std::cout << "Creating new cluster for beats " << a << " and " << b << std::endl;
//                 std::shared_ptr<Cluster> new_cluster = create_new_cluster(beats[a], nullptr);
//                 update_cluster(new_cluster, beats[b]);
//                 beats[a]->cluster = new_cluster;
//                 beats[b]->cluster = new_cluster;
//             } else if (beats[a]->cluster != nullptr && beats[b]->cluster == nullptr) {
//                 std::cout << "Adding beat " << b << " to cluster " << beats[a]->cluster->cluster_id << std::endl;
//                 update_cluster(beats[a]->cluster, beats[b]);
//                 beats[b]->cluster = beats[a]->cluster;
//             } else if (beats[a]->cluster == nullptr && beats[b]->cluster != nullptr) {
//                 std::cout << "Adding beat " << a << " to cluster " << beats[b]->cluster->cluster_id << std::endl;
//                 update_cluster(beats[b]->cluster, beats[a]);
//                 beats[a]->cluster = beats[b]->cluster;
//             } else if (beats[a]->cluster != beats[b]->cluster) {
//                 std::cout << "Merging clusters " << beats[a]->cluster->cluster_id << " and " << beats[b]->cluster->cluster_id << std::endl;
//                 this->merge_clusters(beats[a]->cluster, beats[b]->cluster);
//             }
//         }
//     }

//     for (int i = 0; i < beats.size(); ++i) {
//         if (beats[i]->cluster == nullptr) {
//             std::shared_ptr<Cluster> new_cluster = create_new_cluster(beats[i], nullptr);
//             update_cluster(new_cluster, beats[i]);
//             beats[i]->cluster = new_cluster;
//         }
//     }

//     auto t3 = chrono::high_resolution_clock::now();
//     std:: cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(t3 - t0).count() << " ms" << std::endl;

//     process_clusters();
//     list_clusters();

//     //delete the clusterer
//     delete[] use_beat;
// }

void Clustering::cluster_p(vector<std::shared_ptr<P>> &beats, float _fs) {

    fs = _fs;
    theta = int(0.2 * fs);

    if(beats.size() == 0) {
        //std::cout << "No P waves detected" << std::endl;
        return;
    }
    
    auto t0 = chrono::high_resolution_clock::now();
    for (int i=0; i<beats.size(); ++i) {
        process_p_wave(beats[i]);
        use_beat.push_back(true);
    }
    for (int i = 0; i < beats.size(); ++i) {
        //std::cout << "Processing beat " << i << std::endl;
        this->process_beat(beats[i]);
        //break;
    }

    auto t1 = chrono::high_resolution_clock::now();
    //std:: cout << "Time taken to process beats: " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;

    float max_distance = 0.3f;
    bool done = false;
    int num_clusters = 0;
    int iter = 0;

    std::vector<std::vector<double>> distanceMatrix(beats.size(), std::vector<double>(beats.size(), 0.0));
    for (int i = 0; i < beats.size(); i++) {
        for (int j = i + 1; j < beats.size(); j++) {
            if (use_beat[i] == false || use_beat[j] == false) {
                continue;
            }
            float similarity = 0.0f;
            float norm = 0.0f;
            this->compare_p(beats[i], beats[j], similarity, norm);
            distanceMatrix[i][j] = norm;
            distanceMatrix[j][i] = norm;
        }
    }
    auto t2 = chrono::high_resolution_clock::now();

    // for(int i=0; i<beats.size(); i++) {
    //     for(int j=0; j<beats.size(); j++) {
    //         std::cout << distanceMatrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    //std:: cout << "Time taken to calculate distance matrix: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms" << std::endl;
    std::vector<std::tuple<int, int, double>> merges = completeLinkageClustering(distanceMatrix, true);

    for (int i = 0; i < merges.size(); ++i) {
        int a = std::get<0>(merges[i]);
        int b = std::get<1>(merges[i]);
        double distance = std::get<2>(merges[i]);

        if (distance < max_distance) {
            if (beats[a]->cluster == nullptr && beats[b]->cluster == nullptr) {
                //std::cout << "Creating new cluster for beats " << a << " and " << b << std::endl;
                std::shared_ptr<Cluster> new_cluster = create_new_cluster(beats[a], nullptr);
                update_cluster(new_cluster, beats[b]);
                //std::cout << "Updating cluster ID for " << a << " and " << b << std::endl;
                beats[a]->cluster = new_cluster;
                beats[b]->cluster = new_cluster;
            } else if (beats[a]->cluster != nullptr && beats[b]->cluster == nullptr) {
                //std::cout << "Adding beat " << b << " to cluster " << beats[a]->cluster->cluster_id << std::endl;
                update_cluster(beats[a]->cluster, beats[b]);
                //std::cout << "Updating cluster ID for " << b << " from " << beats[b]->cluster->cluster_id << " to " << beats[a]->cluster->cluster_id << std::endl;
                beats[b]->cluster = beats[a]->cluster;
            } else if (beats[a]->cluster == nullptr && beats[b]->cluster != nullptr) {
                //std::cout << "Adding beat " << a << " to cluster " << beats[b]->cluster->cluster_id << std::endl;
                update_cluster(beats[b]->cluster, beats[a]);
                //std::cout << "Updating cluster ID for " << a << " from " << beats[a]->cluster->cluster_id << " to " << beats[b]->cluster->cluster_id << std::endl;
                beats[a]->cluster = beats[b]->cluster;
            } else if (beats[a]->cluster != beats[b]->cluster) {
                //std::cout << "Merging clusters " << beats[a]->cluster->cluster_id << " and " << beats[b]->cluster->cluster_id << std::endl;
                this->merge_clusters(beats[a]->cluster, beats[b]->cluster);
                //std::cout << "Updating cluster ID for " << std::endl;
            }
        } else {
            if(beats[a]->cluster == nullptr) {
                beats[a]->unclustered = true;
            }
            if(beats[b]->cluster == nullptr) {
                beats[b]->unclustered = true;
            }
        }
    }

    //process_clusters();
    if (verbose) {
        list_clusters();
    }

    //delete[] use_beat;
}

float Clustering::cosine_angle(vector<float> &x, int i, int j, int k) {
    // Calculate the cosine angle between two vectors
    // This function is a placeholder and should be implemented
    double v_ij_x = i - j;
    double v_ij_y = x[i] - x[j];
    double v_jk_x = k - j;
    double v_jk_y = x[k] - x[j];

    double dot = v_ij_x * v_jk_x + v_ij_y * v_jk_y;
    double norm_ij = std::sqrt(v_ij_x * v_ij_x + v_ij_y * v_ij_y);
    double norm_jk = std::sqrt(v_jk_x * v_jk_x + v_jk_y * v_jk_y);
    double denom = norm_ij * norm_jk;

    if (denom == 0.0)
        return 1.0;  // Treat as flat line
    return dot / denom;
}

float Clustering::delta_ecg(vector<float> &x, int i, int j) {
    // Calculate the difference between two points in the ECG signal
    return std::abs(x[i] - x[j]);
}

vector<double> Clustering::delta_ecg(vector<float> &x, int j, vector<int> &indices) {
    // Calculate the difference between point j and all points in indices
    vector<double> delta_q(indices.size());
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        delta_q[idx] = std::abs(x[j] - x[indices[idx]]);
    }
    return delta_q;
}

void Clustering::calculate_I_sets(vector<float> &x, int j, vector<int> &I_minus, vector<int> &I_plus) {

    // // q = np.array(q)
    // // n = len(q)
    int n = x.size();
    I_minus.clear();
    I_plus.clear();

    // Calculate I_j^- 
    for (int i = std::max(0, j - theta); i < j; ++i) {
        float max_diff = std::numeric_limits<float>::lowest();
        for (int i_a = i + 1; i_a < j; ++i_a) {
            for (int i_b = i_a + 1; i_b < j; ++i_b) {
                float delta_b = std::abs(x[j] - x[i_b]);
                float delta_a = std::abs(x[j] - x[i_a]);
                float diff = delta_b - delta_a;
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        if (max_diff < this->rho_min) {
            I_minus.push_back(i);
        }
    }

    // Calculate I_j^+
    for (int k = j + 1; k < std::min(n, j + theta + 1); ++k) {
        float max_diff = std::numeric_limits<float>::lowest();
        for (int i_a = j + 1; i_a < k; ++i_a) {
            for (int i_b = j + 1; i_b < i_a; ++i_b) {
                float delta_b = std::abs(x[j] - x[i_b]);
                float delta_a = std::abs(x[j] - x[i_a]);
                float diff = delta_b - delta_a;
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }
        if (max_diff < this->rho_min) {
            I_plus.push_back(k);
        }
    }


    // int start_i = std::max(0, j - this->theta);
    // for (int i = start_i; i < j; ++i) {
    //     int num_indices_a = j - (i + 1);
    //     if (num_indices_a <= 0) {
    //         I_minus.push_back(i);
    //         continue;
    //     }
    //     std::cout << "Processing I_minus candidate: " << i << " j:" << j << std::endl;

    //     std::vector<float> delta_q_a;
    //     std::vector<float> max_diff;
    //     for (int idx = 0; idx < num_indices_a; ++idx)
    //         delta_q_a.push_back(std::abs(x[j] - x[i + 1 + idx]));

    //     std::cout << "delta_q_a: ";
    //     for (const auto& val : delta_q_a) {
    //         std::cout << val << " ";
    //     }
    //     std::cout << std::endl;

    //     for (int idx = 0; idx < num_indices_a; ++idx) {
    //         double max_val = -1e9;
    //         for (int b = i + 1 + idx; b < j; ++b) {
    //             double delta_b = std::abs(x[j] - x[b]);
    //             double diff = delta_b - delta_q_a[idx];
    //             if (diff > max_val)
    //                 max_val = diff;
    //         }
    //         max_diff[idx] = max_val;
    //     }

    //     bool all_less = true;
    //     for (int idx = 0; idx < num_indices_a; ++idx) {
    //         if (max_diff[idx] >= rho_min) {
    //             all_less = false;
    //             break;
    //         }
    //     }

    //     if (all_less) {
    //         I_minus.push_back(i);
    //     }
    // }

    // // I_plus candidates: (j, min(n, j + theta + 1))
    // int end_k = std::min((int)x.size(), j + this->theta + 1);
    // for (int k = j + 1; k < end_k; ++k) {
    //     int num_indices_a = k - (j + 1);
    //     if (num_indices_a <= 0) {
    //         I_plus.push_back(k);
    //         continue;
    //     }

    //     std::vector<float> delta_q_a(this->theta);
    //     std::vector<float> max_diff(this->theta);
    //     for (int idx = 0; idx < num_indices_a; ++idx)
    //         delta_q_a[idx] = std::abs(x[j] - x[j + 1 + idx]);

    //     for (int idx = 0; idx < num_indices_a; ++idx) {
    //         double max_val = -1e9;
    //         for (int b = j + 1; b <= j + 1 + idx; ++b) {
    //             double delta_b = std::abs(x[j] - x[b]);
    //             double diff = delta_b - delta_q_a[idx];
    //             if (diff > max_val)
    //                 max_val = diff;
    //         }
    //         max_diff[idx] = max_val;
    //     }

    //     bool all_less = true;
    //     for (int idx = 0; idx < num_indices_a; ++idx) {
    //         if (max_diff[idx] >= rho_min) {
    //             all_less = false;
    //             break;
    //         }
    //     }

    //     if (all_less)
    //         I_plus.push_back(k);
    // }
}

void Clustering::calculate_curvature_and_dominance(vector<float> &x, int j, float& max_cosine, int& dominance_min, int& dominance_plus, int& support_win_min, int& support_win_plus) {
    
    std::vector<int> I_minus;
    std::vector<int> I_plus;

    this->calculate_I_sets(x, j, I_minus, I_plus);

    // std::cout << "I_minus: ";
    // for (int i=0; i<I_minus.size(); i++) {
    //     std::cout << I_minus[i] << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "I_plus: ";
    // for (int i=0; i<I_plus.size(); i++) {
    //     std::cout << I_plus[i] << " ";
    // }   
    // std::cout << std::endl;

    max_cosine = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < I_minus.size(); ++i) {
        for (int k = 0; k < I_plus.size(); ++k) {
            float cos_val = this->cosine_angle(x, I_minus[i], j, I_plus[k]);
            if (cos_val > max_cosine) {
                max_cosine = cos_val;
            }
        }
    }

    vector<int> dominance_min_set;
    vector<int> dominance_plus_set;

    for (int k = 0; k < I_plus.size(); ++k) {
        float max_cos = -std::numeric_limits<float>::infinity();
        int max_i = -1;
        for (int i = 0; i < I_minus.size(); ++i) {
            float cos_val = this->cosine_angle(x, I_minus[i], j, I_plus[k]);
            if (cos_val > max_cos) {
                max_cos = cos_val;
                max_i = I_minus[i];
            }
        }
        dominance_min_set.push_back(max_i);
    }

    for (int i = 0; i < I_minus.size(); ++i) {
        float max_cos = -std::numeric_limits<float>::infinity();
        int max_k = -1;
        for (int k = 0; k < I_plus.size(); ++k) {
            float cos_val = this->cosine_angle(x, I_minus[i], j, I_plus[k]);
            if (cos_val > max_cos) {
                max_cos = cos_val;
                max_k = I_plus[k];
            }
        }
        dominance_plus_set.push_back(max_k);
    }

    dominance_min = this->tools.min(dominance_min_set.data(), dominance_min_set.size());
    dominance_plus = this->tools.max(dominance_plus_set.data(), dominance_plus_set.size());

    support_win_min = this->tools.min(I_minus.data(), I_minus.size());
    support_win_plus = this->tools.max(I_plus.data(), I_plus.size());

    //std::cout << j << ", Support win min: " << support_win_min << ", Support win plus: " << support_win_plus << std::endl;
}

void Clustering::qrs_characterization(std::shared_ptr<Component> beat, vector<int> &dominance_min, vector<int> &dominance_max, vector<int> &support_win_min, vector<int> &support_win_max, vector<float> &curvature) {

    //std::cout << "Start:" << beat->start << ", End: " << beat->end << std::endl;
    int n = beat->end - beat->start;
    assert(n == beat->ecg.size());
    for (int j = 1; j < n-1; j++) {
        int dominance_min_val = 0;
        int dominance_max_val = 0;
        int support_win_min_val = 0;
        int support_win_max_val = 0;
        float curvature_val = 0.0;
        this->calculate_curvature_and_dominance(beat->ecg, j, curvature_val, dominance_min_val, dominance_max_val, support_win_min_val, support_win_max_val);
        curvature[j] = curvature_val;
        dominance_min[j] = dominance_min_val;
        dominance_max[j] = dominance_max_val;
        support_win_min[j] = support_win_min_val;
        support_win_max[j] = support_win_max_val;

        //std::cout << "J: " << j << ", Dominance min: " << dominance_min_val << ", Dominance max: " << dominance_max_val << ", Support win min: " << support_win_min_val << ", Support win max: " << support_win_max_val << std::endl;
    }
}

void Clustering::calculate_dominant_points(std::shared_ptr<Component> beat, vector<int> &dominance_min, vector<int> &dominance_max, vector<int> &support_win_min, vector<int> &support_win_max, vector<float> &curvature) {

    int n = beat->end - beat->start;

    //std::cout << "number of dominant points: " << beat->dominant_points.size() << std::endl;

    for (int j = 1; j < n - 1; ++j) {
        int start = dominance_min[j];
        int end = dominance_max[j];
        float max_curvature = curvature[j];
        //std::cout << "J: " << j << ", Dominance min: " << start << ", Dominance max: " << end << ", Curvature: " << max_curvature << std::endl;

        bool is_dominant = true;
        for (int a = start+1; a < end; a++) {
            if (curvature[a] > max_curvature) {
                is_dominant = false;
                break;
            }
        }

        if (is_dominant) {
            //std::cout << "[" << start << ", " << j << ", " << end << "] " << "Curvature: " << max_curvature << std::endl;
            float delta_start = beat->ecg[j] - beat->ecg[start];
            float delta_end = beat->ecg[j] - beat->ecg[end];
            //std::cout << "J: " << beat.start+j << ", Start: " << start << ", End: " << end << ", Delta start: " << delta_start << ", Delta end: " << delta_end << std::endl;

            float abs_start = std::abs(delta_start);
            float abs_end = std::abs(delta_end);

            float min_delta_qj = (abs_start < abs_end) ? abs_start : abs_end;
            float real_delta = (abs_start < abs_end) ? delta_start : delta_end;

            // int dis_from_boundary = std::min(std::abs((j) - beat->wave_start), std::abs((j) - beat->wave_end));
            // if (j < beat->wave_start-5 || j > beat->wave_end+5) {
            //     min_delta_qj = 0.0;
            //     std::cout << "too far from wave boundary: " << dis_from_boundary << std::endl;
            // }
            // if (j < 5 || j > n - 5) {
            //     min_delta_qj = 0.0;
            //     std::cout << "too close to boundary: " << j << std::endl;
            // }
            //std::cout << start << " " << j << " " << end << ", Min delta Qj: " << min_delta_qj << std::endl;
            if (min_delta_qj > this->rho_min*3) {
                //std::cout << "[" << start << ", " << beat->start+j << ", " << end << "] " << "Curvature: " << max_curvature << std::endl;
                DominantPoint dp;
                dp.j = j;
                dp.curvature = max_curvature;
                dp.delta = min_delta_qj;
                dp.real_delta = real_delta;
                dp.convex = (beat->ecg[start] > beat->ecg[j] && beat->ecg[end] > beat->ecg[j]) ? true : false;
                //std::cout << dominance_min[j] << " " << dominance_max[j] << " " << j << " " << support_win_min[j] << " " << support_win_max[j] << std::endl;
                dp.d[0] = start;
                dp.d[1] = end;
                dp.s[0] = support_win_min[j];
                dp.s[1] = support_win_max[j];

                beat->dominant_points.push_back(dp);
            }
        }
    }
    //std::cout << "number of dominant points: " << beat->dominant_points.size() << std::endl;

}

void Clustering::determine_support_region(std::shared_ptr<Component> beat) {

    vector<int> supports_min;
    vector<int> supports_max;

    for (int i = 0; i < beat->dominant_points.size(); ++i) {
        DominantPoint& dp = beat->dominant_points[i];

        int r_min = dp.d[0];
        int r_max = dp.d[1];
        int i_min = dp.s[0];
        int i_max = dp.s[1];
        int j = dp.j;

        int j_min = r_min;
        int j_max = r_max;

        // Search left side
        for (int i = i_min; i < r_min; ++i) {
            bool valid = true;
            for (int a = i; a < r_min; ++a) {
                if (this->delta_ecg(beat->ecg, j, a) < this->delta_ecg(beat->ecg, j, a + 1)) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                j_min = i;
                break;
            }
        }


        for (int k = r_max; k < i_max+1; ++k) {
            bool valid = true;
            for (int a = r_max; a < k+1; ++a) {
                if (this->delta_ecg(beat->ecg, j, a-1) > this->delta_ecg(beat->ecg, j, a)) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                j_max = k;
            }
        }

        dp.sup[0] = j_min;
        dp.sup[1] = j_max;

        supports_min.push_back(j_min);
        supports_max.push_back(j_max);
    }

    beat->support_region_start = this->tools.min(supports_min.data(), supports_min.size());
    beat->support_region_end = this->tools.max(supports_max.data(), supports_max.size());
}

void Clustering::process_p_wave(std::shared_ptr<Component> wave) {
    int x0 = 0;
    int x1 = wave->ecg.size();
    float y0 = wave->ecg[x0];
    float y1 = wave->ecg[x1-1];
    float m = (y1 - y0) / (x1 - x0);
    float b = y0 - m * x0;

    for (int j = x0; j < x1; ++j) {
        wave->ecg[j] -= (m * j + b);
    }
    if (wave->end - wave->start < 0.05*fs) {
        use_beat[wave->id] = false;
        return;
    }

}

void Clustering::process_beat(std::shared_ptr<Component> beat) {
    // Process the beat data here
    // This function should extract the beat data from the ecg signal
    // and perform clustering based on the parameters provided.

    beat->dominant_points.clear();
    int n = beat->end - beat->start;
    assert(n > 0);
    //std::cout << "Processing beat from " << n << std::endl;
    vector<int> dominance_min(n);
    vector<int> dominance_max(n);
    vector<int> support_win_min(n);
    vector<int> support_win_max(n);
    vector<float> curvature(n);

    this->qrs_characterization(beat, dominance_min, dominance_max, support_win_min, support_win_max, curvature);
    this->calculate_dominant_points(beat, dominance_min, dominance_max, support_win_min, support_win_max, curvature);
    
    if (beat->dominant_points.size() == 0) {
        //std::cout << "No dominant points found for beat " << beat->id << std::endl;
        use_beat[beat->id] = false;
        return;
    }

    this->determine_support_region(beat);
    
    for (int i = 0; i < beat->dominant_points.size(); ++i) {
        DominantPoint& dp = beat->dominant_points[i];
        //std::cout << "Dominant point: " << i << ", " << dp.j << ", " << dp.sup[0] << ", " << dp.sup[1] << std::endl;
    }
}

bool Clustering::find_most_similar_cluster(std::shared_ptr<Component> beat, int tstart, int tend, std::shared_ptr<Cluster> &cluster, std::shared_ptr<Cluster> &closest) {

    vector<std::shared_ptr<Cluster>> cluster_ctx;
    this->get_clusters_inside_temporal_ctx(cluster_ctx, tstart, tend);
    //std::cout << "Clusters inside temporal context: " << cluster_ctx.size() << std::endl;

    float norm_similarity = 0.0f;
    int cluster_sim_id = -1;
    if (cluster_ctx.size() == 0) {
        return false;
    }
    for (int i = 0; i < cluster_ctx.size(); ++i) {
        std::shared_ptr<Cluster> clstr = cluster_ctx[i];
        float sim = 0.0f;
        float norm = 0.0f;
        //std::cout << clstr << " " << clstr->cluster_id << std::endl;
        this->get_cluster_similarity(clstr, beat, sim, norm);
        //std::cout << "Cluster " << clstr->cluster_id << " similarity: " << sim << ", norm: " << norm << std::endl;

        if (norm > norm_similarity) {
            norm_similarity = norm;
            cluster_sim_id = i;
        }
    }
    if (cluster_sim_id == -1) {
        return false;
    }
    cluster = cluster_ctx[cluster_sim_id];

    float closest_sim = 0.0f;
    int closest_id = -1;
    for (int i = 0; i < cluster_ctx.size(); ++i) {
        float sim = 0.0f;
        float norm = 0.0f;
        std::shared_ptr<Cluster> clstr = cluster_ctx[i];
        this->get_cluster_similarity(clstr, beat, sim, norm);
        if (norm > 0.3f && i != cluster_sim_id) {
            if (sim > closest_sim) {
                closest_sim = sim;
                closest_id = i;
            }
        }
    }
    if (closest_id != -1) {
        //std::cout << "We found another closest cluster!" << std::endl;
        closest = cluster_ctx[closest_id];
    } else {
        closest = nullptr;
    }

    return true;
}

void Clustering::assign_to_cluster(std::shared_ptr<Component> beat) {
    // Assign the beat to a cluster based on the similarity
    // This function should implement the logic to assign the beat
    // to the appropriate cluster based on the parameters provided.

    //std::cout << "============== Assigning beat " << beat->id << " to a cluster ==============" << std::endl;

    std::shared_ptr<Cluster> closest;
    std::shared_ptr<Cluster> next_closest;
    this->find_most_similar_cluster(beat, 0, beat->id, closest, next_closest);

    std::shared_ptr<Cluster> inside_ctx;
    if(this->find_most_similar_cluster(beat, beat->id-this->tau, beat->id, inside_ctx, next_closest)) {

        float sim = 0.0f;
        float norm = 0.0f;
        this->get_cluster_similarity(inside_ctx, beat, sim, norm);
        

        if (norm > 0.3f) {
            //std::cout << "QRS " << beat->id << " is similar to cluster " << inside_ctx->cluster_id << "(" << sim << "," << norm << "), Next closest " << next_closest << std::endl;
            this->update_cluster(inside_ctx, beat);
            // if (next_closest != nullptr) {
            //     std::shared_ptr<Cluster> newest = (inside_ctx->last_updated > next_closest->last_updated) ? inside_ctx : next_closest;
            //     std::shared_ptr<Cluster> oldest = (inside_ctx->last_updated < next_closest->last_updated) ? inside_ctx : next_closest;
            //     newest->closest = oldest;
            //     std::cout << "More than one cluster matched, possible merging between " << newest->cluster_id << " and " << oldest->cluster_id << std::endl;
            //     this->merge_with_closest(newest, 0.5f);
            // }
        } else {
            //std::cout << "QRS " << beat->id << " is not similar to cluster inside temporal context, looking outside, norm:" << norm << std::endl;
            std::shared_ptr<Cluster> outside_ctx;
            next_closest = nullptr;
            if (this->find_most_similar_cluster(beat, 0, beat->id-this->tau, outside_ctx, next_closest)) {
                float sim = 0.0f;
                float norm = 0.0f;
                this->get_cluster_similarity(outside_ctx, beat, sim, norm);
                if (norm > 0.3f) {
                    //std::cout << "QRS " << beat->id << " is similar to cluster " << outside_ctx->cluster_id << "(" << sim << "," << norm << ")" << std::endl;
                    this->update_cluster(outside_ctx, beat);
                    // if (next_closest != nullptr) {
                    //     std::shared_ptr<Cluster> newest = (outside_ctx->last_updated > next_closest->last_updated) ? outside_ctx : next_closest;
                    //     std::shared_ptr<Cluster> oldest = (outside_ctx->last_updated < next_closest->last_updated) ? outside_ctx : next_closest;
                    //     newest->closest = oldest;
                    //     std::cout << "More than one cluster matched, possible merging between " << newest->cluster_id << " and " << oldest->cluster_id << std::endl;
                    //     this->merge_with_closest(newest, 0.5f);
                    // }
                } else {
                    //std::cout << "QRS " << beat->id << " is not similar to any cluster, creating new cluster, norm:" << norm << std::endl;
                    this->create_new_cluster(beat, closest);
                }
            } else {
                //std::cout << "No other clusters found, creating new cluster" << std::endl;
                this->create_new_cluster(beat, closest);
            }
        }
    } else {
        //std::cout << "No other clusters found, creating new cluster" << std::endl;
        this->create_new_cluster(beat, closest);
    }
}

void Clustering::compare_qrs(std::shared_ptr<Component> self, std::shared_ptr<Component> other, float &similarity, float &normalized_similarity) {
    // Compare two beats and determine if they are similar
    // This function should implement the logic to compare the beats
    // based on the parameters provided.

    //std::cout << "Comparing beats " << self->id << " and " << other->id << std::endl;
    float ps_self_other = this->piecewise_simmilarity(self, other);
    //std::cout << "Piecewise similarity between beats (self to other): " << ps_self_other << std::endl;
    float ps_other_self = this->piecewise_simmilarity(other, self);
    //std::cout << "Piecewise similarity between beats (other to self): " << ps_other_self << std::endl;
    similarity = (ps_self_other + ps_other_self);
    //float simmilarity = (ps_other_self);
    normalized_similarity = similarity / (self->dominant_points.size() + other->dominant_points.size());
}

void Clustering::compare_p(std::shared_ptr<Component> self, std::shared_ptr<Component> other, float &similarity, float &normalized_similarity) {

    vector<pair<int, int>> path;
    vector<float> aligned_x;
    vector<float> aligned_y;
    vector<float> sig_x = self->ecg;
    vector<float> sig_y = other->ecg;
    float dtw = this->tools.dtwPath(sig_x, sig_y, path, aligned_x, aligned_y);
    //normalized_similarity = dtw;

    //calculate correlation between the two aligned sigs
    // float correlation = this->tools.pearson_correlation(aligned_x, aligned_y);
    // normalized_similarity = correlation;

    float mse = 0.0f;
    for (int i = 0; i < aligned_x.size(); ++i) {
        mse += std::pow(aligned_x[i] - aligned_y[i], 2);
    }
    mse /= aligned_x.size();
    mse = sqrt(mse);

    float mse1 = mse/tools.absmax(aligned_x.data(), aligned_x.size());
    float mse2 = mse/tools.absmax(aligned_y.data(), aligned_y.size());
    normalized_similarity = (mse1 + mse2) / 2.0f;

    // rmse = np.sqrt(mean_squared_error(tmp1, avg_wave))/np.max(np.abs(avg_wave))
    // rmse += np.sqrt(mean_squared_error(avg_wave, tmp1))/np.max(np.abs(tmp1))
    // rmse /= 2

}

float Clustering::piecewise_simmilarity(std::shared_ptr<Component> self, std::shared_ptr<Component> other) {
    // Calculate the piecewise similarity between two beats
    // This function should implement the logic to calculate the piecewise
    // similarity based on the parameters provided.

    vector<float> nonconcordant_dissimilarities;
    float piecewise_similarity = 0.0f;

    vector<pair<int, int>> path;
    vector<float> aligned_x;
    vector<float> aligned_y;
    vector<float> sig_x = self->ecg;
    vector<float> sig_y = other->ecg;
    float dtw = this->tools.ddtwPath(sig_x, sig_y, path, aligned_x, aligned_y);

    for (int i = 0; i < path.size(); ++i) {
        int x = path[i].first;
        int y = path[i].second;
        //std::cout << "Path: " << i << ": " << x << ", " << y << std::endl;
        
        aligned_x[i] = sig_x[x];
        aligned_y[i] = sig_y[y];
    }


    for (int i = 0; i < self->dominant_points.size(); ++i) {
        DominantPoint& dp = self->dominant_points[i];
        int j_mid = dp.j;
        int support_min = dp.sup[0];
        int support_max = dp.sup[1];
        //std::cout << "support_min: " << support_min << ", support_max: " << support_max << std::endl;

        int qhat_j = 0;
        int qhat_j_min = 0;
        int qhat_j_max = 0;
        int qc_j = 0;
        int qc_j_min = 0;
        int qc_j_max = 0;
        for (int j = 0; j < path.size(); ++j) {
            if (path[j].first == j_mid && path[j+1].first != j_mid) {
                qhat_j = j;
                qc_j = path[j].second;
                break;
            }
        }
        for (int j = 0; j < path.size(); ++j) {
            if (path[j].first == support_min) {
                qhat_j_min = j;
                qc_j_min = path[j].second;
                break;
            }
        }
        for (int j = 0; j < path.size(); ++j) {
            if (path[j].first == support_max && j == path.size() - 1) {
                qhat_j_max = j;
                qc_j_max = path[j].second;
                break;
            } else {
                if (path[j].first == support_max && path[j+1].first != support_max) {
                    qhat_j_max = j;
                    qc_j_max = path[j].second;
                    break;
                }
            }
        }

        //std::cout << "Dp.j: " << dp.j << ", Qhat j: " << qhat_j << ", Qhat j min: " << qhat_j_min << ", Qhat j max: " << qhat_j_max << std::endl;


        
        float concordance = 0.0f;
        float local_dissimilarity = 0.0f;

        concordance = this->concordance_ratio(sig_x, sig_y, dp.sup[0], dp.j, dp.sup[1], qc_j_min, qc_j_max, dp.convex);
        //std::cout << i << " Concordance: " << concordance << ", convex:" << dp.convex << std::endl;
        local_dissimilarity = this->local_dissimilarity(aligned_x, aligned_y, qhat_j_min, qhat_j, qhat_j_max, dp.convex);
        //std::cout << i << " Local dissimilarity: " << local_dissimilarity << std::endl;


        if (local_dissimilarity != 0.0f) {
            piecewise_similarity += concordance * this->sigmoid(local_dissimilarity);
            //std::cout << "Addition: " << concordance * this->sigmoid(local_dissimilarity) << ", Concordance: " << concordance << ", Sigmoid: " << this->sigmoid(local_dissimilarity) << std::endl;
        } else {
            nonconcordant_dissimilarities.push_back(local_dissimilarity);
        }

        // std::cout << "Convex: " << dp.convex << std::endl;
        // std::cout << "Concordance: " << concordance << std::endl;
        // std::cout << "Local dissimilarity: " << local_dissimilarity << std::endl;

    }
    if (nonconcordant_dissimilarities.size() > 0)
        piecewise_similarity -= this->tools.max(nonconcordant_dissimilarities.data(), nonconcordant_dissimilarities.size());


    return piecewise_similarity;
}

float Clustering::concordance_ratio(vector<float> &q, vector<float> &qc, int q_j_min, int q_j, int q_j_max, int qc_j_min, int qc_j_max, bool isconvex) {
    // Calculate the concordance ratio between beats
    // This function should implement the logic to calculate the concordance
    // ratio based on the parameters provided.

    //std::cout << "Q j min: " << q_j_min << ", Q j: " << q_j << ", Q j max: " << q_j_max << std::endl;
    float q_j_min_val = q[q_j_min];
    float q_j_val = q[q_j];
    float q_j_max_val = q[q_j_max];
    //std::cout << "Q j min value: " << q_j_min_val << ", Q j value: " << q_j_val << ", Q j max value: " << q_j_max_val << std::endl;

    float delta_q = min(abs(q_j_min_val - q_j_val), abs(q_j_max_val - q_j_val));

    int qc_peak_index = 0;
    int qc_min_index = 0;
    int qc_max_index = 0;
    //std::cout << isconvex << std::endl;
    if (isconvex) {
        qc_peak_index = qc_j_min + this->tools.argmin(qc.data() + qc_j_min, qc_j_max - qc_j_min + 1);
        qc_min_index = qc_j_min + this->tools.argmax(qc.data() + qc_j_min, qc_peak_index - qc_j_min + 1);
        qc_max_index = qc_peak_index + this->tools.argmax(qc.data() + qc_peak_index, qc_j_max - qc_peak_index + 1);
    } else {
        qc_peak_index = qc_j_min + this->tools.argmax(qc.data() + qc_j_min, qc_j_max - qc_j_min + 1);
        qc_min_index = qc_j_min + this->tools.argmin(qc.data() + qc_j_min, qc_peak_index - qc_j_min + 1);
        qc_max_index = qc_peak_index + this->tools.argmin(qc.data() + qc_peak_index, qc_j_max - qc_peak_index + 1);
    }

    float qc_peak_val = qc[qc_peak_index];
    float qc_min_val = qc[qc_min_index];
    float qc_max_val = qc[qc_max_index];
    //std::cout << "QC peak index: " << qc_peak_index << ", QC min index: " << qc_min_index << ", QC max index: " << qc_max_index << std::endl;
    //std::cout << "QC peak value: " << qc_peak_val << ", QC min value: " << qc_min_val << ", QC max value: " << qc_max_val << std::endl;
    float delta_qc = min(abs(qc_min_val - qc_peak_val), abs(qc_max_val - qc_peak_val));

    //std::cout << "Delta Q: " << delta_q << ", Delta Qc: " << delta_qc << std::endl;


    if (delta_qc <= this->rho_min) return 0;

    return min(delta_q, delta_qc) / max(delta_q, delta_qc);
}

float Clustering::sigmoid(float x) {
    // Calculate the sigmoid function for the beats
    // This function should implement the logic to calculate the sigmoid
    // based on the parameters provided.
    return 1 - ((this->alpha*x) / sqrt((1 + (this->alpha*x)*(this->alpha*x))));
}

float Clustering::local_dissimilarity(vector<float> &q, vector<float> &qc, int qhat_j_min, int qhat_j, int qhat_j_max, bool isconvex) {
    // Calculate the local dissimilarity between beats
    // This function should implement the logic to calculate the local
    // dissimilarity based on the parameters provided.

    vector<float> a(q.size());
    for (int i = 0; i < q.size(); ++i) {
        a[i] = abs(q[i] - qc[i]);
    }

    //std::cout << "Qhat j min: " << qhat_j_min << ", Qhat j: " << qhat_j << ", Qhat j max: " << qhat_j_max << std::endl;

    float median_min = this->tools.median(a.data() + qhat_j_min, qhat_j - qhat_j_min);
    //std::cout << "Median min: " << median_min << std::endl;
    float median_plus = this->tools.median(a.data() + qhat_j, qhat_j_max - qhat_j);
    //std::cout << "Median plus: " << median_plus << std::endl;

    //std::cout << "Median min: " << median_min << ", Median plus: " << median_plus << std::endl;

    float delta_Aj_min = 0.0f;
    for (int k = qhat_j_min; k <= qhat_j; ++k) {
        delta_Aj_min += a[k];
    }
    delta_Aj_min += ((a[qhat_j_min] + a[qhat_j]) * 0.5f - (qhat_j - qhat_j_min + 1) * median_min);
    //std::cout << "Delta Aj min: " << delta_Aj_min << std::endl;

    // delta_Aj_plus
    float delta_Aj_plus = 0.0f;
    for (int k = qhat_j; k <= qhat_j_max; ++k) {
        delta_Aj_plus += a[k];
    }
    delta_Aj_plus += ((a[qhat_j_max] + a[qhat_j]) * 0.5f - (qhat_j_max - qhat_j + 1) * median_plus);
    //std::cout << "Delta Aj plus: " << delta_Aj_plus << std::endl;

    // A_min
    float A_min = 0.0f;
    for (int k = qhat_j_min; k <= qhat_j; ++k)
        A_min += q[k];
    //std:: cout << "A min: " << A_min << std::endl;
    float seg_min_max = *std::max_element(q.begin() + qhat_j_min, q.begin() + qhat_j + 1);
    if (!isconvex)
        seg_min_max = *std::min_element(q.begin() + qhat_j_min, q.begin() + qhat_j + 1);
    //std::cout << "Seg min max: " << seg_min_max << std::endl;
    A_min += (q[qhat_j_min] + q[qhat_j]) * 0.5f - (qhat_j - qhat_j_min + 1) * seg_min_max;
    A_min = std::fabs(A_min);
    //std::cout << "A min: " << A_min << std::endl;

    // A_plus
    float A_plus = 0.0f;
    for (int k = qhat_j; k <= qhat_j_max; ++k)
        A_plus += q[k];
    seg_min_max = *std::max_element(q.begin() + qhat_j, q.begin() + qhat_j_max + 1);
    if (!isconvex)
        seg_min_max = *std::min_element(q.begin() + qhat_j, q.begin() + qhat_j_max + 1);
    //std::cout << "Seg min max: " << seg_min_max << std::endl;
    A_plus += (q[qhat_j] + q[qhat_j_max]) * 0.5f - (qhat_j_max - qhat_j + 1) * seg_min_max;
    A_plus = std::fabs(A_plus);
    //std::cout << "A min: " << A_min << ", A plus: " << A_plus << std::endl;

    float local_dissimilarity = ((delta_Aj_min * delta_Aj_min) / A_min) + ((delta_Aj_plus * delta_Aj_plus) / A_plus);
    local_dissimilarity *= 1.0f / (A_min + A_plus);

    return local_dissimilarity;
}


void Clustering::get_clusters_inside_temporal_ctx(vector<std::shared_ptr<Cluster>> &cluster_ctx, int tstart, int tend) {

    // Get the clusters inside the temporal context
    // This function should implement the logic to get the clusters
    // based on the parameters provided.

    for (int i = 0; i < this->clusters.size(); ++i) {
        //std::cout << "Cluster " << clusters[i].cluster_id << ", last updated: " << clusters[i].last_updated << std::endl;
        if (clusters[i]->last_updated >= tstart && clusters[i]->last_updated <= tend) {
            cluster_ctx.push_back(clusters[i]);
        }
    }

}

void Clustering::get_clusters_similarity(std::shared_ptr<Cluster> self, std::shared_ptr<Cluster> other, float &similarity, float &normalized_similarity) {

    float ps_self_other = this->piecewise_simmilarity(self->template_beat, other->template_beat);
    //std::cout << "Self: new beat, other: cluster" << std::endl;
    float ps_other_self = this->piecewise_simmilarity(other->template_beat, self->template_beat);
    //std::cout << "Piecewise similarity between cluster and beat (self to other): " << ps_self_other << std::endl;
    //std::cout << "Piecewise similarity between cluster and beat (other to self): " << ps_other_self << std::endl;
    similarity = (ps_self_other + ps_other_self);
    normalized_similarity = similarity / (self->template_beat->dominant_points.size() + other->template_beat->dominant_points.size());
}

void Clustering::get_cluster_similarity(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat, float &similarity, float &normalized_similarity) {
    // Get the cluster similarity between the cluster and the beat
    // This function should implement the logic to get the cluster
    // similarity based on the parameters provided.
    //std::cout << cluster->template_beat->id << " " << cluster->template_beat->start << " " << cluster->template_beat->end << std::endl;
    //std::cout << "Self: cluster, other: new beat" << std::endl;
    float ps_self_other = this->piecewise_simmilarity(cluster->template_beat, beat);
    //std::cout << "Self: new beat, other: cluster" << std::endl;
    float ps_other_self = this->piecewise_simmilarity(beat, cluster->template_beat);
    //std::cout << "Piecewise similarity between cluster and beat (self to other): " << ps_self_other << std::endl;
    //std::cout << "Piecewise similarity between cluster and beat (other to self): " << ps_other_self << std::endl;
    similarity = (ps_self_other + ps_other_self);
    normalized_similarity = similarity / (cluster->template_beat->dominant_points.size() + beat->dominant_points.size());
    //std::cout << "Similarity between cluster and beat: " << normalized_similarity << std::endl;

    if (auto qrs_beat = std::dynamic_pointer_cast<QRS>(beat)) {
        float cluster_p_qrs_ratio = cluster->p_qrs_ratio;
        float beat_has_p = (float)(qrs_beat->p_wave != nullptr);
        //std::cout << "Cluster P-QRS ratio: " << cluster_p_qrs_ratio << ", Beat has P wave: " << beat_has_p << std::endl;
        float p_ratio_diff = std::abs(cluster_p_qrs_ratio - beat_has_p) * 0.3;
        normalized_similarity -= p_ratio_diff;
    }
    //std::cout << "Adjusted norm after P-QRS ratio: " << similarity << ", " << normalized_similarity << std::endl;
}

std::shared_ptr<Cluster> Clustering::create_new_cluster(std::shared_ptr<Component> beat, std::shared_ptr<Cluster> closest) {
    // Create a new cluster for the beat
    // This function should implement the logic to create a new cluster
    // based on the parameters provided.

    auto new_cluster = std::make_shared<Cluster>();
    new_cluster->cluster_id = cluster_index++;
    new_cluster->last_updated = beat->id;
    new_cluster->template_beat = std::make_shared<Component>(*beat);
    //std::cout << "New cluster created with beat ID: " << new_cluster->template_beat->id << std::endl;

    new_cluster->beats.push_back(beat);
    new_cluster->calc_p_qrs_ratio();
    new_cluster->determine_peak();
    beat->cluster = new_cluster;
    this->clusters.push_back(new_cluster);
    //std::cout << "New cluster created with ID: " << new_cluster->cluster_id << std::endl;
    return new_cluster;
}

void Clustering::merge_clusters(std::shared_ptr<Cluster> self, std::shared_ptr<Cluster> other) {
    
    this->update_cluster_template(self, other->template_beat);
    self->last_updated = other->last_updated;
    self->beats.insert(self->beats.end(), other->beats.begin(), other->beats.end());
    for (int i = 0; i < other->beats.size(); ++i) {
        other->beats[i]->cluster = self;
    }
    //remove other cluster from the list
    //std::cout << "Remove others from the list, " << other->cluster_id << ", size: " << this->clusters.size() << std::endl;
    for (int i = 0; i < this->clusters.size(); ++i) {
        if (this->clusters[i]->cluster_id == other->cluster_id) {
            //std::cout << "Found cluster to remove: " << i << std::endl;
            this->clusters.erase(this->clusters.begin() + i);
            break;
        }
    }
    //std::cout << "Cluster " << other->cluster_id << " removed from the list" << std::endl;
}

void Clustering::merge_with_closest(std::shared_ptr<Cluster> cluster, float norm_thres) {
    float sim = 0.0f;
    float norm = 0.0f;
    this->get_clusters_similarity(cluster->closest, cluster, sim, norm);
    if (norm > norm_thres) {
        //std::cout << "Clusters are similar, merging!" << std::endl;
        this->merge_clusters(cluster->closest, cluster);
    } else {
        //std::cout << "Clusters are not similar, not merging!" << std::endl;
    }
}

void Clustering::update_cluster_template(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat) {

    // Update the cluster with the new beat
    // This function should implement the logic to update the cluster
    // based on the parameters provided.

    beat->cluster = cluster;

    vector<pair<int, int>> path;
    vector<float> aligned_x;
    vector<float> aligned_y;
    vector<float> sig_x = beat->ecg;
    vector<float> dsig_x(sig_x.size());
    vector<float> sig_y = cluster->template_beat->ecg;
    vector<float> dsig_y(sig_y.size());

    for (int i = 0; i < sig_y.size()-1; ++i) {
        dsig_y[i] = sig_y[i+1] - sig_y[i];
    }
    for (int i = 0; i < sig_x.size()-1; ++i) {
        dsig_x[i] = sig_x[i+1] - sig_x[i];
    }


    float dtw = this->tools.dtwPath(dsig_x, dsig_y, path, aligned_x, aligned_y);

    int cur_ind = 0;
    int cur_n = 0;
    float running_sum = 0.0f;

    for (int i = 0; i < path.size(); ++i) {
        int x = path[i].first;
        int y = path[i].second;
        cur_n++;
        running_sum += dsig_x[x];
        if (cur_ind != y) {
            dsig_y[cur_ind] = (1-this->beta)*dsig_y[cur_ind] + this->beta*(running_sum / cur_n);
            if (cur_ind > 0)
                cluster->template_beat->ecg[cur_ind] = cluster->template_beat->ecg[cur_ind-1] + dsig_y[cur_ind-1];
            cur_ind++;
            cur_n = 0;
            running_sum = 0.0f;
        } 
        //std::cout << cluster->template_beat->ecg[i] << " ";
        //std::cout << "Path: " << x << ", " << y << std::endl;
        
        //aligned_x[i] = this->ecg[old_beat->start + x];
        //aligned_y[i] = this->ecg[beat.start + y];
    }
    //std::cout << std::endl;
    //std::cout << "Processing new beat " << std::endl;

    this->process_beat(cluster->template_beat);
    //std::cout << "Cluster template updated" << std::endl;
}

void Clustering::update_cluster(std::shared_ptr<Cluster> cluster, std::shared_ptr<Component> beat) {
    // Update the cluster with the new beat
    // This function should implement the logic to update the cluster
    // based on the parameters provided.

    this->update_cluster_template(cluster, beat);
    cluster->determine_peak();
    cluster->last_updated = beat->id;
    cluster->beats.push_back(beat);

    cluster->calc_p_qrs_ratio();

    // plt::figure_size(1500, 780);
    // plt::plot(sig_y);
    // plt::save("cluster_"+std::to_string(cluster->cluster_id)+".png");

}

void Clustering::process_clusters() {
    // Process the clusters and perform any necessary operations
    // This function should implement the logic to process the clusters
    // based on the parameters provided.

    for (int i = 0; i < this->clusters.size(); ++i) {
        this->clusters[i]->set_width();
    }
}

void Clustering::list_clusters() {

    std::cout << "======================= Clusters: ======================" << std::endl;
    py::print("======================= Clusters: ======================");

    for(int i=0; i<this->clusters.size(); i++) {
        std::shared_ptr<Cluster> cluster = this->clusters[i];
        std::cout << "Cluster ID: " << cluster->cluster_id << ", Last updated: " << cluster->last_updated << std::endl;
        for (int j=0; j<cluster->beats.size(); j++) {
            if(std::shared_ptr<QRS> beat = std::dynamic_pointer_cast<QRS>(cluster->beats[j])) {
                std::cout << "QRS ID: " << beat->id << ", Start: " << beat->start << ", End: " << beat->end << "abnormal: " << beat->abnormal << std::endl;
            }
            if(std::shared_ptr<P> beat = std::dynamic_pointer_cast<P>(cluster->beats[j])) {
                std::cout << "ECG ID: " << beat->id << ", Start: " << beat->start << ", End: " << beat->end << std::endl;
            }
        }
    }

}
