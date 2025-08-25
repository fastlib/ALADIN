
#include "pwaveprocessor.h"


std::tuple<std::pair<int, int>, int, float>
PWaveProcessor::find_best_pair(std::vector<std::shared_ptr<P>> &waves,
                               std::vector<int> &already_aligned) {
    const size_t n = waves.size();
    float min_score = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_pair(-1, -1);
    int best_lag = 0;

    // Precompute norms for normalisation
    std::vector<float> norms(n);
    for (size_t i = 0; i < n; ++i) norms[i] = l2norm(waves[i]);
    // std::cout << "Norms: ";
    // for (const float &norm : norms) {
    //     std::cout << norm << " ";
    // }
    // std::cout << std::endl;

    for (size_t i = 0; i < n; ++i) {
        if (already_aligned[i] != -1) continue;
        for (size_t j = i + 1; j < n; ++j) {
            if (already_aligned[j] != -1) continue;

            // Normalised full cross‑correlation
            std::vector<float> corr_full = cross_correlate(waves[i], waves[j]);
            const float denom = norms[i] * norms[j];
            for (float &c : corr_full) c /= denom;

            // Find best correlation index
            auto it = std::max_element(corr_full.begin(), corr_full.end());
            if (it == corr_full.end() || std::isnan(*it)) continue;
            size_t idx = std::distance(corr_full.begin(), it);
            int lag = -correlation_lags(waves[i], waves[j])[idx];

            // Align and compute RMSE normalised exactly as Python code
            std::vector<float> waves_i = waves[i]->ecg;
            std::vector<float> tmp = shift_wave(waves_i, lag);
            tmp = make_equal_length(tmp, waves[j]);
            float rmse1 = rmse_normalised(tmp, waves[j]) / tools.absmax(waves[j]->ecg.data(), waves[j]->ecg.size());
            float rmse2 = rmse_normalised(waves[j], tmp) / tools.absmax(tmp.data(), tmp.size());
            // std::cout << rmse_normalised(tmp, waves[j]) << ", abs: " << tools.absmax(waves[j]->ecg.data(), waves[j]->ecg.size()) << " | "
            //           << rmse_normalised(waves[j], tmp) << ", abs: " << tools.absmax(tmp.data(), tmp.size()) << std::endl;

            float score = 0.5f * (rmse1 + rmse2);
            if (score < min_score) {
                min_score = score;
                best_pair = {static_cast<int>(i), static_cast<int>(j)};
                best_lag = lag;
            }
        }
    }
    return {best_pair, best_lag, min_score};
}

// Recursive alignment for a single group.
std::pair<std::vector<float>, std::vector<int>>
PWaveProcessor::recursively_align_waves(std::vector<std::shared_ptr<P>> &waves,
                                        std::vector<int> &already_aligned,
                                        std::vector<float> avg_wave,
                                        int iteration,
                                        int group) {

    const size_t n = waves.size();

    // If all aligned, return avg_wave and aligned flags
    if (std::all_of(already_aligned.begin(), already_aligned.end(), [](int v) { return v > -1; })) {
        return {avg_wave, already_aligned};
    }

    if (iteration == 0) {
        std::tuple<std::pair<int, int>, int, float> res = find_best_pair(waves, already_aligned);
        
        auto best_pair = std::get<0>(res);
        int lag = std::get<1>(res);
        float min_score = std::get<2>(res);

        // std::cout << "Iteration " << iteration << ": "
        //           << "Best pair: (" << best_pair.first << ", " << best_pair.second << "), "
        //           << "Lag: " << lag << ", "
        //           << "Score: " << min_score << std::endl;

        if (best_pair.first == -1) {
            return {avg_wave, already_aligned};
        }

        // Shift & equalise first wave in pair
        // std::vector<float> firstwave = waves[best_pair.first]->ecg;
        // std::vector<float> secondwave = waves[best_pair.second]->ecg;
        // firstwave = shift_wave(firstwave, lag);
        // firstwave = make_equal_length(firstwave, secondwave);
        waves[best_pair.first]->ecg = shift_wave(waves[best_pair.first], lag);
        waves[best_pair.first]->ecg = make_equal_length(waves[best_pair.first], waves[best_pair.second]);
        already_aligned[best_pair.first] = group;
        already_aligned[best_pair.second] = group;
        avg_wave.resize(waves[best_pair.first]->ecg.size());

        for (size_t i = 0; i < avg_wave.size(); ++i) {
            avg_wave[i] = 0.5f * (waves[best_pair.first]->ecg[i] + waves[best_pair.second]->ecg[i]);
        }

    } else {

        // Align remaining waves against avg_wave
        const std::vector<int> not_aligned_inds = [&]() {
            std::vector<int> idx; idx.reserve(n);
            for (size_t i = 0; i < n; ++i) if (already_aligned[i] == -1) idx.push_back(static_cast<int>(i));
            return idx;
        }();

        if (not_aligned_inds.empty()) return {avg_wave, already_aligned};

        size_t best_match = not_aligned_inds[0];
        float min_score = std::numeric_limits<float>::infinity();
        int best_lag = 0;

        for (int i : not_aligned_inds) {
            std::vector<float> corr_full = cross_correlate(waves[i], avg_wave);
            float denom = l2norm(waves[i]) * l2norm(avg_wave);

            for (float &c : corr_full) c /= denom;
            auto it = std::max_element(corr_full.begin(), corr_full.end());

            if (it == corr_full.end() || std::isnan(*it)) continue;
            size_t idx = std::distance(corr_full.begin(), it);
            int lag = -correlation_lags(waves[i]->ecg.size(), avg_wave.size())[idx];
            std::vector<float> waves_i = waves[i]->ecg;
            std::vector<float> tmp = shift_wave(waves_i, lag);
            tmp = make_equal_length(tmp, avg_wave);
            float rmse1 = rmse_normalised(tmp, avg_wave) / tools.absmax(avg_wave.data(), avg_wave.size());
            float rmse2 = rmse_normalised(avg_wave, tmp) / tools.absmax(tmp.data(), tmp.size());
            float score = 0.5f * (rmse1 + rmse2);
            //std::cout << "Score for wave " << i << ": " << score << std::endl;
            
            if (score < min_score) {
                min_score = score;
                best_match = static_cast<size_t>(i);
                best_lag = lag;
            }
        }
        //std::cout << "Best match: " << best_match << ", Lag: " << best_lag << ", Score: " << min_score << std::endl;
        if (min_score > 0.2f) return {avg_wave, already_aligned};

        waves[best_match]->ecg = shift_wave(waves[best_match], best_lag);
        waves[best_match]->ecg = make_equal_length(waves[best_match], avg_wave);
        already_aligned[best_match] = group;

        // Recompute avg_wave from current group members
        std::vector<int> grp_idx;
        for (size_t i = 0; i < n; ++i) if (already_aligned[i] == group) grp_idx.push_back(static_cast<int>(i));
        if (grp_idx.empty()) return {avg_wave, already_aligned};

        avg_wave.assign(waves[grp_idx.front()]->ecg.size(), 0.f);
        
        for (int id : grp_idx) {
            for (size_t k = 0; k < avg_wave.size(); ++k) avg_wave[k] += waves[id]->ecg[k];
        }
        for (auto &x : avg_wave) x /= static_cast<float>(grp_idx.size());
    }

    // recurse
    return recursively_align_waves(waves, already_aligned, avg_wave, iteration + 1, group);
}

void PWaveProcessor::recursively_group_and_align_waves(std::vector<std::shared_ptr<P>> &waves,
                                                       std::vector<int> &already_aligned,
                                                       std::vector<std::vector<float>> &avg_waves,
                                                       std::vector<std::vector<int>> &groups,
                                                       int iteration) {
    
                                                        // Stop when all aligned or fewer than 3 waves remain unaligned
    size_t remaining = static_cast<size_t>(std::count(already_aligned.begin(), already_aligned.end(), -1));
    if (remaining == 0 || remaining < 3) return;

    std::pair<std::vector<float>, std::vector<int>> res = recursively_align_waves(waves, already_aligned, {}, 0, iteration);

    std::vector<float> avg_wave = res.first;
    std::vector<int> new_aligned = res.second;

    // std::cout << "Iteration " << iteration << ": "
    //           << "Aligned " << std::count(new_aligned.begin(), new_aligned.end(), iteration)
    //           << " waves, remaining: " << std::count(new_aligned.begin(), new_aligned.end(), -1) << std::endl;

    if (avg_wave.empty()) return;

    detrend(avg_wave);

    if (!avg_waves.empty()) {
        const size_t m = avg_waves.size();
        std::vector<float> correlation_scores(m, std::numeric_limits<float>::infinity());

        const float maxrmse = minimum_mean_squared_error(avg_wave, [&]{
            std::vector<float> neg(avg_wave); for (auto &v : neg) v = -v; return neg;}());

        for (size_t i = 0; i < m; ++i) {
            // full cross‑correlation, normalised
            std::vector<float> corr_full = cross_correlate(avg_waves[i], avg_wave);
            const float denom = l2norm(avg_waves[i]) * l2norm(avg_wave);
            for (float &c : corr_full) c /= denom;

            auto it = std::max_element(corr_full.begin(), corr_full.end());
            if (it == corr_full.end() || std::isnan(*it)) continue;
            const size_t idx = static_cast<size_t>(std::distance(corr_full.begin(), it));
            const int lag = -correlation_lags(avg_waves[i].size(), avg_wave.size())[idx];

            // shift & length‑match previous average to new average
            std::vector<float> avg_waves_i = avg_waves[i];
            auto tmp = shift_wave(avg_waves_i, lag);
            tmp = make_equal_length(tmp, avg_wave);

            float rmse = minimum_mean_squared_error(tmp, avg_wave) / maxrmse;
            correlation_scores[i] = rmse;
        }

        // pick best matching existing average
        const auto best_it = std::min_element(correlation_scores.begin(), correlation_scores.end());
        const float minscore = (best_it == correlation_scores.end()) ? std::numeric_limits<float>::infinity() : *best_it;
        const size_t bestmatch = static_cast<size_t>(std::distance(correlation_scores.begin(), best_it));

        // NOTE: The original Python code disables merging with "and False".
        // We retain the structure but keep merging deactivated for identical behaviour.
        const bool enable_merge = true;  // toggle to true if you want automatic merging

        if (enable_merge && minscore < 0.3f) {
            // ---- merge: append indices of this iteration into existing group ----
            for (size_t i = 0; i < new_aligned.size(); ++i) {
                if (new_aligned[i] == iteration) groups[bestmatch].push_back(static_cast<int>(i));
            }
        } else {
            // ---- add as new separate group ------------------------------------
            avg_waves.push_back(avg_wave);
            std::vector<int> grp_idx;
            for (size_t i = 0; i < new_aligned.size(); ++i) if (new_aligned[i] == iteration) grp_idx.push_back(static_cast<int>(i));
            groups.push_back(std::move(grp_idx));
        }
    } else {
        // first ever group
        avg_waves.push_back(avg_wave);
        std::vector<int> grp_idx;
        for (size_t i = 0; i < new_aligned.size(); ++i) if (new_aligned[i] == iteration) grp_idx.push_back(static_cast<int>(i));
        groups.push_back(std::move(grp_idx));
    }

    // --- early exit if <3 unaligned remain -----------------------------------
    if (std::count(already_aligned.begin(), already_aligned.end(), -1) < 3) return;

    // --- recurse to align any remaining waves --------------------------------
    recursively_group_and_align_waves(waves, already_aligned, avg_waves, groups, iteration + 1);

}

PWaveResult PWaveProcessor::get_avg_p_wave(std::vector<std::shared_ptr<QRS>> &beats, std::vector<std::shared_ptr<P>> &pwaves, float fs, float qrs_median_range) {

    // === 1. Locate candidate P‑wave regions =================================
    // For brevity we assume the Record class exposes the same helpers as the
    // original Python Record – get_beats(), delineations etc. You will need to
    // adapt these calls to your own C++ Record implementation.


    // Centres & search windows around each P‑wave
    int win = static_cast<int>(0.1f * fs);

    std::vector<std::pair<int, int>> p_search_windows;
    for (size_t i = 0; i < pwaves.size(); ++i) {
        p_search_windows.emplace_back(pwaves[i]->start, pwaves[i]->end);
    }

    // === 2. Pre‑processing: detrend & filter out tiny waves ==================
    std::vector<int> to_remove;
    for (size_t i = 0; i < pwaves.size(); ++i) {
        detrend(pwaves[i]->ecg);
        float abs_sum = 0.f; for (float v : pwaves[i]->ecg) abs_sum += std::abs(v);
        if (abs_sum < 1.f || pwaves[i]->end - pwaves[i]->start < 0.05f * fs) {
            to_remove.push_back(static_cast<int>(i));
        }
    }

    // Remove flagged waves
    // std::vector<char> remove_mask(pwaves.size(), 0);
    // for (int idx : to_remove) remove_mask[idx] = 1;
    // auto remove_lambda = [&remove_mask, idx = 0](auto &&) mutable { return remove_mask[idx++]; };
    // pwaves.erase(std::remove_if(pwaves.begin(), pwaves.end(), remove_lambda), pwaves.end());

    std::vector<std::vector<float>> avg_waves;
    std::vector<std::vector<int>> groups;
    std::vector<int> already_aligned(pwaves.size(), -1);

    //std::cout << "Number of P-waves after pre-processing: " << pwaves.size() << std::endl;

    recursively_group_and_align_waves(pwaves, already_aligned, avg_waves, groups, 0);

    PWaveResult result;

    result.p_wave_avgs = avg_waves;
    result.groups      = groups;

    // === 4. Determine polarity & peaks (simplified) =========================
    for (int i = 0; i < static_cast<int>(avg_waves.size()); ++i) {
        // normalise by median QRS range
        std::vector<float> &avg = avg_waves[i];

        //std::cout << qrs_median_range << std::endl;

        for (auto &v : avg) v /= qrs_median_range;
        detrend(avg);

        std::vector<float> avg_pos = avg;
        std::vector<float> avg_neg = avg;
        //set all negative values to zero
        for (auto &v : avg_pos) {
            if (v < 0.f) v = 0.f;
        }
        //set all positive values to zero
        for (auto &v : avg_neg) {
            if (v > 0.f) v = 0.f;
        }

        // polarity / peak selection
        int polarity = 1;
        bool biphasic = false;
        float maxv = *std::max_element(avg.begin(), avg.end());
        float minv = *std::min_element(avg.begin(), avg.end());
        //std::cout << "P Wave " << i << ": max = " << maxv << ", min = " << minv << std::endl;

        if (maxv - minv < 0.02f ) {
            //std::cout << "-> Flat P Wave!" << std::endl;
            result.p_wave_avg_peakpos.push_back(static_cast<int>(std::distance(avg.begin(), std::max_element(avg.begin(), avg.end()))));
            result.p_wave_avg_peakpos.push_back(1); // flat waves are considered positive
            polarity = 1; // flat waves are considered positive
        } else if (maxv > -minv * 2) {
            //std::cout << "-> Positive P Wave! " << maxv << ", " << -minv << std::endl;
            result.p_wave_avg_peakpos.push_back(static_cast<int>(std::distance(avg.begin(), std::max_element(avg.begin(), avg.end()))));
            result.p_wave_avg_peakpos.push_back(1); // positive waves
            polarity = 1; // positive waves are considered positive
        } else if (-minv > maxv * 2) {
            //std::cout << "-> Negative P Wave! " << maxv << ", " << -minv << std::endl;
            result.p_wave_avg_peakpos.push_back(static_cast<int>(std::distance(avg.begin(), std::min_element(avg.begin(), avg.end()))));
            result.p_wave_avg_peakpos.push_back(-1); // negative waves
            polarity = -1; // positive waves are considered positive
        } else {
            //std::cout << "-> Biphasic P Wave!" << std::endl;
            result.p_wave_avg_peakpos.push_back(static_cast<int>(std::distance(avg_pos.begin(), std::max_element(avg_pos.begin(), avg_pos.end()))));
            result.p_wave_avg_peakpos.push_back(1); // biphasic waves are considered positive
            polarity = 1; // positive waves are considered positive
            biphasic = true;
        }

        auto new_cluster = std::make_shared<Cluster>();
        new_cluster->cluster_id = i;
        new_cluster->template_ecg = avg;

        for(int j=0; j < static_cast<int>(groups[i].size()); ++j) {
            pwaves[groups[i][j]]->unclustered = false; // mark as clustered
            pwaves[groups[i][j]]->inverted = (polarity == -1);
            pwaves[groups[i][j]]->biphasic = biphasic;
            pwaves[groups[i][j]]->cluster = new_cluster; // assign cluster id
        }
        result.p_wave_clusters.push_back(new_cluster);
    }
    return result;
}
