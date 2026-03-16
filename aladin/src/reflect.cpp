
#include "reflect.h"


using namespace indicators;

Reflection::Reflection() {
    verbose = false;
}

Reflection::Reflection(std::shared_ptr<Record> record) {
    this->record = record;
    this->tools = Tools();
    verbose = false;
}

void Reflection::initialize(std::shared_ptr<Record> record) {
    this->record = record;
    this->tools = Tools();
    verbose = false;
}

void Reflection::match_peaks(std::vector<int> &asra_peaks, std::vector<int> &aladin_peaks) {
    // Match ASRA peaks with Aladin peaks

    int n = record->size;
    int nbatches = n / (30 * record->fs); // 30 seconds at fs Hz
    for (int st=0; st<record->size; st += (30*record->fs)) {
        int end = min(st + (30 * (int)(record->fs)), n);

        int tp = 0;
        int fp = 0;
        int fn = 0;
        std::vector<int> mismatch_asra;
        std::vector<int> mismatch_aladin;

        for (int i = 0; i < aladin_peaks.size(); ++i) {
            if (aladin_peaks[i] >= st && aladin_peaks[i] < end) {
                bool found = false;
                for (int j = 0; j < asra_peaks.size(); ++j) {
                    if (asra_peaks[j] >= st && asra_peaks[j] < end && abs(asra_peaks[j] - aladin_peaks[i]) <= record->fs*0.15) { // Allow a tolerance of 50 samples
                        found = true;
                        break;
                    }
                }
                if (found) {
                    tp++;
                } else {
                    fp++;
                    mismatch_aladin.push_back(aladin_peaks[i]);
                }
            }
        }
        for (int j = 0; j < asra_peaks.size(); ++j) {
            if (asra_peaks[j] >= st && asra_peaks[j] < end) {
                bool found = false;
                for (int i = 0; i < aladin_peaks.size(); ++i) {
                    if (aladin_peaks[i] >= st && aladin_peaks[i] < end && abs(asra_peaks[j] - aladin_peaks[i]) <= record->fs*0.15) { // Allow a tolerance of 50 samples
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    fn++;
                    mismatch_asra.push_back(asra_peaks[j]);
                }
            }
        }
        
        float sensitivity = (float)tp / (tp + fn);
        float precision = (float)tp / (tp + fp);
        if (tp+fn == 0) {
            sensitivity = 0.0f;
        }
        if (tp+fp == 0) {
            precision = 0.0f;
        }
        float f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
        if (sensitivity == 0.0f && precision == 0.0f) {
            f1_score = 0.0f;
        }

        // std::cout << "Batch " << st / (30 * record->fs) << ": TP: " << tp << ", FP: " << fp << ", FN: " << fn 
        //           << ", Sensitivity: " << sensitivity << ", Precision: " << precision 
        //           << ", F1 Score: " << f1_score << std::endl;

        // std::cout << "Batch " << st / (10 * record->fs) << ": TP: " << tp << ", FP: " << fp << ", FN: " << fn 
        //             << ", Sensitivity: " << sensitivity << ", Precision: " << precision 
        //             << ", F1 Score: " << f1_score << std::endl;
        if (f1_score < 0.5 || sensitivity < 0.5 || precision < 0.5) {
            // std::cout << "Low F1 score detected, marking noise region from " << st << " to " << end << std::endl;
            // std::cout << mismatch_aladin.size() << " aladin mismatches: ";
            // for (int i = 0; i < mismatch_aladin.size(); ++i) {
            //     std::cout << mismatch_aladin[i]/record->fs << " ";
            // }
            // std::cout << std::endl;
            // std::cout << mismatch_asra.size() << " asra mismatches: ";
            // for (int i = 0; i < mismatch_asra.size(); ++i) {
            //     std::cout << mismatch_asra[i]/record->fs << " ";
            // }
            // std::cout << std::endl;
            
            int first_asra_mismatch = mismatch_asra.empty() ? end : mismatch_asra[0];
            int first_aladin_mismatch = mismatch_aladin.empty() ? end : mismatch_aladin[0];
            int last_asra_mismatch = mismatch_asra.empty() ? st : mismatch_asra.back();
            int last_aladin_mismatch = mismatch_aladin.empty() ? st : mismatch_aladin.back();
            int st = min(first_asra_mismatch, first_aladin_mismatch);
            int end = max(last_asra_mismatch, last_aladin_mismatch);
            // std::cout << "Marking noise region from " << st << " to " << end << std::endl;

            for (int i = st; i < end; ++i) {
                record->delineations->noise->binary[i] = 1.0f;
            }
            
            // //remove peaks from aladin_peaks that are in the noise region
            for (int i = 0; i < beats.size(); ++i) {
                if (beats[i]->get_r_wave() >= st && beats[i]->get_r_wave() < end) {
                    //remove beat from beats
                    // std::cout << "Removed beat at " << beats[i]->get_r_wave() << " due to low F1 score." << std::endl;
                    beats.erase(beats.begin() + i);
                    i--; // Adjust index after removal
                }
            }
        }
    }

    // std::cout << "beats size: " << beats.size() << std::endl;

    // tools.closingcentered(is_matched, 3);

    // std::cout << "match: ";
    // for(int i=0; i<is_matched.size(); i++) {
    //     if (is_matched[i]) {
    //         std::cout << ".";
    //     } else {
    //         std::cout << "#";
    //     }
    // }
    // std::cout << std::endl;

    // std::cout << "Matched " << std::count(is_matched.begin(), is_matched.end(), true) << " peaks out of " << aladin_peaks.size() << std::endl;
}

void Reflection::reflect_on_noise() {
    // Placeholder for noise correction logic
    //std::cout << "Correcting noise..." << std::endl;

    for(int i=0; i<record->delineations->noise->size; i++) {
        record->delineations->noise->binary[i] = (bool)(record->delineations->noise->logits[i] > 0.5); 
        if (record->delineations->noise->binary[i]) {
            record->delineations->noise->uncertainty[i] = 0.0f;
        }
        if (log(record->delineations->noise->uncertainty[i]+0.0001) > -1) {
            record->delineations->noise->binary[i] = 1.0f;
        } 
    }
    tools.closingcentered(record->delineations->noise->binary, record->delineations->noise->size, 0.5*record->fs, record->delineations->noise->binary);
    //tools.openingcentered(record->delineations->noise->binary, record->delineations->noise->size, 2*record->fs, record->delineations->noise->binary);



    //std::cout << "Noise:";
    for(int i=0; i<record->delineations->noise->size; i++) {
        if (record->delineations->noise->binary[i] == 1.0f) {
            //std::cout << "#";
            //record->filtered_ecg[i] = 0;
            //record->ecg[i] = 0;
        } else {
            //std::cout << ".";
        }
    }
    //std::cout << std::endl;
}

// void Reflection::merge_noise_regions() {

//     tools.closingcentered(record->delineations->noise->binary, record->delineations->noise->size, 2*record->fs, record->delineations->noise->binary);
//     tools.openingcentered(record->delineations->noise->binary, record->delineations->noise->size,2*record->fs, record->delineations->noise->binary);
// }

void Reflection::extend_afib_if_possible() {
    // Placeholder for extending AFIB logic
    //std::cout << "Extending AFIB if possible..." << std::endl;

    int sumafib = tools.sum(record->delineations->afib->binary, record->delineations->afib->size);
    if (sumafib == 0) return;

    //Get average number of P waves present in the local region
    std::vector<bool> no_p(record->delineations->qrs->size);
    int no_p_size = record->delineations->qrs->size;

    //Get regions of absence of P waves
    for (int i = 0; i < beats.size(); ++i) {
        int start_ind = max(0,i-2);
        int end_ind = min(i+2, (int)beats.size()-1);
        int len = end_ind - start_ind + 1;
        int num=0;
        for(int j=start_ind; j<=end_ind; ++j) {
            if (beats[j]->p_wave != nullptr) {
                num++;
            }
        }
        if ((float)num/(float)len < 0.5) {
            int st = 0;
            if (i>0) {
                st = beats[i-1]->get_r_wave();
            }

            int end = beats[i]->get_r_wave();
            for(int j=st; j<=end; ++j) {
                no_p[j] = true;
            }
        } 
    }   

    //Mask out regions where afib is already present or where we already removed it
    for(int i=0; i<record->delineations->afib->size; i++) {
        if (record->delineations->afib->binary[i] == 1) {
            no_p[i] = false;
        }
        if (afib_false_positive[i] == true) {
            no_p[i] = false;
        }
    }
    tools.openingcentered(no_p, (int)record->fs);

    //Filter out regions where there are less then 3 qrs
    vector<tuple<int, int>> no_p_regions = tools.get_regions(no_p);
    for(int i=0; i<no_p_regions.size(); i++) {
        int num_qrs = get_number_of_qrs_waves_inside_region(no_p_regions[i]);
        if (num_qrs < 3) {
            int st = std::get<0>(no_p_regions[i]);
            int end = std::get<1>(no_p_regions[i]);
            for(int j=st; j<=end; ++j) {
                no_p[j] = false;
            }
        }
    }

    //Only keep regions where there is a high certainty of afib
    for(int i=0; i<record->delineations->afib->size; i++) {
        no_p[i] = no_p[i] && (bool)(record->delineations->afib->uncertainty[i] > 0.9);
    }

    //Get median heart rate
    float mean_hr_afib = 60.0 / get_median_hr_afib();

    //Get regions of absence of P waves
    no_p_regions = tools.get_regions(no_p);

    for(int i=0; i<no_p_regions.size(); i++) {
        int st = std::get<0>(no_p_regions[i]);
        int end = std::get<1>(no_p_regions[i]);
        float mean_hr = 60.0 / get_median_hr_inside_region(no_p_regions[i]);

        bool equal_hr = true;
        bool equal_hrv = true;
        int num_qrs = get_number_of_qrs_waves_inside_region(no_p_regions[i]);
        if (std::abs(mean_hr_afib - mean_hr) > mean_hr_afib * 0.1) {
            equal_hr = false;
        }

        if (num_qrs > 8) {
            float cosen_afib = get_cosen_afib();
            float cosen = get_cosen_inside_region(no_p_regions[i]);
            if (cosen_afib > -1.4 && cosen > -1.4) {
                equal_hrv = true;
            } else if (cosen_afib < -1.4 && cosen < -1.4) {
                equal_hrv = true;
            } else {
                equal_hrv = false;
            }
            //std::cout << cosen_afib << " " << cosen << std::endl;
        } else {
            float cv_afib = get_cv_afib();
            float cv = get_cv_inside_region(no_p_regions[i]);
            if (cv_afib < 0.12 && cv < 0.12) {
                equal_hrv = true;
            } else if (cv_afib > 0.12 && cv > 0.12) {
                equal_hrv = true;
            } else {
                equal_hrv = false;
            }
            //std::cout << cv_afib << " " << cv << std::endl;
        }

        if (equal_hr && equal_hrv) {
            //std::cout << "Region has similar HR and HRV to AFIB, extending from " << st << " to " << end << std::endl;
            for(int j=st; j<=end; ++j) {
                record->delineations->afib->binary[j] = true;
            }
        } else {
            //std::cout << "Region has different HR or HRV to AFIB, skipping" << std::endl;
        }
    }

    tools.closingcentered(record->delineations->afib->binary, record->delineations->afib->size, record->fs, record->delineations->afib->binary);


}

void Reflection::correct_afib_for_pattern(std::regex pat) {

    std::string beat_types;

    for (int i = 0; i < beats.size(); ++i) {
        if (beats[i]->abnormal) {
            beat_types += 'V';
        } else {
            beat_types += 'N';
        }
    }
    //std::cout << "Beat types: " << beat_types << std::endl;

    //std::string test("NNNNVNVNVNVNNNNNNVVVVNNNNNVNNVNNVNNVNNNNN");
    auto begin = std::sregex_iterator(beat_types.begin(), beat_types.end(), pat);
    auto end   = std::sregex_iterator();

    for (auto i = begin; i != end; ++i) {
        std::smatch m = *i;
        int start = beats[m.position(0)]->get_r_wave();
        int end = beats[m.position(0) + m.length(0) - 1]->get_r_wave();
        
        for(int j = start; j <= end; ++j) {
            afib_false_positive[j] = true;
            record->delineations->afib->binary[j] = false;
        }
    }

}

void Reflection::correct_afib_for_ivr_or_vt() {
    // Placeholder for IVR or VT correction logic
    if (verbose) {
        std::cout << "Correcting for IVR or VT..." << std::endl;
    }

    std::regex ivr(R"(V{3,})");
    //std::cout << "REGEX for IVR..." << std::endl;
    std::regex big(R"((NV){3,})");
    //std::cout << "REGEX for BIG..." << std::endl;
    std::regex tri(R"((NNV){3,})");
    //std::cout << "REGEX for TRI..." << std::endl;

    //std::cout << "Correcting for IVR..." << std::endl;
    correct_afib_for_pattern(ivr);
    //std::cout << "Correcting for bigeminy..." << std::endl;
    correct_afib_for_pattern(big);
    //std::cout << "Correcting for trigeminy..." << std::endl;
    correct_afib_for_pattern(tri);
}

void Reflection::correct_afib_for_p() {

    if (verbose) {
        std::cout << "Correcting for presence of P waves..." << std::endl;
    }

    //If too little QRS waves are detected, do not correct
    if (beats.size() < 5) return;

    //Get average number of P waves present in the local region
    std::vector<bool> hasp(beats.size(), false);

    for (int i = 0; i < beats.size(); ++i) {
        int start_ind = max(0,i-2);
        int end_ind = min(i+2, (int)beats.size()-1);
        int len = end_ind - start_ind + 1;
        int num=0;
        for(int j=start_ind; j<=end_ind; ++j) {
            if (beats[j]->p_wave != nullptr) {
                num++;
            }
        }
        if ((float)num/(float)len > 0.5) {
            hasp[i] = true;
        } else {
            hasp[i] = false;
        }
    }

    //Print output for test
    // std::cout << "Presence of P waves: ";
    // for(int i=0; i<beats.size(); ++i) {
    //     std::cout << hasp[i];
    // }
    // std::cout << std::endl;

    //Get regions where P waves are present on average
    vector<tuple<int, int>> regions = tools.get_regions(hasp);

    //Correct for the presence of P waves
    for (int i = 0; i < regions.size(); ++i) {
        //std::cout << "Region: " << std::get<0>(regions[i]) << ", " << std::get<1>(regions[i]) << std::endl;
        int st = beats[std::get<0>(regions[i])]->get_r_wave();
        int end = beats[std::get<1>(regions[i])-1]->get_r_wave();
        //std::cout << "Region has P waves, correcting from " << st << " to " << end << std::endl;
        for(int j=st; j<=end; ++j) {
            afib_false_positive[j] = true;
            record->delineations->afib->binary[j] = false;
        }
    }

}

void Reflection::correct_afib_for_number_of_qrs(vector<tuple<int, int>> regions) {
    for (int i=0; i < regions.size(); ++i) {
        int num_qrs = get_number_of_qrs_waves_inside_region(regions[i]);
        //std::cout << "Number of QRS waves in region: " << num_qrs << std::endl;

        if (num_qrs < 3) {
            //std::cout << "Region has too few QRS waves, correcting from " << std::get<0>(regions[i]) << " to " << std::get<1>(regions[i]) << std::endl;
            for(int j=std::get<0>(regions[i]); j<std::get<1>(regions[i]); j++) {
                afib_false_positive[j] = true;
                record->delineations->afib->binary[j] = false;
            }
        }
    }
}

void Reflection::correct_afib_for_uncertainty(vector<tuple<int, int>> regions) {

    for (int i=0; i < regions.size(); ++i) {
        float mean_uncertainty = 0.0f;
        for (int j=std::get<0>(regions[i]); j<std::get<1>(regions[i]); j++) {
            mean_uncertainty += log(record->delineations->afib->uncertainty[j] + 0.0001);
        }
        mean_uncertainty /= (std::get<1>(regions[i]) - std::get<0>(regions[i]));

        //std::cout << "Mean uncertainty in region: " << mean_uncertainty << std::endl;

        if (mean_uncertainty >= -5 && (std::get<1>(regions[i]) - std::get<0>(regions[i])) < 3 * record->fs) {
            //std::cout << "Region is uncertain and is short, correcting from " << std::get<0>(regions[i]) << " to " << std::get<1>(regions[i]) << std::endl;
            for(int j=std::get<0>(regions[i]); j<std::get<1>(regions[i]); j++) {
                record->delineations->afib->binary[j] = false;
            }
        }
    }
}

int Reflection::get_number_of_qrs_waves_inside_region(tuple<int, int> region) {
    int count = 0;
    for (int i=0; i<beats.size(); i++) {
        if (beats[i]->get_r_wave() >= std::get<0>(region) && beats[i]->get_r_wave() <= std::get<1>(region)) {
            count++;
        }
    }
    return count;
}

float Reflection::get_median_hr_inside_region(tuple<int, int> region) {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (beats[i]->get_r_wave() >= std::get<0>(region) && beats[i]->get_r_wave() <= std::get<1>(region) && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 1.0f;
    }

    return tools.median(rrs.data(), rrs.size());
}

float Reflection::get_median_hr_afib() {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (record->delineations->afib->binary[beats[i]->get_r_wave()] && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 1.0f;
    }

    return tools.median(rrs.data(), rrs.size());
}

float Reflection::get_cosen_afib() {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (record->delineations->afib->binary[beats[i]->get_r_wave()] && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 0.0f;
    }

    return tools.cosen(rrs);
}

float Reflection::get_cosen_inside_region(tuple<int, int> region) {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (beats[i]->get_r_wave() >= std::get<0>(region) && beats[i]->get_r_wave() < std::get<1>(region) && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 0.0f;
    }

    return tools.cosen(rrs);
}

float Reflection::get_cv_afib() {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (record->delineations->afib->binary[beats[i]->get_r_wave()] && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 0.0f;
    }

    return tools.cv(rrs);
}

float Reflection::get_cv_inside_region(tuple<int, int> region) {
    vector<float> rrs;
    for (int i=0; i<beats.size(); i++) {
        if (beats[i]->get_r_wave() >= std::get<0>(region) && beats[i]->get_r_wave() <= std::get<1>(region) && !isnan(beats[i]->rr)) {
            rrs.push_back(beats[i]->rr);
        }
    }
    if (rrs.size() == 0) {
        return 0.0f;
    }

    return tools.cv(rrs);
}

void Reflection::print_regions(vector<tuple<int, int>> regions) {
    for (const auto& region : regions) {
        std::cout << "Region: " << std::get<0>(region) << ", " << std::get<1>(region) << std::endl;
    }
}

void Reflection::reflect_on_afib() {
    // Placeholder for AFib correction logic
    if (verbose) {
        std::cout << "Correcting AFib..." << std::endl;
    }

    //Correct afib mask using morphological operations
    for (int i = 0; i < record->delineations->afib->size; i++) {
        afib_false_positive.push_back(false);
    }

    // tools.closingcentered(record->delineations->afib->binary, record->delineations->afib->size, 3 * record->fs, record->delineations->afib->binary);
    // tools.openingcentered(record->delineations->afib->binary, record->delineations->afib->size, 3 * record->fs, record->delineations->afib->binary);

    vector<tuple<int, int>> regions = tools.get_regions(record->delineations->afib->binary, record->delineations->afib->size);
    //print_regions(regions);

    //Correct for the presence of ventricular rhyhtms
    correct_afib_for_ivr_or_vt();
    regions = tools.get_regions(record->delineations->afib->binary, record->delineations->afib->size);
    //print_regions(regions);

    //Correct for the presence of P waves
    correct_afib_for_p();
    regions = tools.get_regions(record->delineations->afib->binary, record->delineations->afib->size);
    //print_regions(regions);

    //Noisy regions can also introduce false positives
    for(int i = 0; i < record->delineations->afib->size; i++) {
        if (record->delineations->noise->binary[i] == 1) {
            record->delineations->afib->binary[i] = false;
        }
    }

    //Smooth out the binary signal
    tools.closingcentered(record->delineations->afib->binary, record->delineations->afib->size, record->fs, record->delineations->afib->binary);

    //Get regions of interest
    regions = tools.get_regions(record->delineations->afib->binary, record->delineations->afib->size);

    //Correct for the presence of QRS waves
    correct_afib_for_number_of_qrs(regions);

    //Smooth out binary signal
    tools.openingcentered(record->delineations->afib->binary, record->delineations->afib->size, record->fs, record->delineations->afib->binary);
    tools.closingcentered(record->delineations->afib->binary, record->delineations->afib->size, record->fs, record->delineations->afib->binary);

    //Get regions of interest
    regions = tools.get_regions(record->delineations->afib->binary, record->delineations->afib->size);

    //Correct for afib uncertainty
    correct_afib_for_uncertainty(regions);

    //Check if we could extend the region
    //extend_afib_if_possible();
}

void Reflection::reflect_on_p_waves() {
    // Placeholder for P-wave correction logic
    if (verbose) {
        std::cout << "Correcting P-wave... " << record->fs << std::endl;
    }

    process_p_wave_uncertainty();

    //get possible peaks
    vector<int> possible_peaks = identify_possible_p_waves();

    //Reset P waves
    for(int i=0; i<beats.size(); i++) {
        beats[i]->p_wave = nullptr;
    }
    p_waves.clear();

    float best_threshold = 0.0;
    float best_f1 = 0.0;
    float lowest_error = std::numeric_limits<float>::max();
    for(float i=0.5; i>=0.05; i-=0.01) {
        float error = analyse_threshold(i, possible_peaks);
        //std::cout << "Threshold: " << i << ", error: " << error << std::endl;
        if (error < lowest_error) {
            lowest_error = error;
            best_threshold = i;
        }
    }
    if (verbose) {
        std::cout << "Best threshold: " << best_threshold << ", F1: " << lowest_error << std::endl;
    }

    //Correct p wave mask using morphological operations
    for(int i=0; i<record->delineations->p_wave->size; i++) {
        record->delineations->p_wave->binary[i] = (record->delineations->p_wave->logits[i] > best_threshold);
        if (record->delineations->noise->binary[i]) {
            record->delineations->p_wave->binary[i] = false;
        }
        if (record->delineations->afib->binary[i]) {
            record->delineations->p_wave->binary[i] = false;
        }
    }

    //tools.openingcentered(record->delineations->p_wave->binary, record->delineations->p_wave->size, 0.01 * record->fs, record->delineations->p_wave->binary);
    //tools.closingcentered(record->delineations->p_wave->binary, record->delineations->p_wave->size, 0.05 * record->fs, record->delineations->p_wave->binary);

    //Calculate an ecg withou QRS waves
    get_ecg_no_qrst();

    //Identify P waves
    vector<shared_ptr<P>> waves = identify_p_waves();
    for (int i = 0; i < waves.size(); ++i) {
        p_waves.push_back(waves[i]);
    }
    //std::cout << "Number of P waves: " << p_waves.size() << std::endl;

    float qrs_median = qrs_median_range(true);

    PWaveProcessor processor = PWaveProcessor();
    PWaveResult res = processor.get_avg_p_wave(beats, p_waves, record->fs, qrs_median);

    for(int i=0; i<res.p_wave_clusters.size(); i++) {
        record->p_clusters.push_back(res.p_wave_clusters[i]);
    }

    // std::cout << "Number of p wave groups: " << p_wave_clusters.groups.size() << std::endl;
    // std::cout << "Number of trimmed pwaves: " << p_wave_clusters.p_waves.size() << std::endl;

    // if (false) {
    //     float median_range = p_median_range();

    //     float rho_min = 0.01*median_range;
    //     float alpha = 4;
    //     float beta = 0.125;

    //     clusterer_p = std::make_unique<Clustering>(rho_min, alpha, beta);
    //     clusterer_p->cluster_p(p_waves, record->fs);

    //     //Identify p wave polarity within clusters
    //     for (int i = 0; i < clusterer_p->get_number_of_clusters(); ++i) {
    //         identify_polarity(clusterer_p->get_cluster(i), clusterer_p->get_number_of_clusters());
    //     }

    //     //Identify p wave polarity within unclustered beats
    //     identify_polarity_of_unclustered_beats();
    // }

    //Match P waves to QRS waves
    match_p_waves_to_qrs();


    // for(int i=0; i<p_waves.size(); i++) {
    //     std::cout << "P wave " << i << ": " << p_waves[i]->start << ", " << p_waves[i]->end << " unclustered:" << p_waves[i]->unclustered << " polarity: " << p_waves[i]->inverted << " cluster_id: " << p_waves[i]->cluster_id << std::endl;
    // }

    //copy to record
    record->p.clear();
    for(int i=0; i<p_waves.size(); i++) {
        record->p.push_back(p_waves[i]);
    }
    // record->p_clusters.clear();
    // for(int i=0; i<clusterer_p->get_number_of_clusters(); i++) {
    //     record->p_clusters.push_back(clusterer_p->get_cluster(i));
    // }

    // return record
    //delete clusterer_p;
}

void Reflection::process_p_wave_uncertainty() {
    // Placeholder for processing P-wave uncertainty
    if (verbose) {
        std::cout << "Processing P-wave uncertainty..." << std::endl;
    }
    int size = record->delineations->p_wave->size;

    //Take the log of the uncertainty
    for (int i = 0; i < size; ++i) {
        record->delineations->p_wave->uncertainty[i] = log(record->delineations->p_wave->uncertainty[i] + 0.0001);
    }

    //Smooth the uncertainty signal
    tools.median_filter(record->delineations->p_wave->uncertainty, size, 0.1 * record->fs);

    //Filter with a 3rd order Savitzky-Golay filter
    float coeffs[21] = {-0.05590062, -0.02484472, 0.00294214, 0.02745995, 0.04870873, 0.06668846,
        0.08139915,0.0928408, 0.1010134, 0.10591697, 0.10755149, 0.10591697,
        0.1010134, 0.0928408, 0.08139915, 0.06668846, 0.04870873, 0.02745995,
        0.00294214, -0.02484472, -0.05590062};

    tools.iir_filter(record->delineations->p_wave->uncertainty, size, coeffs, 21);
    
    //Remove QRS and T from the uncertainty signal
    for (int i=0; i < beats.size(); ++i) {
        for (int j = beats[i]->get_global_start(); j < beats[i]->get_global_end(); ++j) {
            record->delineations->p_wave->uncertainty[j] = -10.0f;
        }
        if (beats[i]->t_wave != nullptr) {
            for (int j = beats[i]->t_wave->get_global_start(); j < beats[i]->t_wave->get_global_end(); ++j) {
                record->delineations->p_wave->uncertainty[j] = -10.0f;
            }
        }
    }
}

vector<int> Reflection::identify_possible_p_waves() {
    // Placeholder for identifying possible P waves
    if (verbose) {
        //std::cout << "Identifying possible P waves..." << std::endl;
    }
    vector<int> initial_peaks = tools.find_peaks(record->delineations->p_wave->uncertainty, record->delineations->p_wave->size, -2.5, 0.2 * record->fs, 3.0);
    vector<int> peaks;

    for (int i = 0; i < initial_peaks.size(); ++i) {
        int mindist = record->delineations->p_wave->size;
        for (int j = 0; j < p_waves.size(); ++j) {
            int dist = abs(initial_peaks[i] - (p_waves[j]->get_global_start() + p_waves[j]->get_global_end()) / 2);
            if (dist < mindist) {
                mindist = dist;
            }
        }
        if (mindist > 0.2 * record->fs && record->delineations->afib->binary[initial_peaks[i]] == false) {
            peaks.push_back(initial_peaks[i]);
        }
    }

    return peaks;
}

float Reflection::analyse_threshold(float threshold, vector<int> possible_peaks) {

    //Correct p wave mask using morphological operations
    for(int i=0; i<record->delineations->p_wave->size; i++) {
        record->delineations->p_wave->binary[i] = (record->delineations->p_wave->logits[i] > threshold);
        if (record->delineations->noise->logits[i] > 0.5) {
            record->delineations->p_wave->binary[i] = false;
        }
        if (record->delineations->afib->binary[i]) {
            record->delineations->p_wave->binary[i] = false;
        }
    }
    //tools.openingcentered(record->delineations->p_wave->binary, record->delineations->p_wave->size, 0.01 * record->fs, record->delineations->p_wave->binary);
    //tools.closingcentered(record->delineations->p_wave->binary, record->delineations->p_wave->size, 0.05 * record->fs, record->delineations->p_wave->binary);

    vector<shared_ptr<P>> waves = identify_p_waves();
    vector<int> pcenters;
    vector<float> pps;
    vector<float> prs;
    

    for(int i=0; i<waves.size(); i++) {
        int cur = (waves[i]->get_global_start() + waves[i]->get_global_end()) / 2;
        pcenters.push_back(cur);
        //std::cout << cur << ", ";
    }
    //std::cout << std::endl;

    for(int i=1; i<waves.size(); i++) {
        pps.push_back((float)(pcenters[i] - pcenters[i-1]));
        int mindist = record->delineations->p_wave->size;
        for(int j=0; j<beats.size(); j++) {
            if (beats[j]->get_global_start() >= pcenters[i] && beats[j]->get_global_start()-pcenters[i-1] < mindist) {
                mindist = beats[j]->get_global_start() - pcenters[i-1];
            }
        }
        prs.push_back((float)mindist);
        //std::cout << (pcenters[i] - pcenters[i-1]) << ", ";
    }
    //std::cout << std::endl;

    if (pps.size() == 0) {
        return std::numeric_limits<float>::max();
    }

    //calculate RMSSD of pps and prs
    float rmssd_pps = tools.rmssd(pps.data(), pps.size());
    float rmssd_prs = tools.rmssd(prs.data(), prs.size());
    //std::cout << "RMSSD PPs: " << rmssd_pps << ", RMSSD PRs: " << rmssd_prs << std::endl;

    return rmssd_pps + rmssd_prs;
    
    tuple<int, int> longeststreak = get_p_wave_streak(pps);

    int start = std::get<0>(longeststreak);
    int end = std::get<1>(longeststreak);
    tuple<int, int, int> res;
    // std::cout << "Longest streak: " << start << ", " << end << std::endl;
    // std::cout << "Possible peaks: ";
    // for(int i=0; i<possible_peaks.size(); i++) {
    //     std::cout << possible_peaks[i] << ", ";
    // }
    // std::cout << std::endl;

    if (end-start > 3) {
        vector<int> preds = get_predicted_locations(pcenters, pps, longeststreak);
        // std::cout << "P: ";
        // for(int i=0; i<preds.size(); i++) {
        //     std::cout << preds[i] << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << "T: ";
        // for(int i=0; i<pcenters.size(); i++) {
        //     std::cout << pcenters[i] << ", ";
        // }
        // std::cout << std::endl;
        res = analyse_predictions(preds, pcenters);
    } else {
        int tp = 0;
        int fp = 0;
        int fn = 0;

        for(int i=0; i<pcenters.size(); i++) {
            if (record->delineations->afib->binary[pcenters[i]] == 0 && record->delineations->noise->binary[pcenters[i]] == 0) {
                tp++;
            } else {
                fp++;
            }
        }
        res = std::make_tuple(tp, fp, 0);
    }

    for(int i=0; i<possible_peaks.size(); i++) {
        bool found = false;
        for(int j=0; j<pcenters.size(); j++) {
            if (abs(possible_peaks[i] - pcenters[j]) < int(0.1*record->fs)) {
                found = true;
                break;
            }
        }
        if (!found) {
            std::get<2>(res)++;
        }
    }

    if (std::get<0>(res) == 0) {
        return 0;
    }

    //std::cout << "TP: " << std::get<0>(res) << ", FP: " << std::get<1>(res) << ", FN: " << std::get<2>(res) << std::endl;
    float se = (float)std::get<0>(res) / (float)(std::get<0>(res) + std::get<2>(res));
    float ppv = (float)std::get<0>(res) / (float)(std::get<0>(res) + std::get<1>(res));
    float f1 = 2 * (se * ppv) / (se + ppv);

    return f1;
}

vector<int> Reflection::get_predicted_locations(vector<int> &pcenters, vector<float> &pibis, tuple<int, int> longeststreak) {
    
    vector<int> predicted_locations;
    int start = std::get<0>(longeststreak);
    int end = std::get<1>(longeststreak);
    int running_median = tools.median(pibis.data()+start, end-start);
    int running_std = running_median/8;
    float alpha = 0.33;

    for(int i=start; i<end; i++) {
        predicted_locations.push_back(pcenters[i]);
    }

    int i = end;

    while(i < pcenters.size()) {
        int pred = predicted_locations[i-1] + running_median;
        int actual = pcenters[i];

        float p = tools.check_hypothesis(pred, actual, running_std, true);

        if (p < 0.05) {
            if (pred > actual) {
                // Too early, look ahead for a better fit
                int m = 1;
                int g = 0;
                double s = running_std;

                //Take different steps and check all future pcenters if they match one of these predictions
                while (m < 4) {
                    pred = static_cast<int>(pcenters[i - 1] + m * running_median);
                    for (size_t k = i; k < pcenters.size(); ++k) {
                        float p = tools.check_hypothesis(pred, pcenters[k], s, true);
                        if (p > 0.05) {
                            //If the null hypothesis is not rejected, we have a match
                            g = static_cast<int>(k);
                            break;
                        }
                    }

                    if (g == 0) { //If no match was found, increase the step size and increase standard deviation
                        s *= 1.5;
                        ++m;
                    } else {
                        break;
                    }
                }

                if (m == 4) { //If no match was found, add the actual pcenter to the predicted locations
                    predicted_locations.push_back(actual);
                    running_std = running_median / 8;
                    ++i;
                    continue;
                }

                //If a match was found, add the predicted pcenter to the predicted locations
                predicted_locations.push_back(pcenters[g]);
                ++i;
            } else {
                //Beat is too late, look further for a better fit
                int m = 1;
                double s = running_std;

                while (m < 4) {
                    pred = static_cast<int>(pcenters[i - 1] + m * running_median);
                    s *= 1.5;
                    float p = tools.check_hypothesis(pred, actual, s, true);
                    if (p > 0.05) //if we have a match, break
                        break;
                    ++m;
                }
                
                if (m == 4) { //If no match was found, add the actual pcenter to the predicted locations
                    predicted_locations.push_back(actual);
                    running_std = running_median / 8;
                    ++i;
                    continue;
                }
                
                for(int j=0; j<m; j++) {
                    pred = int(pcenters[i-1] + (j+1)*running_median);
                    predicted_locations.push_back(pred);
                }
                ++i;
            }
        } else {
            running_median = (1-alpha)*running_median + alpha*(pcenters[i] - pcenters[i-1]);
            predicted_locations.push_back(actual);
            running_std = running_median / 8;
            ++i;
        }
    }

    return predicted_locations;
}

tuple<int, int, int> Reflection::analyse_predictions(vector<int> &predictions, vector<int> pcenters) {

    int tp = 0;
    int fp = 0;
    int fn = 0;
    
    for(int i=0; i<pcenters.size(); i++) {
        bool found = false;
        for(int j=0; j<predictions.size(); j++) {
            if (abs(predictions[j] - pcenters[i]) < int(0.1*record->fs) && record->delineations->afib->binary[predictions[j]] == 0 && record->delineations->noise->binary[predictions[j]] == 0) {
                tp++;
                found = true;
                break;
            }
        }
        if (!found) {
            fp++;
        }
    }
    
    return std::make_tuple(tp, fp, fn);
}

tuple<int, int> Reflection::get_p_wave_streak(vector<float> &pibis) {
    float med = tools.median(pibis.data(), pibis.size());
    int n = pibis.size();

    int start = 0;
    int end = 0;
    vector<tuple<int, int>> streaks;

    while (start < n) {
        end = start;
        for (int i = start; i < n; ++i) {
            end = i;
            if (pibis[i] > med * 1.25 || pibis[i] < med * 0.75) {
                break;
            }
        }
        streaks.emplace_back(start, end);
        start += 1;
    }

    tuple<int, int> longeststreak = std::make_tuple(0, 0);
    int longest = 0;
    for (const auto& streak : streaks) {
        int len = std::get<1>(streak) - std::get<0>(streak);
        if (len >= 3 && len > longest) {
            longest = len;
            longeststreak = streak;
            break;
        }
    }

    return longeststreak;
}

void Reflection::get_ecg_no_qrst() {
    // Placeholder for getting ECG without QRS waves
    //std::cout << "Getting ECG without QRS waves..." << std::endl;

    // Remove QRST segments from ECG signal

    for (int i = 0; i < beats.size(); ++i) {
        int x0 = beats[i]->start + beats[i]->wave_start;
        int x1 = beats[i]->start + beats[i]->wave_end;
        if (beats[i]->t_wave != nullptr) {
            x1 = beats[i]->t_wave->start + beats[i]->t_wave->wave_end;
        }
        float y0 = record->filtered_ecg[x0];
        float y1 = record->filtered_ecg[x1];
        float m = (y1 - y0) / (x1 - x0);
        float b = y0 - m * x0;

        for (int j = x0; j < x1; ++j) {
            record->ecg_no_qrst[j] = m * j + b;
        }
    }
}

vector<shared_ptr<P>> Reflection::identify_p_waves() {

    int pstart = -1;
    int pbeatid = 0;
    bool *pmask = record->delineations->p_wave->binary;
    bool *noise_mask = record->delineations->noise->binary;
    bool *t_mask = record->delineations->t_wave->binary;
    bool phasnoise = false;

    int win_min = record->fs * 0.1;
    int win_plus = record->fs * 0.1;

    vector<shared_ptr<P>> waves;

    for(int i=0; i < record->size-1; i++) {

        if (pmask[i] && pstart == -1) {
            pstart = i;
            phasnoise = false;
        }
        if (noise_mask[i]) {
            phasnoise = true;
        }
        if (pmask[i] == false && pstart != -1) {

            if (phasnoise==false && pstart > win_min && i < record->size - win_plus) {
                if (t_mask[pstart] && t_mask[i]) {
                    // If the P wave overlaps with a T wave, skip it
                    pstart = -1;
                    continue;
                }
                int mid = (pstart + i) / 2;
                std::shared_ptr<P> p = std::make_shared<P>();
                p->id = pbeatid;
                p->start = mid-win_min;
                p->end = mid+win_plus;
                p->wave_start = pstart-p->start;
                p->wave_end = i-p->start;
                p->center = mid;
                p->ecg = std::vector<float>(record->filtered_ecg + pstart, record->filtered_ecg + i);
                p->cluster_id = -1;

                waves.push_back(p);
                pbeatid++;
            }
            pstart = -1;
        }
    }

    if (waves.size() > 1) {
        for(int i=0; i<waves.size()-1; i++) {
            if (waves[i]->end + win_plus > waves[i+1]->start - win_min) {
                waves[i]->isdouble = true;
                waves[i+1]->isdouble = true;
            }
        }
    }
    //std::cout << "Number of P waves identified: " << waves.size() << std::endl;

    return waves;
}

void Reflection::identify_polarity(std::shared_ptr<Cluster> cluster, int number_of_clusters) {

    float median_range = p_median_range();

    for(int i = 0; i < cluster->template_beat->ecg.size(); ++i) {
        cluster->template_beat->ecg[i] /= median_range;
    }
    bool inverted = false;
    bool biphasic = false;

    float maximum = this->tools.max(cluster->template_beat->ecg.data(), cluster->template_beat->ecg.size());
    float minimum = this->tools.min(cluster->template_beat->ecg.data(), cluster->template_beat->ecg.size());

    if (maximum - minimum < 0.02) {
        //std::cout << "-> Flat P Wave!" << std::endl;
        inverted = false;
        biphasic = false;
    } else if (maximum > -minimum * 2) {
        //std::cout << "-> Positive P Wave!" << std::endl;
        inverted = false;
        biphasic = false;
    } else if (-minimum > maximum * 2) {
        //std::cout << "-> Negative P Wave!" << std::endl;
        inverted = true;
        biphasic = false;
    } else {
        //std::cout << "-> Biphasic P Wave!" << std::endl;
        inverted = false;
        biphasic = true;
    }

    for(int i=0; i<cluster->beats.size(); ++i) {
        if(std::shared_ptr<P> p = std::dynamic_pointer_cast<P>(cluster->beats[i])) {
            p->inverted = inverted;
            p->biphasic = biphasic;
        }
    }
}

void Reflection::identify_polarity_of_unclustered_beats() {

    float median_range = p_median_range();

    for(int i=0; i<p_waves.size(); ++i) {
        if(p_waves[i]->unclustered) {
            for(int j=0; j<p_waves[i]->ecg.size(); ++j) {
                p_waves[i]->ecg[j] /= median_range;
            }

            bool inverted = false;
            bool biphasic = false;

            float maximum = this->tools.max(p_waves[i]->ecg.data(), p_waves[i]->ecg.size());
            float minimum = this->tools.min(p_waves[i]->ecg.data(), p_waves[i]->ecg.size());

            if (maximum - minimum < 0.02) {
                //std::cout << "-> Flat P Wave!" << std::endl;
                inverted = false;
                biphasic = false;
            } else if (maximum > -minimum * 2) {
                //std::cout << "-> Positive P Wave!" << std::endl;
                inverted = false;
                biphasic = false;
            } else if (-minimum > maximum * 2) {
                //std::cout << "-> Negative P Wave!" << std::endl;
                inverted = true;
                biphasic = false;
            } else {
                //std::cout << "-> Biphasic P Wave!" << std::endl;
                inverted = false;
                biphasic = true;
            }

            p_waves[i]->inverted = inverted;
            p_waves[i]->biphasic = biphasic;
        }
    }
}

void Reflection::match_p_waves_to_qrs() {

    for (int i=0; i<beats.size(); ++i) {
        
        //std::cout << "QRS wave " << beats[i]->get_id() << ": " << beats[i]->get_global_start() << ", " << beats[i]->get_global_end() << std::endl;
        //Determine search windows to search for p waves and to determine noise ratio
        int p_search_start = 0;
        int p_search_end = beats[i]->get_global_start();
        if (i > 0) {
            p_search_start = min(beats[i-1]->get_global_end() + 0.2 * record->fs, beats[i]->get_global_start() - 0.2 * record->fs);
        }
        int p_noise_win_start = 0;
        int p_noise_win_end = p_search_end;
        if (i > 0) {
            p_noise_win_start = beats[i-1]->get_global_end();
            if (beats[i-1]->t_wave != nullptr) {
                p_noise_win_start = min(beats[i-1]->t_wave->get_global_end() + 0.0, beats[i]->get_global_start() - 0.2 * record->fs);
            }
        }
        
        //std::cout << "Determine foreground signal from " << std::endl;

        //Determine foreground signal
        int fore_size = beats[i]->get_global_end() - beats[i]->get_global_start();
        std::vector<float> fore(fore_size);
        for (int j=0; j<beats[i]->get_global_end() - beats[i]->get_global_start(); ++j) {
            fore[j] = record->filtered_ecg[beats[i]->get_global_start()+j];
        }
        float foremedian = this->tools.median(fore.data(), fore_size);
        for (int j=0; j<beats[i]->get_global_end() - beats[i]->get_global_start(); ++j) {
            fore[j] -= foremedian;
        }
        //std::cout << "Determine background signal from " << std::endl;
        //std::cout << p_noise_win_start << ", " << p_noise_win_end << std::endl;

        //Determine background signal
        int back_size = p_noise_win_end - p_noise_win_start;
        std::vector<float> back(back_size);
        for (int j=0; j<p_noise_win_end - p_noise_win_start; ++j) {
            back[j] = record->ecg_noise[p_noise_win_start + j];
        }
        //std::cout << "Determine SNR" << std::endl;

        //Determine SNR
        float snr = tools.snr_log(fore, back);
        beats[i]->snr = snr;

        //Get potential P wave candidates
        vector<std::shared_ptr<P>> candidates;
        for (int j=0; j<p_waves.size(); ++j) {
            if (p_waves[j]->get_global_start() > p_search_start && p_waves[j]->get_global_start() < p_search_end) {
                candidates.push_back(p_waves[j]);
            }
        }
        //std::cout << "Number of candidates: " << candidates.size() << std::endl;

        //Check if there are any candidates, otherwise stop
        if (candidates.size() > 0) {
            if (candidates.size() > 1 and candidates.back()->isdouble) {
                candidates.erase(candidates.end() - 1);
            }

            //Examine last candidate, closest to the QRS wave
            std::shared_ptr<P> lastcandidate = candidates.back();
            int pwave_size = lastcandidate->get_global_end() - lastcandidate->get_global_start();
            std::vector<float> pwave(pwave_size);
            for (int j=0; j<lastcandidate->get_global_end() - lastcandidate->get_global_start(); ++j) {
                pwave[j] = record->filtered_ecg[lastcandidate->get_global_start() + j];
            }
            float pmedian = tools.median(pwave.data(), pwave_size);
            for (int j=0; j<lastcandidate->get_global_end() - lastcandidate->get_global_start(); ++j) {
                pwave[j] -= pmedian;
            }

            //Determine the ratio of the P wave
            float pratio = tools.snr_lin(fore, pwave);
            beats[i]->pratio = pratio;


            beats[i]->p_wave = lastcandidate;
            beats[i]->pr = (beats[i]->get_global_start() - lastcandidate->get_global_start())/record->fs;

            //Check if the P wave is too far away from the QRS wave
            
            if (beats[i]->pr > 0.5) {
                //P wave is too far, so we note that this beat is unmatched
                lastcandidate->unmatched = true;
            }

            if (beats[i]->pr > 0.075 && beats[i]->pr < 0.2 && beats[i]->abnormal_uncertainty > -1) {
                beats[i]->abnormal = false;
            }

            if (candidates.size() > 1) {
                beats[i]->double_p = true;
                //Check if the P wave is too far away from the QRS wave
                for (int j=0; j<candidates.size()-1; ++j) {
                    candidates[j]->unmatched = true;
                }
            }
        } else {
            //Search behind QRS wave=
            int p_search_start = beats[i]->start + beats[i]->wave_end;
            int p_search_end = beats[i]->start + beats[i]->wave_end + 0.2 * record->fs;

            if (i < beats.size() - 1) {
                p_search_end = min(p_search_end, (int)(beats[i+1]->start - 0.2 * record->fs));
            }

            //Get potential P wave candidates
            vector<std::shared_ptr<P>> candidates;
            for (int j=0; j<p_waves.size(); ++j) {
                if (p_waves[j]->get_global_start() > p_search_start && p_waves[j]->get_global_start() < p_search_end) {
                    candidates.push_back(p_waves[j]);
                    break;
                }
            }

            //Check if there are any candidates, otherwise stop
            if (candidates.size() == 0) {
                continue;
            }

            std::shared_ptr<P> lastcandidate = candidates.back();
            if (abs(beats[i]->end - beats[i]->start) > 0.025 * record->fs) {
                beats[i]->p_wave = lastcandidate;
            }
        }
    }

}

void Reflection::reflect_on_qrs() {

    if (verbose) {
        std::cout << "Correcting QRS waves..." << std::endl;
    }

    //Identify QRS waves
    identify_qrs();
    //std::cout << "Number of QRS waves: " << beats.size() << std::endl;
    float median_range = qrs_median_range();
    float rho_min = 0.01 * median_range;
    //std::cout << "Median range: " << median_range << std::endl;
    //std::cout << "Rho min: " << rho_min << std::endl;   
    float alpha = 4;
    float beta = 0.125;

    //Cluster QRS waves using complete-linkage hierarchical clustering
    clusterer_qrs = std::make_unique<Clustering>(rho_min, alpha, beta);
    clusterer_qrs->cluster_qrs(beats, record->fs);

    //Identify beat abnormality within clusters
    // for (int i = 0; i < clusterer_qrs->get_number_of_clusters(); ++i) {
    //     std::shared_ptr<Cluster> cluster = clusterer_qrs->get_cluster(i);
    //     //identify_abnormality(cluster, clusterer_qrs->get_number_of_clusters());
    // }
    
    //determine r peak
    for (int i=0; i < beats.size(); ++i) {
        //std::cout << "Determine peak" << std::endl;
        beats[i]->determine_peak();
        //std::shared_ptr<QRS> beat = beats[i];
        //std::cout << "QRS " << i << ": " << beat->wave_start << ", " << beat->wave_end << ", " << beat->abnormal << std::endl;
    }


    //search for first decent part of 5s without noise
    bool *clean = new bool[record->size];
    for (int i=0; i<record->size; ++i) {
        clean[i] = !record->delineations->noise->binary[i];
    }
    tools.openingcentered(clean, record->size, record->fs*5, clean);

    calculate_rr_intervals();

    //if there is a large interval between two beats, that is suspicious and we want to recheck if we can find the R-peaks again
    bool abnormal_rmssd = false;
    for (int i=0; i < record->size; i+= 30*record->fs) {
        std::vector<float> ibis;
        for (int j=0; j<beats.size(); ++j) {
            if (beats[j]->get_global_start() > i && beats[j]->get_global_end() < i + 30*record->fs) {
                ibis.push_back(beats[j]->rr_raw);
            }
        }
        float rmssd = tools.rmssd(ibis.data(), ibis.size());
        //std::cout << "RMSSD of RR intervals: " << rmssd << std::endl;
        if (rmssd > 0.5) {
            abnormal_rmssd = true;
            //std::cout << "Abnormal RMSSD detected: " << rmssd << std::endl;
        }
    }

    bool large_pause = false;
    for (int i=0; i<beats.size(); ++i) {
        if (beats[i]->rr_raw > 3) {
            large_pause = true;
            break;
        }
    }

    int start = 0;
    bool foundstart = false;
    for (int i=0; i<record->size; i++) {
        if (clean[i]) {
            start = i;
            foundstart = true;
            break;
        }
    }
    delete[] clean;
    if (foundstart) {
        float *asra_ecg = new float[record->size - start];
        memcpy(asra_ecg, record->filtered_ecg, (record->size - start) * sizeof(float));
        ECGdetector peakdetector(asra_ecg, (record->size - start), record->fs);
        std::vector<int> rpeaks = peakdetector.getRPeaks();
        delete[] asra_ecg;

        for (int i=0; i<rpeaks.size(); i++) {
            rpeaks[i] += start; // Adjust the R-peaks to the original signal
            //std::cout << "R-peak detected at: " << rpeaks[i]/record->fs << std::endl;
        }
        std::vector<int> aladin_peaks;
        for (int i=0; i<beats.size(); i++) {
            aladin_peaks.push_back((int)beats[i]->get_r_wave());
        }

        if (large_pause || abnormal_rmssd) {
            //std::cout << "Large pause detected, re-matching peaks..." << std::endl;
            match_peaks(rpeaks, aladin_peaks);
        }
    }

    for(int i=0; i<record->delineations->noise->size; i++) {
        if (record->delineations->noise->binary[i] == 1.0f) {
            //std::cout << "#";
            record->filtered_ecg[i] = 0;
            record->ecg[i] = 0;
        } else {
            //std::cout << ".";
        }
    }

    calculate_rr_intervals();

    //copy qrs waves to record
    record->qrs.clear();
    for(int i=0; i<beats.size(); ++i) {
        std::shared_ptr<QRS> beat = beats[i];
        //std::cout << beat->get_r_wave() << std::endl;
        record->qrs.push_back(beat);
    }

    //copy t waves to record
    record->t.clear();
    for(int i=0; i<beats.size(); ++i) {
        if (beats[i]->t_wave != nullptr) {
            record->t.push_back(beats[i]->t_wave);
        }
    }
    record->qrs_clusters.clear();
    for(int i=0; i<clusterer_qrs->get_number_of_clusters(); i++) {
        //std::cout << "Cluster " << i << ": " << clusterer_qrs->get_cluster(i)->beats.size() << " beats" << std::endl;
        record->qrs_clusters.push_back(clusterer_qrs->get_cluster(i));
    }
}

float Reflection::qrs_median_range(bool norm) {

    std::vector<float> ranges;

    for (int i = 0; i < beats.size(); ++i) {
        int start = beats[i]->start;
        int end = beats[i]->end;

        float min_val = 0;
        float max_val = 0;

        if (norm) {
            min_val = this->tools.min(beats[i]->ecg_norm.data(), end - start);
            max_val = this->tools.max(beats[i]->ecg_norm.data(), end - start);
        } else {
            min_val = this->tools.min(beats[i]->ecg.data(), end - start);
            max_val = this->tools.max(beats[i]->ecg.data(), end - start);
        }

        float range = max_val - min_val;

        ranges.push_back(range);
    }

    if (ranges.size() == 0) {
        return 0.0;
    }

    float median = this->tools.median(ranges.data(), ranges.size());

    return median;
}

float Reflection::p_median_range() {

    std::vector<float> ranges;

    for (int i = 0; i < p_waves.size(); ++i) {
        int start = p_waves[i]->start;
        int end = p_waves[i]->end;
        float min_val = this->tools.min(p_waves[i]->ecg.data(), end - start);
        float max_val = this->tools.max(p_waves[i]->ecg.data(), end - start);
        float range = max_val - min_val;

        ranges.push_back(range);
    }

    if (ranges.size() == 0) {
        return 0.0;
    }

    float median = this->tools.median(ranges.data(), ranges.size());

    return median;
}

void Reflection::identify_qrs() {

    float *qrs_normal_logit = record->delineations->qrs->logits;
    float *qrs_abnormal_logit = record->delineations->abnormal_qrs->logits;
    std::vector<bool> qrsmask(record->delineations->qrs->size);
    bool *pmask = record->delineations->p_wave->binary;
    bool *tmask = record->delineations->t_wave->binary;
    bool *noise_mask = record->delineations->noise->binary;

    for(int i=0; i<record->delineations->qrs->size; i++) {
        qrsmask[i] = ((qrs_normal_logit[i]+qrs_abnormal_logit[i]) > 0.25);
        //std::cout << "QRS logit at " << i << ": normal=" << qrs_normal_logit[i] << ", abnormal=" << qrs_abnormal_logit[i] << ", mask=" << qrsmask[i] << std::endl;
    }

    //std::cout << record->fs << std::endl;
    tools.closingcentered(qrsmask, (int)(0.12*record->fs));

    auto t0 = chrono::high_resolution_clock::now();
    int win_min = record->fs * 0.1;
    int win_plus = record->fs * 0.2;
    //std::cout << "Win min: " << win_min << ", Win plus: " << win_plus << std::endl;
    //std::cout << "Record size: " << record->size << std::endl;

    int qrsstart = -1;
    int pstart = -1;
    int tstart = -1;
    bool qrshasnoise = false;
    bool phasnoise = false;
    bool thasnoise = false;
    int qrsbeatid = 0;
    int pbeatid = 0;
    int tbeatid = 0;

    for(int i=0; i < record->size-1; i++) {
        if (qrsmask[i] && qrsstart == -1) {
            qrsstart = i;
            qrshasnoise = false;
        }
        if (pmask[i] && pstart == -1) {
            pstart = i;
            phasnoise = false;
        }
        if (tmask[i] && tstart == -1) {
            tstart = i;
            thasnoise = false;
        }
        if (noise_mask[i]) {
            qrshasnoise = true;
            phasnoise = true;
            thasnoise = true;
        }

        if (qrsmask[i] == false && qrsstart != -1) {
            if (qrshasnoise==false && qrsstart > (win_min) && i < record->size - (win_plus)) {
                int mid = (qrsstart + i) / 2;
                std::shared_ptr<QRS> beat = std::make_shared<QRS>();
                beat->id = qrsbeatid;
                beat->start = mid - win_min;
                beat->end = mid + win_plus;
                beat->wave_start = qrsstart-beat->start;
                beat->wave_end = i-beat->start;
                beat->width = beat->wave_end - beat->wave_start + 1;
                beat->ecg = vector<float>(record->ecg_bandpass + beat->start, record->ecg_bandpass + beat->end);
                beat->ecg_norm = vector<float>(record->filtered_ecg + beat->start, record->filtered_ecg + beat->end);
                beat->abnormal_logit = tools.max(record->delineations->abnormal_qrs->logits + beat->start, beat->end-beat->start);
                beat->abnormal_uncertainty = log(tools.max(record->delineations->abnormal_qrs->uncertainty + beat->start, beat->end-beat->start)+0.0001);
                //std::cout << "QRS uncertainty: " << beat->abnormal_uncertainty << "from " << beat->start << " to " << beat->end << std::endl;
                beat->abnormal = (beat->abnormal_logit > 0.1);
                
                //std::cout << "QRS detected from " << beat->start << " to " << beat->end << std::endl;
                if(p_waves.size() > 0) {
                    std::shared_ptr<P> lastp = p_waves.back();
                    //std::cout << "Last P wave: " << lastp->wave_start << ", " << lastp->wave_end << std::endl;
                    if ((lastp->start + lastp->wave_start) > (beat->start + beat->wave_start - 0.5*record->fs) && 
                       (lastp->start + lastp->wave_start) < (beat->start + beat->wave_start - 0.075*record->fs) && 
                        lastp->wave_end - lastp->wave_start > 0.05*record->fs) {
                            //std::cout << "P wave is inside QRS" << std::endl;
                        beat->p_wave = p_waves.back();
                    }
                }
                //std::cout << "QRS detected from " << beat->start << " to " << beat->end << " with abnormality logit: " << beat->abnormal_logit << " and uncertainty: " << beat->abnormal_uncertainty << " and has p_wave: " << (beat->p_wave != nullptr) << std::endl;

                beats.push_back(beat);
                qrsbeatid++;
            } 
            qrsstart = -1;
        }
        if (pmask[i] == false && pstart != -1) {
            if (phasnoise==false) {
                std::shared_ptr<P> p = std::make_shared<P>();
                p->id = pbeatid;
                p->start = pstart;
                p->end = i;
                p->wave_start = 0;
                p->wave_end = i-pstart;
                //std::cout << "P wave detected from " << p->start << " to " << p->end << std::endl;
                
                p_waves.push_back(p);
                pbeatid++;
            } 
            pstart = -1;
        }
        if (tmask[i] == false && tstart != -1) {
            if (thasnoise==false) {
                std::shared_ptr<T> t = std::make_shared<T>();
                t->id = tbeatid;
                t->start = tstart;
                t->end = i;
                t->wave_start = 0;
                t->wave_end = i-tstart;
                //std::cout << "T wave detected from " << t->start << " to " << t->end << std::endl;
                
                t_waves.push_back(t);

                if (beats.size() > 0) {
                    std::shared_ptr<QRS> lastqrs = beats[beats.size()-1];
                    if (lastqrs->t_wave == nullptr) {
                        lastqrs->t_wave = t;
                        //std::cout << "T wave " << tbeatid <<  " matched to QRS wave " << lastqrs->id << std::endl;
                    }
                }
                tbeatid++;
            } 
            tstart = -1;
        }
    }

    if (beats.size() == 0) {
        return;
    }

    for(int i=0; i<beats.size()-1; i++) {
        int curbeat = (beats[i]->start + beats[i]->end)/2;
        int nextbeat = (beats[i+1]->start + beats[i+1]->end)/2;
        float average_noise_level = tools.mean(record->delineations->noise->logits + curbeat, nextbeat - curbeat);

        if (nextbeat - curbeat < 0.2 * record->fs && average_noise_level > 0.5) {
            // If the next beat is too close, merge them
            beats.erase(beats.begin() + i + 1);
            i--;
            beats.erase(beats.begin() + i + 1);
            i--;
        }
    }
    //std::cout << "Number of qrs beats detected: " << beats.size() << std::endl;
    
}

void Reflection::identify_abnormality(std::shared_ptr<Cluster> cluster, int number_of_clusters) {
    // Placeholder for abnormality identification logic
    // std::cout << "Identifying abnormality..." << std::endl;
    // std::cout << "Cluster ID: " << cluster->get_id() << std::endl;
    // std::cout << "Cluster last updated: " << cluster->get_last_updated() << std::endl;
    // std::cout << "Cluster number of beats: " << cluster->get_number_of_beats() << std::endl;

    int number_of_abnormal_beats = 0;
    int number_of_no_atrial_activity = 0;
    float mean_abnormal_uncertainty = 0.0f;

    //Identify per-beat abnormality
    for (int i = 0; i < cluster->get_number_of_beats(); ++i) {
        if(auto beat = std::dynamic_pointer_cast<QRS>(cluster->get_beat(i))) {

            if (beat->p_wave == nullptr) {
                beat->no_atrial_activity = true;
                number_of_no_atrial_activity++;
            } else if (beat->p_wave != nullptr && (beat->get_global_start()) < (beat->p_wave->get_global_start())) {
                beat->no_atrial_activity = true;
                number_of_no_atrial_activity++;
            } else if (beat->p_wave != nullptr && ((beat->get_global_start()) - (beat->p_wave->get_global_start())) < 0.075*record->fs) {
                beat->no_atrial_activity = true;
                number_of_no_atrial_activity++;
            } else if (beat->p_wave != nullptr && ((beat->get_global_start()) - (beat->p_wave->get_global_start())) > 0.3*record->fs) {
                beat->no_atrial_activity = true;
                number_of_no_atrial_activity++;
            } else {
                beat->no_atrial_activity = false;
            }
            
            //mean_abnormal_uncertainty += beat->abnormal_uncertainty;
            if (beat->abnormal && !beat->no_atrial_activity) {
                beat->abnormal = false;
            }
            // if (beat->abnormal_logit > 0.25) {
            //     beat->abnormal = true;
            //     number_of_abnormal_beats++;
            // } else if (beat->abnormal_logit > 0.1 && beat->no_atrial_activity) {
            //     beat->abnormal = true;
            //     number_of_abnormal_beats++;
            // } else if (beat->abnormal_uncertainty > -2.5 && beat->no_atrial_activity) {
            //     beat->abnormal = true;
            //     number_of_abnormal_beats++;
            // } else {
            //     beat->abnormal = false;
            // }
        }
    }


    //mean_abnormal_uncertainty /= cluster->get_number_of_beats();

    //Identify cluster abnormality
    // if (number_of_abnormal_beats > 1) {
    //     for (int i = 0; i < cluster->get_number_of_beats(); ++i) {
    //         if(auto beat = std::dynamic_pointer_cast<QRS>(cluster->get_beat(i))) {
    //             //std::cout << "Beat " << beat->id << ": " << beat->abnormal_logit << ", " << beat->abnormal_uncertainty << std::endl;
    //             if (beat->abnormal_uncertainty > -2.5) {
    //                 beat->abnormal = true;
    //             }
    //         }
    //     }
    // }

    //std::cout << "Number of abnormal beats: " << number_of_abnormal_beats << std::endl;
    //std::cout << "Ratio of abnormal beats: " << ((float)number_of_abnormal_beats/(float)cluster->get_number_of_beats()) << std::endl;

    //Correct for cluster majority vote
    // if (((float)number_of_abnormal_beats/(float)cluster->get_number_of_beats()) >= 0.66) {
    //     for (int i = 0; i < cluster->get_number_of_beats(); ++i) {
    //         if(auto beat = std::dynamic_pointer_cast<QRS>(cluster->get_beat(i))) {
    //             beat->abnormal = true;
    //         }
    //     }
    // }
    // std::cout << "Number of abnormal beats: " << number_of_abnormal_beats << std::endl;
    // std::cout << "Ratio of abnormal beats: " << ((float)number_of_abnormal_beats/(float)cluster->get_number_of_beats()) << std::endl;
    // std::cout << "Number of beats with no atrial activity: " << number_of_no_atrial_activity << std::endl;
    // std::cout << "Ratio of beats with no atrial activity: " << ((float)number_of_no_atrial_activity/(float)cluster->get_number_of_beats()) << std::endl;

    //Check mean uncertainty when there are more than one cluster
    // if (number_of_clusters > 1) {
    //     if (mean_abnormal_uncertainty > -5) {
    //         for (int i = 0; i < cluster->get_number_of_beats(); ++i) {
    //             if(auto beat = std::dynamic_pointer_cast<QRS>(cluster->get_beat(i))) {
    //                 if (beat->abnormal_uncertainty > -5 && beat->width >= 0.12*record->fs && beat->no_atrial_activity) {
    //                     beat->abnormal = true;
    //                 }
    //             }
    //         }
    //     }
    // } else {
    if (number_of_clusters == 0) {
        // Last defence for LBBB and RBBB
        if (((float)number_of_no_atrial_activity/(float)cluster->get_number_of_beats()) < 0.25) {
            for (int i = 0; i < cluster->get_number_of_beats(); ++i) {
                if(auto beat = std::dynamic_pointer_cast<QRS>(cluster->get_beat(i))) {
                    beat->abnormal = false;
                }
            }
        }
    }
}

void Reflection::calculate_rr_intervals() {
    // Placeholder for RR interval calculation logic
    //std::cout << "Calculating RR intervals..." << std::endl;

    if (beats.size() < 2) {
        if (verbose) {
            std::cout << "Not enough beats to calculate RR intervals." << std::endl;
        }
        return;
    }

    for (int i=1; i < beats.size(); ++i) {
        if (!beats[i]->abnormal) {
            beats[i]->rr = (beats[i]->get_r_wave()-beats[i-1]->get_r_wave())/record->fs;
        } else {
            beats[i]->rr = set_nan();
        }

        float last_good_rr = set_nan();
        int k = i;
        while (std::isnan(last_good_rr) && k > 0) {
            k -= 1;
            last_good_rr = beats[k]->rr;
        }

        beats[i]->rr = beats[i]->rr;
        beats[i]->rr_raw = (beats[i]->get_r_wave()-beats[i-1]->get_r_wave())/record->fs;
        beats[i]->rr_smooth = beats[i]->rr;
    }

    // Handle the first beat
    if (beats.size() > 1) {
        beats[0]->rr = beats[1]->rr;
        beats[0]->rr_raw = beats[1]->rr_raw;
        beats[0]->rr_smooth = beats[1]->rr_smooth;
    }

    for (int i=1; i< beats.size()-1; ++i) {
        if (std::isnan(beats[i]->rr_smooth) && !std::isnan(beats[i-1]->rr_smooth) && !std::isnan(beats[i+1]->rr_smooth)) {
            beats[i]->rr_smooth = (beats[i-1]->rr_smooth + beats[i+1]->rr_smooth)/2;
        } else if (!std::isnan(beats[i]->rr_smooth)) {
            if (std::isnan(beats[i-1]->rr_smooth) && std::isnan(beats[i+1]->rr_smooth)) {
                beats[i]->rr_smooth = beats[i]->rr_smooth;
            } else if (std::isnan(beats[i-1]->rr_smooth) && !std::isnan(beats[i+1]->rr_smooth)) {
                beats[i]->rr_smooth = (beats[i+1]->rr_smooth + beats[i]->rr_smooth)/2;
            } else if (std::isnan(beats[i+1]->rr_smooth) && !std::isnan(beats[i-1]->rr_smooth)) {
                beats[i]->rr_smooth = (beats[i-1]->rr_smooth + beats[i]->rr_smooth)/2;
            } else {
                beats[i]->rr_smooth = (beats[i-1]->rr_smooth + beats[i+1]->rr_smooth + beats[i]->rr_smooth)/3;
            }
        } else {
            beats[i]->rr_smooth = beats[i]->rr_smooth;
        }
    }

    // Handle the last beat
    // if (beats.size() > 1) {
    //     beats[beats.size()-1]->rr = beats[beats.size()-2]->rr;
    //     beats[beats.size()-1]->rr_raw = beats[beats.size()-2]->rr_raw;
    //     beats[beats.size()-1]->rr_smooth = beats[beats.size()-2]->rr_smooth;
    // }
}


void Reflection::reflect(std::shared_ptr<Record> record) {
    // Placeholder for reflection logic
    initialize(record);
    reflect();
}

void Reflection::reflect() {
    // Placeholder for reflection logic
    if (verbose) {
        std::cout << "Reflecting..." << std::endl;
    }
    assert (record != nullptr);
    this->reflect_on_noise();
    this->reflect_on_qrs();
    this->reflect_on_afib();
    //this->correct_p_wave();
}

void Reflection::reset() {
    // Placeholder for reset logic
    record = nullptr;
    beats.clear();
    p_waves.clear();
    t_waves.clear();
    clusterer_qrs = nullptr;
    clusterer_p = nullptr;
}


void BatchReflection::reflect_on_record(std::shared_ptr<Record> record) {
    std::unique_ptr<Reflection> reflection = std::make_unique<Reflection>(record);
    reflection->reflect_on_noise();
    reflection->reflect_on_qrs();
    reflection->reflect_on_afib();
    reflection->reflect_on_p_waves();
    reflection->reset();
}

void BatchReflection::reflect_on_batch(std::vector<std::shared_ptr<Record>> records, std::function<void(int)> progress_callback) {

    //omp_set_num_threads(24);

    // #pragma omp parallel
    // {
    //     int thread_id = omp_get_thread_num();
    //     int num_threads = omp_get_num_threads();

    //     if (thread_id == 0) {
    //         std::cout << "Starting batch reflection with " << num_threads << " threads." << std::endl;
    //     }
    // }
    indicators::ProgressBar bar{
        option::BarWidth{80},
        option::Start{" ["},
        option::Fill{"█"},
        option::Lead{"█"},
        option::Remainder{"-"},
        option::End{"]"},
        option::PrefixText{"Reflecting on records"},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::MaxProgress{records.size()}
        //option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    #pragma omp parallel for
    for (size_t i = 0; i < records.size(); ++i) {
        //std::cout << "Reflecting on record " << i << " of " << records.size() << std::endl;
        reflect_on_record(records[i]);
        bar.tick();
    }
}