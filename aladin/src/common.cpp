
#include "common.h"
 

void Component::determine_peak() {

    // vector<pair<int, int>> path;
    // vector<float> aligned_x;
    // vector<float> aligned_y;
    // vector<float> sig_x = ecg;
    // vector<float> dsig_x(sig_x.size());
    // vector<float> sig_y = cluster->template_beat->ecg;
    // vector<float> dsig_y(sig_y.size());

    // for (int i = 0; i < sig_y.size()-1; ++i) {
    //     dsig_y[i] = sig_y[i+1] - sig_y[i];
    // }
    // for (int i = 0; i < sig_x.size()-1; ++i) {
    //     dsig_x[i] = sig_x[i+1] - sig_x[i];
    // }

    // float dtw = Tools::dtwPath(dsig_x, dsig_y, path, aligned_x, aligned_y);

    // int cur_ind = 0;
    // int cur_n = 0;
    // float running_sum = 0.0f;

    // for (int i = 0; i < path.size(); ++i) {
    //     if (path[i].second == cluster->peak) {
    //         peak = path[i].first;
    //         break;
    //     }
    // }

    float max_delta = 0.0f;
    for (int i = 0; i < dominant_points.size(); ++i) {
        int pos = dominant_points[i].j;
        float delta = dominant_points[i].delta;
        float factor = 0;
        if (pos < wave_start) {
            factor = abs(wave_start - pos) / (float)(wave_start - wave_end);
        } else if (pos > wave_end) {
            factor = abs(pos - wave_end) / (float)(wave_start - wave_end);
        } else {
            factor = 1.0f;
        }

        if (delta*factor > max_delta) {
            max_delta = delta;
            peak = dominant_points[i].j;    
        }
    }

}

void Cluster::calc_p_qrs_ratio() {
    float p = 0;
    float qrs = 0;
    for (int i = 0; i < beats.size(); ++i) {
        if(auto beat = std::dynamic_pointer_cast<QRS>(beats[i])) {
            if (beat->get_p_wave() != nullptr) {
                p += 1.0f;   
            }
            qrs += 1.0f;
        }
    }
    if (qrs > 0) {
        //std::cout << "P: " << p << ", QRS: " << qrs << std::endl;
        p_qrs_ratio = p / qrs;
    } else {
        p_qrs_ratio = set_nan();
    }
}

Record::Record(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg, float _fs, std::shared_ptr<Delineations> _delineations) {
    ecg_python = _ecg;
    ecg = _ecg.mutable_data();
    size = _ecg.shape(0);
    fs = _fs;
    delineations = _delineations;

    filtered_ecg_python = py::array_t<float>(size);
    filtered_ecg = filtered_ecg_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        filtered_ecg[i] = 0.0f;
        ecg_no_qrst[i] = 0.0f;
        ecg_noise[i] = 0.0f;
    }

    //preprocess();
}

Record::Record(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg, float _fs) {
    ecg_python = _ecg;
    ecg = _ecg.mutable_data();
    size = _ecg.shape(0);
    fs = _fs;

    filtered_ecg_python = py::array_t<float>(size);
    filtered_ecg = filtered_ecg_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        filtered_ecg[i] = 0.0f;
    }

    normalized_ecg_python = py::array_t<float>(size);
    normalized_ecg = normalized_ecg_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        normalized_ecg[i] = 0.0f;
    }

    ecg_no_qrst_python = py::array_t<float>(size);
    ecg_no_qrst = ecg_no_qrst_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        ecg_no_qrst[i] = 0.0f;
    }

    ecg_noise_python = py::array_t<float>(size);
    ecg_noise = ecg_noise_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        ecg_noise[i] = 0.0f;
    }

    ecg_bandpass_python = py::array_t<float>(size);
    ecg_bandpass = ecg_bandpass_python.mutable_data();
    for (int i = 0; i < size; ++i) {
        ecg_bandpass[i] = 0.0f;
    }

    //preprocess();
}

void Record::preprocess() {

    Tools tls = Tools();

    memcpy(ecg_bandpass, ecg, size * sizeof(float));
    tls.filtfilt_lowpass(ecg_bandpass, size, 30.0f, fs);
    tls.remove_baseline(ecg_bandpass, size, fs);
    

    memcpy(normalized_ecg, ecg, size * sizeof(float));
    tls.normalize_zscore(normalized_ecg, size);

    // Calculate filtered ecg
    memcpy(filtered_ecg, ecg, size * sizeof(float));
    // tls.filtfilt_lowpass(filtered_ecg, size, 40.0f, fs);
    // tls.remove_baseline(filtered_ecg, size, fs);
    tls.normalize_zscore(filtered_ecg, size);
    

    // Copy filtered_ecg to ecg_no_qrst
    memcpy(ecg_no_qrst, filtered_ecg, size * sizeof(float));
    memcpy(ecg_noise, filtered_ecg, size * sizeof(float));

    // Apply high pass filter
    tls.filtfilt_highpass(ecg_noise, size, 10.0f, fs);

}

void Record::add_diagnosis(std::shared_ptr<Diagnosis> _diagnosis) { 
    //Add diagnosis
    diagnosis.push_back(_diagnosis); 

    //Add diagnosis to beats
    int count = 0;
    for (int i=0; i<qrs.size(); i++) {
        if (qrs[i]->get_r_wave() >= _diagnosis->onset && qrs[i]->get_r_wave() <= _diagnosis->offset) {
            qrs[i]->diagnosis = _diagnosis->name;
        }
    }
}

void Record::add_subdiagnosis(std::shared_ptr<Diagnosis> _diagnosis) { 
    //Add diagnosis
    subdiagnosis.push_back(_diagnosis); 
}

void Record::reverse() {

    // Reverse the ECG signal
    for (int i = 0; i < size; ++i) {
        ecg[i] = -ecg[i];
        filtered_ecg[i] = -filtered_ecg[i];
        ecg_bandpass[i] = -ecg_bandpass[i];
        ecg_no_qrst[i] = -ecg_no_qrst[i];
        ecg_noise[i] = -ecg_noise[i];
    }
}
