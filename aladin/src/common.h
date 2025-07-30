
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>
#include <cassert>
#include <complex.h>
#include <cmath>
#include <deque>
#include <chrono>
#include <regex>
#include <limits>
#include <functional>
#include "helpers.h"
#include "omp.h"
#include <chrono>
#include <chrono>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <thread>

namespace py = pybind11;
using namespace std;

static float set_nan() {
    return std::numeric_limits<float>::quiet_NaN();
}

class DominantPoint {
    public:
        DominantPoint() {};

        int get_midpoint() const { return j; }
        py::array_t<int> get_support() const { return py::array_t<int>({ 2 }, { sizeof(int) }, sup); }

        int j;              // Midpoint index
        float curvature;   // Curvature at j
        float delta;       // abs(signal[mid] - signal[start or end])
        float real_delta;  // real difference: can be positive or negative
        bool convex;        // Convexity check
        int d[2];     // r_min, r_max
        int s[2];     // i_min, i_max
        int sup[2];   // j_min, j_max (to be determined)

};

class Component;

class Cluster {
    public:
        Cluster();
        Cluster(int _cluster_id, int _last_updated, std::shared_ptr<Component> _template_beat);
        void set_width();
        void determine_peak();

        void add_beat(std::shared_ptr<Component> beat) { beats.push_back(beat); }
        void calc_p_qrs_ratio();
        void set_template(std::shared_ptr<Component> beat);
        void set_closest(std::shared_ptr<Cluster> c) { closest = c; }

        int get_id() const { return cluster_id; }
        int get_last_updated() const { return last_updated; }
        int get_number_of_beats() const { return beats.size(); }
        std::shared_ptr<Component> get_beat(int i) { return beats[i]; }
        float get_wave_onset() const { return onset; }
        float get_wave_offset() const { return offset; }
        std::shared_ptr<Component> get_template() const { std::shared_ptr<Component> shared(template_beat); return shared; }

        int cluster_id, last_updated;
        int peak;
        float onset, offset, width, p_qrs_ratio;
        std::shared_ptr<Component> template_beat;
        std::shared_ptr<Cluster> closest;
        vector<std::shared_ptr<Component>> beats;
};

class Component {
    public:
        Component() : cluster(nullptr), peak(0) {};
        virtual ~Component() = default;
        py::array_t<float> get_ecg() const { return py::array_t<float>({ ecg.size() }, { sizeof(float) }, ecg.data());}
        int get_id() const { return id; }
        int get_cluster_id() const { return cluster->cluster_id; }
        std::shared_ptr<Cluster> get_cluster() const { return cluster; }
        int get_start() const { return start; }
        int get_end() const { return end; }
        int get_wave_start() const { return wave_start; }
        int get_wave_end() const { return wave_end; }
        int get_global_start() const { return start + wave_start; }
        int get_global_end() const { return start + wave_end; }
        float get_width() const { return width; }
        int get_peak() const { return peak; }
        void determine_peak();
        int get_support_region_start() const { return support_region_start; }
        int get_support_region_end() const { return support_region_end; }
        int get_number_of_dominant_points() const { return dominant_points.size(); }
        std::shared_ptr<DominantPoint> get_dominant_point(int i) { return std::make_shared<DominantPoint>(dominant_points[i]);}

        std::shared_ptr<Cluster> cluster;
        int id, start, end, wave_start, wave_end, wave_onset, wave_offset, peak; // Start and end indices of the beat
        float width;
        vector<DominantPoint> dominant_points; // Array of dominant points
        int dominant_count = 0;
        int support_region_start, support_region_end; // Support region
        vector<float> ecg;
        
};

class P : public Component {
    public:
        P() : Component() {
            inverted = false;
            biphasic = false;
            unmatched = false;
            unclustered = false;
            isdouble = false;
        }

        bool get_inverted() const { return inverted; }
        bool get_biphasic() const { return biphasic; }
        bool get_unmatched() const { return unmatched; }
        bool get_unclustered() const { return unclustered; }

        bool inverted;
        bool biphasic;
        bool unmatched;
        bool unclustered;
        bool isdouble;
};

class T : public Component {
    public:
        T() : Component() {}
};

class QRS : public Component {
    public:
        QRS() : Component() {
            abnormal_uncertainty = 0.0;
            abnormal_logit = 0.0;
            no_atrial_activity = false;
            abnormal = false;
            junctional = false;
            double_p = false;
            isearly = false;
            startingposition = false;
            pratio = 0.0;
            snr = 0.0;
            prediction_m = 0.0;
            prediction_std = 0.0;
            uncertain = 0.0;
            hr = 0.0;
            hrv = 0.0;
            ibi = 0.0;
            pr = set_nan();
            rr = set_nan();
            rr_raw = set_nan();
            rr_smooth = set_nan();
            diagnosis = "";
        }
        int get_r_wave() const { return start + peak; }
        float get_rr() const { return rr; }
        float get_rr_raw() const { return rr_raw; }
        float get_rr_smooth() const { return rr_smooth; }
        bool get_no_atrial_activity() const { return no_atrial_activity; }
        bool get_abnormal() const { return abnormal; }

        std::shared_ptr<P> get_p_wave() const { return p_wave; }
        std::shared_ptr<T> get_t_wave() const { return t_wave; }

        std::shared_ptr<P> p_wave;
        std::shared_ptr<T> t_wave;
        float abnormal_uncertainty;
        float abnormal_logit;
        float pr;
        bool no_atrial_activity, abnormal, junctional, double_p, isearly, startingposition;
        float rr, rr_raw, rr_smooth;
        float pratio;
        float snr;
        float prediction_m, prediction_std, uncertain, hr, hrv, ibi;
        std::string diagnosis;
};

class Diagnosis {
    public:
        Diagnosis() : name(""), explanation(""), onset(0.0), offset(0.0) {}
        Diagnosis(std::string _name, std::string _explanation, float _onset, float _offset) : name(_name), explanation(_explanation), onset(_onset), offset(_offset) {}

        std::string name;
        std::string explanation;
        float onset, offset;
};

class Delineation {
    public:
        Delineation(py::array_t<float, py::array::c_style | py::array::forcecast> _logits, 
                    py::array_t<float, py::array::c_style | py::array::forcecast> _uncertainty,
                    py::array_t<bool, py::array::c_style | py::array::forcecast> _binary) {
            size = _logits.shape(0);
            logits_python = _logits; // Keep ownership alive
            logits = _logits.mutable_data();
            uncertainty_python = _uncertainty; // Keep ownership alive
            uncertainty = _uncertainty.mutable_data();
            binary_python = _binary; // Keep ownership alive
            binary = _binary.mutable_data();
        }

        py::array_t<float> get_logits() const { return logits_python; }
        py::array_t<float> get_uncertainty() const { return uncertainty_python; }
        void set_uncertainty(py::array_t<float, py::array::c_style | py::array::forcecast> _uncertainty) {
            uncertainty_python = _uncertainty; // Keep ownership alive
            uncertainty = _uncertainty.mutable_data();
        }
        void set_binary(py::array_t<bool, py::array::c_style | py::array::forcecast> _binary) {
            binary_python = _binary; // Keep ownership alive
            binary = _binary.mutable_data();
        }
        py::array_t<bool> get_binary() const { return binary_python; }
        int get_size() const { return size; }

        float *logits;
        float *uncertainty;
        bool *binary;
        int size;

        py::array_t<float> logits_python;
        py::array_t<float> uncertainty_python;
        py::array_t<bool> binary_python;
};

class Delineations {
    public:
        Delineations(std::shared_ptr<Delineation> _p_wave, std::shared_ptr<Delineation> _qrs, std::shared_ptr<Delineation> _abnormal_qrs, std::shared_ptr<Delineation> _t_wave, std::shared_ptr<Delineation> _noise, std::shared_ptr<Delineation> _afib) {
            p_wave = _p_wave;
            qrs = _qrs;
            abnormal_qrs = _abnormal_qrs;
            t_wave = _t_wave;
            noise = _noise;
            afib = _afib;
        }

        std::shared_ptr<Delineation> get_pwave() const { return p_wave; }
        std::shared_ptr<Delineation> get_qrs() const { return qrs; }
        std::shared_ptr<Delineation> get_abnormal_qrs() const { return abnormal_qrs; }
        std::shared_ptr<Delineation> get_twave() const { return t_wave; }
        std::shared_ptr<Delineation> get_noise() const { return noise; }
        std::shared_ptr<Delineation> get_afib() const { return afib; }

        std::shared_ptr<Delineation> p_wave;
        std::shared_ptr<Delineation> qrs;
        std::shared_ptr<Delineation> abnormal_qrs;
        std::shared_ptr<Delineation> t_wave;
        std::shared_ptr<Delineation> noise;
        std::shared_ptr<Delineation> afib;
};

class Record {
    public:
        Record(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg, float _fs, shared_ptr<Delineations> _delineations);
        Record(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg, float _fs);
        void preprocess();
        void reverse();

        float *ecg;
        float *filtered_ecg;
        float *ecg_no_qrst;
        float *ecg_noise;
        float *ecg_bandpass; 
        int size;
        float fs;
        
        std::shared_ptr<Delineations> delineations;
        vector<std::shared_ptr<Diagnosis>> diagnosis;
        vector<std::shared_ptr<Diagnosis>> subdiagnosis;
        vector<std::shared_ptr<QRS>> qrs;
        vector<std::shared_ptr<P>> p;
        vector<std::shared_ptr<T>> t;
        vector<std::shared_ptr<Cluster>> qrs_clusters;
        vector<std::shared_ptr<Cluster>> p_clusters;

        py::array_t<float> ecg_python;
        py::array_t<float> filtered_ecg_python;
        py::array_t<float> ecg_bandpass_python; // ECG without QRS complexes
        py::array_t<float> ecg_no_qrst_python;
        py::array_t<float> ecg_noise_python;

        py::array_t<float> get_ecg() const { return ecg_python; }
        py::array_t<float> get_filtered_ecg() const { return filtered_ecg_python; }
        void set_filtered_ecg(py::array_t<float, py::array::c_style | py::array::forcecast> _filtered_ecg) {
            filtered_ecg_python = _filtered_ecg; // Keep ownership alive
            filtered_ecg = _filtered_ecg.mutable_data();
        }
        void set_ecg_no_qrst(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg_no_qrst) {
            ecg_no_qrst_python = _ecg_no_qrst; // Keep ownership alive
            ecg_no_qrst = _ecg_no_qrst.mutable_data();
        }
        void set_ecg_noise(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg_noise) {
            ecg_noise_python = _ecg_noise; // Keep ownership alive
            ecg_noise = _ecg_noise.mutable_data();
        }
        void set_ecg_bandpass(py::array_t<float, py::array::c_style | py::array::forcecast> _ecg_bandpass) {
            ecg_bandpass_python = _ecg_bandpass; // Keep ownership alive
            ecg_bandpass = _ecg_bandpass.mutable_data();
        }
        py::array_t<float> get_ecg_no_qrst() const { return ecg_no_qrst_python; }
        py::array_t<float> get_ecg_noise() const { return ecg_noise_python; }
        py::array_t<float> get_ecg_bandpass() const { return ecg_bandpass_python; }
        std::shared_ptr<Delineations> get_delineations() const { return delineations; }
        void set_delineations(std::shared_ptr<Delineations> _delineations) { delineations = _delineations; }
        void set_p(std::shared_ptr<P> _p) { p.push_back(_p); }
        vector<std::shared_ptr<Diagnosis>> get_diagnosis() const { return diagnosis; }
        vector<std::shared_ptr<Diagnosis>> get_subdiagnosis() const { return subdiagnosis; }
        void add_diagnosis(std::shared_ptr<Diagnosis> _diagnosis);
        void add_subdiagnosis(std::shared_ptr<Diagnosis> _diagnosis);
        int get_size() const { return size; }
        float get_fs() const { return fs; }

        vector<std::shared_ptr<QRS>> get_qrs() const { return qrs; }
        vector<std::shared_ptr<P>> get_p() const { return p; }
        vector<std::shared_ptr<T>> get_t() const { return t; }
        vector<std::shared_ptr<Cluster>> get_qrs_clusters() const { return qrs_clusters; }
        vector<std::shared_ptr<Cluster>> get_p_clusters() const { return p_clusters; }

};

class RecordCollection {
    public:
        RecordCollection() = default;
        void add_record(std::shared_ptr<Record> record) { records.push_back(record); }
        std::shared_ptr<Record> get_record(int i) { return records[i]; }
        int get_size() const { return records.size(); }
        vector<std::shared_ptr<Record>> get_records() const { return records; }
        void clear() { records.clear(); }
        void preprocess() {
            //omp_set_num_threads(48);
            #pragma omp parallel for
            for (auto &record : records) {
                record->preprocess();
            }
        }

    private:
        vector<std::shared_ptr<Record>> records;
};