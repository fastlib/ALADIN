

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "common.h"
#include "reflect.h"

namespace py = pybind11;

PYBIND11_MODULE(_main, m) {

    py::class_<Diagnosis, std::shared_ptr<Diagnosis>>(m, "Diagnosis")
        .def(py::init<>())
        .def(py::init<std::string, std::string, int, int>(), py::arg("name"), py::arg("explanation"), py::arg("onset"), py::arg("offset"))
        .def_readwrite("name", &Diagnosis::name)
        .def_readwrite("explanation", &Diagnosis::explanation)
        .def_readwrite("onset", &Diagnosis::onset)
        .def_readwrite("offset", &Diagnosis::offset);

    py::class_<Delineation, std::shared_ptr<Delineation>>(m, "Delineation")
        .def(py::init<
             py::array_t<float, py::array::c_style | py::array::forcecast>,
             py::array_t<float, py::array::c_style | py::array::forcecast>,
             py::array_t<bool, py::array::c_style | py::array::forcecast>>(),
             py::arg("logits"), py::arg("uncertainty"), py::arg("binary"))
        .def_property_readonly("logits", &Delineation::get_logits)
        .def_property_readonly("uncertainty", &Delineation::get_uncertainty)
        .def("set_uncertainty", &Delineation::set_uncertainty)
        .def("set_binary", &Delineation::set_binary)
        .def_property_readonly("binary", &Delineation::get_binary)
        .def_property_readonly("size", &Delineation::get_size);

    py::class_<Delineations, std::shared_ptr<Delineations>>(m, "Delineations")
        .def(py::init<
            std::shared_ptr<Delineation>, 
            std::shared_ptr<Delineation>, 
            std::shared_ptr<Delineation>, 
            std::shared_ptr<Delineation>, 
            std::shared_ptr<Delineation>,
            std::shared_ptr<Delineation>>(),
            py::arg("p_wave"), py::arg("qrs"), py::arg("abnormal_qrs"), py::arg("t_wave"), py::arg("noise"), py::arg("afib"))
        .def_property_readonly("p", &Delineations::get_pwave)
        .def_property_readonly("qrs", &Delineations::get_qrs)
        .def_property_readonly("abnormal_qrs", &Delineations::get_abnormal_qrs)
        .def_property_readonly("t", &Delineations::get_twave)
        .def_property_readonly("noise", &Delineations::get_noise)
        .def_property_readonly("afib", &Delineations::get_afib);

    py::class_<BatchReflection, std::shared_ptr<BatchReflection>>(m, "BatchReflection")
        .def(py::init<>())
        .def("reflect_on_record", &BatchReflection::reflect_on_record)
        .def("reflect_on_batch", &BatchReflection::reflect_on_batch);

    py::class_<Reflection, std::shared_ptr<Reflection>>(m, "Reflection")
        .def(py::init<>())
        .def(py::init<std::shared_ptr<Record>>(), 
            py::arg("record"))
        .def("initialize", &Reflection::initialize)
        .def("reflect", py::overload_cast<>(&Reflection::reflect))
        .def("reflect", py::overload_cast<std::shared_ptr<Record>>(&Reflection::reflect))
        .def("reflect_on_noise", &Reflection::reflect_on_noise)
        .def("reflect_on_qrs", &Reflection::reflect_on_qrs)
        .def("reflect_on_afib", &Reflection::reflect_on_afib)
        .def("reflect_on_p_waves", &Reflection::reflect_on_p_waves)
        .def_property_readonly("number_of_qrs_clusters", &Reflection::get_number_of_qrs_clusters)
        .def("get_qrs_cluster", &Reflection::get_qrs_cluster)
        .def_property_readonly("number_of_qrs_beats", &Reflection::get_number_of_qrs_beats)
        .def("get_qrs_beat", &Reflection::get_qrs_beat)
        .def_property_readonly("number_of_p_clusters", &Reflection::get_number_of_p_clusters)
        .def("get_p_cluster", &Reflection::get_p_cluster)
        .def_property_readonly("number_of_p_beats", &Reflection::get_number_of_p_beats)
        .def("get_p_beat", &Reflection::get_p_beat)
        .def("reset", &Reflection::reset);
        
    py::class_<Cluster, std::shared_ptr<Cluster>>(m, "Cluster")
        .def_property_readonly("template", &Cluster::get_template)
        .def_property_readonly("id", &Cluster::get_id)
        .def_property_readonly("last_updated", &Cluster::get_last_updated)
        .def_property_readonly("wave_onset", &Cluster::get_wave_onset)
        .def_property_readonly("wave_offset", &Cluster::get_wave_offset);

    py::class_<Component, std::shared_ptr<Component>>(m, "Component")
        .def_property_readonly("ecg", &Component::get_ecg)
        .def_property_readonly("id", &Component::get_id)
        .def_property_readonly("cluster_id", &Component::get_cluster_id)
        .def_property_readonly("start", &Component::get_start)
        .def_property_readonly("end", &Component::get_end)
        .def_property_readonly("wave_start", &Component::get_wave_start)
        .def_property_readonly("wave_end", &Component::get_wave_end)
        .def_property_readonly("support_region_start", &Component::get_support_region_start)
        .def_property_readonly("support_region_end", &Component::get_support_region_end)
        .def_property_readonly("number_of_dominant_points", &Component::get_number_of_dominant_points)
        .def("get_dominant_point", &Component::get_dominant_point);

    py::class_<DominantPoint, std::shared_ptr<DominantPoint>>(m, "DominantPoint")
        .def_property_readonly("j", &DominantPoint::get_midpoint)
        .def_property_readonly("support", &DominantPoint::get_support);

    py::class_<Record, std::shared_ptr<Record>>(m, "Record")
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            float,
            std::shared_ptr<Delineations>>(),
            py::arg("ecg"), py::arg("fs"), py::arg("delineations"))
        .def(py::init<
            py::array_t<float, py::array::c_style | py::array::forcecast>,
            float>(),
            py::arg("ecg"), py::arg("fs"))
        .def("preprocess", &Record::preprocess)
        .def("reverse", &Record::reverse)
        .def_property_readonly("ecg", &Record::get_ecg)
        .def_property_readonly("fs", &Record::get_fs)
        .def_property("filtered_ecg", &Record::get_filtered_ecg, &Record::set_filtered_ecg)
        .def_property("ecg_no_qrst", &Record::get_ecg_no_qrst, &Record::set_ecg_no_qrst)
        .def_property("ecg_noise", &Record::get_ecg_noise, &Record::set_ecg_noise)
        .def_property("ecg_bandpass", &Record::get_ecg_bandpass, &Record::set_ecg_bandpass)
        .def_property("delineations", &Record::get_delineations, &Record::set_delineations)
        .def_property_readonly("qrs", &Record::get_qrs)
        .def_property("p", &Record::get_p, &Record::set_p)
        .def_property_readonly("t", &Record::get_t)
        .def_property_readonly("qrs_clusters", &Record::get_qrs_clusters)
        .def_property_readonly("p_clusters", &Record::get_p_clusters)
        .def_property_readonly("diagnosis", &Record::get_diagnosis)
        .def_property_readonly("subdiagnosis", &Record::get_subdiagnosis)
        .def("add_subdiagnosis", &Record::add_subdiagnosis)
        .def("add_diagnosis", &Record::add_diagnosis);

    py::class_<RecordCollection, std::shared_ptr<RecordCollection>>(m, "RecordCollection")
        .def(py::init<>())
        .def("add_record", &RecordCollection::add_record)
        .def("get_record", &RecordCollection::get_record)
        .def_property_readonly("size", &RecordCollection::get_size)
        .def_property_readonly("records", &RecordCollection::get_records)
        .def("preprocess", &RecordCollection::preprocess);

    py::class_<QRS, std::shared_ptr<QRS>>(m, "QRS")
        .def_property_readonly("id", &QRS::get_id)
        .def_property_readonly("start", &QRS::get_start)
        .def_property_readonly("end", &QRS::get_end)
        .def_property_readonly("cluster_id", &QRS::get_cluster_id)
        .def_property_readonly("wave_start", &QRS::get_wave_start)
        .def_property_readonly("wave_end", &QRS::get_wave_end)
        .def_property_readonly("onset", &QRS::get_global_start)
        .def_property_readonly("offset", &QRS::get_global_end)
        .def_property_readonly("ecg", &QRS::get_ecg)
        .def_property_readonly("support_region_start", &QRS::get_support_region_start)
        .def_property_readonly("support_region_end", &QRS::get_support_region_end)
        .def_property_readonly("number_of_dominant_points", &QRS::get_number_of_dominant_points)
        .def("get_dominant_point", &QRS::get_dominant_point)
        .def_property_readonly("peak", &QRS::get_peak)
        .def_property_readonly("r", &QRS::get_r_wave)
        .def_property_readonly("width", &QRS::get_width)
        .def_property_readonly("rr", &QRS::get_rr)
        .def_property_readonly("rr_raw", &QRS::get_rr_raw)
        .def_property_readonly("rr_smooth", &QRS::get_rr_smooth)
        .def_property_readonly("p", &QRS::get_p_wave)
        .def_property_readonly("t", &QRS::get_t_wave)
        .def_property_readonly("abnormal", &QRS::get_abnormal)
        .def_readwrite("junctional", &QRS::junctional)
        .def_readwrite("double_p", &QRS::double_p)
        .def_readwrite("startingposition", &QRS::startingposition)
        .def_readwrite("isearly", &QRS::isearly)
        .def_readwrite("snr", &QRS::snr)
        .def_readwrite("pr", &QRS::pr)
        .def_readwrite("pratio", &QRS::pratio)
        .def_readwrite("diagnosis", &QRS::diagnosis)
        .def_readwrite("prediction_m", &QRS::prediction_m)
        .def_readwrite("prediction_std", &QRS::prediction_std)
        .def_readwrite("uncertain", &QRS::uncertain)
        .def_readwrite("hr", &QRS::hr)
        .def_readwrite("hrv", &QRS::hrv)
        .def_readwrite("ibi", &QRS::ibi);

    
    py::class_<P, std::shared_ptr<P>>(m, "P")
        .def_property_readonly("id", &P::get_id)
        .def_property_readonly("start", &P::get_start)
        .def_property_readonly("end", &P::get_end)
        .def_property_readonly("cluster_id", &P::get_cluster_id)
        .def_property_readonly("cluster", &P::get_cluster)
        .def_property_readonly("wave_start", &P::get_wave_start)
        .def_property_readonly("wave_end", &P::get_wave_end)
        .def_property_readonly("onset", &P::get_global_start)
        .def_property_readonly("offset", &P::get_global_end)
        .def_property_readonly("ecg", &P::get_ecg)
        .def_property_readonly("support_region_start", &P::get_support_region_start)
        .def_property_readonly("support_region_end", &P::get_support_region_end)
        .def_property_readonly("number_of_dominant_points", &P::get_number_of_dominant_points)
        .def("get_dominant_point", &P::get_dominant_point)
        .def_property_readonly("peak", &P::get_peak)
        .def_property_readonly("width", &P::get_width)
        .def_property_readonly("inverted", &P::get_inverted)
        .def_property_readonly("biphasic", &P::get_biphasic)
        .def_readwrite("unmatched", &P::unmatched)
        .def_property_readonly("unclustered", &P::get_unclustered);

    py::class_<T, std::shared_ptr<T>>(m, "T")
        .def_property_readonly("id", &T::get_id)
        .def_property_readonly("start", &T::get_start)
        .def_property_readonly("end", &T::get_end)
        .def_property_readonly("cluster_id", &T::get_cluster_id)
        .def_property_readonly("wave_start", &T::get_wave_start)
        .def_property_readonly("wave_end", &T::get_wave_end)
        .def_property_readonly("onset", &T::get_global_start)
        .def_property_readonly("offset", &T::get_global_end)
        .def_property_readonly("ecg", &T::get_ecg)
        .def_property_readonly("support_region_start", &T::get_support_region_start)
        .def_property_readonly("support_region_end", &T::get_support_region_end)
        .def_property_readonly("number_of_dominant_points", &T::get_number_of_dominant_points)
        .def("get_dominant_point", &T::get_dominant_point)
        .def_property_readonly("peak", &T::get_peak)
        .def_property_readonly("width", &T::get_width);

};