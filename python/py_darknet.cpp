#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <signal.h>
#include <stdlib.h>     /* srand, rand */
#include <unistd.h>
#include <pybind11/stl.h>
#include <mutex>

#include <vector>
#include <opencv2/core/core.hpp>

#include "Yolo3Detection.h"

namespace py = pybind11;
using namespace std;
using namespace cv;


Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input) {
    if (input.ndim() != 2)
        throw std::runtime_error("1-channel image must be 2 dims ");
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC1, (unsigned char*)buf.ptr);
    return mat;
}

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input) {

    if (input.ndim() != 3)
        throw std::runtime_error("3-channel image must be 3 dims ");

    py::buffer_info buf = input.request();

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);

    return mat;
}


class Tkdnn_darknet
{
public:
    Tkdnn_darknet(const std::string &name, const int n_classes, const int n_batch, const int min_h, const double min_conf) 
    {
        this->n_batch = n_batch;
        this->n_classes = n_classes;
        this->d_confident = min_conf;
        this->min_detection_height = min_h;
        yolo.init(name, n_classes, n_batch);
    }
    
    void queue_image(py::array_t<unsigned char>& input) {
        cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
        batch_dnn_input.push_back(img);
    }

    std::vector <std::vector <int32_t>> inference(int32_t frameid) {   
        yolo.update(batch_dnn_input, n_batch);
        std::vector <std::vector <int32_t>> detections; 
        for(int bi=0; bi<batch_dnn_input.size(); ++bi){
                // draw dets
                for(int i=0; i<yolo.batchDetected[bi].size(); i++) {
                    tk::dnn::box b = yolo.batchDetected[bi][i];
                    if (b.prob < this->d_confident || b.w < this->min_detection_height)
                    {
                        continue;
                    }
                    std::vector <int32_t> _det;
                    _det.push_back(frameid);
                    _det.push_back(-1);
                    _det.push_back((int32_t) b.x);
                    _det.push_back((int32_t) b.y);
                    _det.push_back((int32_t) b.w);
                    _det.push_back((int32_t) b.h);
                    _det.push_back((int32_t) (b.prob * 1000));
                    _det.push_back(b.cl);
                    _det.push_back(-1);
                    _det.push_back(-1);
                    detections.push_back(_det);
                }
        }
        batch_dnn_input.clear();
        return detections;
    }
    
    ~Tkdnn_darknet() {}
private:
    std::vector<cv::Mat> batch_dnn_input;
    int n_batch;
    int n_classes;
    double d_confident;
    int min_detection_height;
    tk::dnn::Yolo3Detection yolo;
};   


PYBIND11_MODULE(py_darknet, m) 
{
    // optional module docstring
    m.doc() = "pybind11 yolo v4 tkDNN fast inference";

    // define add function

    // m.def("length", &py_length, "Calculate the length of an array of vectors");

    // bindings to Pet class
    py::class_<Tkdnn_darknet>(m, "Tkdnn_darknet")
        .def(py::init<const std::string &, int, int, int, double> ())
        .def("queue_image", &Tkdnn_darknet::queue_image)
        .def("inference", &Tkdnn_darknet::inference);
}
