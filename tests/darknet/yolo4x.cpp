#include<iostream>
#include<vector>
#include "tkdnn.h"
#include "test.h"
#include "DarknetParser.h"

int main() {
    std::string bin_path  = "yolo4x";
    std::vector<std::string> input_bins = { 
        bin_path + "/layers/input.bin"
    };
    std::vector<std::string> output_bins = {
        bin_path + "/debug/layer168_out.bin",
        bin_path + "/debug/layer185_out.bin",
        bin_path + "/debug/layer202_out.bin"
    };
    std::string wgs_path  = bin_path + "/layers";
    std::string cfg_path  = std::string(TKDNN_PATH) + "/tests/darknet/cfg/yolo4x.cfg";
    std::string name_path = std::string(TKDNN_PATH) + "/tests/darknet/names/coco.names";
    downloadWeightsifDoNotExist(input_bins[0], bin_path, "https://cloud.hipert.unimore.it/s/BLPpiAigZJLorQD/download");



    // parse darknet network
    tk::dnn::Network *net = tk::dnn::darknetParser(cfg_path, wgs_path, name_path);
    net->print();

    //convert network to tensorRT
    tk::dnn::NetworkRT *netRT = new tk::dnn::NetworkRT(net, net->getNetworkRTName(bin_path.c_str()));
    
    int ret = testInference(input_bins, output_bins, net, netRT);
    net->releaseLayers();
    delete net;
    delete netRT;
    return ret;
}
