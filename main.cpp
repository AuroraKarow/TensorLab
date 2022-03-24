#pragma once

#include "dataset"
#include "neunet"

using namespace std;
using namespace mtx;
using namespace dataset;
using namespace neunet;
using namespace layer;



class NetMNISTIm2Col final : public NetClassify
{
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    // MNIST demo
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte");
    // dataset.output_bitmap("E:\\VS Code project data\\MNIST_out", BMIO_BMP);
    NetBNMNIST LeNet(0.1, 32, false);
    LeNet.AddLayer<LAYER_CONV>(20, 1, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN>(20);
    LeNet.AddLayer<LAYER_ACT_FT>(RELU);
    LeNet.AddLayer<LAYER_POOL>(POOL_MAX, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_CONV>(50, 20, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN>(50);
    LeNet.AddLayer<LAYER_ACT_FT>(RELU);
    LeNet.AddLayer<LAYER_POOL>(POOL_MAX, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_TRANS>();
    LeNet.AddLayer<LAYER_FC>(800, 500);
    LeNet.AddLayer<LAYER_FC_BN>();
    LeNet.AddLayer<LAYER_ACT_VT>(SIGMOID);
    LeNet.AddLayer<LAYER_FC>(500, 10);
    LeNet.AddLayer<LAYER_ACT_VT>(SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    LeNet.Run(dataset);
    return EXIT_SUCCESS;
}