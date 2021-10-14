#include "dataset"
#include "neunet"

using namespace std;
using namespace dataset;
using namespace neunet;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    MNIST dataset(".../MNIST/train-images.idx3-ubyte", ".../MNIST/train-labels.idx1-ubyte", true, 20, true);
    Basnet net;
    /* Need to appoint the first layer
     * It needn't appoint current layer's activate function if next one is BN layer
     */
    net.AddLayer<LAYER_CONV_ADA>(6, 1, 5, 5, 1, 1, NULL_FUNC, CONV_ADA, true);
    // BN layer, appoint the previous layer's activate function.
    net.AddLayer<LAYER_BN_CONV_ADA>(6, RELU);
    net.AddLayer<LAYER_POOL>(2, 2, 2, 2);
    net.AddLayer<LAYER_CONV_ADA>(16, 6, 5, 5, 1, 1, NULL_FUNC);
    net.AddLayer<LAYER_BN_CONV_ADA>(16, RELU);
    net.AddLayer<LAYER_POOL>(2, 2, 2, 2);
    net.AddLayer<LAYER_CONV_ADA>(120, 16, 5, 5, 1, 1, NULL_FUNC);
    net.AddLayer<LAYER_TRAN_TO_VECT>();
    net.AddLayer<LAYER_FC_ADA>(120, 80, NULL_FUNC);
    net.AddLayer<LAYER_BN_FC_ADA>(SIGMOID);
    net.AddLayer<LAYER_FC>(84, 10, SOFTMAX);
    net.Run(set<vect>(), dataset.elem, dataset.orgn());

    return EXIT_SUCCESS;
}