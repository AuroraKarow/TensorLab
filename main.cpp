#include "dataset"
#include "neunet"
#include "thread_pool"

using namespace std;
using namespace dataset;
using namespace neunet;

class NetBNMNIST final : public NetClassify
{
private:
    /* BN amount            - Calling function AddLayer count the BN quantity, first dimension
     * BN index             - Counting with the iterating
     * BN mini-batch amount - Mini-batch size
     */
    vect_t<BN_PTR> mapBNData;
    uint64_t iBNLayerSize = 0;

    void ValueAssign(NetBNMNIST &netSrc) { iBNLayerSize = netSrc.iBNLayerSize; }

    bool ForwProp(MNIST &mnistDataset)
    {
        return true;
    }
    bool BackProp()
    {
        return true;
    }
    set<vect> Deduce(MNIST &mnistDataset)
    {
        set<vect> setDeduceOutput;
        return setDeduceOutput;
    }
public:
    NetBNMNIST(NetBNMNIST &netSrc) : NetClassify(netSrc), mapBNData(netSrc.mapBNData) { ValueAssign(netSrc); }
    NetBNMNIST(NetBNMNIST &&netSrc) : NetClassify(move(netSrc)), mapBNData(move(netSrc.mapBNData)) { ValueAssign(netSrc); }
    void operator=(NetBNMNIST &netSrc) { new (this)NetBNMNIST(netSrc); }
    void operator=(NetBNMNIST &&netSrc) { new (this)NetBNMNIST(move(netSrc)); }

    NetBNMNIST(double dNetAcc = 1e-2, bool bShowIter = false) : NetClassify(dNetAcc, bShowIter) {}
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", 20, true, true, 2);
    
    return EXIT_SUCCESS;
}