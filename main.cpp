#pragma once

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
    vect_t<BN_PTR> vecBNData;
    uint64_t iBNLayerSize = 0;

    void ValueAssign(NetBNMNIST &netSrc) { iBNLayerSize = netSrc.iBNLayerSize; }

    set<vect> ForwProp(set<feature> &setInput)
    {
        set<vect> setOutput;
        set<feature> setTemp;
        for(auto i=0; i<lsLayer.size(); ++i) switch (lsLayer[i].get() -> iLayerType)
        {
        case FC:
            setOutput = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]).get() -> ForwProp(setOutput);
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case FC_BN:
            setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> ForwProp(setOutput);
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case CONV:
            if(i) setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> ForwProp(setTemp);
            else setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> ForwProp(setInput, true);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case CONV_BN:
            if(i) setTemp = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setTemp);
            else setTemp = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setInput, true);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case POOL:
            setTemp = INSTANCE_DERIVE<LAYER_POOL>(lsLayer[i]).get() -> ForwProp(setTemp);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case TRANS:
            if(INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get()->bFeatToVec)
            {
                if(i) setOutput = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get() -> ForwProp(setInput);
                else setOutput = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get() -> ForwProp(setTemp);
                if(setOutput.size()) break;
                else return blank_vect_seq;
            }
            else
            {
                setTemp = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get() -> ForwProp(setOutput);
                if(setTemp.size()) break;
                else return blank_vect_seq;
            }
        default: return blank_vect_seq;
        }
        return setOutput;
    }
    bool BackProp(set<vect> &setOutput, set<vect> &Origin)
    {
        set<vect> setGradVec;
        set<feature> setGradFt;
        for(auto i=lsLayer.size()-1; i>0; --i) switch (lsLayer[i].get() -> iLayerType)
        {
        case FC:
            if(i==lsLayer.size()-1) setGradVec = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]).get() -> BackProp(setOutput, Origin);
            else setGradVec = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]).get() -> BackProp(setGradVec);
            if(setOutput.size()) break;
            else return false;
        case FC_BN:
            if(i==lsLayer.size()-1) setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> BackProp(setOutput, Origin);
            else setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> BackProp(setGradVec);
            if(setOutput.size()) break;
            else return false;
        case CONV:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> BackProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case CONV_BN:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case POOL:
            setGradFt = INSTANCE_DERIVE<LAYER_POOL>(lsLayer[i]).get() -> BackProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case TRANS:
            if(INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get()->bFeatToVec)
            {
                setGradFt = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get() -> BackProp(setGradVec);
                if(setGradFt.size()) break;
                else return false;
            }
            else
            {
                setGradVec = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]).get() -> BackProp(setGradFt);
                if(setGradVec.size()) break;
                else return false;
            }
        default: return false;
        }
        return true;
    }
public:
    NetBNMNIST(NetBNMNIST &netSrc) : NetClassify(netSrc), vecBNData(netSrc.vecBNData) { ValueAssign(netSrc); }
    NetBNMNIST(NetBNMNIST &&netSrc) : NetClassify(move(netSrc)), vecBNData(move(netSrc.vecBNData)) { ValueAssign(netSrc); }
    void operator=(NetBNMNIST &netSrc) { new (this)NetBNMNIST(netSrc); }
    void operator=(NetBNMNIST &&netSrc) { new (this)NetBNMNIST(move(netSrc)); }

    NetBNMNIST(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, double dLearnRate = 0, uint64_t iMiniBatch = 0, bool bShowIter = true) : NetClassify(iDscType, dNetAcc, dLearnRate, iMiniBatch, bShowIter) {}
    void Run(MNIST &mnistDataset)
    {
        auto bTrainFlag = false;
        auto setOrigin = mnistDataset.orgn();
        
        set<vect> setPreOutput;
        vect_t<vect> batPreOutput;
        do for(auto i=0; i<mnistDataset.elem.size(); ++i)
        {
            auto setOutput = ForwProp(mnistDataset.elem[i]);
            if(bShowIterFlag) IterShow(setPreOutput, setOutput, setOrigin[i]);
            if(IterFlag(setOutput, setOrigin[i])) bTrainFlag = true;
            if(bTrainFlag) bTrainFlag = BackProp(setOutput, setOrigin[IDX_ZERO]);
        }
        while (bTrainFlag);
    }
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", {20});
    NetBNMNIST LeNet;
    LeNet.AddLayer<LAYER_CONV>(20, 3, 5, 5, 1, 1, NULL);
    LeNet.AddLayer<LAYER_CONV_BN>(20, 0, 1, RELU);
    LeNet.AddLayer<LAYER_POOL>(POOL_MAX, 2, 2, 1, 1);
    LeNet.AddLayer<LAYER_CONV>(50, 20, 5, 5, 1, 1, NULL);
    LeNet.AddLayer<LAYER_CONV_BN>(50, 0, 1, RELU);
    LeNet.AddLayer<LAYER_POOL>(POOL_MAX, 2, 2, 1, 1);
    LeNet.AddLayer<LAYER_TRANS>();
    LeNet.AddLayer<LAYER_FC>(80, 500, NULL);
    LeNet.AddLayer<LAYER_FC_BN>(0, 1, SIGMOID);
    LeNet.AddLayer<LAYER_FC>(500, 10, SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    LeNet.Run(dataset);
    return EXIT_SUCCESS;
}