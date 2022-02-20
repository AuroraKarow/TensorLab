#pragma once

#include "dataset"
#include "neunet"
#include "thread_pool"

using namespace std;
using namespace dataset;
using namespace neunet;
using namespace layer;

class NetBNMNIST final : public NetClassify
{
private:
    NET_MAP<uint64_t, set<BN_PTR>> mapBNData;

    set<vect> ForwProp(set<feature> &setInput, uint64_t iMiniBatchIdx = 0)
    {
        set<vect> setOutput;
        set<feature> setTemp;
        for(auto i=0Ui64; i<lsLayer.size(); ++i) switch (lsLayer[i].get() -> iLayerType)
        {
        case FC:
            setOutput = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]).get() -> ForwProp(setOutput);
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case FC_BN:
            setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> ForwProp(setOutput, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_FC>(mapBNData[i][iMiniBatchIdx])));
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case CONV:
            if(i) setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> ForwProp(setTemp);
            else setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> ForwProp(setInput, true);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case CONV_BN:
            if(i) setTemp = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setTemp, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_CONV>(mapBNData[i][iMiniBatchIdx])));
            else setTemp = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setInput, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_CONV>(mapBNData[i][iMiniBatchIdx])), true);
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
    bool BackProp(set<vect> &setOutput, set<vect> &Origin, uint64_t iMiniBatchIdx = 0)
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
            if(i==lsLayer.size()-1) setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> BackProp(setOutput, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_FC>(mapBNData[i][iMiniBatchIdx])), Origin);
            else setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]).get() -> BackProp(setGradVec, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_FC>(mapBNData[i][iMiniBatchIdx])));
            if(setOutput.size()) break;
            else return false;
        case CONV:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]).get() -> BackProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case CONV_BN:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]).get() -> ForwProp(setGradFt, INSTANCE_QUOTE(INSTANCE_DERIVE<BN_CONV>(mapBNData[i][iMiniBatchIdx])));
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
    NetBNMNIST(NetBNMNIST &netSrc) : NetClassify(netSrc), mapBNData(netSrc.mapBNData) { ValueAssign(netSrc); }
    NetBNMNIST(NetBNMNIST &&netSrc) : NetClassify(move(netSrc)), mapBNData(move(netSrc.mapBNData)) { ValueAssign(netSrc); }
    void operator=(NetBNMNIST &netSrc) { new (this)NetBNMNIST(netSrc); }
    void operator=(NetBNMNIST &&netSrc) { new (this)NetBNMNIST(move(netSrc)); }

    NetBNMNIST(uint64_t iDscType = GD_BGD, double dNetAcc = 1e-2, double dLearnRate = 0, uint64_t iMiniBatch = 0, bool bShowIter = true) : NetClassify(iDscType, dNetAcc, dLearnRate, iMiniBatch, bShowIter) {}
    /* FC
    uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5
    * FC_BN
    double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dDmt = 1e-10
    * CONV
    * uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0
    * CONV_BN
    * uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dDmt = 1e-10
    * POOL
    * uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iActFuncTypeVal = NULL
    * LayerTrans - 
    * uint64_t iActFuncTypeVal = NULL
    * uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iActFuncIdx = NULL
     */
    template<typename LayerType, typename ... Args,  typename = enable_if_t<is_base_of_v<Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs)
    {
        auto lyrCurrTemp = std::make_shared<LayerType>(pacArgs...);
        if(lyrCurrTemp.get()->iLayerType==FC_BN || lyrCurrTemp.get()->iLayerType == CONV_BN) mapBNData.insert(lsLayer.size(), set<BN_PTR>());
        return lsLayer.emplace_back(lyrCurrTemp);
    }
    void Run(MNIST &mnistDataset)
        {
            for(auto i=0; i<mapBNData.size(); ++i)
            {
                mapBNData.index(i).value.init(mnistDataset.elem.size());
                for(auto j=0; j<mnistDataset.elem.size(); ++j)
                    if(lsLayer[mapBNData.index(i).key].get()->iLayerType == FC_BN) mapBNData.index(i).value[j] = make_shared<BN_FC>();
                    else mapBNData.index(i).value[j] = make_shared<BN_CONV>();
            }
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