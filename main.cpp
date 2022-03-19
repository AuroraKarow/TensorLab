#pragma once

#include "dataset"
#include "neunet"
#include "thread_pool"

using namespace std;
using namespace mtx;
using namespace dataset;
using namespace neunet;
using namespace layer;

class NetBNMNIST final : public NetClassify
{
private:
    NET_MAP<uint64_t, BN_PTR> mapBNData;

    set<vect> ForwProp(set<feature> &setInput)
    {
        set<vect> setOutput;
        set<feature> setTemp;
        for(auto i=0Ui64; i<lsLayer.size(); ++i) switch (lsLayer[i] -> iLayerType)
        {
        case ACT_VT:
            setOutput = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> ForwProp(setOutput);
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case ACT_FT:
            setTemp = INSTANCE_DERIVE<LAYER_ACT_FT>(lsLayer[i]) -> ForwProp(setTemp);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case FC:
            setOutput = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> ForwProp(setOutput);
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case FC_BN:
            setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> ForwProp(setOutput);
            
            if(setOutput.size()) break;
            else return blank_vect_seq;
        case CONV:
            if(!i) setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]) -> ForwProp(setInput);
            else setTemp = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]) -> ForwProp(setTemp);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case CONV_BN:
            if(!i) INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]) -> ForwProp(setInput);
            else setTemp = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]) -> ForwProp(setTemp);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case POOL:
            setTemp = INSTANCE_DERIVE<LAYER_POOL>(lsLayer[i]) -> ForwProp(setTemp);
            if(setTemp.size()) break;
            else return blank_vect_seq;
        case TRANS:
            if(INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> bFeatToVec)
            {
                if(i) setOutput = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> ForwProp(setTemp);
                else setOutput = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> ForwProp(setInput);
                if(setOutput.size()) break;
                else return blank_vect_seq;
            }
            else
            {
                setTemp = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> ForwProp(setOutput);
                if(setTemp.size()) break;
                else return blank_vect_seq;
            }
        default: return blank_vect_seq;
        }
        return setOutput;
    }
    bool BackProp(set<vect> &setOutput, set<vect> &Origin, uint64_t iBatchIdx = 0)
    {
        set<vect> setGradVec;
        set<feature> setGradFt;
        for(auto i=lsLayer.size()-1; i>0; --i) switch (lsLayer[i] -> iLayerType)
        {
        case ACT_FT:
            setGradFt = INSTANCE_DERIVE<LAYER_ACT_FT>(lsLayer[i]) -> BackProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case ACT_VT:
            if(i+1 == lsLayer.size()) setGradVec = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> BackProp(setOutput, Origin);
            else setGradVec = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> BackProp(setGradVec);
            if(setGradVec.size()) break;
            else return false;
        case FC:
            setGradVec = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> BackProp(setGradVec);
            if(setGradVec.size() && INSTANCE_DERIVE<LAYER_FC>(lsLayer[i])->UpdatePara()) break;
            else return false;
        case FC_BN:
            setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> BackProp(setGradVec);
            if(setGradVec.size() && INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->UpdatePara())
            {
                if(iBatchIdx)
                {
                    INSTANCE_DERIVE<BN_FC>(mapBNData[iBatchIdx])->vecMiuBeta += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecMiuBeta;
                    INSTANCE_DERIVE<BN_FC>(mapBNData[iBatchIdx])->vecSigmaSqr += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecSigmaSqr;   
                }
                else INSTANCE_DERIVE<BN_FC>(mapBNData[iBatchIdx]) = std::make_shared<BN_FC>(std::move(INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData));
                break;
            }
            else return false;
        case CONV:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i]) -> BackProp(setGradFt);
            if(setGradFt.size() && INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i])->UpdatePara()) break;
            else return false;
        case CONV_BN:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]) -> BackProp(setGradFt);
            if(setGradFt.size() && INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i])->UpdatePara())
            {
                if(iBatchIdx) for(auto j=0; j<setGradFt[IDX_ZERO].size(); ++j)
                {
                    INSTANCE_DERIVE<BN_CONV>(mapBNData[iBatchIdx])->vecMiuBeta[j] += INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i])->BNData.vecMiuBeta[j];
                    INSTANCE_DERIVE<BN_CONV>(mapBNData[iBatchIdx])->vecSigmaSqr[j] += INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i])->BNData.vecSigmaSqr[j];   
                }
                else INSTANCE_DERIVE<BN_CONV>(mapBNData[iBatchIdx]) = std::make_shared<BN_CONV>(std::move(INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i])->BNData));
                break;
            }
            else return false;
        case POOL:
            setGradFt = INSTANCE_DERIVE<LAYER_POOL>(lsLayer[i]) -> BackProp(setGradFt);
            if(setGradFt.size()) break;
            else return false;
        case TRANS:
            if(INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> bFeatToVec)
            {
                setGradFt = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> BackProp(setGradVec);
                if(setGradFt.size()) break;
                else return false;
            }
            else
            {
                setGradVec = INSTANCE_DERIVE<LAYER_TRANS>(lsLayer[i]) -> BackProp(setGradFt);
                if(setGradVec.size()) break;
                else return false;
            }
        default: return false;
        }
        return true;
    }
    NetClassfyInput GetCurrInput(set<feature> &setInput, set<vect> &setOrigin, uint64_t iCurrBatchIdx)
    {
        NetClassfyInput NetCurrInput;
        auto setCurrBatchDatasetIdx = CurrBatchDatasetIdx(iCurrBatchIdx);
        if(setCurrBatchDatasetIdx.size())
        {
            NetCurrInput.setInput = setInput.sub_queue(setCurrBatchDatasetIdx);
            NetCurrInput.setOrigin = setOrigin.sub_queue(setCurrBatchDatasetIdx);
        }
        else
        {
            NetCurrInput.setInput = setInput;
            NetCurrInput.setOrigin = setOrigin;
        }
        return NetCurrInput;
    }
public:
    void ValueCopy(NetBNMNIST &netSrc) { mapBNData = netSrc.mapBNData; }
    void ValueMove(NetBNMNIST &&netSrc) { mapBNData = move(netSrc.mapBNData); }
    NetBNMNIST(NetBNMNIST &netSrc) : NetClassify(netSrc) { ValueCopy(netSrc); }
    NetBNMNIST(NetBNMNIST &&netSrc) : NetClassify(move(netSrc)) { ValueMove(move(netSrc)); }
    void operator=(NetBNMNIST &netSrc) { NetClassify::operator=(netSrc);  ValueCopy(netSrc); }
    void operator=(NetBNMNIST &&netSrc) { NetClassify::operator=(move(netSrc));  ValueMove(move(netSrc)); }

    NetBNMNIST(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bShowIter = true) : NetClassify(dNetAcc, iMinibatch, bShowIter) {}
    /* ACT_FT
    * uint64_t iActFuncType
    * ACT_VT
    * uint64_t iActFuncType
    *FC
    uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05
    * FC_BN
    double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10
    * CONV
    * uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0
    * CONV_BN
    * uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10
    * POOL
    * uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0
    * LayerTrans -
    * uint64_t iChannLnCnt, uint64_t iChannColCnt
    */
    template<typename LayerType, typename ... Args,  typename = enable_if_t<is_base_of_v<Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs)
    {
        auto iCurrLayerSize = lsLayer.size();
        auto bAddFlag = lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...));
        if(lsLayer[iCurrLayerSize]->iLayerType==FC_BN || lsLayer[iCurrLayerSize]->iLayerType==CONV_BN) mapBNData.insert(lsLayer.size(), std::make_shared<BN>());
        return bAddFlag;
    }
    bool Run(MNIST &mnistDataset)
    {
        if(mnistDataset.valid())
        {
            InitDatasetIdx(mnistDataset.size());
            auto setOrigin = mnistDataset.orgn();      
            auto iEpoch = 0, dTestAcc = 0, iBatchTrainCnt = 0;
            do
            {
                // Batch shuffling
                ShuffleIdx();
                for(auto i=0; i<iNetBatchCnt; ++i)
                {
                    auto CurrInput = GetCurrInput(mnistDataset.elem, setOrigin, i);
                    set<vect> setPreOutput;
                    // Train
                    auto setCurrOutput = ForwProp(CurrInput.setInput);
                    if(setCurrOutput.size())
                    {
                        auto bBatchIterFlag = IterFlag(setCurrOutput, CurrInput.setOrigin);
                        if(!bBatchIterFlag) ++ iBatchTrainCnt;
                        if(bShowIterFlag) IterShow(setCurrOutput, CurrInput.setOrigin);
                        std::cout << "[Epoch][" << iEpoch << "][Batch Index][" << i << ']' << std::endl;
                        if(bBatchIterFlag) if(!BackProp(setCurrOutput, CurrInput.setOrigin)) return false;
                    }
                    else return false;
                }
                ++ iEpoch;
            }
            while(iBatchTrainCnt < iNetBatchCnt);
            return true;
        }
        else return false;    
    }
};

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    // MNIST demo
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", {64});
    // dataset.output_bitmap("E:\\VS Code project data\\MNIST_out", BMIO_BMP);
    NetBNMNIST LeNet(0.1, 32);
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