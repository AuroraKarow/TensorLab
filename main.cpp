#pragma once

#include "dataset"
#include "neunet"

void monitor(int td_idx, int lyr_idx, std::string lyr_name, bool is_forw) { std::printf("Thread[%d]Layer[%d - %s][%s]\n", td_idx, lyr_idx, lyr_name.c_str(), (is_forw ? "FP" : "BP")); }

NEUNET_BEGIN

class NetMNISTIm2ColThread final : public NetMNISTIm2Col
{
private:
    vect ForwProp(uint64_t iInputLnCnt, uint64_t iThreadIdx)
    {
        auto vecOutput = std::move(setCurrBatchInput[iThreadIdx]);
        for(auto i=0; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i] -> iLayerType)
            {
            case ACT: vecOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->ForwProp(vecOutput, iThreadIdx); break;
            case FC: vecOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->ForwProp(vecOutput, iThreadIdx); break;
            case FC_BN: vecOutput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->ForwProp(vecOutput, iThreadIdx); break;
            case CONV_IM2COL:
                vecOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->ForwProp(vecOutput, iInputLnCnt, iThreadIdx);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerOutputLnCnt;
                break;
            case CONV_BN_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->ForwProp(vecOutput, iThreadIdx); break;
            case POOL_IM2COL:
                vecOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->ForwProp(vecOutput, iInputLnCnt, iThreadIdx);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->iLayerOutputLnCnt;
                break;
            case TRANS_IM2COL:
                if(INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->bFeatToVec) vecOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(vecOutput, iInputLnCnt, iThreadIdx);
                else vecOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(vecOutput);
                break;
            default: return blank_vect;
            }
            if(!vecOutput.is_matrix())  return blank_vect;
        }
        return vecOutput;
    }
    bool BackProp(vect &vecOutput, uint64_t iThreadIdx)
    {
        for(int i=seqLayer.size()-1; i>=0; --i)
        {
            switch (seqLayer[i] -> iLayerType)
            {
            case ACT: vecOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->BackProp(vecOutput, iThreadIdx, setCurrBatchOrigin[iThreadIdx]); break;
            case FC: vecOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            case FC_BN: vecOutput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            case CONV_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            case CONV_BN_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            case POOL_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            case TRANS_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->BackProp(vecOutput, iThreadIdx); break;
            default: return false;
            }
            if(!vecOutput.is_matrix()) return false;
        }
        return true;
    }
    void UpdatePara(uint64_t iCurrBatchIdx = 0)
    {
        for(auto i=0; i<seqLayer.size(); ++i) switch (seqLayer[i] -> iLayerType)
        {
        case ACT: continue;
        case POOL_IM2COL: INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->UpdatePara(); break;
        case TRANS_IM2COL: INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->UpdatePara(); break;
        case FC: INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->UpdatePara(); break;
        case FC_BN:
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecMiuBeta += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMiuBeta;
                INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecSigmaSqr += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr;
            }
            else mapBNData[i] = std::make_shared<BN_FC>(std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData));
            break;
        case CONV_IM2COL: INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->UpdatePara(); break;
        case CONV_BN_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColMuBeta += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColMuBeta;
                INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColSigmaSqr += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColSigmaSqr;
            }
            else mapBNData[i] = std::make_shared<BN_CONV_IM2COL>(std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData));
            break;
        default: continue;
        }
    }
    bool PassTest(vect &vecOutput, uint64_t iThreadIdx)
    {
        for(auto i=0; i<vecOutput.ELEM_CNT; ++i) if(setCurrBatchOrigin[iThreadIdx].pos_idx(i))
        {
            if(1-vecOutput.pos_idx(i) > dAcc) return false;
            else return true;
        }
        return false;
    }
    friend void TrainRound(NetMNISTIm2ColThread *netSrc, bool &bTrainFlag, uint64_t iInputLnCnt, uint64_t iThreadIdx = 0)
    {
        auto vecOutput = netSrc->ForwProp(iInputLnCnt, iThreadIdx);
        auto bPassSgn = netSrc->PassTest(vecOutput, iThreadIdx);
        bTrainFlag = netSrc->BackProp(vecOutput, iThreadIdx);
        if(bPassSgn) netSrc->iPassCnt.increment();
        netSrc->iTrainCnt.increment();
    }
    lock::lock_counter iPassCnt, iTrainCnt;
    uint64_t iIterCnt = 0;
    bool iBlockThread = false, iActThread = true;
    NET_SEQ<LAYER_PTR> seqLayer;
public:
    NetMNISTIm2ColThread(NetMNISTIm2ColThread &netSrc) : NetMNISTIm2Col(netSrc) { seqLayer = netSrc.seqLayer; }
    NetMNISTIm2ColThread(NetMNISTIm2ColThread &&netSrc) : NetMNISTIm2Col(std::move(netSrc)) { seqLayer = std::move(netSrc.seqLayer); }
    void operator=(NetMNISTIm2ColThread &netSrc) { NetMNISTIm2Col::operator=(netSrc); seqLayer = netSrc.seqLayer; }
    void operator=(NetMNISTIm2ColThread &&netSrc) { NetMNISTIm2Col::operator=(std::move(netSrc)); seqLayer = std::move(netSrc.seqLayer); }

    NetMNISTIm2ColThread(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bMonitor = true) : NetMNISTIm2Col(dNetAcc, iMinibatch, bMonitor) {}
    /*LAYER_ACT_SGL
    * uint64_t iActFuncType
    * LAYER_FC
    * uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05
    * LAYER_FC_BN
    * double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10
    * LAYER_CONV_IM2COL
    * uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, double dLearnRate = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5
    * LAYER_CONV_BN_IM2COL
    * uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10
    * LAYER_POOL_IM2COL
    * uint64_t iPoolTypeVal = POOL_MAX_IM2COL, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0
    * LAYER_TRANS_IM2COL
    * 
    * uint64_t iChannLnCnt, uint64_t iChannColCnt
    */
    template<typename LayerType, typename ... Args,  typename = enable_if_t<is_base_of_v<Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs)
    {
        auto iCurrLayerSize = seqLayer.size();
        auto bAddFlag = seqLayer.emplace_back(std::make_shared<LayerType>(pacArgs...));
        switch (seqLayer[iCurrLayerSize] -> iLayerType)
        {
        case ACT: INSTANCE_DERIVE<LAYER_ACT>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch); break;
        case FC:
            INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
            INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerGradWeight.init(iNetMiniBatch);
            break;
        case FC_BN:
            mapBNData.insert(iCurrLayerSize, std::make_shared<BN>());
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            break;
        case CONV_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setGradKernel.init(iNetMiniBatch);
            INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setPrepInput.init(iNetMiniBatch);
            break;
        case CONV_BN_IM2COL:
            mapBNData.insert(iCurrLayerSize, std::make_shared<BN>());
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            break;
        case POOL_IM2COL:
            if(INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->iPoolType == POOL_MAX_IM2COL)INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->setInputMaxPosList.init(iNetMiniBatch); break;
        default: break;
        }
        return bAddFlag;
    }
    bool Run(dataset::MNIST &mnistTrainSet)
    {
        if(mnistTrainSet.valid())
        {
            InitDatasetIdx(mnistTrainSet.size());
            auto setOrigin = mnistTrainSet.orgn();
            auto iEpoch = 0, iTestPassCnt = 0;
            do
            {
                ShuffleIdx();
                iTestPassCnt = 0;
                for(auto i=0; i<iNetBatchCnt; ++i)
                {
                    InitCurrBatchTrainSet(mnistTrainSet.elem_im2col, setOrigin, i);
                    auto iCurrBatchSize = CurrBatchSize(i);
                    CLOCK_BEGIN
                    set<std::thread> setTd(iNetMiniBatch);
                    for(auto j=0; j<iCurrBatchSize; ++j)
                    {
                        bool bTrainFlag = true;
                        setTd[j] = std::thread(TrainRound, this, std::ref(bTrainFlag), mnistTrainSet.ln_cnt(), j);
                        if(!bTrainFlag) return false;
                    }
                    for(auto j=0; j<iCurrBatchSize; ++j) setTd[j].join();
                    while(iTrainCnt.get_cnt() != iCurrBatchSize);
                    iTrainCnt.set_cnt();
                    UpdatePara(i);
                    CLOCK_END
                    std::printf("\r[Epoch][%d][Batch Index][%d/%d][Accuracy][%.2f][Duration][%dms]  ", iEpoch+1, i+1, (int)iNetBatchCnt, iPassCnt.get_cnt()*1.0/iCurrBatchSize, CLOCK_DURATION);
                    iTestPassCnt += iPassCnt.get_cnt();
                    iPassCnt.set_cnt();
                }
                std::printf("\r[Epoch][%d][Accuracy][%lf]\n", iEpoch+1, iTestPassCnt*1.0/setOrigin.size());
                ++ iEpoch;
            } while (iTestPassCnt < mnistTrainSet.size());
            return true;
        }
        else return false;
    }
    void Reset() { NetMNISTIm2Col::Reset(); seqLayer.reset(); }
    ~NetMNISTIm2ColThread() { Reset(); }
};

NEUNET_END

using namespace std;
using namespace mtx;
using namespace dataset;
using namespace neunet;
using namespace layer;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;
    // MNIST demo
    string root_dir = "D:\\code\\cpp\\mnist\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", true);
    // dataset.output_bitmap("E:\\VS Code project data\\MNIST_out\\train", BMIO_BMP);
    // MNIST testset(root_dir + "t10k-images.idx3-ubyte", root_dir + "t10k-labels.idx1-ubyte");
    // testset.output_bitmap("E:\\VS Code project data\\MNIST_out\\test", BMIO_BMP);
    NetMNISTIm2ColThread LeNet(0.1, 32, false);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(20, 1, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(20);
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(50, 20, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(50);
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_TRANS_IM2COL>();
    LeNet.AddLayer<LAYER_FC>(800, 500);
    LeNet.AddLayer<LAYER_FC_BN>();
    LeNet.AddLayer<LAYER_ACT>(SIGMOID);
    LeNet.AddLayer<LAYER_FC>(500, 10);
    LeNet.AddLayer<LAYER_ACT>(SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    if(LeNet.Run(dataset)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}