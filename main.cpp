#pragma once

#include "dataset"
#include "neunet"

NEUNET_BEGIN

class NetMNISTIm2Col final : public NetBase
{
private:
    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        for(auto i=0; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: setInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->ForwProp(setInput); break;
            case POOL_IM2COL: 
                setInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->ForwProp(setInput, iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->iLayerOutputLnCnt; break;
            case TRANS_IM2COL:
                setInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(setInput, iInputLnCnt); break;
                iInputLnCnt = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iLnCnt; break;
            case FC: setInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->ForwProp(setInput); break;
            case FC_BN: setInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->ForwProp(setInput); break;
            case CONV_IM2COL: 
                setInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->ForwProp(setInput, iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerOutputLnCnt; break;
            case CONV_BN_IM2COL: setInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->ForwProp(setInput); break;
            default: return blank_vect_seq;
            }
            if(!setInput.size()) return blank_vect_seq;
        }
        return setInput;
    }
    bool BackProp(set<vect> &setOutput, set<vect> &setOrgn)
    {
        for(int i=seqLayer.size()-1; i>=0; --i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: setOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->BackProp(setOutput, setOrgn); break;
            case POOL_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case TRANS_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case FC: setOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->BackProp(setOutput); break;
            case FC_BN: setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BackProp(setOutput); break;
            case CONV_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            case CONV_BN_IM2COL: setOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BackProp(setOutput); break;
            default: return false;
            }
            if(!setOutput.size()) return false;
        }
        return true;
    }
    vect ForwProp(vect &vecInput, uint64_t iInputLnCnt, uint64_t iIdx)
    {
        for(auto i=0; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case POOL_IM2COL: 
                vecInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->ForwProp(vecInput, iInputLnCnt, iIdx);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->iLayerOutputLnCnt; break;
            case TRANS_IM2COL:
                vecInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->ForwProp(vecInput, iInputLnCnt, iIdx);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->iLnCnt; break;
            case FC: vecInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case FC_BN: vecInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            case CONV_IM2COL: 
                vecInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->ForwProp(vecInput, iInputLnCnt, iIdx);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->iLayerOutputLnCnt; break;
            case CONV_BN_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->ForwProp(vecInput, iIdx); break;
            default: return blank_vect;
            }
            if(!vecInput.is_matrix()) return blank_vect;
        }
        return vecInput;
    }
    bool BackProp(vect &vecOutput, vect &vecOrgn, uint64_t iIdx)
    {
        for(int i=seqLayer.size()-1; i>=0; --i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecOutput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->BackProp(vecOutput, iIdx, vecOrgn); break;
            case POOL_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case TRANS_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case FC: vecOutput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case FC_BN: vecOutput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case CONV_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            case CONV_BN_IM2COL: vecOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BackProp(vecOutput, iIdx); break;
            default: return false;
            }
            if(!vecOutput.is_matrix()) return false;
        }
        return true;
    }
    void UpdatePara(uint64_t iCurrBatchIdx = 0)
    {
        for(auto i=0ui64; i<seqLayer.size(); ++i) switch (seqLayer[i] -> iLayerType)
        {
        case ACT: continue;
        case POOL_IM2COL: if(bThreadFlag) INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->RefreshAsyncSgn(); break;
        case TRANS_IM2COL: if(bThreadFlag) INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->RefreshAsyncSgn(); break;
        case FC: INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->UpdatePara(); break;
        case FC_BN:
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecMiuBeta += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMiuBeta;
                INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecSigmaSqr += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr;
            }
            else mapBNData[i] = std::make_shared<BN_FC>(std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData)); break;
        case CONV_IM2COL: INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->UpdatePara(); break;
        case CONV_BN_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColMuBeta += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColMuBeta;
                INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColSigmaSqr += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColSigmaSqr;
            }
            else mapBNData[i] = std::make_shared<BN_CONV_IM2COL>(std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData)); break;
        default: continue;
        }
    }
    bool DeduceInit(uint64_t iTrainBatchCnt)
    {
        iAccCnt = 0; iPrecCnt = 0;
        for(auto i=0ui64; i<seqLayer.size(); ++i) 
        {
            bool bFlag = true;
            switch (seqLayer[i]->iLayerType)
            {
            case FC_BN: bFlag = _FC BNDataExp(INSTANCE_DERIVE<BN_FC>(mapBNData[i]), iNetMiniBatch, iTrainBatchCnt); break;
            case CONV_BN_IM2COL: bFlag = _CONV BNDataExpIm2Col(INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i]), iNetMiniBatch, iTrainBatchCnt); break;
            default: continue;
            }
            if(!bFlag) return false;
        }
        return true;
    }
    void DeduceAcc(vect &vecOutput, uint64_t iLbl)
    {
        auto dDeduceAcc = 1 - vecOutput.pos_idx(iLbl);
        if(dDeduceAcc < 0.5) { ++ iAccCnt; if(dDeduceAcc < dAcc) ++ iPrecCnt; }
    }
    void Deduce(vect &vecInput, uint64_t iLbl)
    {
        for(auto i=0ui64; i<seqLayer.size(); ++i) switch (seqLayer[i] -> iLayerType)
        {
        case ACT: vecInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->Deduce(vecInput); break;
        case POOL_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
        case TRANS_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
        case FC: vecInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->Deduce(vecInput); break;
        case FC_BN: vecInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->Deduce(vecInput, INSTANCE_DERIVE<BN_FC>(mapBNData[i])); break;
        case CONV_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
        case CONV_BN_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->Deduce(vecInput, INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])); break;
        default: continue;
        }
        DeduceAcc(vecInput, iLbl);
    }
    void Deduce(set<vect> &setInput, set<uint64_t> &setLbl) { for(auto i=0; i<setInput.size(); ++i) Deduce(setInput[i], setLbl[i]); }
    void TrainAcc(set<vect> &Output, set<uint64_t> &setLbl) { for(auto i=0; i<Output.size(); ++i) DeduceAcc(Output[i], setLbl[i]); }
    void ValueAssign(NetMNISTIm2Col &netSrc)
    {
        iAccCnt = netSrc.iAccCnt;
        iPrecCnt = netSrc.iPrecCnt;
        bThreadFlag = netSrc.bThreadFlag;
    }
public:
    void ValueCopy(NetMNISTIm2Col &netSrc) { ValueAssign(netSrc); seqLayer = netSrc.seqLayer; mapBNData = netSrc.mapBNData; }
    void ValueMove(NetMNISTIm2Col &&netSrc) { ValueAssign(netSrc); seqLayer = std::move(netSrc.seqLayer); mapBNData = std::move(netSrc.mapBNData); }
    NetMNISTIm2Col(NetMNISTIm2Col &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetMNISTIm2Col(NetMNISTIm2Col &&netSrc) : NetBase(netSrc) { ValueMove(std::move(netSrc)); }
    void operator=(NetMNISTIm2Col &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetMNISTIm2Col &&netSrc) { NetBase::operator=(netSrc); ValueMove(std::move(netSrc)); }

    NetMNISTIm2Col(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bMultiThread = true) : NetBase(dNetAcc, iMinibatch), bThreadFlag(bMultiThread), task_batch(bMultiThread?iMinibatch:0) {}
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
        case ACT: if(bThreadFlag) INSTANCE_DERIVE<LAYER_ACT>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch); break;
        case FC:
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerGradWeight.init(iNetMiniBatch);
            } break;
        case FC_BN:
            mapBNData.insert(iCurrLayerSize, std::make_shared<BN>());
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            }
            break;
        case CONV_IM2COL:
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setGradKernel.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setPrepInput.init(iNetMiniBatch);
            } break;
        case CONV_BN_IM2COL:
            mapBNData.insert(iCurrLayerSize, std::make_shared<BN>());
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            }
            break;
        case POOL_IM2COL:
            if(bThreadFlag && INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->iPoolType == POOL_MAX_IM2COL) INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->setInputMaxPosList.init(iNetMiniBatch); break;
        default: break;
        }
        return bAddFlag;
    }
    bool Run(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        mnistTrainSet.init_batch(iNetMiniBatch);
        auto iEpoch = 0;
        do
        {
            CLOCK_BEGIN(0)
            async::shared_counter asyCnt = 0;
            mnistTrainSet.shuffle_batch(); ++ iEpoch;
            for(auto i=0; i<mnistTrainSet.batch_cnt(); ++i)
            {
                CLOCK_BEGIN(1)
                mnistTrainSet.init_curr_set(i); iAccCnt = 0; iPrecCnt = 0; asyCnt = 0;
                if(bThreadFlag)
                {
                    auto bTrainFlag = true;
                    for(auto j=0; j<mnistTrainSet.batch_size(i); ++j) task_batch.set_task(j, 
                        [this](vect &input, vect &orgn, int lbl, bool &flag, async::shared_counter &cnt, int ln_cnt, int idx){
                            auto output = this->ForwProp(input, ln_cnt, idx);
                            this->DeduceAcc(output, lbl);
                            flag = this->BackProp(output, orgn, idx);
                            ++ cnt;
                        }, std::ref(mnistTrainSet.curr_input_im2col[j]), std::ref(mnistTrainSet.curr_orgn[j]), mnistTrainSet.curr_lbl[j], std::ref(bTrainFlag), std::ref(asyCnt), mnistTrainSet.ln_cnt(), j);
                    if(!bTrainFlag) return false;
                }
                else
                {
                    auto setCurrOutput = ForwProp(mnistTrainSet.curr_input_im2col, mnistTrainSet.ln_cnt());
                    TrainAcc(setCurrOutput, mnistTrainSet.curr_lbl);
                    if(!BackProp(setCurrOutput, mnistTrainSet.curr_orgn)) return false;
                }
                while(asyCnt != mnistTrainSet.batch_size(i)); UpdatePara(i);
                CLOCK_END(1)
                std::printf("\r[Epoch][%d][Progress][%d/%d][Train Acuuracy][%.2f][Train Precision][%.2f][Duration][%dms]", (int)iEpoch, (int)i, (int)mnistTrainSet.batch_cnt(), FRACTOR_RATE(iAccCnt, mnistTrainSet.curr_input_im2col.size()), FRACTOR_RATE(iPrecCnt, mnistTrainSet.curr_input_im2col.size()), (int)CLOCK_DURATION(1));
            }
            DeduceInit(mnistTrainSet.batch_cnt());
            if(bThreadFlag)
            {
                mnistTestSet.init_batch(iNetMiniBatch);
                for(auto i=0; i<mnistTestSet.batch_cnt(); ++i)
                {
                    mnistTestSet.init_curr_set(i);
                    for(auto j=0; j<mnistTestSet.batch_size(); ++j) task_batch.set_task(j,
                        [this](vect &input, int lbl, async::shared_counter &cnt){
                            this->Deduce(input, lbl);
                            ++ cnt;
                        }, std::ref(mnistTestSet.curr_input_im2col[j]), mnistTestSet.curr_lbl[j], std::ref(asyCnt));
                    while(asyCnt != mnistTestSet.batch_size()); asyCnt = 0;
                }
            }
            else Deduce(mnistTestSet.elem_im2col, mnistTestSet.elem_lbl);
            CLOCK_END(0)
            std::printf("\r[Epoch][%d][Deduce Acuuracy][%lf][Deduce Precision][%lf][Duration][%dms]", iEpoch, FRACTOR_RATE(iAccCnt, mnistTestSet.size()), FRACTOR_RATE(iPrecCnt, mnistTestSet.size()), CLOCK_DURATION(0));
        } while (iAccCnt == mnistTestSet.size());
        return true;
    }
    void Reset() { mapBNData.reset(); seqLayer.reset(); }
    uint64_t Depth() { return seqLayer.size(); }
    ~NetMNISTIm2Col() { Reset(); }
private:
    async::async_batch task_batch;
    async::shared_counter iAccCnt = 0, iPrecCnt = 0;
    bool bThreadFlag = false;
    NET_SEQ<LAYER_PTR> seqLayer;
    NET_MAP<uint64_t, BN_PTR> mapBNData;
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
    string root_dir = "E:\\VS Code project data\\MNIST\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", true);
    // dataset.output_bitmap("E:\\VS Code project data\\MNIST_out\\train", BMIO_BMP);
    MNIST testset(root_dir + "t10k-images.idx3-ubyte", root_dir + "t10k-labels.idx1-ubyte", true);
    // testset.output_bitmap("E:\\VS Code project data\\MNIST_out\\test", BMIO_BMP);
    NetMNISTIm2Col LeNet(0.1, 125);
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
    if(LeNet.Run(dataset, testset)) return EXIT_FAILURE;
    else return EXIT_SUCCESS;
}