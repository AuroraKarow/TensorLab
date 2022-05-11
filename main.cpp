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
            if(!vecInput.is_matrix()) {std::cout << "Layer - " << i << ", " << seqLayer[i]->iLayerType; return blank_vect;}
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
    void UpdatePara(uint64_t iCurrBatchIdx = 0, uint64_t iBatchCnt = 0)
    {
        auto bIsLastBatch = ((iCurrBatchIdx + 1) == iBatchCnt);
        for(auto i=0ui64; i<seqLayer.size(); ++i) switch (seqLayer[i] -> iLayerType)
        {
        case ACT: continue;
        case POOL_IM2COL: if(bThreadFlag) INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->RefreshAsyncSgn(); break;
        case TRANS_IM2COL: if(bThreadFlag) INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->RefreshAsyncSgn(); break;
        case FC: INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->UpdatePara(bThreadFlag); break;
        case FC_BN:
            INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                mapBNData[i].vecExp += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMuBeta;
                mapBNData[i].vecVar += INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr;
                if(bIsLastBatch) _FC BNDeduceInit(mapBNData[i], iBatchCnt, iNetMiniBatch);
            }
            else
            {
                mapBNData[i].vecExp = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecMuBeta);
                mapBNData[i].vecVar = std::move(INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->BNData.vecSigmaSqr);
            } break;
        case CONV_IM2COL: INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->UpdatePara(bThreadFlag); break;
        case CONV_BN_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->UpdatePara();
            if(iCurrBatchIdx)
            {
                mapBNData[i].vecExp += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColMuBeta;
                mapBNData[i].vecVar += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData.vecIm2ColSigmaSqr;
                if(bIsLastBatch) _CONV BNDeduceIm2ColInit(mapBNData[i], iBatchCnt, iNetMiniBatch);
            }
            else
            {
                mapBNData[i].vecExp = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData).vecIm2ColMuBeta;
                mapBNData[i].vecVar = std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->BNData).vecIm2ColSigmaSqr;
            } break;
        default: continue;
        }
    }
    vect Deduce(vect &vecInput)
    {
        for(auto i=0ui64; i<seqLayer.size(); ++i)
        {
            switch (seqLayer[i]->iLayerType)
            {
            case ACT: vecInput = INSTANCE_DERIVE<LAYER_ACT>(seqLayer[i])->Deduce(vecInput); break;
            case POOL_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case TRANS_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case FC: vecInput = INSTANCE_DERIVE<LAYER_FC>(seqLayer[i])->Deduce(vecInput); break;
            case FC_BN: vecInput = INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[i])->Deduce(vecInput, mapBNData[i]); break;
            case CONV_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[i])->Deduce(vecInput); break;
            case CONV_BN_IM2COL: vecInput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[i])->Deduce(vecInput, mapBNData[i]); break;
            default: continue;
            }
            if(!vecInput.is_matrix()) break;
        }
        return vecInput;
    }
    template<typename tCnt> void OutputAcc(vect &vecOutput, uint64_t iLbl, tCnt &iAccCnt, tCnt &iPrecCnt)
    {
        static_assert(std::is_integral<tCnt>::value || 
                    std::is_same<tCnt, async::shared_counter>::value,
                    "Counter value only.");
        auto dCurrAcc = 1 - vecOutput.pos_idx(iLbl);
        if(dCurrAcc < 0.5) { ++ iAccCnt; if(dCurrAcc < dAcc) ++ iPrecCnt; }
    }
    void TrainAcc(set<vect> &Output, set<uint64_t> &setLbl, uint64_t &iAccCnt, uint64_t &iPrecCnt) { for(auto i=0; i<Output.size(); ++i) OutputAcc(Output[i], setLbl[i], iAccCnt, iPrecCnt); }
    bool RunLinear(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        mnistTrainSet.init_batch(iNetMiniBatch);
        uint64_t iAccCnt = 0, iPrecCnt = 0;
        auto iEpoch = 0;
        do
        {
            CLOCK_BEGIN(0)
            mnistTrainSet.shuffle_batch(); ++ iEpoch;
            // Train
            for(auto i=0; i<mnistTrainSet.batch_cnt(); ++i)
            {
                CLOCK_BEGIN(1)
                mnistTrainSet.init_curr_set(i); iAccCnt = 0; iPrecCnt = 0;
                // FP
                auto setOutput = ForwProp(mnistTrainSet.curr_input_im2col, mnistTrainSet.ln_cnt());
                TrainAcc(setOutput, mnistTrainSet.curr_lbl, iAccCnt, iPrecCnt);
                // EPOCH_TRAIN_STATUS(setOutput, mnistTrainSet.curr_orgn);
                // BP & update parameters
                if(BackProp(setOutput, mnistTrainSet.curr_orgn)) UpdatePara(i, mnistTrainSet.batch_cnt());
                else return false;
                CLOCK_END(1)
                EPOCH_TRAIN_STATUS(iEpoch, i+1, mnistTrainSet.batch_cnt(), FRACTOR_RATE(iAccCnt, mnistTrainSet.batch_size()), FRACTOR_RATE(iPrecCnt, mnistTrainSet.batch_size()), CLOCK_DURATION(1));
            }
            // Deduce
            iAccCnt = 0; iPrecCnt = 0;
            for(auto i=0; i<mnistTestSet.size(); ++i) OutputAcc(Deduce(mnistTestSet.elem_im2col[i]), mnistTestSet.elem_lbl[i], iAccCnt, iPrecCnt);
            CLOCK_END(0)
            EPOCH_DEDUCE_STATUS(iEpoch, FRACTOR_RATE(iAccCnt, mnistTestSet.batch_size()), FRACTOR_RATE(iPrecCnt, mnistTestSet.batch_size()), CLOCK_DURATION(0));
        } while (iAccCnt != mnistTestSet.size());
        return true;
    }
    bool RunThread(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        // Batch thread
        async::async_batch asyncBatch(iNetMiniBatch);
        async::shared_counter iAccCnt = 0, iPrecCnt = 0, iProcCnt = 0;
        async::shared_signal bRtnFlag = true, 
                            bTrainMode = true, // true - train; false - deduce
                            bIterFlag = true; // true - Iteration; false - convergence
        std::condition_variable condBatch, condMain;
        std::mutex tdmtxBatch, tdmtxMain;
        // Load in batch threads
        for(auto i=0; i<iNetMiniBatch; ++i) asyncBatch.set_task(i,
            [this, &bRtnFlag, &bIterFlag, &bTrainMode, &condMain, &condBatch, &tdmtxBatch, &iAccCnt, &iPrecCnt, &iProcCnt, &mnistTrainSet, &mnistTestSet]
            (int idx)
            {
                while(true)
                {
                    {
                        // Wait for preparation of train set or test set in each batch
                        std::unique_lock<std::mutex> lk(tdmtxBatch);
                        condBatch.wait(lk);
                    }
                    if(!(bRtnFlag && bIterFlag)) break;
                    if(bTrainMode)
                    {
                        // FP
                        auto output = this->ForwProp(mnistTrainSet.curr_input_im2col[idx], mnistTrainSet.ln_cnt(), idx);
                        this->OutputAcc(output, mnistTrainSet.curr_lbl[idx], iAccCnt, iPrecCnt);
                        // BP
                        if(!this->BackProp(output, mnistTrainSet.curr_orgn[idx], idx)) bRtnFlag = false;
                    }
                    // Deduce
                    else this->OutputAcc(this->Deduce(mnistTestSet.curr_input_im2col[idx]), mnistTestSet.curr_lbl[idx], iAccCnt, iPrecCnt);
                    if((++iProcCnt) == iNetMiniBatch) condMain.notify_one();
                }
            }, i);
        // Dataset initialization
        mnistTrainSet.init_batch(iNetMiniBatch);
        mnistTestSet.init_batch(iNetMiniBatch);
        auto iEpoch = 0; bIterFlag = true;
        // Main thread
        do
        {
            CLOCK_BEGIN(0)
            // Train set shuffle for the epoch increment
            mnistTrainSet.shuffle_batch(); ++ iEpoch;
            // Switch on train
            bTrainMode = true;
            // Train
            for(auto i=0; i<mnistTrainSet.batch_cnt(); ++i)
            {
                // Current batch initialization
                CLOCK_BEGIN(1)
                mnistTrainSet.init_curr_set(i);
                // Wake up the batch threads for current training
                condBatch.notify_all();
                // Wait for this round
                {
                    std::unique_lock<std::mutex> lk(tdmtxMain);
                    condMain.wait(lk);
                    iProcCnt = 0;
                }
                if(bRtnFlag) UpdatePara(i, mnistTrainSet.batch_cnt());
                else 
                {
                    condBatch.notify_all();
                    return false;
                }
                CLOCK_END(1)
                auto dAcc = FRACTOR_RATE(iAccCnt, mnistTrainSet.batch_size()),
                    dPrec = FRACTOR_RATE(iPrecCnt, mnistTrainSet.batch_size());
                EPOCH_TRAIN_STATUS(iEpoch, i+1, mnistTrainSet.batch_cnt(), dAcc, dPrec, CLOCK_DURATION(1));
                iAccCnt = 0; iPrecCnt = 0;
            }
            // Deduce
            // Switch off train mode
            bTrainMode = false;
            for(auto i=0; i<mnistTestSet.batch_cnt(); ++i)
            {
                // Current batch initialization
                mnistTestSet.init_curr_set(i);
                condBatch.notify_all();
                // Wait for this round
                {
                    std::unique_lock<std::mutex> lk(tdmtxMain);
                    condMain.wait(lk);
                    iProcCnt = 0;
                }
                EPOCH_DEDUCE_PROG(i+1, mnistTestSet.batch_cnt());
            }
            CLOCK_END(0)
            auto dAcc = FRACTOR_RATE(iAccCnt, mnistTestSet.size()),
                dPrec = FRACTOR_RATE(iPrecCnt, mnistTestSet.size());
            EPOCH_DEDUCE_STATUS(iEpoch, dAcc, dPrec, CLOCK_DURATION(0));
            PRINT_ENTER
            bIterFlag = (iAccCnt != mnistTestSet.size());
            iAccCnt = 0; iPrecCnt = 0;
        } while (bIterFlag);
        return bRtnFlag;
    }
public:
    void ValueCopy(NetMNISTIm2Col &netSrc) { ValueAssign(netSrc); seqLayer = netSrc.seqLayer; mapBNData = netSrc.mapBNData; }
    void ValueMove(NetMNISTIm2Col &&netSrc) { ValueAssign(netSrc); seqLayer = std::move(netSrc.seqLayer); mapBNData = std::move(netSrc.mapBNData); }
    NetMNISTIm2Col(NetMNISTIm2Col &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetMNISTIm2Col(NetMNISTIm2Col &&netSrc) : NetBase(netSrc) { ValueMove(std::move(netSrc)); }
    void operator=(NetMNISTIm2Col &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetMNISTIm2Col &&netSrc) { NetBase::operator=(netSrc); ValueMove(std::move(netSrc)); }

    NetMNISTIm2Col(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bMultiThread = true) : NetBase(dNetAcc, iMinibatch), bThreadFlag(bMultiThread) {}
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
            INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerGradWeight.init(iNetMiniBatch);
            if(bThreadFlag) INSTANCE_DERIVE<LAYER_FC>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch); break;
        case FC_BN:
            mapBNData.insert(iCurrLayerSize, BN_EXP_VAR());
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_FC_BN>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            }
            break;
        case CONV_IM2COL:
            INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setGradKernel.init(iNetMiniBatch);
            INSTANCE_DERIVE<LAYER_CONV_IM2COL>(seqLayer[iCurrLayerSize])->setPrepInput.init(iNetMiniBatch); break;
        case CONV_BN_IM2COL:
            mapBNData.insert(iCurrLayerSize, BN_EXP_VAR());
            if(bThreadFlag)
            {
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setLayerInput.init(iNetMiniBatch);
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(seqLayer[iCurrLayerSize])->setGradLossToOutput.init(iNetMiniBatch);
            }
            break;
        case POOL_IM2COL:
            if(INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->iPoolType == POOL_MAX_IM2COL) INSTANCE_DERIVE<LAYER_POOL_IM2COL>(seqLayer[iCurrLayerSize])->setInputMaxPosList.init(iNetMiniBatch); break;
        default: break;
        }
        return bAddFlag;
    }
    bool Run(dataset::MNIST &mnistTrainSet, dataset::MNIST &mnistTestSet)
    {
        if(bThreadFlag) return RunThread(mnistTrainSet, mnistTestSet);
        else return RunLinear(mnistTrainSet, mnistTestSet);
    }
    void Reset() { mapBNData.reset(); seqLayer.reset(); }
    uint64_t Depth() { return seqLayer.size(); }
    ~NetMNISTIm2Col() { Reset(); }
private:
    bool bThreadFlag = false;

    NET_SEQ<LAYER_PTR> seqLayer;
    NET_MAP<uint64_t, BN_EXP_VAR> mapBNData;
    void ValueAssign(NetMNISTIm2Col &netSrc) { bThreadFlag = netSrc.bThreadFlag; }
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
    auto dLearnRate = 0.1;
    NetMNISTIm2Col LeNet(0.1, 125, true);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(20, 1, 5, 5, 1, 1, dLearnRate);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(20);
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(50, 20, 5, 5, 1, 1, dLearnRate);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(50);
    LeNet.AddLayer<LAYER_ACT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_TRANS_IM2COL>();
    LeNet.AddLayer<LAYER_FC>(800, 500, dLearnRate);
    LeNet.AddLayer<LAYER_FC_BN>();
    LeNet.AddLayer<LAYER_ACT>(SIGMOID);
    LeNet.AddLayer<LAYER_FC>(500, 10, dLearnRate);
    LeNet.AddLayer<LAYER_ACT>(SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    if(LeNet.Run(dataset, testset)) return EXIT_FAILURE;
    else return EXIT_SUCCESS;
}