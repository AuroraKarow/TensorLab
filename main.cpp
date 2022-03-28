#pragma once

#include "dataset"
#include "neunet"

#include "Iomp.h"

class NetMNISTIm2Col final : public neunet::NetBase
{
private:
    set<vect> ForwProp(uint64_t iInputLnCnt)
    {
        set<vect> setOutput = std::move(setCurrBatchInput);
        for(auto i=0ui64; i<lsLayer.size(); ++i)
        {
            switch (lsLayer[i] -> iLayerType)
            {
            case ACT_VT:
                setOutput = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> ForwProp(setOutput);
                break;
            case FC:
                setOutput = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> ForwProp(setOutput);
                break;
            case FC_BN:
                setOutput = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> ForwProp(setOutput);
                break;
            case CONV_IM2COL:
                setOutput = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> ForwProp(setOutput, iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> iLayerOutputLnCnt;
                break;
            case CONV_BN_IM2COL:
                setOutput = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> ForwProp(setOutput);
                break;
            case POOL_IM2COL:
                setOutput = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(lsLayer[i]) -> ForwProp(setOutput, iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(lsLayer[i]) -> iLayerOutputLnCnt;
                break;
            case TRANS_IM2COL:
                if(INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(lsLayer[i]) -> bFeatToVec) setOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(lsLayer[i]) -> ForwProp(setOutput, iInputLnCnt);
                else setOutput = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(lsLayer[i]) -> ForwProp(setOutput);
                break;
            default: return blank_vect_seq;
            }
            if(!setOutput.size())  return blank_vect_seq;
        }
        return setOutput;
    }
    bool BackProp(set<vect> &setOutput, uint64_t iCurrBatchIdx = 0)
    {
        set<vect> setGrad = std::move(setOutput);
        for(auto i=lsLayer.size()-1; i>0; --i)
        {
            auto bUpdateFlag = true;
            switch (lsLayer[i] -> iLayerType)
            {
            case ACT_VT:
                setGrad = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> BackProp(setGrad, setCurrBatchOrigin);
                break;
            case FC:
                setGrad = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> BackProp(setGrad);
                bUpdateFlag = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> UpdatePara();
                break;
            case FC_BN:
                setGrad = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> BackProp(setGrad);
                bUpdateFlag = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> UpdatePara();
                if(iCurrBatchIdx)
                {
                    INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecMiuBeta += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecMiuBeta;
                    INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecSigmaSqr += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecSigmaSqr;
                }
                else mapBNData[i] = std::make_shared<BN_FC>(std::move(INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData));
                break;
            case CONV_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                bUpdateFlag = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i])->UpdatePara();
                break;
            case CONV_BN_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                bUpdateFlag = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> UpdatePara();
                if(iCurrBatchIdx)
                {
                    INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColMuBeta += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i])->BNData.vecIm2ColMuBeta;
                    INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i])->vecIm2ColSigmaSqr += INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i])->BNData.vecIm2ColSigmaSqr;
                }
                else mapBNData[i] = std::make_shared<BN_CONV_IM2COL>(std::move(INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i])->BNData));
                break;
            case POOL_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                break;
            case TRANS_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                break;
            default: return false;
            }
            if(!setGrad.size() || !bUpdateFlag) return false;
        }
        return true;
    }
    set<vect> Deduce(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        set<vect> setOutput(setInput.size());
        for(auto i=0ui64; i<setInput.size(); ++i) 
        {
            setOutput[i] = setInput[i];
            for(auto j=0; j<lsLayer.size(); ++j) switch (lsLayer[i] -> iLayerType)
            {
            case ACT_VT:
                setOutput[i] = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> Deduce(setOutput[i]);
                break;
            case FC:
                setOutput[i] = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> Deduce(setOutput[i]);
                break;
            case FC_BN:
                if(i) setOutput[i] = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> Deduce(setOutput[i], INSTANCE_DERIVE<BN_FC>(mapBNData[i]));
                else setOutput[i] = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> Deduce(setOutput[i], INSTANCE_DERIVE<BN_FC>(mapBNData[i]), iNetMiniBatch, iNetBatchCnt);
                break;
            case CONV_IM2COL:
                setOutput[i] = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> Deduce(setOutput[i], iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> iLayerOutputLnCnt;
                break;
            case CONV_BN_IM2COL:
                if(i) setOutput[i] = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> Deduce(setOutput[i], INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i]));
                else setOutput[i] = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> Deduce(setOutput[i], INSTANCE_DERIVE<BN_CONV_IM2COL>(mapBNData[i]), iNetMiniBatch, iNetBatchCnt);
                break;
            case POOL_IM2COL:
                setOutput[i] = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(lsLayer[i]) -> Deduce(setOutput[i], iInputLnCnt);
                iInputLnCnt = INSTANCE_DERIVE<LAYER_POOL_IM2COL>(lsLayer[i]) -> iLayerOutputLnCnt;
                break;
            case TRANS_IM2COL:
                setOutput[i] = INSTANCE_DERIVE<LAYER_TRANS_IM2COL>(lsLayer[i]) -> Deduce(setOutput[i]);
                break;
            default: return blank_vect_seq;
            }
            if(!setOutput[i].is_matrix()) return blank_vect_seq;
        }
        return setOutput;
    }
    void InitCurrBatchTrainSet(set<vect> &setInput, set<vect> &setOrigin, uint64_t iCurrBatchIdx)
    {
        auto setCurrBatchDatasetIdx = CurrBatchDatasetIdx(iCurrBatchIdx);
        if(setCurrBatchDatasetIdx.size())
        {
            setCurrBatchInput = setInput.sub_queue(setCurrBatchDatasetIdx);
            setCurrBatchOrigin = setOrigin.sub_queue(setCurrBatchDatasetIdx);
        }
        else
        {
            setCurrBatchInput = setInput;
            setCurrBatchOrigin = setOrigin;
        }
    }
    uint64_t IterPass(set<vect> &setCurrOutput)
    {
        auto iPassCnt = 0;
        for(auto i=0; i<setCurrBatchOrigin.size(); i++) for(auto j=0; j<setCurrBatchOrigin[i].LN_CNT; ++j)
            if(setCurrBatchOrigin[i].pos_idx(j) && setCurrOutput[i].pos_idx(j) > (1-dAcc))
            {
                iPassCnt ++;
                break;
            }
        return iPassCnt;
    }
    void IterShow(set<vect> &setCurrOutput)
    {
        for(auto i=0; i<setCurrBatchOrigin.size(); i++)
        {
            std::cout << " [BarY]\t\t[Label]\t[Y]" << std::endl;
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                if(setCurrBatchOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << setCurrOutput[i].pos_idx(j) << '\t';
                std::cout << j << '\t';
                std::cout << setCurrBatchOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }

    NET_MAP<uint64_t, BN_PTR> mapBNData;
    set<vect> setCurrBatchInput, setCurrBatchOrigin;
    bool bMonitorFlag = false;
public:
    NetMNISTIm2Col(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bMonitor = true) : NetBase(dNetAcc, iMinibatch), bMonitorFlag(bMonitor) {}
    /*LAYER_ACT_VT
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
        auto iCurrLayerSize = lsLayer.size();
        auto bAddFlag = lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...));
        if(lsLayer[iCurrLayerSize]->iLayerType==FC_BN || lsLayer[iCurrLayerSize]->iLayerType==CONV_BN_IM2COL) mapBNData.insert(iCurrLayerSize, std::make_shared<BN>());
        return bAddFlag;
    }
    bool Run(dataset::MNIST &mnistTrainSet/*, dataset::MNIST &mnistTestSet*/)
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
                    CLOCK_BEGIN
                    auto setCurrOutput = ForwProp(mnistTrainSet.ln_cnt());
                    if(setCurrOutput.size())
                    {
                        auto iSglTestPassCnt = IterPass(setCurrOutput);
                        if(bMonitorFlag) IterShow(setCurrOutput);
                        else std::printf("\r[Epoch][%d][Batch Index][%d/%d][Accuracy][%.2f]", iEpoch+1, i+1, (int)iNetBatchCnt, iSglTestPassCnt*1.0/setCurrOutput.size());
                        if(iSglTestPassCnt != setCurrOutput.size()) if(!BackProp(setCurrOutput, i)) return false;
                        CLOCK_END
                        std::cout << "[Duration][" << CLOCK_DURATION << ']';
                        iTestPassCnt += iSglTestPassCnt;
                    }
                    else return false;
                }
                std::printf("\r[Epoch][%d][Accuracy][%lf]\n", iEpoch+1, iTestPassCnt*1.0/setOrigin.size());
                ++ iEpoch;
            } while (iTestPassCnt < mnistTrainSet.size());
            return true;
        }
        else return false;
    }
};

using namespace std;
using namespace mtx;
using namespace dataset;
using namespace neunet;
using namespace layer;

int main(int argc, char *argv[], char *envp[])
{
    cout << "hello, world." << endl;

    
    //// this only works in visual stdio
    //cout << "max can use thead cnt:" << omps::Iomp::getMaxCanUseThreadCnt() << endl
    //     << "max processor cnt on this machine:" << omps::Iomp::getMaxProcessorCnt() << endl;
    //

    // MNIST demo
    string root_dir = "D:\\code\\cpp\\mnist\\";
    MNIST dataset(root_dir + "train-images.idx3-ubyte", root_dir + "train-labels.idx1-ubyte", true);
    // dataset.output_bitmap("E:\\VS Code project data\\MNIST_out\\train", BMIO_BMP);
    // MNIST testset(root_dir + "t10k-images.idx3-ubyte", root_dir + "t10k-labels.idx1-ubyte");
    // testset.output_bitmap("E:\\VS Code project data\\MNIST_out\\test", BMIO_BMP);
    NetMNISTIm2Col LeNet(0.1, 32, false);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(20, 1, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(20);
    LeNet.AddLayer<LAYER_ACT_VT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_CONV_IM2COL>(50, 20, 5, 5, 1, 1);
    LeNet.AddLayer<LAYER_CONV_BN_IM2COL>(50);
    LeNet.AddLayer<LAYER_ACT_VT>(RELU);
    LeNet.AddLayer<LAYER_POOL_IM2COL>(POOL_MAX_IM2COL, 2, 2, 2, 2);
    LeNet.AddLayer<LAYER_TRANS_IM2COL>();
    LeNet.AddLayer<LAYER_FC>(800, 500);
    LeNet.AddLayer<LAYER_FC_BN>();
    LeNet.AddLayer<LAYER_ACT_VT>(SIGMOID);
    LeNet.AddLayer<LAYER_FC>(500, 10);
    LeNet.AddLayer<LAYER_ACT_VT>(SOFTMAX);
    cout << "[LeNet depth][" << LeNet.Depth() << ']' << endl;
    LeNet.Run(dataset);
    return EXIT_SUCCESS;
}