NEUNET_BEGIN

class NetBase
{
protected:
    double dAcc = 1e-5;
    uint64_t iNetMiniBatch = 0, iNetBatchCnt = 0, iNetRearBatchSize = 0;
    NET_LIST<LAYER_PTR> lsLayer;
    set<uint64_t> setDatasetIdx;
    
    void InitDatasetIdx(uint64_t iDataSize)
    {
        if(!iNetMiniBatch) iNetMiniBatch = iDataSize;
        iNetBatchCnt = iDataSize / iNetMiniBatch;
                // Last bacth's size
        iNetRearBatchSize = iDataSize % iNetMiniBatch;
        if(iNetRearBatchSize) ++ iNetBatchCnt;
        if(iNetMiniBatch != iDataSize)
        {
            setDatasetIdx.init(iDataSize);
            for(auto i=0; i<setDatasetIdx.size(); ++i) setDatasetIdx[i] = i;
        }
    }
    void ShuffleIdx() { if(setDatasetIdx.size()) setDatasetIdx.shuffle(); }
    set<uint64_t> CurrBatchDatasetIdx(uint64_t iCurrBatchIdx)
    {
        if(setDatasetIdx.size())
        {
            auto iBatchSize = iNetMiniBatch;
            if(iNetRearBatchSize && iCurrBatchIdx+1==iNetBatchCnt) iBatchSize = iNetRearBatchSize;
            // Dataset shuffled indexes for current batch
            return setDatasetIdx.sub_queue(mtx::mtx_elem_pos(iCurrBatchIdx, 0, iNetMiniBatch), mtx::mtx_elem_pos(iCurrBatchIdx, iBatchSize-1, iNetMiniBatch));
        }
        else return set<uint64_t>::blank_queue();
    }

    virtual void ValueAssign(NetBase &netSrc)
    {
        dAcc = netSrc.dAcc;
        iNetMiniBatch = netSrc.iNetMiniBatch;
        iNetBatchCnt = netSrc.iNetBatchCnt;
        iNetRearBatchSize = netSrc.iNetRearBatchSize;
    }
    void ShowIter() {}
    bool IterFlag() { return true; }
    bool ForwProp() { return true; }
    bool BackProp() { return true; }
    set<vect> Deduce() { return blank_vect_seq; }
public:
    virtual void ValueCopy(NetBase &netSrc)
    {
        ValueAssign(netSrc);
        lsLayer = netSrc.lsLayer;
        setDatasetIdx = netSrc.setDatasetIdx;
    }
    virtual void ValueMove(NetBase &&netSrc)
    {
        ValueAssign(netSrc);
        lsLayer = std::move(netSrc.lsLayer);
        setDatasetIdx = std::move(netSrc.setDatasetIdx);
    }
    NetBase(NetBase &netSrc) { ValueCopy(netSrc); }
    NetBase(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    void operator=(NetBase &netSrc) { ValueCopy(netSrc); }
    void operator=(NetBase &&netSrc) { ValueMove(std::move(netSrc)); }
    
    NetBase(double dNetAcc = 1e-5, uint64_t iMinibatch = 0) : dAcc(dNetAcc), iNetMiniBatch(iMinibatch) {}
    template<typename LayerType, typename ... Args,  typename = std::enable_if_t<std::is_base_of_v<_LAYER Layer, LayerType>>> bool AddLayer(Args&& ... pacArgs) { return lsLayer.emplace_back(std::make_shared<LayerType>(pacArgs...)); }
    uint64_t Depth() { return lsLayer.size(); }
    bool Run() { return true; }
    void Reset()
    {
        setDatasetIdx.reset();
        lsLayer.reset();
    }
    ~NetBase() { Reset(); }
};

class NetClassify : public NetBase
{
protected:
    bool bShowIterFlag = false;

    void ValueAssign(NetClassify &netSrc) { bShowIterFlag = netSrc.bShowIterFlag; }
    void IterShow(set<vect> &setPreOutput, set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        if(!setPreOutput.size()) setPreOutput = set<vect>(setOrigin.size());
        for(auto i=0; i<setOrigin.size(); i++)
        {
            if(!setPreOutput[i].is_matrix()) setPreOutput[i] = vect(setCurrOutput[i].LN_CNT, setCurrOutput[i].COL_CNT);
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                auto dCurrVal = setCurrOutput[i].pos_idx(j);
                if(setOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << dCurrVal << '\t';
                std::cout << j << '\t';
                double dif = dCurrVal - setPreOutput[i].pos_idx(j);
                setPreOutput[i].pos_idx(j) = dCurrVal;
                if(dif < 0) std::cout << dif;
                else std::cout << '+' << dif;
                std::cout << '\t';
                std::cout << setOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }
    void IterShow(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        for(auto i=0; i<setOrigin.size(); i++)
        {
            std::cout << " [BarY]\t\t[Label]\t[Y]" << std::endl;
            for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j)
            {
                if(setOrigin[i].pos_idx(j)) std::cout << '>';
                else std::cout << ' ';
                std::cout << setCurrOutput[i].pos_idx(j) << '\t';
                std::cout << j << '\t';
                std::cout << setOrigin[i].pos_idx(j) << std::endl;
            }
            std::cout << std::endl;
        }
    }
    uint64_t IterPass(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        auto iPassCnt = 0;
        for(auto i=0; i<setOrigin.size(); i++) for(auto j=0; j<setOrigin[i].LN_CNT; ++j)
            if(setOrigin[i].pos_idx(j) && setCurrOutput[i].pos_idx(j) > (1-dAcc))
            {
                iPassCnt ++;
                break;
            }
        return iPassCnt;
    }
    bool IterFlag(set<vect> &setCurrOutput, set<vect> &setOrigin)
    {
        for(auto i=0; i<setCurrOutput.size(); ++i) for(auto j=0; j<setCurrOutput[i].LN_CNT; ++j) if(setOrigin[i][j][IDX_ZERO]) if(std::abs(1-setCurrOutput[i][j][IDX_ZERO]) > dAcc) return true;
        return false;
    }
public:
    struct NetClassfyInput
    {
        set<feature> setInput;
        set<vect> setOrigin;
    };
    virtual void ValueCopy(NetClassify &netSrc) { ValueAssign(netSrc); }
    virtual void ValueMove(NetClassify &&netSrc) { ValueAssign(netSrc); }
    NetClassify(NetClassify &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetClassify(NetClassify &&netSrc) : NetBase(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetClassify &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetClassify &&netSrc) { NetBase::operator=(std::move(netSrc)); ValueMove(std::move(netSrc)); }
    
    NetClassify(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bShowIter = false) : NetBase(dNetAcc, iMinibatch), bShowIterFlag(bShowIter) {}
};

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
            INSTANCE_DERIVE<LAYER_FC>(lsLayer[i])->UpdatePara();
            if(setGradVec.size()) break;
            else return false;
        case FC_BN:
            setGradVec = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> BackProp(setGradVec);
            INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->UpdatePara();
            if(setGradVec.size())
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
            INSTANCE_DERIVE<LAYER_CONV>(lsLayer[i])->UpdatePara();
            if(setGradFt.size()) break;
            else return false;
        case CONV_BN:
            setGradFt = INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i]) -> BackProp(setGradFt);
            INSTANCE_DERIVE<LAYER_CONV_BN>(lsLayer[i])->UpdatePara();
            if(setGradFt.size())
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
    void ValueMove(NetBNMNIST &&netSrc) { mapBNData = std::move(netSrc.mapBNData); }
    NetBNMNIST(NetBNMNIST &netSrc) : NetClassify(netSrc) { ValueCopy(netSrc); }
    NetBNMNIST(NetBNMNIST &&netSrc) : NetClassify(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetBNMNIST &netSrc) { NetClassify::operator=(netSrc);  ValueCopy(netSrc); }
    void operator=(NetBNMNIST &&netSrc) { NetClassify::operator=(std::move(netSrc));  ValueMove(std::move(netSrc)); }

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
    bool Run(dataset::MNIST &mnistDataset)
    {
        if(mnistDataset.valid())
        {
            InitDatasetIdx(mnistDataset.size());
            auto setOrigin = mnistDataset.orgn();      
            auto iEpoch = 0, iTestPassCnt = 0;;
            do
            {
                // Batch shuffling
                ShuffleIdx();
                iTestPassCnt = 0;
                CLOCK_BEGIN(0)
                for(auto i=0; i<iNetBatchCnt; ++i)
                {
                    auto CurrInput = GetCurrInput(mnistDataset.elem, setOrigin, i);
                    // Train
                    auto setCurrOutput = ForwProp(CurrInput.setInput);
                    if(setCurrOutput.size())
                    {
                        if(bShowIterFlag) IterShow(setCurrOutput, CurrInput.setOrigin);
                        auto iSglTestPassCnt = IterPass(setCurrOutput, CurrInput.setOrigin);
                        if(IterFlag(setCurrOutput, CurrInput.setOrigin)) if(!BackProp(setCurrOutput, CurrInput.setOrigin)) return false;
                        std::printf("\r[Epoch][%d][Batch Index][%d/%d][Accuracy][%.2f]", iEpoch+1, i+1, (int)iNetBatchCnt, iSglTestPassCnt*1.0/setCurrOutput.size());
                        iTestPassCnt += iSglTestPassCnt;
                    }
                    else return false;
                }
                CLOCK_END(0)
                std::printf("\r[Epoch][%d][Accuracy][%lf]", iEpoch+1, iTestPassCnt*1.0/setOrigin.size());
                std::cout << "[Duration][" << CLOCK_DURATION(0) << ']' << std::endl;
                ++ iEpoch;
            }
            while(iTestPassCnt < mnistDataset.size());
            return true;
        }
        else return false;    
    }
};

class NetMNISTIm2Col : public neunet::NetBase
{
protected:
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
        for(int j=lsLayer.size()-1; j>=0; --j)
        {
            uint64_t i = j;
            switch (lsLayer[i] -> iLayerType)
            {
            case ACT_VT:
                setGrad = INSTANCE_DERIVE<LAYER_ACT_VT>(lsLayer[i]) -> BackProp(setGrad, setCurrBatchOrigin);
                break;
            case FC:
                setGrad = INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> BackProp(setGrad);
                INSTANCE_DERIVE<LAYER_FC>(lsLayer[i]) -> UpdatePara();
                break;
            case FC_BN:
                setGrad = INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> BackProp(setGrad);
                INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i]) -> UpdatePara();
                if(iCurrBatchIdx)
                {
                    INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecMiuBeta += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecMiuBeta;
                    INSTANCE_DERIVE<BN_FC>(mapBNData[i])->vecSigmaSqr += INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData.vecSigmaSqr;
                }
                else mapBNData[i] = std::make_shared<BN_FC>(std::move(INSTANCE_DERIVE<LAYER_FC_BN>(lsLayer[i])->BNData));
                break;
            case CONV_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                INSTANCE_DERIVE<LAYER_CONV_IM2COL>(lsLayer[i])->UpdatePara();
                break;
            case CONV_BN_IM2COL:
                setGrad = INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> BackProp(setGrad);
                INSTANCE_DERIVE<LAYER_CONV_BN_IM2COL>(lsLayer[i]) -> UpdatePara();
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
            if(!setGrad.size()) return false;
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
    uint64_t CurrBatchSize(uint64_t iCurrBatchIdx)
    {
        if(iCurrBatchIdx+1==iNetBatchCnt && iNetRearBatchSize) return iNetRearBatchSize;
        else return iNetMiniBatch;
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
    void ValueAssign(NetMNISTIm2Col &netSrc) { bMonitorFlag = netSrc.bMonitorFlag; }

    NET_MAP<uint64_t, BN_PTR> mapBNData;
    set<vect> setCurrBatchInput, setCurrBatchOrigin;
    bool bMonitorFlag = false;
public:
    void ValueCopy(NetMNISTIm2Col &netSrc)
    {
        NetBase::ValueCopy(netSrc);
        ValueAssign(netSrc);
        mapBNData = netSrc.mapBNData;
        setCurrBatchInput = netSrc.setCurrBatchInput;
        setCurrBatchOrigin = netSrc.setCurrBatchOrigin;
    }
    void ValueMove(NetMNISTIm2Col &&netSrc)
    {
        NetBase::ValueMove(std::move(netSrc));
        ValueAssign(netSrc);
        mapBNData = std::move(netSrc.mapBNData);
        setCurrBatchInput = std::move(netSrc.setCurrBatchInput);
        setCurrBatchOrigin = std::move(netSrc.setCurrBatchOrigin);
    }
    NetMNISTIm2Col(NetMNISTIm2Col &netSrc) : NetBase(netSrc) { ValueCopy(netSrc); }
    NetMNISTIm2Col(NetMNISTIm2Col &&netSrc) : NetBase(std::move(netSrc)) { ValueMove(std::move(netSrc)); }
    void operator=(NetMNISTIm2Col &netSrc) { NetBase::operator=(netSrc); ValueCopy(netSrc); }
    void operator=(NetMNISTIm2Col &&netSrc) { NetBase::operator=(std::move(netSrc)); ValueMove(std::move(netSrc)); }

    NetMNISTIm2Col(double dNetAcc = 1e-5, uint64_t iMinibatch = 0, bool bMonitor = true) : NetBase(dNetAcc, iMinibatch), bMonitorFlag(bMonitor) {}
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
                    CLOCK_BEGIN(0)
                    auto setCurrOutput = ForwProp(mnistTrainSet.ln_cnt());
                    if(setCurrOutput.size())
                    {
                        auto iSglTestPassCnt = IterPass(setCurrOutput);
                        if(bMonitorFlag) IterShow(setCurrOutput);
                        else std::printf("\r[Epoch][%d][Batch Index][%d/%d][Accuracy][%.2f]", iEpoch+1, i+1, (int)iNetBatchCnt, iSglTestPassCnt*1.0/setCurrOutput.size());
                        if(iSglTestPassCnt != setCurrOutput.size()) if(!BackProp(setCurrOutput, i)) return false;
                        CLOCK_END(0)
                        std::cout << "[Duration][" << CLOCK_DURATION(0) << ']';
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

    void Reset()
    {
        NetBase::Reset();
        mapBNData.reset();
        setCurrBatchInput.reset();
        setCurrBatchOrigin.reset();
    }
    ~NetMNISTIm2Col() { Reset(); }
};

NEUNET_END