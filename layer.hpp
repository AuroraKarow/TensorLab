LAYER_BEGIN

struct Layer
{
    uint64_t iLayerType = ACT;
    double dLayerLearnRate = 0;

    virtual void ValueAssign(Layer &lyrSrc)
    {
        iLayerType = lyrSrc.iLayerType;
        dLayerLearnRate = lyrSrc.dLayerLearnRate;
    }
    virtual void ValueCopy(Layer &lyrSrc) { ValueAssign(lyrSrc); }
    virtual void ValueMove(Layer &&lyrSrc) {}
    Layer(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(Layer &lyrSrc) { ValueCopy(lyrSrc); }
    
    Layer(uint64_t iLayerTypeVal = ACT, double dLearnRate = 0) : iLayerType(iLayerTypeVal), dLayerLearnRate(dLearnRate) {}
    void ForwProp() {}
    void BackProp() {}
    virtual void UpdatePara() {}

    virtual void Reset() {}
    ~Layer() {}
};

struct LayerAct : Layer
{
    uint64_t iLayerActFuncType = NULL;
    set<vect> setLayerInput;
    
    void ValueAssign(LayerAct &lyrSrc) { iLayerActFuncType = lyrSrc.iLayerActFuncType; }
    void ValueCopy(LayerAct &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerAct &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerAct(LayerAct &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerAct(LayerAct &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerAct &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerAct &&lyrSrc) { Layer::operator=(lyrSrc); ValueMove(std::move(lyrSrc)); }

    LayerAct(uint64_t iActFuncType, uint64_t iLayerTypeVal = ACT, double dLearnRate = 0) : Layer(iLayerTypeVal, dLearnRate), iLayerActFuncType(iActFuncType) {}

    vect ForwProp(vect &vecInput, uint64_t iTdIdx)
    {
        if(vecInput.is_matrix()) setLayerInput[iTdIdx] = std::move(vecInput);
        switch (iLayerActFuncType)
        {
        case SIGMOID:   return sigmoid(setLayerInput[iTdIdx]);
        case RELU:      return ReLU(setLayerInput[iTdIdx]);
        case SOFTMAX:   return softmax(setLayerInput[iTdIdx]);
        default: return setLayerInput[iTdIdx];
        }
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx, vect &vecOrgn = blank_vect)
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID:   return hadamard_product(sigmoid_dv(setLayerInput[iTdIdx]), vecGrad);
        case RELU:      return hadamard_product(ReLU_dv(setLayerInput[iTdIdx]), vecGrad);
        case SOFTMAX:   return softmax_cec_grad(vecGrad, vecOrgn);
        default: return vecGrad;
        }
    }

    template<typename elemT> set<elemT> ForwProp(set<elemT> &setLayerInput)
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID:   return sigmoid(setLayerInput);
        case RELU:      return ReLU(setLayerInput);
        case SOFTMAX:   return softmax(setLayerInput);
        default: return setLayerInput;
        }
    }
    // Gradient is activation output as the last layer
    template<typename elemT> set<elemT> BackProp(set<elemT> &setLayerInput, set<elemT> &setGrad, set<elemT> &setOrigin = set<elemT>())
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID:   return hadamard_product(sigmoid_dv(setLayerInput), setGrad);
        case RELU:      return hadamard_product(ReLU_dv(setLayerInput), setGrad);
        case SOFTMAX:   return softmax_cec_grad(setGrad, setOrigin);
        default: return setGrad;
        }     
    }
    template<typename T> T Deduce(T &vecInput)
    {
        switch (iLayerActFuncType)
        {
        case SIGMOID:   return sigmoid(vecInput);
        case RELU:      return ReLU(vecInput);
        case SOFTMAX:   return softmax(vecInput);
        default: return vecInput;
        }
    }
    void Reset() { setLayerInput.reset(); }
    ~LayerAct() { Reset(); }
};

struct LayerActVect : LayerAct
{
    set<vect> setLayerInput;

    void ValueAssign(LayerActVect &lyrSrc) {}
    void ValueCopy(LayerActVect &lyrSrc) { setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerActVect &&lyrSrc) { setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerActVect(LayerActVect &lyrSrc) : LayerAct(lyrSrc) { ValueCopy(lyrSrc); }
    LayerActVect(LayerActVect &&lyrSrc) : LayerAct(std::move(lyrSrc)) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerActVect &lyrSrc) { LayerAct::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerActVect &&lyrSrc) { LayerAct::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerActVect(uint64_t iActFuncType) : LayerAct(ACT_VT, iActFuncType) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        return LayerAct::ForwProp(setLayerInput);
    }
    set<vect> BackProp(set<vect> &setGrad, set<vect> &setOrigin = set<vect>()) { return LayerAct::BackProp(setLayerInput, setGrad, setOrigin); }
    vect Deduce(vect &vecInput) { return LayerAct::Deduce(vecInput); }
    ~LayerActVect() { setLayerInput.reset(); }
};

struct LayerActFt : LayerAct
{
    set<feature> setLayerInput;

    void ValueAssign(LayerActFt &lyrSrc) {}
    void ValueCopy(LayerActFt &lyrSrc) { setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerActFt &&lyrSrc) { setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerActFt(LayerActFt &lyrSrc) : LayerAct(lyrSrc) { ValueCopy(lyrSrc); }
    LayerActFt(LayerActFt &&lyrSrc) : LayerAct(std::move(lyrSrc)) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerActFt &lyrSrc) { LayerAct::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerActFt &&lyrSrc) { LayerAct::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerActFt(uint64_t iActFuncType) : LayerAct(ACT_FT, iActFuncType) {}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        return LayerAct::ForwProp(setLayerInput);
    }
    set<feature> BackProp(set<feature> &setGrad, set<feature> &setOrigin = set<feature>()) { return LayerAct::BackProp(setLayerInput, setGrad, setOrigin); }
    feature Deduce(feature &vecInput) { return LayerAct::Deduce(vecInput); }
    ~LayerActFt() { setLayerInput.reset(); }
};

struct LayerFC : Layer
{
    vect vecLayerWeight, vecLayerGradWeight;
    _ADA AdaDeltaVect advLayerDelta;
    _ADA AdaNesterovVect anvLayerMomt;

    set<vect> setLayerInput;
    set<vect> setLayerGradWeight;

    void ValueAssign(LayerFC &lyrSrc) {}
    void ValueCopy(LayerFC &lyrSrc)
    {
        vecLayerWeight = lyrSrc.vecLayerWeight; setLayerInput = lyrSrc.setLayerInput; vecLayerGradWeight = lyrSrc.vecLayerGradWeight; anvLayerMomt = lyrSrc.anvLayerMomt; setLayerGradWeight = lyrSrc.setLayerGradWeight;
    }
    void ValueMove(LayerFC &&lyrSrc)
    {
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight); setLayerInput = std::move(lyrSrc.setLayerInput); vecLayerGradWeight = std::move(lyrSrc.vecLayerGradWeight); anvLayerMomt = std::move(lyrSrc.anvLayerMomt); setLayerGradWeight = std::move(lyrSrc.setLayerGradWeight);
    }
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFC &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFC &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }
    
    LayerFC() : Layer(FC) {}
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05) : Layer(FC, dLearnRate) { vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc); }

    vect ForwProp(vect &vecInput, uint64_t iTdIdx)
    {
        if(vecInput.is_matrix()) setLayerInput[iTdIdx] = std::move(vecInput);
        return Deduce(setLayerInput[iTdIdx]);
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        setLayerGradWeight[iTdIdx] = _FC GradLossToWeight(vecGrad, setLayerInput[iTdIdx]);
        return _FC GradLossToInput(vecGrad, vecLayerWeight);
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        return _FC Output(setLayerInput, vecLayerWeight);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        vecLayerGradWeight = _FC GradLossToWeight(setGrad, setLayerInput);
        return _FC GradLossToInput(setGrad, vecLayerWeight);
    }

    void UpdatePara()
    {
        if(setLayerGradWeight.size()) vecLayerGradWeight = setLayerGradWeight.sum();
        if(dLayerLearnRate) vecLayerWeight = _FC AdaNesterovUpdateWeight(vecLayerWeight, vecLayerGradWeight, dLayerLearnRate, anvLayerMomt);
        else vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, vecLayerGradWeight, advLayerDelta);
    }
    vect Deduce(vect &vecInput) { return _FC Output(vecInput, vecLayerWeight); }

    void ResetAda() { advLayerDelta.Reset(); anvLayerMomt.Reset(); }
    void Reset() { vecLayerWeight.reset(); setLayerInput.reset(); vecLayerGradWeight.reset(); setLayerGradWeight.reset(); ResetAda(); }
    ~LayerFC() { Reset(); }
};

struct LayerFCBN : Layer
{
    // Shift, Scale, Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-10, dGradBeta = 0, dGradGamma = 0;
    _ADA AdaDeltaVal advBeta, advGamma;
    _ADA AdaNesterovVal anvBeta, anvGamma;
    BN_FC BNData;

    set<vect> setLayerInput, setGradLossToOutput, setGradLossToInput/*Need not initialize*/;
    std::mutex tdmtxFCBN;
    std::condition_variable condFCBN;
    async::lock_counter lkCnt;
    bool bForwSgn = false, bBackSgn = false;

    void ValueAssign(LayerFCBN &lyrSrc)
    {
        dBeta = lyrSrc.dBeta; dGamma = lyrSrc.dGamma; dEpsilon = lyrSrc.dEpsilon; dGradBeta = lyrSrc.dGradBeta; dGradGamma = lyrSrc.dGradGamma; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma; bForwSgn = lyrSrc.bForwSgn; bBackSgn = lyrSrc.bBackSgn;
    }
    void ValueCopy(LayerFCBN &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; BNData = lyrSrc.BNData; setGradLossToOutput = lyrSrc.setGradLossToOutput; setGradLossToInput = lyrSrc.setGradLossToInput; }
    void ValueMove(LayerFCBN &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); BNData = std::move(lyrSrc.BNData); setGradLossToOutput = std::move(lyrSrc.setGradLossToOutput); setGradLossToInput = std::move(lyrSrc.setGradLossToInput); }
    LayerFCBN(LayerFCBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFCBN(LayerFCBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFCBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFCBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerFCBN(double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10) : Layer(FC_BN, dLearnRate), dBeta(dShift), dGamma(dScale), dEpsilon(dDmt) {}

    vect ForwProp(vect &vecInput, uint64_t iTdIdx)
    {
        lkCnt.increment();
        if(vecInput.is_matrix()) setLayerInput[iTdIdx] = std::move(vecInput);
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(!bForwSgn) condFCBN.wait(lkFCBN);
        }
        else
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(lkCnt.get_cnt() != setLayerInput.size());
            BNData = _FC BNTrain(setLayerInput, dBeta, dGamma, dEpsilon);
            bForwSgn = true;
            condFCBN.notify_all();
            lkCnt.set_cnt();
        }
        return BNData.setY[iTdIdx];
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        lkCnt.increment();
        if(vecGrad.is_matrix()) setGradLossToOutput[iTdIdx] = std::move(vecGrad);
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(!bBackSgn) condFCBN.wait(lkFCBN);
        }
        else
        {
            while(lkCnt.get_cnt() != setGradLossToOutput.size());
            setGradLossToInput = _FC BNGradLossToInput(BNData, setLayerInput, setGradLossToOutput, dGamma, dEpsilon);
            bBackSgn = true;
            condFCBN.notify_all();
            dGradGamma = _FC BNGradLossToScale(setGradLossToOutput, BNData);
            dGradBeta = _FC BNGradLossToShift(setGradLossToOutput);
            lkCnt.set_cnt();
        }
        return setGradLossToInput[iTdIdx];
    }

    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        BNData = _FC BNTrain(setLayerInput, dBeta, dGamma, dEpsilon);
        return BNData.setY;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        dGradGamma = _FC BNGradLossToScale(setGrad, BNData);
        dGradBeta = _FC BNGradLossToShift(setGrad);
        return _FC BNGradLossToInput(BNData, setLayerInput, setGrad, dGamma, dEpsilon);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate)
        {
            dGamma = _FC BNAdaNesterovUpdateScaleShift(dGamma, dGradGamma, dLayerLearnRate,  anvGamma);
            dBeta = _FC BNAdaNesterovUpdateScaleShift(dBeta, dGradBeta, dLayerLearnRate, anvBeta);
        }
        else 
        {
            dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, dGradGamma, advGamma);
            dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, dGradBeta, advBeta);
        }
        bForwSgn = false;
        bBackSgn = false;
    }
    vect Deduce(vect &vecInput, BN_FC_PTR &pBNData, uint64_t iBatchSize = 0, uint64_t iBatchCnt = 0) { return _FC BNDeduce(vecInput, dBeta, dGamma, pBNData, iBatchSize, iBatchCnt, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { setLayerInput.reset(); BNData.reset(); setGradLossToOutput.reset(); setGradLossToInput.reset(); ResetAda(); }
    ~LayerFCBN() { Reset(); }
};

struct LayerConv : Layer
{
    uint64_t iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;
    tensor tenKernel, tenGradKernel;
    set<feature> setLayerInput;
    _ADA ada_tensor<_ADA AdaDeltaVect> advLayerDelta;
    _ADA ada_tensor<_ADA AdaNesterovVect> anvLayerMomt;

    void ValueAssign(LayerConv &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerInputPadTop = lyrSrc.iLayerInputPadTop; iLayerInputPadRight = lyrSrc.iLayerInputPadRight; iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom; iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft; iLayerLnDistance = lyrSrc.iLayerLnDistance; iLayerColDistance = lyrSrc.iLayerColDistance;
    }
    void ValueCopy(LayerConv &lyrSrc)
    {
        ValueAssign(lyrSrc); tenKernel = lyrSrc.tenKernel; setLayerInput = lyrSrc.setLayerInput; tenGradKernel = lyrSrc.tenGradKernel; advLayerDelta = lyrSrc.advLayerDelta; anvLayerMomt = lyrSrc.anvLayerMomt;
    }
    void ValueMove(LayerConv &&lyrSrc)
    {
        ValueAssign(lyrSrc); tenKernel = std::move(lyrSrc.tenKernel); setLayerInput = std::move(lyrSrc.setLayerInput); tenGradKernel = std::move(lyrSrc.tenGradKernel); advLayerDelta = std::move(lyrSrc.advLayerDelta); anvLayerMomt = std::move(lyrSrc.anvLayerMomt);
    }
    LayerConv(LayerConv &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConv(LayerConv &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConv &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConv &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConv() : Layer(CONV) {}
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(CONV, dLearnRate), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom( iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) { tenKernel = _CONV InitKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dRandBoundryAcc); }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        return _CONV Conv(setLayerInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        tenGradKernel = _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate) tenKernel = _CONV AdaNesterovUpdateKernel(tenKernel, tenGradKernel, dLayerLearnRate, anvLayerMomt);
        else tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, tenGradKernel, advLayerDelta);
    }
    feature Deduce(feature &vecInput) { return _CONV Conv(vecInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance); }

    void ResetAda() { advLayerDelta.reset(); anvLayerMomt.reset(); }
    void Reset() { setLayerInput.reset(); tenGradKernel.reset(); tenKernel.reset(); ResetAda(); }
    ~LayerConv() { Reset(); }
};

struct LayerConvIm2Col : Layer
{
    uint64_t iLayerOutputLnCnt = 0, iLayerKernelLnCnt = 0, iLayerKernelColCnt = 0 ,iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;
    vect vecKernel, vecGradKernel, vecPrepInput;
    _ADA AdaDeltaVect advKernel;
    _ADA AdaNesterovVect anvKernel;

    set<vect> setPrepInput, setGradKernel;
    std::mutex tdmtxConvIm2Col;
    std::condition_variable condConvIm2Col;
    bool bActSgn = false;

    void ValueAssign(LayerConvIm2Col &lyrSrc)
    {
        iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerKernelLnCnt = lyrSrc.iLayerKernelLnCnt; iLayerKernelColCnt = lyrSrc.iLayerKernelColCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerInputPadTop = lyrSrc.iLayerInputPadTop; iLayerInputPadRight = lyrSrc.iLayerInputPadRight; iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom; iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft; iLayerLnDistance = lyrSrc.iLayerLnDistance; iLayerColDistance = lyrSrc.iLayerColDistance; bActSgn = lyrSrc.bActSgn;
    }
    void ValueCopy(LayerConvIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = lyrSrc.vecKernel; setPrepInput = lyrSrc.setPrepInput; vecGradKernel = lyrSrc.vecGradKernel; vecPrepInput = lyrSrc.vecPrepInput; advKernel = lyrSrc.advKernel; anvKernel = lyrSrc.anvKernel; setGradKernel = lyrSrc.setGradKernel;
    }
    void ValueMove(LayerConvIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = std::move(lyrSrc.vecKernel); setPrepInput = std::move(lyrSrc.setPrepInput); vecGradKernel = std::move(lyrSrc.vecGradKernel); vecPrepInput = std::move(lyrSrc.vecPrepInput); advKernel = std::move(lyrSrc.advKernel); anvKernel = std::move(lyrSrc.anvKernel); setGradKernel = std::move(lyrSrc.setGradKernel);
    }
    LayerConvIm2Col(LayerConvIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvIm2Col(LayerConvIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvIm2Col() : Layer(CONV_IM2COL) {}
    LayerConvIm2Col(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, double dLearnRate = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5) : Layer(CONV_IM2COL, dLearnRate), iLayerKernelLnCnt(iKernelLnCnt), iLayerKernelColCnt(iKernelColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom(iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) { vecKernel = _CONV InitKernelIm2Col(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dRandBoundryAcc); }

    vect ForwProp(vect &vecInput, uint64_t iInputLnCnt, uint64_t iTdIdx)
    {
        uint64_t iOutputLnCnt = 0;
        if(vecInput.is_matrix()) setPrepInput[iTdIdx] = _CONV Im2ColInputTransform(vecInput, iOutputLnCnt, iInputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lkConvIm2Col(tdmtxConvIm2Col);
            while(!bActSgn) condConvIm2Col.wait(lkConvIm2Col);            
        }
        else
        {
            std::unique_lock<std::mutex> lkConvIm2Col(tdmtxConvIm2Col);
            if(iOutputLnCnt) iLayerOutputLnCnt = iOutputLnCnt;
            bActSgn = true;
            condConvIm2Col.notify_all();
        }
        return _CONV ConvIm2Col(setPrepInput[iTdIdx], vecKernel);
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        setGradKernel[iTdIdx] = _CONV GradLossToKernelIm2Col(vecGrad, setPrepInput[iTdIdx]);
        return _CONV GradLossToInputIm2Col(vecGrad, vecKernel, iLayerOutputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }

    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        if(setInput.size()) setPrepInput = _CONV Im2ColInputTransform(setInput, iLayerOutputLnCnt, iInputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return _CONV ConvIm2Col(setPrepInput, vecKernel);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        vecGradKernel = _CONV GradLossToKernelIm2Col(setGrad, setPrepInput);
        return _CONV GradLossToInputIm2Col(setGrad, vecKernel, iLayerOutputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    void UpdatePara()
    {
        if(!vecGradKernel.is_matrix()) vecGradKernel = setGradKernel.sum();
        if(dLayerLearnRate) vecKernel = _FC AdaNesterovUpdateWeight(vecKernel, vecGradKernel, dLayerLearnRate, anvKernel);
        else vecKernel = _FC AdaDeltaUpdateWeight(vecKernel, vecGradKernel, advKernel);
        bActSgn = false;
    }
    vect Deduce(vect &vecInput, uint64_t iInputLnCnt) { return _CONV ConvIm2Col(_CONV Im2ColInputTransform(vecInput, iLayerOutputLnCnt, iInputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), vecKernel); }

    void ResetAda() { advKernel.Reset(); anvKernel.Reset(); }
    void Reset() { setPrepInput.reset(); vecPrepInput.reset(); vecGradKernel.reset(); vecKernel.reset(); ResetAda(); }
    ~LayerConvIm2Col() { Reset(); }
};

struct LayerConvBN : Layer
{
    // Dominant
    double dEpsilon = 1e-5;
    // Shift, Scale
    vect vecBeta, vecGamma, vecGradBeta, vecGradGamma;
    set<feature> setLayerInput;
    _ADA AdaDeltaVect advBeta, advGamma;
    _ADA AdaNesterovVect anvBeta, anvGamma;
    BN_CONV BNData;

    void ValueAssign(LayerConvBN &lyrSrc) { dEpsilon = lyrSrc.dEpsilon;}
    void ValueCopy(LayerConvBN &lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = lyrSrc.vecBeta; vecGamma = lyrSrc.vecGamma; setLayerInput = lyrSrc.setLayerInput; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; BNData = lyrSrc.BNData; vecGradBeta = lyrSrc.vecGradBeta; vecGradGamma = lyrSrc.vecGradGamma;
    }
    void ValueMove(LayerConvBN &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = std::move(lyrSrc.vecBeta); vecGamma = std::move(lyrSrc.vecGamma); setLayerInput = std::move(lyrSrc.setLayerInput); advBeta = std::move(lyrSrc.advBeta); advGamma = std::move(lyrSrc.advGamma); anvBeta = std::move(lyrSrc.anvBeta); anvGamma = std::move(lyrSrc.anvGamma); BNData = std::move(lyrSrc.BNData); vecGradBeta = std::move(lyrSrc.vecGradBeta); vecGradGamma = std::move(lyrSrc.vecGradGamma);
    }
    LayerConvBN(LayerConvBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvBN(LayerConvBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvBN(uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10) : Layer(CONV_BN, dLearnRate), dEpsilon(dDmt)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        BNData = _CONV BNTrain(setLayerInput, vecBeta, vecGamma, dEpsilon);
        return BNData.setY;
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        vecGradGamma = _CONV BNGradLossToScale(setGrad, BNData);
        vecGradBeta = _CONV BNGradLossToShift(setGrad);
        return _CONV BNGradLossToInput(BNData, setLayerInput, setGrad, vecGamma, dEpsilon);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate)
        {
            vecGamma = _CONV BNAdaNesterovUpdateScaleShift(vecGamma, vecGradGamma, dLayerLearnRate, anvGamma);
            vecBeta = _CONV BNAdaNesterovUpdateScaleShift(vecBeta, vecGradBeta, dLayerLearnRate, anvBeta);
        }
        else 
        {
            vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, vecGradGamma, advGamma);
            vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, vecGradBeta, advBeta);
        }
    }
    feature Deduce(feature &vecInput, BN_CONV_PTR &pBNData, uint64_t iBatchSize = 0, uint64_t iBatchCnt = 0) { _CONV BNDeduce(vecInput, vecBeta, vecGamma, pBNData, iBatchSize, iBatchCnt, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { vecBeta.reset(); vecGamma.reset(); setLayerInput.reset(); vecGradBeta.reset(); vecGradGamma.reset(); BNData.Reset(); ResetAda(); }
    ~LayerConvBN() { Reset(); }
};

struct LayerConvBNIm2Col : Layer
{
    // Dominant
    double dEpsilon = 1e-5;
    // Shift, Scale
    vect vecBeta, vecGamma, vecGradBeta, vecGradGamma;
    _ADA AdaDeltaVect advBeta, advGamma;
    _ADA AdaNesterovVect anvBeta, anvGamma;
    BN_CONV_IM2COL BNData;

    set<vect> setLayerInput, setGradLossToOutput, setGradLossToInput/*Need not initialize*/;
    std::mutex tdmtxFCBN;
    std::condition_variable condFCBN;
    async::lock_counter lkCnt;
    bool bForwSgn = false, bBackSgn = false;

    void ValueAssign(LayerConvBNIm2Col &lyrSrc) { dEpsilon = lyrSrc.dEpsilon; bForwSgn = lyrSrc.bForwSgn; bBackSgn = lyrSrc.bBackSgn; }
    void ValueCopy(LayerConvBNIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = lyrSrc.vecBeta; vecGamma = lyrSrc.vecGamma; setLayerInput = lyrSrc.setLayerInput; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma; BNData = lyrSrc.BNData; vecGradBeta = lyrSrc.vecGradBeta; vecGradGamma = lyrSrc.vecGradGamma; setGradLossToOutput = lyrSrc.setGradLossToOutput; setGradLossToInput = lyrSrc.setGradLossToInput;
    }
    void ValueMove(LayerConvBNIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = std::move(lyrSrc.vecBeta); vecGamma = std::move(lyrSrc.vecGamma); setLayerInput = std::move(lyrSrc.setLayerInput); advBeta = std::move(lyrSrc.advBeta); advGamma = std::move(lyrSrc.advGamma); BNData = std::move(lyrSrc.BNData); vecGradBeta = std::move(lyrSrc.vecGradBeta); vecGradGamma = std::move(lyrSrc.vecGradGamma); setGradLossToOutput = std::move(lyrSrc.setGradLossToOutput); setGradLossToInput = std::move(lyrSrc.setGradLossToInput);
    }
    LayerConvBNIm2Col(LayerConvBNIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvBNIm2Col(LayerConvBNIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvBNIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvBNIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvBNIm2Col(uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10) : Layer(CONV_BN_IM2COL, dLearnRate), dEpsilon(dDmt)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }

    vect ForwProp(vect &vecInput, uint64_t iTdIdx)
    {
        lkCnt.increment();
        if(vecInput.is_matrix()) setLayerInput[iTdIdx] = std::move(vecInput);
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(!bForwSgn) condFCBN.wait(lkFCBN);
        }
        else
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(lkCnt.get_cnt() != setLayerInput.size());
            BNData = _CONV BNTrainIm2Col(setLayerInput, vecBeta, vecGamma, dEpsilon);
            bForwSgn = true;
            condFCBN.notify_all();
            lkCnt.set_cnt();
        }
        return BNData.setIm2ColY[iTdIdx];
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        lkCnt.increment();
        if(vecGrad.is_matrix()) setGradLossToOutput[iTdIdx] = std::move(vecGrad);
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lkFCBN(tdmtxFCBN);
            while(!bBackSgn) condFCBN.wait(lkFCBN);
        }
        else
        {
            while(lkCnt.get_cnt() != setGradLossToOutput.size());
            setGradLossToInput = _CONV BNGradLossToInputIm2Col(setGradLossToOutput, BNData, setLayerInput, vecGamma, dEpsilon);
            bBackSgn = true;
            condFCBN.notify_all();
            vecGradGamma = _CONV BNGradLossToScaleIm2Col(setGradLossToOutput, BNData);
            vecGradBeta = _CONV BNGradLossToShiftIm2Col(setGradLossToOutput);
            lkCnt.set_cnt();
        }
        return setGradLossToInput[iTdIdx];
    }
    
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        BNData = _CONV BNTrainIm2Col(setLayerInput, vecBeta, vecGamma, dEpsilon);
        return BNData.setIm2ColY;
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        vecGradGamma = _CONV BNGradLossToScaleIm2Col(setGrad, BNData);
        vecGradBeta = _CONV BNGradLossToShiftIm2Col(setGrad);
        return _CONV BNGradLossToInputIm2Col(setGrad, BNData, setLayerInput, vecGamma, dEpsilon);
    }
    void UpdatePara()
    {
        if(dLayerLearnRate)
        {
            vecGamma = _CONV BNAdaNesterovUpdateScaleShift(vecGamma, vecGradGamma, dLayerLearnRate, anvGamma);
            vecBeta = _CONV BNAdaNesterovUpdateScaleShift(vecBeta, vecGradBeta, dLayerLearnRate, anvBeta);
        }
        else 
        {
            vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, vecGradGamma, advGamma);
            vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, vecGradBeta, advBeta);
        }
        bForwSgn = false;
        bBackSgn = false;
    }
    vect Deduce(vect &vecInput, BN_CONV_IM2COL_PTR &pBNData, uint64_t iBatchSize = 0, uint64_t iBatchCnt = 0) { _CONV BNDeduceIm2Col(vecInput, vecBeta, vecGamma, pBNData, iBatchCnt, iBatchSize, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { vecBeta.reset(); vecGamma.reset(); setLayerInput.reset(); vecGradBeta.reset(); vecGradGamma.reset(); BNData.Reset(); setGradLossToOutput.reset(); setGradLossToInput.reset(); ResetAda(); }
    ~LayerConvBNIm2Col() { Reset(); }
};

struct LayerPool : Layer
{
    uint64_t iPoolType = POOL_MAX, iLayerFilterLnCnt = 0, iLayerFilterColCnt = 0, iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0;
    set<feature> setLayerInput;

    void ValueAssign(LayerPool &lyrSrc)
    {
        iPoolType = lyrSrc.iPoolType; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt; iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt;
    }
    void ValueCopy(LayerPool &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerPool &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerPool(LayerPool &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerPool(LayerPool &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerPool &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerPool &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    uint64_t PoolUpType(uint64_t iPoolIdx)
    {
        switch (iPoolIdx)
        {
        case POOL_AVG: return POOL_UP_AVG;
        case POOL_GAG: return POOL_UP_GAG;
        case POOL_MAX: return POOL_UP_MAX;
        default: return (uint64_t)NAN;
        }
    }
    LayerPool(uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(POOL, 0), iPoolType(iPoolTypeVal), iLayerFilterLnCnt(iFilterLnCnt), iLayerFilterColCnt(iFilterColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation) {}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(setInput.size()) setLayerInput = std::move(setInput);
        return _CONV Pool(setLayerInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
    }
    set<feature> BackProp(set<feature> &setGrad) { return _CONV Pool(setGrad, PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation); }
    feature Deduce(feature &vecInput) { return _CONV PoolDown(vecInput, iPoolType, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation); }

    void Reset() { setLayerInput.reset(); }
    ~LayerPool() { Reset(); }
};

struct LayerPoolIm2Col : Layer
{
    uint64_t iPoolType = POOL_MAX_IM2COL, iLayerInputLnCnt = 0, iLayerInputColCnt = 0, iLayerOutputLnCnt = 0, iLayerFilterLnCnt = 0, iLayerFilterColCnt = 0, iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;

    vect_t<bagrt::net_list<mtx::mtx_pos>> setInputMaxPosList/* Max */;
    std::mutex tdmtxPoolIm2Col;
    std::condition_variable condPoolIm2Col;
    bool bActSgn = false;

    void ValueAssign(LayerPoolIm2Col &lyrSrc)
    {
        iPoolType = lyrSrc.iPoolType; iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt; iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt; iLayerInputPadTop = lyrSrc.iLayerInputPadTop; iLayerInputPadRight = lyrSrc.iLayerInputPadRight; iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom; iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft; iLayerLnDistance = lyrSrc.iLayerLnDistance; iLayerColDistance = lyrSrc.iLayerColDistance; iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt; iLayerInputColCnt = lyrSrc.iLayerInputColCnt; bActSgn = lyrSrc.bActSgn;
    }
    void ValueCopy(LayerPoolIm2Col &lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = lyrSrc.setInputMaxPosList; }
    void ValueMove(LayerPoolIm2Col &&lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = std::move(lyrSrc.setInputMaxPosList); }
    LayerPoolIm2Col(LayerPoolIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerPoolIm2Col(LayerPoolIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerPoolIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerPoolIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerPoolIm2Col(uint64_t iPoolTypeVal = POOL_MAX_IM2COL, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(POOL_IM2COL, 0), iPoolType(iPoolTypeVal), iLayerFilterLnCnt(iFilterLnCnt), iLayerFilterColCnt(iFilterColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom(iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) {}

    vect ForwProp(vect &vecInput, uint64_t iInputLnCnt, uint64_t iTdIdx)
    {
        if(iPoolType == POOL_GAG_IM2COL)
        {
            if(iTdIdx)
            {
                std::unique_lock<std::mutex> lkPoolIm2Col(tdmtxPoolIm2Col);
                while(!bActSgn) condPoolIm2Col.wait(lkPoolIm2Col);
            }
            else
            {
                std::unique_lock<std::mutex> lkPoolIm2Col(tdmtxPoolIm2Col);
                iLayerInputLnCnt = iInputLnCnt;
                iLayerInputColCnt = vecInput.LN_CNT / iLayerInputLnCnt;
                iLayerOutputLnCnt = 1;
                bActSgn = true;
                condPoolIm2Col.notify_all();
            }
            return _CONV PoolGlbAvgIm2Col(vecInput);
        }
        else
        {
            uint64_t iOutputLnCnt = 0;
            auto vecAns =  _CONV PoolMaxAvgIm2Col(iPoolType, vecInput, setInputMaxPosList[iTdIdx], iOutputLnCnt, iInputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
            if(iTdIdx)
            {
                std::unique_lock<std::mutex> lkPoolIm2Col(tdmtxPoolIm2Col);
                while(!bActSgn) condPoolIm2Col.wait(lkPoolIm2Col);
            }
            else
            {
                std::unique_lock<std::mutex> lkPoolIm2Col(tdmtxPoolIm2Col);
                if(iOutputLnCnt) iLayerOutputLnCnt = iOutputLnCnt;
                bActSgn = true;
                condPoolIm2Col.notify_all();
            }
            return vecAns;
        }
    }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        if(iPoolType == POOL_GAG_IM2COL) return _CONV GradLossToPoolGlbAvgInputIm2Col(vecGrad, iLayerInputLnCnt*iLayerInputColCnt);
        else
        { 
            return _CONV GradLossToPoolMaxAvgInputIm2Col(iPoolType, vecGrad, setInputMaxPosList[iTdIdx], iLayerOutputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        }
    }

    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        iLayerInputLnCnt = iInputLnCnt;
        iLayerInputColCnt = setInput[IDX_ZERO].LN_CNT / iLayerInputLnCnt;
        setInputMaxPosList.init(setInput.size()); 
        return _CONV PoolIm2Col(iPoolType, setInput, setInputMaxPosList, iLayerOutputLnCnt, iLayerInputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    set<vect> BackProp(set<vect> &setGrad) { return _CONV GradLossToPoolIm2ColInput(iPoolType, setGrad, setInputMaxPosList, iLayerOutputLnCnt, iLayerInputLnCnt, iLayerInputColCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance); }
    vect Deduce(vect &vecInput, uint64_t iInputLnCnt)
    {
        if(iPoolType == POOL_GAG_IM2COL) { iLayerOutputLnCnt = 1; return _CONV PoolGlbAvgIm2Col(vecInput); }
        else return _CONV PoolMaxAvgIm2Col(iPoolType, vecInput, set<bagrt::net_list<mtx::mtx_pos>>(), iLayerOutputLnCnt, iInputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    void UpdatePara() { bActSgn = false; }

    void Reset() { setInputMaxPosList.reset(); }
    ~LayerPoolIm2Col() { Reset(); }
};

struct LayerTrans : Layer
{
    uint64_t iLnCnt = 0, iColCnt = 0;
    bool bFeatToVec = true;

    void ValueAssign(LayerTrans &lyrSrc) { iLnCnt = lyrSrc.iLnCnt; iColCnt = lyrSrc.iColCnt; bFeatToVec = lyrSrc.bFeatToVec; }
    void ValueCopy(LayerTrans &lyrSrc) { ValueAssign(lyrSrc); }
    void ValueMove(LayerTrans &lyrSrc) {}
    LayerTrans(LayerTrans &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerTrans &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerTrans() : Layer(TRANS, 0) {}
    set<vect> ForwProp(set<feature> &setInput)
    {
        iLnCnt = setInput[IDX_ZERO][IDX_ZERO].LN_CNT;
        iColCnt = setInput[IDX_ZERO][IDX_ZERO].COL_CNT;
        return _FC FeatureTransform(setInput);
    }
    set<feature> BackProp(set<vect> &setGrad) {return _FC FeatureTransform(setGrad, iLnCnt, iColCnt);}
    vect Deduce(feature &vecInput) { return _FC FeatureTransform(vecInput); }
    
    LayerTrans(uint64_t iChannLnCnt, uint64_t iChannColCnt) : Layer(TRANS, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt), bFeatToVec(false) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLnCnt, iColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}
    feature Deduce(vect &vecInput) { return _FC FeatureTransform(vecInput, iLnCnt, iColCnt); }

    void Reset() {}
    ~LayerTrans() {}
};

struct LayerTransIm2Col : Layer
{
    uint64_t iLnCnt = 0, iColCnt = 0;
    bool bFeatToVec = true;

    std::mutex tdmtxTransIm2Col;
    std::condition_variable condTransIm2Col;
    bool bActSgn = false;

    void ValueAssign(LayerTransIm2Col &lyrSrc) { iLnCnt = lyrSrc.iLnCnt; iColCnt = lyrSrc.iColCnt; bFeatToVec = lyrSrc.bFeatToVec; bActSgn = lyrSrc.bActSgn; }
    void ValueCopy(LayerTransIm2Col &lyrSrc) { ValueAssign(lyrSrc); }
    LayerTransIm2Col(LayerTransIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerTransIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerTransIm2Col() : Layer(TRANS_IM2COL, 0) {}
    LayerTransIm2Col(uint64_t iChannLnCnt, uint64_t iChannColCnt) : Layer(TRANS_IM2COL, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt), bFeatToVec(false) {}

    vect ForwProp(vect &vecInput, uint64_t iInputLnCnt, uint64_t iTdIdx)
    {
        if(iTdIdx)
        {
            std::unique_lock<std::mutex> lckTransIm2Col(tdmtxTransIm2Col);
            while(!bActSgn) condTransIm2Col.wait(lckTransIm2Col);
        }
        else
        {
            std::unique_lock<std::mutex> lckTransIm2Col(tdmtxTransIm2Col);
            iLnCnt = iInputLnCnt;
            iColCnt = vecInput.LN_CNT / iLnCnt;
            bActSgn = true;
            condTransIm2Col.notify_all();
        }
        return Deduce(vecInput); 
    }
    vect ForwProp(vect &vecInput) { return Deduce(vecInput); }
    vect BackProp(vect &vecGrad, uint64_t iTdIdx)
    {
        if(bFeatToVec) return _FC FeatureTransformIm2Col(vecGrad, iLnCnt, iColCnt);
        else return _FC FeatureTransformIm2Col(vecGrad);
    }

    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        iLnCnt = iInputLnCnt;
        iColCnt = setInput[IDX_ZERO].LN_CNT / iLnCnt;
        return _FC FeatureTransformIm2Col(setInput);
    }
    set<vect> ForwProp(set<vect> &setInput) {return _FC FeatureTransformIm2Col(setInput, iLnCnt, iColCnt);}    
    set<vect> BackProp(set<vect> &setGrad)
    {
        if(bFeatToVec) return _FC FeatureTransformIm2Col(setGrad, iLnCnt, iColCnt);
        else return _FC FeatureTransformIm2Col(setGrad);
    }
    void UpdatePara() { if(bFeatToVec) bActSgn = false; }

    vect Deduce(vect &vecInput)
    {
        if(bFeatToVec) return _FC FeatureTransformIm2Col(vecInput);
        else return _FC FeatureTransformIm2Col(vecInput, iLnCnt, iColCnt);
    }
};

LAYER_END