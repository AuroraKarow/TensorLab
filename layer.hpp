LAYER_BEGIN

struct Layer
{
    uint64_t iLayerType = ACT_VT;
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
    
    Layer(uint64_t iLayerTypeVal = ACT_VT, double dLearnRate = 0) : iLayerType(iLayerTypeVal), dLayerLearnRate(dLearnRate) {}
    void ForwProp() {}
    void BackProp() {}
    virtual bool UpdatePara() {return true;}

    virtual void Reset() {}
    ~Layer() {}
};

struct LayerAct : Layer
{
    uint64_t iLayerActFuncType = NULL;
    void ValueAssign(LayerAct &lyrSrc) { iLayerActFuncType = lyrSrc.iLayerActFuncType; }
    void ValueCopy(LayerAct &lyrSrc) { ValueAssign(lyrSrc); }
    void ValueMove(LayerAct &&lyrSrc) {}
    LayerAct(LayerAct &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerAct &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerAct(uint64_t iLayerTypeVal, uint64_t iActFuncType) : Layer(iLayerTypeVal, 0), iLayerActFuncType(iActFuncType) {}
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
    void Reset() {}
    ~LayerAct() { Reset(); }
};

struct LayerActVect : LayerAct
{
    set<vect> setLayerInput;

    void ValueAssign(LayerActVect &lyrSrc) {}
    void ValueCopy(LayerActVect &lyrSrc) { setLayerInput = lyrSrc.setLayerInput; }
    void ValueMove(LayerActVect &&lyrSrc) { setLayerInput = std::move(lyrSrc.setLayerInput); }
    LayerActVect(LayerActVect &lyrSrc) : LayerAct(lyrSrc) { ValueCopy(lyrSrc); }
    LayerActVect(LayerActVect &&lyrSrc) : LayerAct(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerActVect &lyrSrc) { LayerAct::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerActVect &&lyrSrc) { LayerAct::operator=(lyrSrc); ValueMove(std::move(lyrSrc)); }

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
    LayerActFt(LayerActFt &&lyrSrc) : LayerAct(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerActFt &lyrSrc) { LayerAct::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerActFt &&lyrSrc) { LayerAct::operator=(lyrSrc); ValueMove(std::move(lyrSrc)); }

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
    set<vect> setLayerInput;
    // Default
    _ADA AdaDeltaVect advLayerDelta;
    _ADA AdaNesterovVect anvLayerMomt;

    void ValueAssign(LayerFC &lyrSrc) {}
    void ValueCopy(LayerFC &lyrSrc)
    {
        vecLayerWeight = lyrSrc.vecLayerWeight; setLayerInput = lyrSrc.setLayerInput; vecLayerGradWeight = lyrSrc.vecLayerGradWeight; anvLayerMomt = lyrSrc.anvLayerMomt;
    }
    void ValueMove(LayerFC &&lyrSrc)
    {
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight); setLayerInput = std::move(lyrSrc.setLayerInput); vecLayerGradWeight = std::move(lyrSrc.vecLayerGradWeight); anvLayerMomt = std::move(lyrSrc.anvLayerMomt);
    }
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFC &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFC &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }
    
    LayerFC() : Layer(FC) {}
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05) : Layer(FC, dLearnRate) { vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc); }
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
    bool UpdatePara()
    {
        if(vecLayerGradWeight.is_matrix())
        {
            if(dLayerLearnRate) vecLayerWeight = _FC AdaNesterovUpdateWeight(vecLayerWeight, vecLayerGradWeight, dLayerLearnRate, anvLayerMomt);
            else vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, vecLayerGradWeight, advLayerDelta);
            return true;
        }
        else return false;
    }
    vect Deduce(vect &vecInput) { return _FC Output(vecInput, vecLayerWeight); }

    void ResetAda() { advLayerDelta.Reset(); anvLayerMomt.Reset(); }
    void Reset() { vecLayerWeight.reset(); setLayerInput.reset(); vecLayerGradWeight.reset(); ResetAda(); }
    ~LayerFC() { Reset(); }
};

struct LayerFCBN : Layer
{
    // Shift, Scale, Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-10, dGradBeta = 0, dGradGamma = 0;
    set<vect> setLayerInput;
    _ADA AdaDeltaVal advBeta, advGamma;
    _ADA AdaNesterovVal anvBeta, anvGamma;
    BN_FC BNData;

    void ValueAssign(LayerFCBN &lyrSrc)
    {
        dBeta = lyrSrc.dBeta; dGamma = lyrSrc.dGamma; dEpsilon = lyrSrc.dEpsilon; dGradBeta = lyrSrc.dGradBeta; dGradGamma = lyrSrc.dGradGamma; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma;
    }
    void ValueCopy(LayerFCBN &lyrSrc) { ValueAssign(lyrSrc); setLayerInput = lyrSrc.setLayerInput; BNData = lyrSrc.BNData; }
    void ValueMove(LayerFCBN &&lyrSrc) { ValueAssign(lyrSrc); setLayerInput = std::move(lyrSrc.setLayerInput); BNData = std::move(lyrSrc.BNData); }
    LayerFCBN(LayerFCBN &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerFCBN(LayerFCBN &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerFCBN &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerFCBN &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerFCBN(double dShift = 0, double dScale = 1, double dLearnRate = 0, double dDmt = 1e-10) : Layer(FC_BN, dLearnRate), dBeta(dShift), dGamma(dScale), dEpsilon(dDmt) {}
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
    bool UpdatePara()
    {
        if(dGradGamma && dGradBeta)
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
            return true;
        }
        else return false;
    }
    vect Deduce(vect &vecInput, BN_FC_PTR &pBNData, uint64_t iBatchSize = 0, uint64_t iBatchCnt = 0) { return _FC BNDeduce(vecInput, dBeta, dGamma, pBNData, iBatchSize, iBatchCnt, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { setLayerInput.reset(); BNData.reset(); }
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
    bool UpdatePara()
    {
        if(tenGradKernel.size())
        {
            if(dLayerLearnRate) tenKernel = _CONV AdaNesterovUpdateKernel(tenKernel, tenGradKernel, dLayerLearnRate, anvLayerMomt);
            else tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, tenGradKernel, advLayerDelta);
            return true;
        }
        else return false;
    }
    feature Deduce(feature &vecInput) { return _CONV Conv(vecInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance); }

    void ResetAda() { advLayerDelta.reset(); anvLayerMomt.reset(); }
    void Reset() { setLayerInput.reset(); tenGradKernel.reset(); tenKernel.reset(); ResetAda(); }
    ~LayerConv() { Reset(); }
};

struct LayerConvIm2Col : Layer
{
    uint64_t iLayerOutputLnCnt = 0, iLayerKernelLnCnt = 0, iLayerKernelColCnt = 0 ,iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;
    set<vect> setPrepInput;
    vect vecKernel, vecGradKernel, vecPrepInput;
    _ADA AdaDeltaVect advKernel;
    _ADA AdaNesterovVect anvKernel;

    void ValueAssign(LayerConvIm2Col &lyrSrc)
    {
        iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerKernelLnCnt = lyrSrc.iLayerKernelLnCnt; iLayerKernelColCnt = lyrSrc.iLayerKernelColCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerInputPadTop = lyrSrc.iLayerInputPadTop; iLayerInputPadRight = lyrSrc.iLayerInputPadRight; iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom; iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft; iLayerLnDistance = lyrSrc.iLayerLnDistance; iLayerColDistance = lyrSrc.iLayerColDistance;
    }
    void ValueCopy(LayerConvIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = lyrSrc.vecKernel; setPrepInput = lyrSrc.setPrepInput; vecGradKernel = lyrSrc.vecGradKernel; vecPrepInput = lyrSrc.vecPrepInput; advKernel = lyrSrc.advKernel; anvKernel = lyrSrc.anvKernel;
    }
    void ValueMove(LayerConvIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecKernel = std::move(lyrSrc.vecKernel); setPrepInput = std::move(lyrSrc.setPrepInput); vecGradKernel = std::move(lyrSrc.vecGradKernel); vecPrepInput = std::move(lyrSrc.vecPrepInput); advKernel = std::move(lyrSrc.advKernel); anvKernel = std::move(lyrSrc.anvKernel);
    }
    LayerConvIm2Col(LayerConvIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerConvIm2Col(LayerConvIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerConvIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerConvIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerConvIm2Col() : Layer(CONV_IM2COL) {}
    LayerConvIm2Col(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, double dLearnRate = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dRandBoundryAcc = 1e-5) : Layer(CONV, dLearnRate), iLayerKernelLnCnt(iKernelLnCnt), iLayerKernelColCnt(iKernelColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom(iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) { vecKernel = _CONV InitKernelIm2Col(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dRandBoundryAcc); }
    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        setPrepInput = _CONV Im2ColInputTransform(setInput, iLayerOutputLnCnt, iInputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return _CONV ConvIm2Col(setPrepInput, vecKernel);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        vecGradKernel = _CONV GradLossToKernelIm2Col(setGrad, setPrepInput);
        return _CONV GradLossToInputIm2Col(setGrad, vecKernel, iLayerOutputLnCnt, iLayerKernelLnCnt, iLayerKernelColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    bool UpdatePara()
    {
        if(vecGradKernel.is_matrix())
        {
            if(dLayerLearnRate) vecKernel = _FC AdaNesterovUpdateWeight(vecKernel, vecGradKernel, dLayerLearnRate, anvKernel);
            else vecKernel = _FC AdaDeltaUpdateWeight(vecKernel, vecGradKernel, advKernel);
            return true;
        }
        else return false;
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
    bool UpdatePara()
    {
        if(vecGradGamma.is_matrix() && vecGradBeta.is_matrix())
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
            return true;
        }
        else return false;
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
    set<vect> setLayerInput;
    _ADA AdaDeltaVect advBeta, advGamma;
    _ADA AdaNesterovVect anvBeta, anvGamma;
    BN_CONV_IM2COL BNData;

    void ValueAssign(LayerConvBNIm2Col &lyrSrc) { dEpsilon = lyrSrc.dEpsilon;}
    void ValueCopy(LayerConvBNIm2Col &lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = lyrSrc.vecBeta; vecGamma = lyrSrc.vecGamma; setLayerInput = lyrSrc.setLayerInput; advBeta = lyrSrc.advBeta; advGamma = lyrSrc.advGamma; anvBeta = lyrSrc.anvBeta; anvGamma = lyrSrc.anvGamma; BNData = lyrSrc.BNData; vecGradBeta = lyrSrc.vecGradBeta; vecGradGamma = lyrSrc.vecGradGamma;
    }
    void ValueMove(LayerConvBNIm2Col &&lyrSrc)
    {
        ValueAssign(lyrSrc); vecBeta = std::move(lyrSrc.vecBeta); vecGamma = std::move(lyrSrc.vecGamma); setLayerInput = std::move(lyrSrc.setLayerInput); advBeta = std::move(lyrSrc.advBeta); advGamma = std::move(lyrSrc.advGamma); BNData = std::move(lyrSrc.BNData); vecGradBeta = std::move(lyrSrc.vecGradBeta); vecGradGamma = std::move(lyrSrc.vecGradGamma);
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
    bool UpdatePara()
    {
        if(vecGradGamma.is_matrix() && vecGradBeta.is_matrix())
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
            return true;
        }
        else return false;
    }
    vect Deduce(vect &vecInput, BN_CONV_IM2COL_PTR &pBNData, uint64_t iBatchSize = 0, uint64_t iBatchCnt = 0) { _CONV BNDeduceIm2Col(vecInput, vecBeta, vecGamma, pBNData, iBatchCnt, iBatchSize, dEpsilon); }

    void ResetAda() { advBeta.Reset(); advGamma.Reset(); anvBeta.Reset(); anvGamma.Reset(); }
    void Reset() { vecBeta.reset(); vecGamma.reset(); setLayerInput.reset(); vecGradBeta.reset(); vecGradGamma.reset(); BNData.Reset(); ResetAda(); }
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
    vect_t<bagrt::net_list<mtx::mtx_pos>> setInputMaxPosList;

    void ValueAssign(LayerPoolIm2Col &lyrSrc)
    {
        iPoolType = lyrSrc.iPoolType; iLayerOutputLnCnt = lyrSrc.iLayerOutputLnCnt; iLayerLnStride = lyrSrc.iLayerLnStride; iLayerColStride = lyrSrc.iLayerColStride; iLayerLnDilation = lyrSrc.iLayerLnDilation; iLayerColDilation = lyrSrc.iLayerColDilation; iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt; iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt; iLayerInputPadTop = lyrSrc.iLayerInputPadTop; iLayerInputPadRight = lyrSrc.iLayerInputPadRight; iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom; iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft; iLayerLnDistance = lyrSrc.iLayerLnDistance; iLayerColDistance = lyrSrc.iLayerColDistance; iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt; iLayerInputColCnt = lyrSrc.iLayerInputColCnt;
    }
    void ValueCopy(LayerPoolIm2Col &lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = lyrSrc.setInputMaxPosList; }
    void ValueMove(LayerPoolIm2Col &&lyrSrc) { ValueAssign(lyrSrc); setInputMaxPosList = std::move(lyrSrc.setInputMaxPosList); }
    LayerPoolIm2Col(LayerPoolIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    LayerPoolIm2Col(LayerPoolIm2Col &&lyrSrc) : Layer(lyrSrc) { ValueMove(std::move(lyrSrc)); }
    void operator=(LayerPoolIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }
    void operator=(LayerPoolIm2Col &&lyrSrc) { Layer::operator=(std::move(lyrSrc)); ValueMove(std::move(lyrSrc)); }

    LayerPoolIm2Col(uint64_t iPoolTypeVal = POOL_MAX_IM2COL, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(POOL_IM2COL, 0), iPoolType(iPoolTypeVal), iLayerFilterLnCnt(iFilterLnCnt), iLayerFilterColCnt(iFilterColCnt), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom(iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) {}
    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        iLayerInputLnCnt = iInputLnCnt;
        iLayerInputColCnt = setInput[IDX_ZERO].LN_CNT / iLayerInputLnCnt;
        return _CONV PoolIm2Col(iPoolType, setInput, setInputMaxPosList, iLayerOutputLnCnt, iLayerInputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }
    set<vect> BackProp(set<vect> &setGrad) { return _CONV GradLossToPoolIm2ColInput(iPoolType, setGrad, setInputMaxPosList, iLayerOutputLnCnt, iLayerInputLnCnt, iLayerInputColCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance); }
    vect Deduce(vect &vecInput, uint64_t iInputLnCnt)
    {
        set<bagrt::net_list<mtx::mtx_pos>> setTemp;
        if(iPoolType == POOL_GAG_IM2COL) { iLayerOutputLnCnt = 1; return _CONV PoolGlbAvgIm2Col(vecInput); }
        else return _CONV PoolMaxAvgIm2Col(iPoolType, vecInput, setTemp, iLayerOutputLnCnt, iInputLnCnt, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
    }

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

    void ValueAssign(LayerTransIm2Col &lyrSrc) { iLnCnt = lyrSrc.iLnCnt; iColCnt = lyrSrc.iColCnt; bFeatToVec = lyrSrc.bFeatToVec; }
    void ValueCopy(LayerTransIm2Col &lyrSrc) { ValueAssign(lyrSrc); }
    void ValueMove(LayerTransIm2Col &lyrSrc) {}
    LayerTransIm2Col(LayerTransIm2Col &lyrSrc) : Layer(lyrSrc) { ValueCopy(lyrSrc); }
    void operator=(LayerTransIm2Col &lyrSrc) { Layer::operator=(lyrSrc); ValueCopy(lyrSrc); }

    LayerTransIm2Col() : Layer(TRANS_IM2COL, 0) {}
    set<vect> ForwProp(set<vect> &setInput, uint64_t iInputLnCnt)
    {
        iLnCnt = iInputLnCnt;
        iColCnt = setInput[IDX_ZERO].LN_CNT / iLnCnt;
        return _FC FeatureTransformIm2Col(setInput);
    }
    vect Deduce(vect &vecInput, uint64_t iInputLnCnt) { return _FC FeatureTransformIm2Col(vecInput); }

    LayerTransIm2Col(uint64_t iChannLnCnt, uint64_t iChannColCnt) : Layer(TRANS_IM2COL, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt), bFeatToVec(false) {}
    set<vect> ForwProp(set<vect> &setInput) {return _FC FeatureTransformIm2Col(setInput, iLnCnt, iColCnt);}
    vect Deduce(vect &vecInput) { return _FC FeatureTransformIm2Col(vecInput, iLnCnt, iColCnt); }

    set<vect> BackProp(set<vect> &setGrad)
    {
        if(bFeatToVec) return _FC FeatureTransformIm2Col(setGrad, iLnCnt, iColCnt);
        else return _FC FeatureTransformIm2Col(setGrad);
    }
};

LAYER_END