LAYER_BEGIN

struct Layer
{
    uint64_t iActFuncType = NULL, iLayerType = FC;
    double dLayerLearnRate = 0;

    virtual void ValueAssign(Layer &lyrSrc)
    {
        iLayerType = lyrSrc.iLayerType;
        iActFuncType = lyrSrc.iActFuncType;
        dLayerLearnRate = lyrSrc.dLayerLearnRate;
    }
    Layer(Layer &lyrSrc) { ValueAssign(lyrSrc); }
    void operator=(Layer &lyrSrc) { new (this)Layer(lyrSrc); }
    
    Layer(uint64_t iLayerTypeVal = FC, uint64_t iActFuncTypeVal = NULL, double dLearnRate = 0) : iLayerType(iLayerTypeVal), iActFuncType(iActFuncTypeVal), dLayerLearnRate(dLearnRate) {}
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Activate(set<T> &setInput)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return activate(setInput, sigmoid);
        case RELU:      return activate(setInput, ReLU);
        case SOFTMAX:   return activate(setInput, softmax);  
        default: return setInput;
        }
    }
    template<typename T, typename = std::enable_if_t<true, vect>, typename = std::enable_if_t<true, feature>> set<T> Derivative(set<T> &setActInput, set<T> &setGrad)
    {
        switch (iActFuncType)
        {
        case SIGMOID:   return derivative(setActInput, setGrad, sigmoid_dv);
        case RELU:      return derivative(setActInput, setGrad, ReLU_dv);
        default: return setGrad;
        }
    }
    virtual void ForwProp() {}
    virtual void BackProp() {}
};

struct LayerFC : Layer
{
    vect vecLayerWeight;
    set<vect> setLayerInput, setLayerOutput;
    // Default
    _ADA AdaDeltaVect advLayerDelta;

    void ValueAssign(LayerFC &lyrSrc) {}
    LayerFC() : Layer(FC) {}
    LayerFC(LayerFC &lyrSrc) : Layer(lyrSrc), vecLayerWeight(lyrSrc.vecLayerWeight), setLayerInput(lyrSrc.setLayerInput), setLayerOutput(lyrSrc.setLayerOutput) {}
    LayerFC(LayerFC &&lyrSrc) : Layer(lyrSrc), vecLayerWeight(std::move(lyrSrc.vecLayerWeight)), setLayerInput(std::move(lyrSrc.setLayerInput)), setLayerOutput(std::move(lyrSrc.setLayerOutput)) {}
    void operator=(LayerFC &lyrSrc) { new(this)LayerFC(lyrSrc); }
    void operator=(LayerFC &&lyrSrc) { new(this)LayerFC(std::move(lyrSrc)); }
    
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-05) : Layer(FC, iActFuncTypeVal, dLearnRate) { _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc); }
    set<vect> ForwProp(set<vect> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setLayerOutput = _FC Output(setLayerInput, vecLayerWeight);
        return Activate(setLayerOutput);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        setGrad = Derivative(setLayerOutput, setGrad);
        auto setGradBack = _FC GradLossToInput(setGrad, vecLayerWeight);
        auto vecGradWeight = _FC GradLossToWeight(setGrad, setLayerInput);
        if(dLayerLearnRate) vecLayerWeight -= dLayerLearnRate * vecGradWeight;
        else vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, vecGradWeight, advLayerDelta);
        return setGradBack;
    }
};

struct LayerFCBN : Layer
{
    // Shift, Scale, Dominant
    double dBeta = 0, dGamma = 1, dEpsilon = 1e-10;
    set<vect> setLayerInput;
    BN_FC BNData;
    _ADA AdaDeltaVal advBeta, advGamma;

    void ValueAssign(LayerFCBN &lyrSrc)
    {
        dBeta = lyrSrc.dBeta;
        dGamma = lyrSrc.dGamma;
        dEpsilon = lyrSrc.dEpsilon;
    }
    LayerFCBN() : Layer(FC_BN) {}    
    LayerFCBN(LayerFCBN &lyrSrc) : Layer(lyrSrc), setLayerInput(lyrSrc.setLayerInput), BNData(lyrSrc.BNData), advBeta(lyrSrc.advBeta), advGamma(lyrSrc.advGamma) { ValueAssign(lyrSrc); }
    LayerFCBN(LayerFCBN &&lyrSrc) : Layer(lyrSrc), setLayerInput(std::move(lyrSrc.setLayerInput)), BNData(std::move(lyrSrc.BNData)), advBeta(std::move(lyrSrc.advBeta)), advGamma(std::move(lyrSrc.advGamma)) { ValueAssign(lyrSrc); }
    void operator=(LayerFCBN &lyrSrc) { new(this)LayerFCBN(lyrSrc); }
    void operator=(LayerFCBN &&lyrSrc){ new(this)LayerFCBN(std::move(lyrSrc)); }

    LayerFCBN(double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = SIGMOID, double dLearnRate = 0, double dDmt = 1e-10) : Layer(FC_BN, iActFuncTypeVal, dLearnRate), dBeta(dShift), dGamma(dScale), dEpsilon(dDmt) {}
    set<vect> ForwProp(set<vect> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        BNData = _FC BNTrain(setLayerInput, dBeta, dGamma, dEpsilon);
        return Activate(BNData.setY);
    }
    set<vect> BackProp(set<vect> &setGrad)
    { 
        setGrad = Derivative(BNData.setY, setGrad);
        auto setGradBack = _FC BNGradLossToInput(BNData, setLayerInput, setGrad, dGamma, dEpsilon);
        auto dGradScale = _FC BNGradLossToScale(setGrad, BNData),
            dGradShift = _FC BNGradLossToShift(setGrad);
        if(dLayerLearnRate)
        {
            dGamma = _FC BNUpdateScaleShift(dGamma, dGradScale, dLayerLearnRate);
            dBeta = _FC BNUpdateScaleShift(dBeta, dGradShift, dLayerLearnRate);
        }
        else
        {
            dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, dGradScale, advGamma);
            dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, dGradShift, advBeta);
        }
        return setGradBack;
    }
};

struct LayerConv : Layer
{
    uint64_t iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0, iLayerInputPadTop = 0, iLayerInputPadRight = 0, iLayerInputPadBottom = 0, iLayerInputPadLeft = 0, iLayerLnDistance = 0, iLayerColDistance = 0;
    tensor tenKernel;
    set<feature> setLayerInput, setLayerOutput;
    _ADA ada_tensor<_ADA AdaDeltaVect> advLayerDelta;

    LayerConv() : Layer(CONV) {}
    void ValueAssign(LayerConv &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerInputPadTop = lyrSrc.iLayerInputPadTop;
        iLayerInputPadRight = lyrSrc.iLayerInputPadRight;
        iLayerInputPadBottom = lyrSrc.iLayerInputPadBottom;
        iLayerInputPadLeft = lyrSrc.iLayerInputPadLeft;
        iLayerLnDistance = lyrSrc.iLayerLnDistance;
        iLayerColDistance = lyrSrc.iLayerColDistance;
    }
    LayerConv(LayerConv &lyrSrc) : Layer(lyrSrc), tenKernel(lyrSrc.tenKernel), setLayerInput(lyrSrc.setLayerInput), setLayerOutput(lyrSrc.setLayerOutput), advLayerDelta(lyrSrc.advLayerDelta) { ValueAssign(lyrSrc); }
    LayerConv(LayerConv &&lyrSrc) : Layer(lyrSrc), tenKernel(std::move(lyrSrc.tenKernel)), setLayerInput(std::move(lyrSrc.setLayerInput)), setLayerOutput(std::move(lyrSrc.setLayerOutput)), advLayerDelta(std::move(lyrSrc.advLayerDelta)) { ValueAssign(lyrSrc); }
    void operator=(LayerConv &lyrSrc) { new(this)LayerConv(lyrSrc); }
    void operator=(LayerConv &&lyrSrc) { new(this)LayerConv(std::move(lyrSrc)); }

    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(CONV, iActFuncTypeVal, dLearnRate), iLayerLnStride(iLnStride), iLayerColStride(iColStride), iLayerLnDilation(iLnDilation), iLayerColDilation(iColDilation), iLayerInputPadTop(iInputPadTop), iLayerInputPadRight(iInputPadRight), iLayerInputPadBottom( iInputPadBottom), iLayerInputPadLeft(iInputPadLeft), iLayerLnDistance(iLnDistance), iLayerColDistance(iColDistance) { tenKernel = _CONV InitKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc); }
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setLayerOutput = _CONV Conv(setLayerInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        return Activate(setLayerOutput);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        setGrad = Derivative(setLayerOutput, setGrad);
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        auto tenGradKernel = _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        if(dLayerLearnRate) tenKernel = _CONV UpdateKernel(tenKernel, tenGradKernel, dLayerLearnRate);
        else tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, tenGradKernel, advLayerDelta);
        return setGradBack;
    }
};

struct LayerConvBN : Layer
{
    // Dominant
    double dEpsilon = 1e-5;
    // Shift, Scale
    vect vecBeta, vecGamma;
    BN_CONV BNData;
    set<feature> setLayerInput;
    _ADA AdaDeltaVect advBeta, advGamma;

    void ValueAssign(LayerConvBN &lyrSrc) { dEpsilon = lyrSrc.dEpsilon;}
    LayerConvBN() : Layer(CONV_BN) {}
    LayerConvBN(LayerConvBN &lyrSrc) : Layer(lyrSrc), vecBeta(lyrSrc.vecBeta), vecGamma(lyrSrc.vecGamma), BNData(lyrSrc.BNData), setLayerInput(lyrSrc.setLayerInput), advBeta(lyrSrc.advBeta), advGamma(lyrSrc.advGamma) { ValueAssign(lyrSrc); }
    LayerConvBN(LayerConvBN &&lyrSrc) : Layer(lyrSrc), vecBeta(std::move(lyrSrc.vecBeta)), vecGamma(std::move(lyrSrc.vecGamma)), BNData(std::move(lyrSrc.BNData)), setLayerInput(std::move(lyrSrc.setLayerInput)), advBeta(std::move(lyrSrc.advBeta)), advGamma(std::move(lyrSrc.advGamma)) { ValueAssign(lyrSrc); }
    void operator=(LayerConvBN &lyrSrc) { new(this)LayerConvBN(lyrSrc); }
    void operator=(LayerConvBN &&lyrSrc) { new(this)LayerConvBN(std::move(lyrSrc)); }

    LayerConvBN(uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, uint64_t iActFuncTypeVal = RELU, double dLearnRate = 0, double dDmt = 1e-10) : Layer(CONV_BN, iActFuncTypeVal, dLearnRate)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        BNData = _CONV BNTrain(setLayerInput, vecBeta, vecGamma, dEpsilon);
        return Activate(BNData.setY);
    }
    set<feature> BackProp(set<feature> &setGrad, uint64_t iMiniBatchIdx = 0)
    {
        setGrad = Derivative(BNData.setY, setGrad);
        auto setGradBack = _CONV BNGradLossToInput(BNData, setLayerInput, setGrad, vecGamma, dEpsilon);
        auto dGradScale = _CONV BNGradLossToScale(setGrad, BNData),
            dGradShift = _CONV BNGradLossToShift(setGrad);
        if(dLayerLearnRate)
        {
            vecGamma = _CONV BNUpdateScaleShift(vecGamma, dGradScale, dLayerLearnRate);
            vecBeta = _CONV BNUpdateScaleShift(vecBeta, dGradShift, dLayerLearnRate);
        }
        else
        {
            vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, dGradScale, advGamma);
            vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, dGradShift, advBeta);
        }
        return setGradBack;
    }
};

struct LayerPool : Layer
{
    uint64_t iPoolType = POOL_MAX, iLayerFilterLnCnt = 0, iLayerFilterColCnt = 0, iLayerLnStride = 0, iLayerColStride = 0, iLayerLnDilation = 0, iLayerColDilation = 0;
    set<feature> setLayerInput, setLayerOutput;

    LayerPool() : Layer(POOL) {}
    void ValueAssign(LayerPool &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt;
        iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt;
    }
    LayerPool(LayerPool &lyrSrc) : Layer(lyrSrc), setLayerInput(lyrSrc.setLayerInput) { ValueAssign(lyrSrc); }
    LayerPool(LayerPool &&lyrSrc) : Layer(lyrSrc), setLayerInput(std::move(lyrSrc.setLayerInput)) { ValueAssign(lyrSrc); }
    void operator=(LayerPool &lyrSrc) { new(this)LayerPool(lyrSrc); }
    void operator=(LayerPool &&lyrSrc) { new(this)LayerPool(std::move(lyrSrc)); }

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
    LayerPool(uint64_t iPoolTypeVal = POOL_MAX, uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iActFuncTypeVal = NULL) : Layer(POOL, iActFuncTypeVal, 0), iPoolType(iPoolTypeVal) {}
    set<feature> ForwProp(set<feature> &setInput, bool bFirstLayer = false)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setLayerOutput = _CONV Pool(setLayerInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
        return Activate(setLayerOutput);
    }
    set<feature> BackProp(set<feature> &setGrad) { return _CONV Pool(Derivative(setLayerOutput, setGrad), PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation); }
};

struct LayerTrans : Layer
{
    uint64_t iLnCnt = 0, iColCnt = 0;
    bool bFeatToVec = true;

    void ValueAssign(LayerTrans &lyrSrc)
    {
        iLnCnt = lyrSrc.iLnCnt;
        iColCnt = lyrSrc.iColCnt;
        bFeatToVec = lyrSrc.bFeatToVec;
    }
    LayerTrans() : Layer(TRANS) {}
    LayerTrans(LayerTrans &lyrSrc) : Layer(lyrSrc) { ValueAssign(lyrSrc); }
    void operator=(LayerTrans &lyrSrc) { new (this)LayerTrans(lyrSrc); }

    LayerTrans(uint64_t iActFuncTypeVal = NULL) : Layer(TRANS, iActFuncTypeVal, 0) {}
    set<vect> ForwProp(set<feature> &setInput)
    {
        iLnCnt = setInput[IDX_ZERO][IDX_ZERO].LN_CNT;
        iColCnt = setInput[IDX_ZERO][IDX_ZERO].COL_CNT;
        return _FC FeatureTransform(setInput);
    }
    set<feature> BackProp(set<vect> &setGrad) {return _FC FeatureTransform(setGrad, iLnCnt, iColCnt);}
    LayerTrans(uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iActFuncIdx = NULL) : Layer(TRANS, iActFuncIdx, 0), iLnCnt(iChannLnCnt), iColCnt(iChannColCnt), bFeatToVec(false) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLnCnt, iColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}
};

LAYER_END