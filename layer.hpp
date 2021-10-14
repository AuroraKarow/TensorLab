LAYER_BEGIN

class Layer
{
protected:
    uint64_t iActFuncType = NULL_FUNC;
    bool bFirstLayer = false;
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
public:
    uint64_t iLayerType = 0;
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer(uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = 0, bool bIsFirstLayer = false) : iLayerType(iCurrLayerType), iActFuncType(iActFuncIdx), bFirstLayer(bIsFirstLayer) {}
    void operator=(Layer &lyrSrc)
    {
        iActFuncType = lyrSrc.iActFuncType;
        bFirstLayer = lyrSrc.bFirstLayer;
        iLayerType = lyrSrc.iLayerType;
    }
    Layer(Layer &lyrSrc) {*this = lyrSrc;}
    void ForwProp() {}
    void BackProp() {}
    void InitPara() {}
};

class LayerFC : public Layer
{
protected:
    vect vecLayerWeight;
    set<vect> setLayerInput;
    set<vect> setOutput;
public:
    LayerFC(uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC, bool bIsFirstLayer = false) : Layer(iCurrLayerType, iActFuncIdx, bIsFirstLayer) {}
    bool InitLayerWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecLayerWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iCurrLayerType, iActFuncIdx, bIsFirstLayer) {InitLayerWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer)setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput = _FC Output(setInput, vecLayerWeight);
        return Activate(setOutput);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = Derivative<vect>(setOutput, _FC GradLossToInput(setGrad, vecLayerWeight));
        vecLayerWeight -= dLearnRate * _FC GradLossToWeight(setGrad, setLayerInput);
        return setGradBack;
    }
    LayerFC(LayerFC &lyrSrc) {*this = lyrSrc;}
    LayerFC(LayerFC &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerFC &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        vecLayerWeight = lyrSrc.vecLayerWeight;
        setLayerInput = lyrSrc.setLayerInput;
    }
    void operator=(LayerFC &&lyrSrc)
    {
        Layer::operator=(lyrSrc);
        vecLayerWeight = std::move(lyrSrc.vecLayerWeight);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
};

class LayerFCAda : public LayerFC
{
protected:
    _ADA AdaDeltaVect advLayerDelta;
public:
    LayerFCAda(uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC_ADA, bool bIsFirstLayer = false) : LayerFC(iCurrLayerType, iActFuncIdx, bIsFirstLayer) {}
    LayerFCAda(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, uint64_t iCurrLayerType = FC_ADA, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : LayerFC(iInputLnCnt, iOutputLnCnt, iCurrLayerType, iActFuncIdx, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc) {}
    set<vect> BackProp(set<vect> &setGrad)
    {
        auto setGradBack = Derivative<vect>(setOutput, _FC GradLossToInput(setGrad, vecLayerWeight));
        vecLayerWeight = _FC AdaDeltaUpdateWeight(vecLayerWeight, _FC GradLossToWeight(setGrad, setLayerInput), advLayerDelta);
        return setGradBack;
    }
    LayerFCAda(LayerFCAda &lyrSrc) {*this = lyrSrc;}
    LayerFCAda(LayerFCAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerFCAda &lyrSrc)
    {
        LayerFC::operator=(lyrSrc);
        advLayerDelta = lyrSrc.advLayerDelta;
    }
    void operator=(LayerFCAda &&lyrSrc)
    {
        LayerFC::operator=(std::move(lyrSrc));
        advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
};

class LayerBNFC : public Layer
{
protected:
    _FC FCBN fcbnData;
    // Shift
    double dBeta = 0;
    // Scale
    double dGamma = 1;
    double dEpsilon = 1e-10;
    set<vect> setLayerInput;
    void ValueAssign(LayerBNFC &lyrSrc)
    {
        dBeta = lyrSrc.dBeta;
        dGamma = lyrSrc.dGamma;
    }
public:
    LayerBNFC(uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, bool bIsFirstLayer = false, uint64_t iCurrLayerType = BN_FC, double dBNDominator = 1e-10) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), dBeta(dShift), dGamma(dScale), dEpsilon(dBNDominator) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        fcbnData = _FC BNTrain(setInput, dBeta, dGamma, true, dEpsilon);
        return Activate(fcbnData.setY);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = Derivative<vect>(fcbnData.setY, _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon));
        dGamma = _FC BNUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), dLearnRate);
        dBeta = _FC BNUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), dLearnRate);
        return setGradBack;
    }
    set<vect> Deduce(uint64_t iMiniBatchSize) {return _FC BNDeduce(setLayerInput, dBeta, dGamma, iMiniBatchSize, dEpsilon);}
    LayerBNFC(LayerBNFC &lyrSrc) {*this = lyrSrc;}
    LayerBNFC(LayerBNFC &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNFC &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        fcbnData = lyrSrc.fcbnData;
        ValueAssign(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
    }
    void operator=(LayerBNFC &&lyrSrc)
    {
        Layer::operator=(lyrSrc);
        fcbnData = std::move(lyrSrc.fcbnData);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
};

class LayerBNFCAda : public LayerBNFC
{
protected: 
    _ADA AdaDeltaVal advBeta;
    _ADA AdaDeltaVal advGamma;
public:
    LayerBNFCAda(uint64_t iActFuncIdx = NULL_FUNC, bool bIsFirstLayer = false, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_FC_ADA, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dBNDominator = 1e-10) : LayerBNFC(iActFuncIdx, bIsFirstLayer, dShift, dScale, iCurrLayerType, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        auto setGradBack = Derivative<vect>(fcbnData.setY, _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon));
        dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), advGamma);
        dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), advBeta);
        return setGradBack;
    }
    LayerBNFCAda(LayerBNFCAda &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerBNFCAda &lyrSrc)
    {
        LayerBNFC::operator=(lyrSrc);
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
};

class LayerConv : public Layer
{
protected:
    tensor tenKernel;
    set<feature> setLayerInput;
    set<feature> setOutput;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
    uint64_t iLayerInputPadTop = 0;
    uint64_t iLayerInputPadRight = 0;
    uint64_t iLayerInputPadBottom = 0;
    uint64_t iLayerInputPadLeft = 0;
    uint64_t iLayerLnDistance = 0;
    uint64_t iLayerColDistance = 0;
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
public:
    LayerConv(LayerConv &lyrSrc) {*this = lyrSrc;}
    LayerConv(LayerConv &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerConv &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        tenKernel = lyrSrc.tenKernel;
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    void operator=(LayerConv &&lyrSrc)
    {
        Layer::operator=(lyrSrc);
        tenKernel = std::move(lyrSrc.tenKernel);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    LayerConv(uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV, bool bIsFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer) {}
    bool InitLayerKernel(uint64_t iAmt, uint64_t iChannCnt, uint64_t iLnCnt, uint64_t iColCnt, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5)
    {
        if(iAmt && iChannCnt && iLnCnt && iColCnt)
        {
            tenKernel = _CONV InitKernel(iAmt, iChannCnt, iLnCnt, iColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    bool InitPara(uint64_t iLnStride, uint64_t iColStride, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0)
    {
        if(iLnStride && iColStride)
        {
            iLayerLnStride = iLnStride; iLayerColStride = iColStride;
            iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
            iLayerInputPadTop = iInputPadTop; iLayerInputPadRight = iInputPadRight; iLayerInputPadBottom = iInputPadBottom; iLayerInputPadLeft = iInputPadLeft;
            iLayerLnDistance = iLnDistance; iLayerColDistance = iColDistance;
            return true;
        }
        else return false;
    }
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer)
    {
        InitLayerKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
        InitPara(iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput =  MRG_CHANN(_CONV Conv(setInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));
        return Activate(setOutput);
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate) 
    {
        auto setGradBack = Derivative(setOutput, _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));
        tenKernel = _CONV UpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), dLearnRate);
        return setGradBack;
    }
};

class LayerConvAda : public LayerConv
{
protected:
    _ADA ten_ada<_ADA AdaDeltaVect> advLayerDelta;
public:
    LayerConvAda(LayerConvAda &lyrSrc) {*this = lyrSrc;}
    LayerConvAda(LayerConvAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerConvAda &lyrSrc)
    {
        LayerConv::operator=(lyrSrc);
        advLayerDelta = lyrSrc.advLayerDelta;
    }
    void operator=(LayerConvAda &&lyrSrc)
    {
        LayerConv::operator=(std::move(lyrSrc));
        advLayerDelta = std::move(lyrSrc.advLayerDelta);
    }
    LayerConvAda(uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV_ADA, bool bIsFirstLayer = false) : LayerConv(iActFuncIdx, iCurrLayerType, bIsFirstLayer) {}
    LayerConvAda(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, uint64_t iCurrLayerType = CONV_ADA, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : LayerConv(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx, iCurrLayerType, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance) {}
    set<feature> BackProp(set<feature> &setGrad)
    {
        auto setGradBack = Derivative(setOutput, _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));
        tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), advLayerDelta);
        return setGradBack;
    }
};

class LayerPool : public Layer
{
protected:
    uint64_t iPoolType = POOL_MAX;
    set<feature> setLayerInput;
    set<feature> setOutput;
    uint64_t iLayerFilterLnCnt = 0;
    uint64_t iLayerFilterColCnt = 0;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
    void ValueAssign(LayerPool &lyrSrc)
    {
        iLayerLnStride = lyrSrc.iLayerLnStride;
        iLayerColStride = lyrSrc.iLayerColStride;
        iLayerLnDilation = lyrSrc.iLayerLnDilation;
        iLayerColDilation = lyrSrc.iLayerColDilation;
        iLayerFilterLnCnt = lyrSrc.iLayerFilterLnCnt;
        iLayerFilterColCnt = lyrSrc.iLayerFilterColCnt;
    }
public:
    LayerPool(LayerPool &lyrSrc) {*this = lyrSrc;}
    LayerPool(LayerPool &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerPool &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        ValueAssign(lyrSrc);
    }
    void operator=(LayerPool &&lyrSrc)
    {
        Layer::operator=(lyrSrc);
        ValueAssign(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
    }
    static uint64_t PoolUpType(uint64_t iPoolIdx)
    {
        switch (iPoolIdx)
        {
        case POOL_AVG: return POOL_UP_AVG;
        case POOL_GAG: return POOL_UP_GAG;
        case POOL_MAX: return POOL_UP_MAX;
        default: return (uint64_t)NAN;
        }
    }
    LayerPool(uint64_t iPoolIdx = POOL_MAX, uint64_t iActFuncIdx = NULL_FUNC, uint64_t iCurrLayerType = POOL, bool bIsFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), iPoolType(iPoolIdx) {}
    void InitPara(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
    {
        iLayerFilterLnCnt = iFilterLnCnt; iLayerFilterColCnt = iFilterColCnt;
        iLayerLnStride = iLnStride; iLayerColStride = iColStride;
        iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
    }
    LayerPool(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, bool bIsFirstLayer = false, uint64_t iColStride = 0, uint64_t iPoolIdx = POOL_MAX, uint64_t iActFuncIdx = NULL_FUNC, uint64_t iCurrLayerType = POOL, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), iPoolType(iPoolIdx) {InitPara(iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        setOutput = _CONV Pool(setInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
        return Activate(setOutput);
    }
    set<feature> BackProp(set<feature> &setGrad) {return Derivative(setOutput, _CONV Pool(setGrad, PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation));}
};

class LayerBNConv : public Layer
{
protected:
    _CONV ConvBN convbnData;
    // Shift
    vect vecBeta;
    // Scale
    vect vecGamma;
    double dEpsilon = 1e-5;
    set<feature> setLayerInput;
public:
    LayerBNConv(LayerBNConv &lyrSrc) {*this = lyrSrc;}
    LayerBNConv(LayerBNConv &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNConv &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        setLayerInput = lyrSrc.setLayerInput;
        convbnData = lyrSrc.convbnData;
        vecBeta = lyrSrc.vecBeta;
        vecGamma = lyrSrc.vecGamma;
        dEpsilon = lyrSrc.dEpsilon;
    }
    void operator=(LayerBNConv &&lyrSrc)
    {
        Layer::operator=(lyrSrc);
        setLayerInput = std::move(lyrSrc.setLayerInput);
        convbnData = std::move(lyrSrc.convbnData);
        vecBeta = std::move(lyrSrc.vecBeta);
        vecGamma = std::move(lyrSrc.vecGamma);
        dEpsilon = lyrSrc.dEpsilon;
    }
    LayerBNConv(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, uint64_t iActFuncIdx = NULL_FUNC, double dShift = 0, double dScale = 1, uint64_t iCurrLayerType = BN_CONV, double dBNDominator = 1e-5) : Layer(iActFuncIdx, iCurrLayerType, bIsFirstLayer), dEpsilon(dBNDominator)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        convbnData = _CONV BNTrain(setInput, vecBeta, vecGamma, true, dEpsilon);
        return Activate(convbnData.setY);
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate)
    {
        auto setGradBack = Derivative(convbnData.setY, _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon));
        vecBeta = _CONV BNUpdateScaleShiftBat(vecBeta, _CONV BNGradLossToShift(setGrad), dLearnRate);
        vecGamma = _CONV BNUpdateScaleShiftBat(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), dLearnRate);
        return setGradBack;
    }
    set<feature> Deduce(uint64_t iMiniBatchSize) {return _CONV BNDeduce(setLayerInput, vecBeta, vecGamma, iMiniBatchSize, dEpsilon);}
};

class LayerBNConvAda : public LayerBNConv
{
protected:
    _ADA AdaDeltaVect advBeta;
    _ADA AdaDeltaVect advGamma;
public:
    LayerBNConvAda(LayerBNConvAda &lyrSrc) {*this = lyrSrc;}
    LayerBNConvAda(LayerBNConvAda &&lyrSrc) {*this = std::move(lyrSrc);}
    void operator=(LayerBNConvAda &lyrSrc)
    {
        LayerBNConv::operator=(lyrSrc);
        advBeta = lyrSrc.advBeta;
        advGamma = lyrSrc.advGamma;
    }
    void operator=(LayerBNConvAda &&lyrSrc)
    {
        LayerBNConv::operator=(std::move(lyrSrc));
        advBeta = std::move(lyrSrc.advBeta);
        advGamma = std::move(lyrSrc.advGamma);
    }
    LayerBNConvAda(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, uint64_t iCurrLayerType = BN_CONV, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNConv(bIsFirstLayer, iChannCnt, dShift, dScale, iCurrLayerType, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        auto setGradBack = Derivative(convbnData.setY, _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon));
        vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, _CONV BNGradLossToShift(setGrad), advBeta);
        vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), advGamma);
        return setGradBack;
    }
};

class LayerTransToVect : public Layer
{
protected:
    uint64_t iLayerInputLnCnt = 0;
    uint64_t iLayerInputColCnt = 0;
public:
    set<vect> ForwProp(set<feature> &setInput)
    {
        iLayerInputLnCnt = setInput[ZERO_IDX][ZERO_IDX].LN_CNT;
        iLayerInputColCnt = setInput[ZERO_IDX][ZERO_IDX].COL_CNT;
        return _FC FeatureTransform(setInput);
    }
    set<feature> BackProp(set<vect> &setGrad) {return _FC FeatureTransform(setGrad, iLayerInputLnCnt, iLayerInputColCnt);}
    LayerTransToVect(uint64_t iCurrLayerType = TRANS_TO_VECT, uint64_t iActFuncIdx = NULL_FUNC, bool isFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, isFirstLayer) {}
    LayerTransToVect(LayerTransToVect &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerTransToVect &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        iLayerInputLnCnt = lyrSrc.iLayerInputLnCnt;
        iLayerInputColCnt = lyrSrc.iLayerInputColCnt;
    }
};

class LayerTransToFeat : public Layer
{
protected:
    uint64_t iLayerChannLnCnt = 0;
    uint64_t iLayerChannColCnt = 0;
public:
    LayerTransToFeat(uint64_t iChannLnCnt, uint64_t iChannColCnt, uint64_t iCurrLayerType = TRANS_TO_FEAT, uint64_t iActFuncIdx = NULL_FUNC, bool isFirstLayer = false) : Layer(iActFuncIdx, iCurrLayerType, isFirstLayer), iLayerChannLnCnt(iChannLnCnt), iLayerChannColCnt(iChannColCnt) {}
    set<feature> ForwProp(set<vect> &setInput) {return _FC FeatureTransform(setInput, iLayerChannLnCnt, iLayerChannColCnt);}
    set<vect> BackProp(set<feature> &setGrad) {return _FC FeatureTransform(setGrad);}
    LayerTransToFeat(LayerTransToFeat &lyrSrc) {*this = lyrSrc;}
    void operator=(LayerTransToFeat &lyrSrc)
    {
        Layer::operator=(lyrSrc);
        iLayerChannLnCnt = lyrSrc.iLayerChannLnCnt;
        iLayerChannColCnt = lyrSrc.iLayerChannColCnt;
    }
};

LAYER_END