LAYER_BEGIN

class Layer
{
protected:
    uint64_t iActFuncType = SIGMOID;
    bool bFirstLayer = false;
    template <typename inputT> struct InputData{inputT vecLayerInput;};
public:
    virtual void setActFunc(uint64_t iActFuncIdx) { iActFuncType = iActFuncIdx;}
    Layer(uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false) : iActFuncType(iActFuncIdx), bFirstLayer(bIsFirstLayer) {}
    void operator=(Layer &lyrSrc) {iActFuncType = lyrSrc.iActFuncType;}
    void operator=(Layer &&lyrSrc) {iActFuncType = std::move(lyrSrc.iActFuncType);}
    Layer(Layer &lyrSrc) {*this = lyrSrc;}
    Layer(Layer &&lyrSrc) {*this = std::move(lyrSrc);}
    template<typename T, typename ... Args, typename = std::enable_if_t<std::is_base_of_v<Layer, T>>> std::shared_ptr<T> static InitLayerPtr(Args&& ... args) {return std::make_shared<T>(std::forward<Args>(args)...);}
    void ForwProp() {}
    void BackProp() {}
    void InitPara() {}
};

class LayerFC : public Layer
{
protected:
    vect vecWeight;
    set<vect> vecLayerInput;
public:
    LayerFC(uint64_t iActFuncIdx = SIGMOID) : Layer(iActFuncIdx) {}
    bool InitLayerWeight(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05)
    {
        if(iInputLnCnt && iOutputLnCnt)
        {
            vecWeight = _FC InitWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
            return true;
        }
        else return false;
    }
    LayerFC(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : Layer(iActFuncIdx, bIsFirstLayer) {InitLayerWeight(iInputLnCnt, iOutputLnCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);}
    set<vect> ForwProp(set<vect> &setInput)
    {
        return _FC Output(setInput, vecWeight);
        if(bFirstLayer)vecLayerInput = setInput;
        else vecLayerInput = std::move(setInput);
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = _FC GradLossToInput(setGrad, vecWeight);
        vecWeight -= dLearnRate * _FC GradLossToWeight(setGrad, vecLayerInput);
        return setGradBack;
    }
};

class LayerFCAda : public LayerFC
{
protected:
    _ADA AdaDeltaVect advLayerDelta;
public:
    LayerFCAda(uint64_t iActFuncIdx = SIGMOID) : LayerFC(iActFuncIdx) {}
    LayerFCAda(uint64_t iInputLnCnt, uint64_t iOutputLnCnt, uint64_t iActFuncIdx = SIGMOID, bool bIsFirstLayer = false, double dRandBoundryFirst = 0.0, double dRandBoundrySecond = 0.0, double dAcc = 1e-05) : LayerFC(iInputLnCnt, iOutputLnCnt, iActFuncIdx, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc) {}
    set<vect> BackProp(set<vect> &setInput, set<vect> &setGrad)
    {
        auto setGradBack = _FC GradLossToInput(setGrad, vecWeight);
        vecWeight = _FC AdaDeltaUpdateWeight(vecWeight, _FC GradLossToWeight(setGrad, setInput), advLayerDelta);
        return setGradBack;
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
public:
    LayerBNFC(bool bIsFirstLayer = false, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : Layer(0, bIsFirstLayer), dBeta(dShift), dGamma(dScale), dEpsilon(dBNDominator) {}
    set<vect> ForwProp(set<vect> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        fcbnData = _FC BNTrain(setInput, dBeta, dGamma, true, dEpsilon);
        return fcbnData.setY;
    }
    set<vect> BackProp(set<vect> &setGrad, double dLearnRate)
    {
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), dLearnRate);
        dBeta = _FC BNUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), dLearnRate);
        return setGradBack;
    }
    set<vect> Deduce(uint64_t iMiniBatchSize) {return _FC BNDeduce(setLayerInput, dBeta, dGamma, iMiniBatchSize, dEpsilon);}
};

class LayerBNFCAda : public LayerBNFC
{
protected: 
    _ADA AdaDeltaVal advBeta;
    _ADA AdaDeltaVal advGamma;
public:
    LayerBNFCAda(bool bIsFirstLayer = false, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNFC(bIsFirstLayer, dShift, dScale, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVal(0, 0, dDecayController, dAdaDominator);
    }
    set<vect> BackProp(set<vect> &setGrad)
    {
        auto setGradBack = _FC BNGradLossToInput(fcbnData, setLayerInput, setGrad, dGamma, dEpsilon);
        dGamma = _FC BNAdaDeltaUpdateScaleShift(dGamma, _FC BNGradLossToScale(setGrad, fcbnData), advGamma);
        dBeta = _FC BNAdaDeltaUpdateScaleShift(dBeta, _FC BNGradLossToShift(setGrad), advBeta);
        return setGradBack;
    }
};

class LayerConv : public Layer
{
protected:
    tensor tenKernel;
    set<feature> setLayerInput;
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
public:
    LayerConv(uint64_t iActFuncIdx = RELU) : Layer(iActFuncIdx) {}
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
    LayerConv(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : Layer(iActFuncIdx, bIsFirstLayer)
    {
        InitLayerKernel(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, dRandBoundryFirst, dRandBoundrySecond, dAcc);
        InitPara(iLnStride, iColStride, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        return MRG_CHANN(_CONV Conv(setInput, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance));
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate) 
    {
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = _CONV UpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), dLearnRate);
        return setGradBack;
    }
};

class LayerConvAda : public LayerConv
{
protected:
    _ADA ten_ada<_ADA AdaDeltaVect> advLayerDelta;
public:
    LayerConvAda(uint64_t iActFuncIdx = RELU) : LayerConv(iActFuncIdx) {}
    LayerConvAda(uint64_t iKernelAmt, uint64_t iKernelChannCnt, uint64_t iKernelLnCnt, uint64_t iKernelColCnt, uint64_t iLnStride, uint64_t iColStride, uint64_t iActFuncIdx = RELU, bool bIsFirstLayer = false, double dRandBoundryFirst = 0, double dRandBoundrySecond = 0, double dAcc = 1e-5, uint64_t iLnDilation = 0, uint64_t iColDilation = 0, uint64_t iInputPadTop = 0, uint64_t iInputPadRight = 0, uint64_t iInputPadBottom = 0, uint64_t iInputPadLeft = 0, uint64_t iLnDistance = 0, uint64_t iColDistance = 0) : LayerConv(iKernelAmt, iKernelChannCnt, iKernelLnCnt, iKernelColCnt, iLnStride, iColStride, iActFuncIdx, bIsFirstLayer, dRandBoundryFirst, dRandBoundrySecond, dAcc, iLnDilation, iColDilation, iInputPadTop, iInputPadRight, iInputPadBottom, iInputPadLeft, iLnDistance, iColDistance) {}
    set<feature> BackProp(set<feature> &setGrad)
    {
        auto setGradBack = _CONV GradLossToInput(setGrad, tenKernel, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance);
        tenKernel = _CONV AdaDeltaUpdateKernel(tenKernel, _CONV GradLossToKernel(setGrad, setLayerInput, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation, iLayerInputPadTop, iLayerInputPadRight, iLayerInputPadBottom, iLayerInputPadLeft, iLayerLnDistance, iLayerColDistance), advLayerDelta);
        return setGradBack;
    }
};

class LayerPool : public Layer
{
protected:
    uint64_t iPoolType = POOL_MAX;
    set<feature> setLayerInput;
    uint64_t iLayerFilterLnCnt = 0;
    uint64_t iLayerFilterColCnt = 0;
    uint64_t iLayerLnStride = 0;
    uint64_t iLayerColStride = 0;
    uint64_t iLayerLnDilation = 0;
    uint64_t iLayerColDilation = 0;
public:
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
    LayerPool(uint64_t iPoolIdx = POOL_MAX, bool bIsFirstLayer = false) : Layer(0, bIsFirstLayer), iPoolType(iPoolIdx) {}
    void InitPara(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, uint64_t iColStride = 0, uint64_t iLnDilation = 0, uint64_t iColDilation = 0)
    {
        iLayerFilterLnCnt = iFilterLnCnt; iLayerFilterColCnt = iFilterColCnt;
        iLayerLnStride = iLnStride; iLayerColStride = iColStride;
        iLayerLnDilation = iLnDilation; iLayerColDilation = iColDilation;
    }
    LayerPool(uint64_t iFilterLnCnt = 0, uint64_t iFilterColCnt = 0, uint64_t iLnStride = 0, bool bIsFirstLayer = false, uint64_t iColStride = 0, uint64_t iPoolIdx = POOL_MAX, uint64_t iLnDilation = 0, uint64_t iColDilation = 0) : Layer(0, bIsFirstLayer), iPoolType(iPoolIdx) {InitPara(iFilterLnCnt, iFilterColCnt, iLnStride, iColStride, iLnDilation, iColDilation);}
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        return _CONV Pool(setInput, iPoolType, true, set<feature>(), iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);
    }
    set<feature> BackProp(set<feature> &setGrad) {return _CONV Pool(setGrad, PoolUpType(iPoolType), false, setLayerInput, iLayerFilterLnCnt, iLayerFilterColCnt, iLayerLnStride, iLayerColStride, iLayerLnDilation, iLayerColDilation);}
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
    LayerBNConv(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, double dShift = 0, double dScale = 1, double dBNDominator = 1e-5) : Layer(0, bIsFirstLayer), dEpsilon(dBNDominator)
    {
        vecBeta = _CONV BNInitScaleShift(iChannCnt, dShift);
        vecGamma = _CONV BNInitScaleShift(iChannCnt, dScale);
    }
    set<feature> ForwProp(set<feature> &setInput)
    {
        if(bFirstLayer) setLayerInput = setInput;
        else setLayerInput = std::move(setInput);
        convbnData = _CONV BNTrain(setInput, vecBeta, vecGamma, true, dEpsilon);
        return convbnData.setY;
    }
    set<feature> BackProp(set<feature> &setGrad, double dLearnRate)
    {
        auto setGradBack = _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon);
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
    LayerBNConvAda(bool bIsFirstLayer = false, uint64_t iChannCnt = 1, double dDecayController = 0.95, double dAdaDominator = 1e-3, double dShift = 0, double dScale = 1, double dBNDominator = 1e-10) : LayerBNConv(bIsFirstLayer, iChannCnt, dShift, dScale, dBNDominator)
    {
        advBeta = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
        advGamma = _ADA AdaDeltaVect(iChannCnt, 1, dDecayController, dAdaDominator);
    }
    set<feature> BackProp(set<feature> &setGrad)
    {
        auto setGradBack = _CONV BNGradLossToInput(convbnData, setLayerInput, setGrad, vecGamma, dEpsilon);
        vecBeta = _CONV BNAdaDeltaUpdateScaleShift(vecBeta, _CONV BNGradLossToShift(setGrad), advBeta);
        vecGamma = _CONV BNAdaDeltaUpdateScaleShift(vecGamma, _CONV BNGradLossToScale(setGrad, convbnData), advGamma);
        return setGradBack;
    }
};

LAYER_END